"""
Module: pipeline.py
Dự án: OULAD Big Data Project
Mô tả: Đóng gói toàn bộ quy trình điền khuyết (imputation), mã hóa (encoding)
       và chuẩn hóa (scaling) dữ liệu cho 2 nhóm mô hình:
       - Nhóm 1 (Linear/Clustering: LR, K-Means, PCA): Log1p + StandardScaler + OneHot/Ordinal
       - Nhóm 2 (Tree-based: Random Forest, XGBoost): Giữ nguyên số gốc + OneHot/Ordinal
Quy tắc: Fit CHỈ trên tập Train, sau đó transform cho Dev và Test.
"""

import os
import logging
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config.config import COLUMNS_YAML, PREPROCESSOR_PATH, MISSING_TOKEN

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. ĐỌC DANH SÁCH CỘT VÀ THỨ TỰ TỪ CONFIG/COLUMNS.YAML
# -----------------------------------------------------------------------------
def load_column_specs(yaml_path=COLUMNS_YAML) -> dict:
    """Đọc đặc tả cột từ config/columns.yaml và phân nhóm tự động theo dtype/role."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)

    cols_meta = spec['columns']
    orders = spec['ordinal_orders']

    # Chỉ lấy các cột có role là 'feature' hoặc 'context'
    active_cols = {
        col: meta for col, meta in cols_meta.items()
        if meta.get('role') in ('feature', 'context')
    }

    ordinal_cols = [c for c, m in active_cols.items() if m.get('dtype') == 'ordinal']
    nominal_cols = [c for c, m in active_cols.items() if m.get('dtype') == 'nominal']
    numeric_static_cols = [
        c for c, m in active_cols.items()
        if m.get('dtype') == 'numeric' and m.get('group') == 'demographic'
    ]
    count_cols = [c for c, m in active_cols.items() if m.get('dtype') == 'count']
    other_num_cols = [
        c for c, m in active_cols.items()
        if m.get('dtype') == 'numeric' and m.get('group') != 'demographic'
    ]
    flag_cols = [c for c, m in active_cols.items() if m.get('dtype') == 'flag']

    return {
        'raw_spec': spec,
        'ordinal_cols': ordinal_cols,
        'ordinal_categories': [orders[c] for c in ordinal_cols],
        'nominal_cols': nominal_cols,
        'numeric_static_cols': numeric_static_cols,
        'count_cols': count_cols,
        'other_num_cols': other_num_cols,
        'flag_cols': flag_cols,
        'key_cols': [c for c, m in cols_meta.items() if m.get('role') == 'key'],
        'label_cols': [c for c, m in cols_meta.items() if m.get('role') == 'label'],
        'excluded_cols': [c for c, m in cols_meta.items() if m.get('role') == 'excluded'],
    }


_COL_SPECS = load_column_specs()

ORDINAL_COLS = _COL_SPECS['ordinal_cols']
ORDINAL_CATEGORIES = _COL_SPECS['ordinal_categories']
NOMINAL_COLS = _COL_SPECS['nominal_cols']
NUMERIC_STATIC_COLS = _COL_SPECS['numeric_static_cols']
COUNT_COLS = _COL_SPECS['count_cols']
OTHER_NUM_COLS = _COL_SPECS['other_num_cols']
FLAG_COLS = _COL_SPECS['flag_cols']


# -----------------------------------------------------------------------------
# 2. HÀM LÀM SẠCH TOKEN THIẾU, FIT THAM SỐ VÀ ĐIỀN KHUYẾT (IMPUTATION)
# -----------------------------------------------------------------------------
def _clean_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các chuỗi đại diện giá trị thiếu ('?', khoảng trắng) về np.nan."""
    df_clean = df.copy()
    missing_tokens = [MISSING_TOKEN, ' ', '', 'NaN', 'null', 'None']
    for col in NOMINAL_COLS + ORDINAL_COLS + NUMERIC_STATIC_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(missing_tokens, np.nan)
    return df_clean


def get_impute_values(X_train: pd.DataFrame) -> dict:
    """Tính median (số) và mode (phân loại) chỉ trên tập Train sau khi chuẩn hóa '?'."""
    X_clean = _clean_missing_tokens(X_train)
    impute_dict = {}
    for col in NUMERIC_STATIC_COLS:
        if col in X_clean.columns:
            impute_dict[col] = pd.to_numeric(X_clean[col], errors='coerce').median()
    for col in NOMINAL_COLS + ORDINAL_COLS:
        if col in X_clean.columns:
            impute_dict[col] = X_clean[col].mode()[0]
    return impute_dict


def apply_imputation(df: pd.DataFrame, impute_dict: dict) -> pd.DataFrame:
    """Áp dụng các giá trị điền khuyết đã tính từ Train và cập nhật cờ thiếu."""
    df_out = _clean_missing_tokens(df)

    # Cập nhật chuẩn xác cờ thiếu trước khi điền giá trị (xử lý triệt để cả '?' và NaN)
    if 'imd_band' in df_out.columns:
        df_out['imd_missing'] = df_out['imd_band'].isna().astype(np.int8)
    if 'date_registration' in df_out.columns:
        df_out['date_registration'] = pd.to_numeric(df_out['date_registration'], errors='coerce')
        df_out['date_registration_missing'] = df_out['date_registration'].isna().astype(np.int8)

    # Điền median / mode từ tập Train
    for col, val in impute_dict.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(val)

    # Các biến đếm, tỷ lệ và cờ điền 0
    zero_fill_cols = COUNT_COLS + OTHER_NUM_COLS + FLAG_COLS
    for col in zero_fill_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(0)

    return df_out


# -----------------------------------------------------------------------------
# 3. HÀM TẠO PIPELINE / COLUMNTRANSFORMER VÀ ESTIMATOR CHUẨN SCIKIT-LEARN
# -----------------------------------------------------------------------------
def build_preprocessors():
    """Tạo cấu trúc ColumnTransformer cho cả 2 nhánh mô hình."""
    num_log_scaler = Pipeline([
        ('log1p', FunctionTransformer(np.log1p, validate=True, feature_names_out='one-to-one')),
        ('scaler', StandardScaler())
    ])
    num_standard_scaler = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Lưu ý: OneHotEncoder(drop='first', handle_unknown='ignore') sẽ mã hóa giá trị lạ
    # thành vector toàn số 0 (giống với category đầu tiên bị drop).
    preprocessor_linear = ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(categories=ORDINAL_CATEGORIES,
                               handle_unknown='use_encoded_value', unknown_value=-1), ORDINAL_COLS),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), NOMINAL_COLS),
        ('cnt_scaled', num_log_scaler, COUNT_COLS),
        ('num_scaled', num_standard_scaler, NUMERIC_STATIC_COLS + OTHER_NUM_COLS),
        ('flags', 'passthrough', FLAG_COLS)
    ])

    preprocessor_tree = ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(categories=ORDINAL_CATEGORIES,
                               handle_unknown='use_encoded_value', unknown_value=-1), ORDINAL_COLS),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), NOMINAL_COLS),
        ('num_raw', 'passthrough', COUNT_COLS + NUMERIC_STATIC_COLS + OTHER_NUM_COLS),
        ('flags', 'passthrough', FLAG_COLS)
    ])

    return preprocessor_linear, preprocessor_tree


class OULADPreprocessor(BaseEstimator, TransformerMixin):
    """
    Wrapper chuẩn Scikit-learn đóng gói toàn bộ bước:
    Làm sạch token thiếu -> Điền khuyết (fit trên train) -> Mã hóa & Chuẩn hóa.
    Hỗ trợ gọi trực tiếp .fit(X), .transform(X), .get_feature_names_out().
    """
    def __init__(self, model_type: str = 'linear'):
        if model_type not in ('linear', 'tree'):
            raise ValueError("model_type phải là 'linear' hoặc 'tree'")
        self.model_type = model_type
        self.impute_values_ = {}
        self.transformer_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.impute_values_ = get_impute_values(X)
        X_imp = apply_imputation(X, self.impute_values_)
        prep_linear, prep_tree = build_preprocessors()
        self.transformer_ = prep_linear if self.model_type == 'linear' else prep_tree
        self.transformer_.fit(X_imp)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_imp = apply_imputation(X, self.impute_values_)
        return self.transformer_.transform(X_imp)

    def get_feature_names_out(self, input_features=None):
        return self.transformer_.get_feature_names_out(input_features)

    def get_numeric_feature_names_out(self, input_features=None) -> np.ndarray:
        """Trả về danh sách tên các cột số đã chuẩn hóa (cnt_scaled__* và num_scaled__*)."""
        if self.model_type != 'linear':
            raise ValueError("Chỉ hỗ trợ trích xuất cột số đã scale với model_type='linear'")
        all_names = self.get_feature_names_out(input_features)
        return np.array([
            name for name in all_names
            if name.startswith(('cnt_scaled__', 'num_scaled__'))
        ])

    def transform_numeric(self, X: pd.DataFrame) -> np.ndarray:
        """
        Trích xuất riêng ma trận các cột số đã chuẩn hóa (cnt_scaled__* và num_scaled__*)
        để dùng riêng cho thuật toán PCA (không bao gồm biến phân loại và cờ nhị phân).
        """
        if self.model_type != 'linear':
            raise ValueError("Chỉ hỗ trợ trích xuất cột số đã scale với model_type='linear'")
        X_full = self.transform(X)
        all_names = self.get_feature_names_out()
        num_indices = [
            idx for idx, name in enumerate(all_names)
            if name.startswith(('cnt_scaled__', 'num_scaled__'))
        ]
        return X_full[:, num_indices]


# -----------------------------------------------------------------------------
# 4. HÀM TIỆN ÍCH FIT, SAVE & LOAD CHO TOÀN BỘ QUY TRÌNH
# -----------------------------------------------------------------------------
_PIPELINE_CACHE = {}


def fit_and_save_pipeline(X_train: pd.DataFrame, output_path: str = str(PREPROCESSOR_PATH)) -> dict:
    """
    Fit 2 đối tượng OULADPreprocessor ('linear' và 'tree') trên X_train và lưu ra file joblib.
    """
    output_path = str(output_path)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prep_linear = OULADPreprocessor(model_type='linear').fit(X_train)
    prep_tree = OULADPreprocessor(model_type='tree').fit(X_train)

    bundle = {
        'linear': prep_linear,
        'tree': prep_tree,
        'impute_values': prep_linear.impute_values_,
        'preprocessor_linear': prep_linear.transformer_,
        'preprocessor_tree': prep_tree.transformer_,
        'feature_names_linear': prep_linear.get_feature_names_out(),
        'feature_names_tree': prep_tree.get_feature_names_out()
    }

    joblib.dump(bundle, output_path)
    _PIPELINE_CACHE[output_path] = bundle
    logger.info("Đã lưu bundle tiền xử lý thành công tại: %s", output_path)
    return bundle


def load_preprocessor(model_type: str = 'linear', model_path: str = str(PREPROCESSOR_PATH)) -> OULADPreprocessor:
    """
    Trả về trực tiếp một Scikit-learn Estimator (OULADPreprocessor) có sẵn .transform()
    và .get_feature_names_out() để khớp với giao diện gọi trong train_lr.py / train_rf.py.
    """
    model_path = str(model_path)
    if model_path not in _PIPELINE_CACHE:
        loaded = joblib.load(model_path)
        if isinstance(loaded, OULADPreprocessor):
            return loaded
        _PIPELINE_CACHE[model_path] = loaded

    bundle = _PIPELINE_CACHE[model_path]
    if model_type in bundle:
        return bundle[model_type]
    raise ValueError("model_type phải là 'linear' hoặc 'tree'")


def load_and_transform(df: pd.DataFrame, model_type: str = 'linear', model_path: str = str(PREPROCESSOR_PATH)):
    """
    Hàm tiện ích nạp pipeline và transform trực tiếp df thành mảng numpy + danh sách tên cột.
    """
    prep = load_preprocessor(model_type=model_type, model_path=model_path)
    return prep.transform(df), prep.get_feature_names_out()


def load_and_transform_numeric(df: pd.DataFrame, model_path: str = str(PREPROCESSOR_PATH)):
    """
    Hàm tiện ích dành riêng cho PCA: nạp pipeline 'linear' và chỉ trả về
    ma trận các cột số đã scale (cnt_scaled__* và num_scaled__*) kèm tên cột.
    """
    prep = load_preprocessor(model_type='linear', model_path=model_path)
    return prep.transform_numeric(df), prep.get_numeric_feature_names_out()