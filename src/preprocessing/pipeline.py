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
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# 1. KHAI BÁO CÁC NHÓM CỘT ĐẶC TRƯNG CHUẨN
# -----------------------------------------------------------------------------
ORDINAL_COLS = ['highest_education', 'imd_band', 'age_band']
EDUCATION_ORDER = [
    'No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 
    'HE Qualification', 'Post Graduate Qualification'
]
IMD_ORDER = [
    '0-10%', '10-20', '20-30%', '30-40%', '40-50%', 
    '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'
]
AGE_ORDER = ['0-35', '35-55', '55<=']

NOMINAL_COLS = ['code_module', 'gender', 'region', 'disability']

NUMERIC_STATIC_COLS = ['num_of_prev_attempts', 'studied_credits', 'date_registration']

COUNT_COLS = [
    'clicks_pre_start', 'total_clicks', 'active_days', 'clicks_per_active_day',
    'clicks_w1', 'clicks_w2', 'clicks_w3', 'clicks_w4', 'n_distinct_sites',
    'n_distinct_activity_types', 'n_assess_due', 'n_assess_submitted'
]

OTHER_NUM_COLS = [
    'click_trend', 'first_active_day', 'last_active_day', 
    'days_since_last_active', 'longest_gap_days',
    'assess_submit_rate', 'avg_score', 'weighted_score'
]

FLAG_COLS = ['imd_missing', 'date_registration_missing', 'no_activity_flag', 'no_assess_flag']


# -----------------------------------------------------------------------------
# 2. HÀM FIT THAM SỐ VÀ ĐIỀN KHUYẾT (IMPUTATION)
# -----------------------------------------------------------------------------
def get_impute_values(X_train: pd.DataFrame) -> dict:
    """Tính median (số) và mode (phân loại) chỉ trên tập Train."""
    impute_dict = {}
    for col in NUMERIC_STATIC_COLS:
        impute_dict[col] = X_train[col].median()
    for col in NOMINAL_COLS + ORDINAL_COLS:
        impute_dict[col] = X_train[col].mode()[0]
    return impute_dict


def apply_imputation(df: pd.DataFrame, impute_dict: dict) -> pd.DataFrame:
    """Áp dụng các giá trị điền khuyết đã tính từ Train cho bất kỳ tập dữ liệu nào."""
    df_out = df.copy()
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
# 3. HÀM TẠO PIPELINE / COLUMNTRANSFORMER
# -----------------------------------------------------------------------------
def build_preprocessors():
    """Tạo cấu trúc ColumnTransformer cho cả 2 nhánh mô hình."""
    # Pipeline biến đổi log1p + standard scaling
    num_log_scaler = Pipeline([
        ('log1p', FunctionTransformer(np.log1p, validate=True, feature_names_out='one-to-one')),
        ('scaler', StandardScaler())
    ])
    num_standard_scaler = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Nhánh 1: Linear / Clustering / PCA (Được chuẩn hóa & log1p)
    preprocessor_linear = ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(categories=[EDUCATION_ORDER, IMD_ORDER, AGE_ORDER], 
                               handle_unknown='use_encoded_value', unknown_value=-1), ORDINAL_COLS),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), NOMINAL_COLS),
        ('cnt_scaled', num_log_scaler, COUNT_COLS),
        ('num_scaled', num_standard_scaler, NUMERIC_STATIC_COLS + OTHER_NUM_COLS),
        ('flags', 'passthrough', FLAG_COLS)
    ])

    # Nhánh 2: Tree-based (RF, XGBoost - Giữ nguyên phân phối số gốc)
    preprocessor_tree = ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(categories=[EDUCATION_ORDER, IMD_ORDER, AGE_ORDER], 
                               handle_unknown='use_encoded_value', unknown_value=-1), ORDINAL_COLS),
        ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), NOMINAL_COLS),
        ('num_raw', 'passthrough', COUNT_COLS + NUMERIC_STATIC_COLS + OTHER_NUM_COLS),
        ('flags', 'passthrough', FLAG_COLS)
    ])

    return preprocessor_linear, preprocessor_tree


# -----------------------------------------------------------------------------
# 4. HÀM TIỆN ÍCH FIT & SAVE CHO TOÀN BỘ QUY TRÌNH
# -----------------------------------------------------------------------------
def fit_and_save_pipeline(X_train: pd.DataFrame, output_path: str = "models/preprocessor.joblib") -> dict:
    """
    Fit toàn bộ imputer và 2 pipeline trên X_train, sau đó lưu ra file joblib.
    Trả về bundle chứa các transformer đã fit sẵn.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Tính toán giá trị điền khuyết
    impute_values = get_impute_values(X_train)
    X_train_imp = apply_imputation(X_train, impute_values)

    # 2. Xây dựng và fit pipeline
    prep_linear, prep_tree = build_preprocessors()
    prep_linear.fit(X_train_imp)
    prep_tree.fit(X_train_imp)

    # 3. Lấy tên đặc trưng sau biến đổi
    try:
        feat_linear = prep_linear.get_feature_names_out()
    except Exception:
        feat_linear = [f"f_{i}" for i in range(prep_linear.transform(X_train_imp).shape[1])]

    try:
        feat_tree = prep_tree.get_feature_names_out()
    except Exception:
        feat_tree = [f"f_{i}" for i in range(prep_tree.transform(X_train_imp).shape[1])]

    bundle = {
        'impute_values': impute_values,
        'preprocessor_linear': prep_linear,
        'preprocessor_tree': prep_tree,
        'feature_names_linear': feat_linear,
        'feature_names_tree': feat_tree
    }

    joblib.dump(bundle, output_path)
    print(f"Đã lưu bundle tiền xử lý thành công tại: {output_path}")
    return bundle


def load_and_transform(df: pd.DataFrame, model_type: str = 'linear', model_path: str = "models/preprocessor.joblib"):
    """
    Hàm tiện ích cho các thành viên khác nạp pipeline và transform trực tiếp df:
    - model_type='linear': trả về ma trận đã scale (cho LR, K-Means, PCA)
    - model_type='tree': trả về ma trận số gốc (cho RF, XGBoost)
    """
    bundle = joblib.load(model_path)
    df_imp = apply_imputation(df, bundle['impute_values'])

    if model_type == 'linear':
        X_out = bundle['preprocessor_linear'].transform(df_imp)
        feature_names = bundle['feature_names_linear']
    elif model_type == 'tree':
        X_out = bundle['preprocessor_tree'].transform(df_imp)
        feature_names = bundle['feature_names_tree']
    else:
        raise ValueError("model_type phải là 'linear' hoặc 'tree'")

    return X_out, feature_names