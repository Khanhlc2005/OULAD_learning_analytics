"""Compare RF and XGBoost configurations using the saved train-only group folds."""

from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.baseline_rf_xgb_real import FEATURES, FORBIDDEN, load_data
from src.models.evaluate import metrics_positive
from src.preprocessing.pipeline import OULADPreprocessor


# Small, fixed search. Both baseline settings remain in the comparison.
CONFIGS = {
    "random_forest": {
        "rf_baseline_100": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        "rf_leaf5_200": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
        "rf_depth12_200": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2},
    },
    "xgboost": {
        "xgb_baseline_100": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        "xgb_depth3_200": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
        "xgb_depth4_200": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    },
}


def _make_model(family, params, y_train):
    if family == "random_forest":
        return RandomForestClassifier(
            **params, class_weight="balanced", random_state=42, n_jobs=1
        )
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    if not positives or not negatives:
        raise ValueError("Mỗi phần train cần đủ hai lớp để tính scale_pos_weight")
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negatives / positives,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )


def run_cv(dataset_dir, results_dir):
    """Fit on each fold-train, score fold-validation, and select by mean PR-AUC.

    The saved global preprocessor is deliberately not used here: its imputation
    and category vocabulary saw all of train, including each validation fold.
    """
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    data = load_data(dataset_dir)
    train = data.loc[data.split == "train"].reset_index(drop=True)
    fold_table = pd.read_csv(dataset_dir / "train_cv_folds.csv")
    if list(fold_table.columns) != ["id_student", "fold"]:
        raise ValueError("train_cv_folds.csv cần đúng cột id_student,fold")
    if fold_table.id_student.isna().any() or fold_table.id_student.duplicated().any():
        raise ValueError("Phân fold có sinh viên thiếu hoặc trùng")
    if set(fold_table.id_student) != set(train.id_student):
        raise ValueError("Phân fold không khớp tập train")
    if set(fold_table.fold) != set(range(1, 6)):
        raise ValueError("Cần đúng 5 fold đánh số 1–5")
    fold_by_student = fold_table.set_index("id_student").fold
    train_fold = train.id_student.map(fold_by_student)
    if train_fold.isna().any():
        raise ValueError("Có lượt đăng ký train chưa được phân fold")

    rows = []
    for fold in range(1, 6):
        is_val = train_fold.eq(fold).to_numpy()
        fold_train = train.loc[~is_val]
        fold_val = train.loc[is_val]
        if set(fold_train.id_student) & set(fold_val.id_student):
            raise ValueError(f"Fold {fold} có sinh viên xuất hiện ở hai phía")
        if fold_val.empty or fold_train.empty:
            raise ValueError(f"Fold {fold} rỗng")
        y_train = fold_train.label_withdrawn.astype(int).to_numpy()
        y_val = fold_val.label_withdrawn.astype(int).to_numpy()
        pre = OULADPreprocessor(model_type="tree").fit(fold_train[FEATURES])
        feature_names = list(pre.get_feature_names_out())
        if any(any(token in name for token in FORBIDDEN) for name in feature_names):
            raise ValueError("Preprocessor tạo feature cấm")
        columns = [f"f{i:03d}" for i in range(len(feature_names))]
        x_train = pd.DataFrame(pre.transform(fold_train[FEATURES]), columns=columns)
        x_val = pd.DataFrame(pre.transform(fold_val[FEATURES]), columns=columns)
        if not np.isfinite(x_train.to_numpy(dtype=float)).all() or not np.isfinite(x_val.to_numpy(dtype=float)).all():
            raise ValueError(f"Fold {fold} có NaN/Inf sau tiền xử lý")

        for family, configs in CONFIGS.items():
            for name, params in configs.items():
                model = _make_model(family, params, y_train)
                start = perf_counter()
                model.fit(x_train, y_train)
                fit_seconds = perf_counter() - start
                score = model.predict_proba(x_val)[:, 1]
                rows.append({
                    "model": family, "config": name, "fold": fold,
                    "train_rows": len(fold_train), "validation_rows": len(fold_val),
                    "n_features": len(feature_names), "fit_seconds": fit_seconds,
                    **metrics_positive(y_val, score, 0.5),
                })

    fold_results = pd.DataFrame(rows)
    if len(fold_results) != 30 or fold_results.groupby(["model", "config"]).fold.nunique().ne(5).any():
        raise ValueError("Thiếu kết quả của một cấu hình hoặc fold")
    summary = (
        fold_results.groupby(["model", "config"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"), f1_mean=("f1", "mean"),
            precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"), n_features_min=("n_features", "min"),
            n_features_max=("n_features", "max"),
        )
        .sort_values(["model", "pr_auc_mean", "f1_mean", "config"],
                     ascending=[True, False, False, True])
        .reset_index(drop=True)
    )
    summary["parameters"] = summary.apply(
        lambda row: json.dumps(CONFIGS[row["model"]][row["config"]], sort_keys=True), axis=1
    )
    selected = summary.groupby("model", sort=False).head(1).reset_index(drop=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    selected_path = results_dir / "rf_xgb_cv_selected.csv"
    if (results_dir / "rf_xgb_model_lock.json").exists():
        if not selected_path.exists():
            raise ValueError("Đã khóa mô hình nhưng thiếu tệp cấu hình CV")
        previous = pd.read_csv(selected_path)
        stable = [col for col in selected.columns if col != "fit_seconds_mean"]
        from pandas.testing import assert_frame_equal
        try:
            assert_frame_equal(previous[stable], selected[stable], check_exact=False, rtol=1e-10)
        except AssertionError as exc:
            raise ValueError("Kết quả CV khác bản đã khóa; không ghi đè cấu hình") from exc
    else:
        selected.to_csv(selected_path, index=False)
    fold_results.to_csv(results_dir / "rf_xgb_cv_folds.csv", index=False)
    summary.to_csv(results_dir / "rf_xgb_cv_summary.csv", index=False)
    return fold_results, summary, selected


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    _, summary, selected = run_cv(root / "dataset", root / "results")
    print(summary.to_string(index=False))
    print("\nSelected:\n", selected[["model", "config", "pr_auc_mean"]].to_string(index=False))
