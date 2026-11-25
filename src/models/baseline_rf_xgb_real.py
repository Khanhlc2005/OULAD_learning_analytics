"""RF/XGBoost baseline on the shared OULAD day-28 feature table.

Only train and dev are read into modeling arrays. Test is reserved for the final gate.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.evaluate import metrics_positive
from src.models.train_rf import train_rf
from src.models.train_xgb import train_xgb

KEYS = ["id_student", "code_module", "code_presentation"]
CATEGORICAL = [
    "code_module", "gender", "region", "highest_education", "imd_band",
    "age_band", "disability",
]
NUMERIC = [
    "num_of_prev_attempts", "studied_credits", "date_registration",
    "imd_missing", "date_registration_missing", "total_clicks", "active_days",
    "clicks_per_active_day", "clicks_w1", "clicks_w2", "clicks_w3", "clicks_w4",
    "click_trend", "n_distinct_sites", "n_distinct_activity_types",
    "first_active_day", "last_active_day", "days_since_last_active",
    "longest_gap_days", "no_activity_flag", "n_assess_due",
    "n_assess_submitted", "assess_submit_rate", "avg_score",
    "weighted_score", "no_assess_flag",
]
FEATURES = CATEGORICAL + NUMERIC
FORBIDDEN = {"final_result", "date_unregistration", "label_withdrawn", "split", "id_student", "code_presentation", "clicks_pre_start"}


def load_data(dataset_dir):
    root = Path(dataset_dir)
    features = pd.read_csv(root / "feature_28d.csv")
    manifest = pd.read_csv(root / "split.csv")
    required = set(FEATURES + KEYS + ["label_withdrawn", "split"])
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature table thiếu cột: {sorted(missing)}")
    unexpected = set(features.columns).difference(required)
    if unexpected:
        raise ValueError(f"Feature table có cột chưa duyệt: {sorted(unexpected)}")
    if len(features) != len(features.drop_duplicates(KEYS)) or features[KEYS].isna().any().any():
        raise ValueError("Khóa lượt đăng ký bị thiếu hoặc trùng")
    if manifest.id_student.duplicated().any() or manifest[["id_student", "split"]].isna().any().any():
        raise ValueError("Split manifest bị thiếu hoặc trùng sinh viên")
    if set(manifest.split) != {"train", "dev", "test"}:
        raise ValueError("Split phải gồm train, dev, test")
    merged = features.merge(manifest[["id_student", "split"]], on="id_student", how="left",
                            suffixes=("", "_manifest"), validate="many_to_one")
    if len(merged) != len(features) or merged.split_manifest.isna().any() or (merged.split != merged.split_manifest).any():
        raise ValueError("Feature table và manifest không khớp")
    if set(features.label_withdrawn.dropna().unique()) != {0, 1} or features.label_withdrawn.isna().any():
        raise ValueError("Nhãn phải là 0/1 và không thiếu")
    if any(c in FORBIDDEN for c in FEATURES):
        raise ValueError("Danh sách feature chứa cột cấm")
    return features


def build_preprocessor():
    # Fit only on train. Identical raw information for RF and XGBoost.
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ], sparse_threshold=0)


def run(dataset_dir, results_dir):
    data = load_data(dataset_dir)
    train = data.loc[data.split == "train"]
    dev = data.loc[data.split == "dev"]
    pre = build_preprocessor()
    train_values = pre.fit_transform(train[FEATURES])
    encoded_names = list(pre.get_feature_names_out())
    # XGBoost rejects brackets in names such as imd_band_[0-10).
    safe_names = [f"f{i:03d}" for i in range(train_values.shape[1])]
    x_train = pd.DataFrame(train_values, columns=safe_names)
    x_dev = pd.DataFrame(pre.transform(dev[FEATURES]), columns=safe_names)
    y_train = train.label_withdrawn.astype(int).reset_index(drop=True)
    y_dev = dev.label_withdrawn.astype(int).reset_index(drop=True)
    if not np.isfinite(x_train.to_numpy(dtype=float)).all() or not np.isfinite(x_dev.to_numpy(dtype=float)).all():
        raise ValueError("Feature sau tiền xử lý có NaN/Inf")
    rows, importances = [], []
    for name, fit in [("random_forest", train_rf), ("xgboost", train_xgb)]:
        model = fit(x_train, y_train)
        probability = model.predict_proba(x_dev)[:, 1]
        rows.append({"model": name, "split": "dev", "n_train": len(train), "n_dev": len(dev),
                     "n_features": x_train.shape[1], **metrics_positive(y_dev, probability, 0.5)})
        importances.extend({"model": name, "feature": col, "importance": float(value)}
                           for col, value in zip(encoded_names, model.feature_importances_))
    metrics = pd.DataFrame(rows)
    importance = pd.DataFrame(importances).sort_values(["model", "importance"], ascending=[True, False])
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "rf_xgb_baseline_dev.csv", index=False)
    importance.to_csv(output / "rf_xgb_feature_importance.csv", index=False)
    return metrics, importance, FEATURES
