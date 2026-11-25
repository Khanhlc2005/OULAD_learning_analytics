"""Generate train-only importance and a reproducible RF/XGBoost handoff manifest."""

import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import threadpoolctl
import xgboost
import yaml
from sklearn.exceptions import InconsistentVersionWarning

from src.models.baseline_rf_xgb_real import FEATURES, FORBIDDEN, load_data
from src.models.cv_rf_xgb import _make_model
from src.models.rf_xgb_stage4 import raw_feature_name, sha256_file
from src.models.rf_xgb_stage5 import verify_lock
from src.preprocessing.pipeline import load_preprocessor


def generate_handoff(dataset_dir, results_dir):
    """Fit locked trees on train, save importance and environment metadata.

    No dev/test labels or predictions are used to calculate importance.
    """
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    lock, lock_hash = verify_lock(dataset_dir, results_dir)
    data = load_data(dataset_dir)
    train = data.loc[data.split == "train"].reset_index(drop=True)
    y_train = train.label_withdrawn.astype(int).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        pre = load_preprocessor("tree", str(dataset_dir / "preprocessor.joblib"))
    names = list(pre.get_feature_names_out())
    if len(names) != 49 or any(any(token in name for token in FORBIDDEN) for name in names):
        raise ValueError("Schema feature tree không hợp lệ")
    matrix = np.asarray(pre.transform(train[FEATURES]), dtype=float)
    if matrix.shape != (len(train), len(names)) or not np.isfinite(matrix).all():
        raise ValueError("Ma trận train không khớp tên feature hoặc có NaN/Inf")

    rows = []
    for family, item in lock["models"].items():
        if item["branch"] != "raw":
            raise ValueError("Importance này chỉ áp dụng cho nhánh raw đã khóa")
        model = _make_model(family, item["parameters"], y_train)
        model.fit(matrix, y_train)
        weights = np.asarray(model.feature_importances_, dtype=float)
        if weights.shape != (len(names),) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Feature importance không hợp lệ")
        weights /= weights.sum()
        method = "mean_decrease_impurity" if family == "random_forest" else "xgboost_gain_default"
        rows.extend({
            "model": family,
            "config": item["config"],
            "branch": item["branch"],
            "encoded_feature": name,
            "raw_feature": raw_feature_name(name),
            "importance": float(value),
            "method": method,
            "lock_sha256": lock_hash,
        } for name, value in zip(names, weights))
    encoded = pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])
    raw = (encoded.groupby(["model", "config", "branch", "raw_feature", "method", "lock_sha256"],
                           as_index=False).importance.sum()
           .sort_values(["model", "importance"], ascending=[True, False]))
    if encoded.groupby("model").size().ne(49).any() or raw.groupby("model").size().ne(len(FEATURES)).any():
        raise ValueError("Số feature importance khác hợp đồng")

    manifest = {
        "purpose": "RF/XGBoost day-28 OULAD handoff",
        "label": "label_withdrawn=1: Withdrawn after day 28; 0: Pass/Fail/Distinction",
        "cutoff_day": 28,
        "n_train": len(train),
        "n_dev": int(data.split.eq("dev").sum()),
        "n_test": int(data.split.eq("test").sum()),
        "raw_feature_count": len(FEATURES),
        "tree_feature_count": len(names),
        "selected_models": lock["models"],
        "rf_xgb_lock_sha256": lock_hash,
        "input_sha256": lock["input_sha256"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_logical_count": psutil.cpu_count(),
        "memory_total_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
            "PyYAML": yaml.__version__,
            "psutil": psutil.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "result_sha256": {
            name: sha256_file(results_dir / name)
            for name in ("rf_xgb_test_metrics.csv", "rf_xgb_temporal_2013_to_2014.csv",
                         "rf_xgb_benchmark_runs.csv", "rf_xgb_benchmark_summary.csv")
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(results_dir / "rf_xgb_locked_importance_encoded.csv", index=False)
    raw.to_csv(results_dir / "rf_xgb_locked_importance_raw.csv", index=False)
    (results_dir / "rf_xgb_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return encoded, raw, manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    _, importance, manifest = generate_handoff(root / "dataset", root / "results")
    print(importance.groupby("model", group_keys=False).head(10).to_string(index=False))
    print(json.dumps(manifest["libraries"], indent=2))
