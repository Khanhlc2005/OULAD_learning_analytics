"""Lock LR on current train/dev, then compare one final test with locked RF/XGB."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from src.models.baseline_rf_xgb_real import FEATURES, FORBIDDEN, load_data
from src.models.evaluate import metrics_positive
from src.models.rf_xgb_stage4 import sha256_file
from src.models.rf_xgb_stage5 import verify_lock
from src.models.train_lr import C_GRID, build_model
from src.preprocessing.pipeline import load_preprocessor


METRICS = ["threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
INPUTS = ("feature_28d.csv", "split.csv", "preprocessor.joblib")
THRESHOLDS = np.round(np.arange(0.1, 0.9001, 0.05), 2)


def _data_and_preprocessor(dataset_dir):
    data = load_data(dataset_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        pre = load_preprocessor("linear", str(Path(dataset_dir) / "preprocessor.joblib"))
    names = list(pre.get_feature_names_out())
    if len(names) != 49 or any(any(token in name for token in FORBIDDEN) for name in names):
        raise ValueError("LR cần đúng 49 feature hợp lệ từ 33 cột gốc")
    return data, pre


def _transform(pre, frame):
    values = np.asarray(pre.transform(frame[FEATURES]), dtype=float)
    if values.shape[1] != 49 or not np.isfinite(values).all():
        raise ValueError("Ma trận LR không đúng 49 feature hoặc có NaN/Inf")
    return values


def select_lr_on_dev(dataset_dir, results_dir):
    """Replicate LR's C-grid/F1-at-0.5 and dev threshold grid; lock before test."""
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    data, pre = _data_and_preprocessor(dataset_dir)
    train = data.loc[data.split == "train"].reset_index(drop=True)
    dev = data.loc[data.split == "dev"].reset_index(drop=True)
    if set(train.id_student) & set(dev.id_student):
        raise ValueError("Train/dev có sinh viên trùng")
    x_train, x_dev = _transform(pre, train), _transform(pre, dev)
    y_train = train.label_withdrawn.astype(int).to_numpy()
    y_dev = dev.label_withdrawn.astype(int).to_numpy()
    rows = []
    for c in C_GRID:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            model = build_model(C=float(c)).fit(x_train, y_train)
        probability = model.predict_proba(x_dev)[:, 1]
        rows.append({"C": float(c), "n_train": len(train), "n_dev": len(dev),
                     "n_features": x_train.shape[1], **metrics_positive(y_dev, probability, 0.5)})
    c_table = pd.DataFrame(rows)
    best_c = float(c_table.sort_values(["f1", "C"], ascending=[False, True]).iloc[0].C)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        best_model = build_model(C=best_c).fit(x_train, y_train)
    probability = best_model.predict_proba(x_dev)[:, 1]
    threshold_table = pd.DataFrame([{"C": best_c, **metrics_positive(y_dev, probability, float(t))}
                                    for t in THRESHOLDS])
    selected = threshold_table.sort_values(["f1", "threshold"], ascending=[False, True]).head(1).copy()
    selected.insert(0, "model", "logistic_regression")
    selected.insert(2, "n_train", len(train))
    selected.insert(3, "n_dev", len(dev))
    selected.insert(4, "n_features", x_train.shape[1])
    results_dir.mkdir(parents=True, exist_ok=True)
    lock = {
        "schema_version": 1,
        "model": "logistic_regression",
        "C": best_c,
        "threshold": float(selected.threshold.iloc[0]),
        "raw_features": FEATURES,
        "n_transformed_features": 49,
        "selection": "dev F1 at threshold 0.5 for C; dev F1 for threshold 0.10..0.90 step 0.05",
        "library_versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
        "input_sha256": {name: sha256_file(dataset_dir / name) for name in INPUTS},
    }
    lock_path = results_dir / "lr_model_lock_current.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Khóa LR hiện có khác lựa chọn dev mới")
    elif (results_dir / "lr_test_metrics_current.csv").exists():
        raise ValueError("Đã có kết quả test LR nhưng thiếu khóa")
    c_table.to_csv(results_dir / "lr_dev_c_grid_current.csv", index=False)
    threshold_table.to_csv(results_dir / "lr_dev_thresholds_current.csv", index=False)
    selected.to_csv(results_dir / "lr_dev_selected_current.csv", index=False)
    if not lock_path.exists():
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return c_table, threshold_table, selected, lock


def verify_lr_lock(dataset_dir, results_dir):
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    path = results_dir / "lr_model_lock_current.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("model") != "logistic_regression":
        raise ValueError("Khóa LR không hợp lệ")
    if lock.get("raw_features") != FEATURES or lock.get("n_transformed_features") != 49:
        raise ValueError("Feature LR khác hợp đồng chung")
    if lock.get("C") not in C_GRID or lock.get("threshold") not in THRESHOLDS:
        raise ValueError("C/ngưỡng LR không nằm trong lưới đã chọn")
    for name in INPUTS:
        if sha256_file(dataset_dir / name) != lock["input_sha256"].get(name):
            raise ValueError(f"Dữ liệu/artifact {name} đã đổi sau khi khóa LR")
    rf_lock, _ = verify_lock(dataset_dir, results_dir)
    if any(rf_lock["input_sha256"][name] != lock["input_sha256"][name] for name in INPUTS):
        raise ValueError("LR và RF/XGBoost không cùng dữ liệu/split/preprocessor")
    return lock, sha256_file(path)


def evaluate_lr_test_once(dataset_dir, results_dir):
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    lock, lock_hash = verify_lr_lock(dataset_dir, results_dir)
    output = results_dir / "lr_test_metrics_current.csv"
    if output.exists():
        raise FileExistsError("LR test đã được đánh giá; không chạy lại")
    data, pre = _data_and_preprocessor(dataset_dir)
    train = data.loc[data.split == "train"].reset_index(drop=True)
    test = data.loc[data.split == "test"].reset_index(drop=True)
    if set(train.id_student) & set(test.id_student):
        raise ValueError("Train/test có sinh viên trùng")
    x_train, x_test = _transform(pre, train), _transform(pre, test)
    y_train = train.label_withdrawn.astype(int).to_numpy()
    y_test = test.label_withdrawn.astype(int).to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        model = build_model(C=lock["C"]).fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    result = pd.DataFrame([{"model": "logistic_regression", "config": f"C={lock['C']:g}",
                            "branch": "linear", "split": "test", "lock_sha256": lock_hash,
                            "n_train": len(train), "n_test": len(test), "n_features": x_train.shape[1],
                            "test_positive_rate": float(y_test.mean()),
                            **metrics_positive(y_test, score, lock["threshold"])}])
    with output.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    return result


def build_test_comparison(dataset_dir, results_dir):
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    _, lr_hash = verify_lr_lock(dataset_dir, results_dir)
    _, rf_hash = verify_lock(dataset_dir, results_dir)
    lr = pd.read_csv(results_dir / "lr_test_metrics_current.csv")
    trees = pd.read_csv(results_dir / "rf_xgb_test_metrics.csv")
    if len(lr) != 1 or len(trees) != 2:
        raise ValueError("Cần đúng một dòng LR và hai dòng RF/XGBoost")
    if not lr.lock_sha256.eq(lr_hash).all() or not trees.lock_sha256.eq(rf_hash).all():
        raise ValueError("Metric test không khớp các cấu hình khóa")
    comparison = pd.concat([lr, trees], ignore_index=True)
    if comparison.n_test.nunique() != 1 or comparison.n_train.nunique() != 1:
        raise ValueError("Ba mô hình không dùng cùng số lượt train/test")
    if comparison.test_positive_rate.nunique() != 1 or set(comparison.model) != {"logistic_regression", "random_forest", "xgboost"}:
        raise ValueError("Ba mô hình không dùng cùng nhãn hoặc thiếu mô hình")
    comparison.to_csv(results_dir / "model_comparison_test.csv", index=False)
    return comparison


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(select_lr_on_dev(root / "dataset", root / "results")[2].to_string(index=False))
    print(evaluate_lr_test_once(root / "dataset", root / "results").to_string(index=False))
    print(build_test_comparison(root / "dataset", root / "results").to_string(index=False))
