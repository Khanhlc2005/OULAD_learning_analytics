"""Locked final test, temporal transfer, and three-run RF/XGBoost benchmark."""

import json
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from sklearn.exceptions import InconsistentVersionWarning
import warnings

from src.models.baseline_rf_xgb_real import FEATURES, load_data
from src.models.cv_rf_xgb import CONFIGS, _make_model
from src.models.evaluate import metrics_positive
from src.models.rf_xgb_stage4 import INPUT_FILES, sha256_file
from src.preprocessing.pipeline import OULADPreprocessor, load_preprocessor


def verify_lock(dataset_dir, results_dir):
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    lock_path = results_dir / "rf_xgb_model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("raw_features") != FEATURES:
        raise ValueError("Schema của cấu hình khóa không khớp")
    if set(lock.get("models", {})) != set(CONFIGS):
        raise ValueError("Cấu hình khóa phải chứa RF và XGBoost")
    for family, item in lock["models"].items():
        if item["config"] not in CONFIGS[family] or item["parameters"] != CONFIGS[family][item["config"]]:
            raise ValueError(f"Tham số {family} khác cấu hình CV")
        if item["branch"] not in {"raw", "pca"} or not 0 < item["threshold"] < 1:
            raise ValueError(f"Nhánh/ngưỡng {family} không hợp lệ")
    for filename in INPUT_FILES:
        if sha256_file(dataset_dir / filename) != lock["input_sha256"].get(filename):
            raise ValueError(f"Dữ liệu/artifact {filename} đã đổi sau khi khóa")
    if sha256_file(results_dir / "rf_xgb_cv_selected.csv") != lock["cv_selected_sha256"]:
        raise ValueError("Kết quả CV đã đổi sau khi khóa")
    return lock, sha256_file(lock_path)


def _matrix(pre, frame):
    value = np.asarray(pre.transform(frame[FEATURES]), dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("Ma trận feature có NaN/Inf")
    return value


def _locked_matrix(dataset_dir, branch, frame):
    dataset_dir = Path(dataset_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        if branch == "raw":
            pre = load_preprocessor("tree", str(dataset_dir / "preprocessor.joblib"))
        else:
            import joblib
            pre = load_preprocessor("linear", str(dataset_dir / "preprocessor.joblib"))
            bundle = joblib.load(dataset_dir / "pca.joblib")
    x = _matrix(pre, frame)
    if branch == "raw":
        return x
    names = list(pre.get_feature_names_out())
    numeric = list(pre.get_numeric_feature_names_out())
    if bundle["columns"] != numeric or bundle["pca"].n_features_in_ != len(numeric):
        raise ValueError("PCA không khớp danh sách cột số đã khóa")
    indices = [names.index(name) for name in numeric]
    remaining = [i for i in range(len(names)) if i not in set(indices)]
    return np.column_stack([x[:, remaining], bundle["pca"].transform(x[:, indices])])


def run_test_once(dataset_dir, results_dir):
    """Fit on full train and write the fixed-threshold test metrics only once."""
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    lock, lock_hash = verify_lock(dataset_dir, results_dir)
    output = results_dir / "rf_xgb_test_metrics.csv"
    if output.exists():
        raise FileExistsError("Test đã được đánh giá; không chạy lại")
    data = load_data(dataset_dir)
    train = data.loc[data.split == "train"].reset_index(drop=True)
    test = data.loc[data.split == "test"].reset_index(drop=True)
    if set(train.id_student) & set(test.id_student):
        raise ValueError("Train/test có sinh viên trùng")
    y_train = train.label_withdrawn.astype(int).to_numpy()
    y_test = test.label_withdrawn.astype(int).to_numpy()
    rows = []
    for family, item in lock["models"].items():
        x_train = _locked_matrix(dataset_dir, item["branch"], train)
        x_test = _locked_matrix(dataset_dir, item["branch"], test)
        model = _make_model(family, item["parameters"], y_train)
        model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]
        rows.append({"model": family, "config": item["config"], "branch": item["branch"],
                     "split": "test", "lock_sha256": lock_hash,
                     "n_train": len(train), "n_test": len(test), "n_features": x_train.shape[1],
                     "test_positive_rate": float(y_test.mean()),
                     **metrics_positive(y_test, score, item["threshold"])})
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    return result


def run_temporal_once(dataset_dir, results_dir):
    """Train on 2013 train; evaluate on 2014 test with the locked raw settings."""
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    lock, lock_hash = verify_lock(dataset_dir, results_dir)
    if not (results_dir / "rf_xgb_test_metrics.csv").exists():
        raise ValueError("Chỉ chạy kiểm tra thời gian sau đánh giá test chính")
    output = results_dir / "rf_xgb_temporal_2013_to_2014.csv"
    if output.exists():
        raise FileExistsError("Kiểm tra thời gian đã chạy; không chạy lại")
    if any(item["branch"] != "raw" for item in lock["models"].values()):
        raise ValueError("Kiểm tra thời gian hiện hỗ trợ nhánh raw; PCA cần fit riêng trên 2013")
    data = load_data(dataset_dir)
    years = data.code_presentation.str[:4]
    train = data.loc[(data.split == "train") & (years == "2013")].reset_index(drop=True)
    test = data.loc[(data.split == "test") & (years == "2014")].reset_index(drop=True)
    if train.empty or test.empty or (set(train.id_student) & set(test.id_student)):
        raise ValueError("Tập 2013/2014 rỗng hoặc có sinh viên trùng")
    y_train = train.label_withdrawn.astype(int).to_numpy()
    y_test = test.label_withdrawn.astype(int).to_numpy()
    # Refit preprocessing on 2013 only. The shared full-train artifact saw 2014.
    pre = OULADPreprocessor(model_type="tree").fit(train[FEATURES])
    x_train, x_test = _matrix(pre, train), _matrix(pre, test)
    rows = []
    for family, item in lock["models"].items():
        model = _make_model(family, item["parameters"], y_train)
        model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]
        rows.append({"model": family, "config": item["config"], "branch": "raw",
                     "train_period": "2013", "eval_period": "2014", "eval_split": "test",
                     "lock_sha256": lock_hash, "n_train": len(train), "n_eval": len(test),
                     "n_features": x_train.shape[1], "eval_positive_rate": float(y_test.mean()),
                     **metrics_positive(y_test, score, item["threshold"])})
    result = pd.DataFrame(rows)
    with output.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    return result


def _measure(fn):
    """Wall time, process CPU time, and sampled peak process RSS."""
    process = psutil.Process()
    stop = threading.Event()
    peak = [process.memory_info().rss]

    def sample():
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    worker = threading.Thread(target=sample, daemon=True)
    before = process.cpu_times()
    start = perf_counter()
    worker.start()
    try:
        value = fn()
    finally:
        stop.set()
        worker.join()
    elapsed = perf_counter() - start
    after = process.cpu_times()
    peak[0] = max(peak[0], process.memory_info().rss)
    cpu_seconds = (after.user + after.system) - (before.user + before.system)
    return value, elapsed, cpu_seconds / elapsed * 100, peak[0] / 1024**2


def run_benchmark(dataset_dir, results_dir, repeats=3):
    """Benchmark each locked model on train/dev; do not inspect test labels."""
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    lock, lock_hash = verify_lock(dataset_dir, results_dir)
    if repeats < 3:
        raise ValueError("Benchmark cần ít nhất ba lượt")
    # Limit BLAS/OpenMP pools so CPU and throughput are comparable across models.
    with threadpool_limits(limits=1):
        rows = []
        for family, item in lock["models"].items():
            for repeat in range(1, repeats + 1):
                data, wall, cpu, rss = _measure(lambda: load_data(dataset_dir))
                train = data.loc[data.split == "train"].reset_index(drop=True)
                dev = data.loc[data.split == "dev"].reset_index(drop=True)
                rows.append((family, repeat, "read_validate", len(data), wall, cpu, rss))

                def transform():
                    return (_locked_matrix(dataset_dir, item["branch"], train),
                            _locked_matrix(dataset_dir, item["branch"], dev))

                (x_train, x_dev), wall, cpu, rss = _measure(transform)
                rows.append((family, repeat, "transform", len(train) + len(dev), wall, cpu, rss))
                y_train = train.label_withdrawn.astype(int).to_numpy()
                model = _make_model(family, item["parameters"], y_train)
                _, wall, cpu, rss = _measure(lambda: model.fit(x_train, y_train))
                rows.append((family, repeat, "fit", len(train), wall, cpu, rss))
                def infer_many():
                    for _ in range(20):
                        score = model.predict_proba(x_dev)[:, 1]
                    return score

                _, wall, cpu, rss = _measure(infer_many)
                rows.append((family, repeat, "infer", len(dev) * 20, wall, cpu, rss))
    runs = pd.DataFrame(rows, columns=["model", "repeat", "stage", "rows", "wall_seconds",
                                        "cpu_percent", "peak_rss_mb"])
    runs["calls"] = np.where(runs.stage == "infer", 20, 1)
    runs["throughput_rows_per_second"] = runs.rows / runs.wall_seconds
    runs["lock_sha256"] = lock_hash
    summary = runs.groupby(["model", "stage"], as_index=False).agg(
        runs=("repeat", "nunique"), rows=("rows", "first"), calls=("calls", "first"),
        wall_median_s=("wall_seconds", "median"), wall_min_s=("wall_seconds", "min"),
        wall_max_s=("wall_seconds", "max"),
        cpu_median_percent=("cpu_percent", "median"),
        peak_rss_median_mb=("peak_rss_mb", "median"),
        peak_rss_min_mb=("peak_rss_mb", "min"), peak_rss_max_mb=("peak_rss_mb", "max"),
        throughput_median_rows_s=("throughput_rows_per_second", "median"),
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(results_dir / "rf_xgb_benchmark_runs.csv", index=False)
    summary.to_csv(results_dir / "rf_xgb_benchmark_summary.csv", index=False)
    return runs, summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(run_test_once(root / "dataset", root / "results").to_string(index=False))
    print(run_temporal_once(root / "dataset", root / "results").to_string(index=False))
    _, summary = run_benchmark(root / "dataset", root / "results")
    print(summary.to_string(index=False))
