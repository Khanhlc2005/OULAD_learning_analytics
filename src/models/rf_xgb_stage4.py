"""RF/XGBoost dev experiments: PCA, ablation, threshold, and configuration lock."""

import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from src.models.baseline_rf_xgb_real import FEATURES, FORBIDDEN, load_data
from src.models.cv_rf_xgb import CONFIGS, _make_model
from src.models.evaluate import metrics_positive
from src.preprocessing.pipeline import load_column_specs, load_preprocessor


INPUT_FILES = ("feature_28d.csv", "split.csv", "preprocessor.joblib", "pca.joblib")
THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_inputs(dataset_dir):
    """Validate fitted artifacts and return raw/PCA representations."""
    dataset_dir = Path(dataset_dir)
    data = load_data(dataset_dir)
    train = data.loc[data.split == "train"].reset_index(drop=True)
    dev = data.loc[data.split == "dev"].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        tree = load_preprocessor("tree", str(dataset_dir / "preprocessor.joblib"))
        linear = load_preprocessor("linear", str(dataset_dir / "preprocessor.joblib"))
        pca_bundle = joblib.load(dataset_dir / "pca.joblib")
    if set(pca_bundle) != {"pca", "columns"}:
        raise ValueError("pca.joblib cần chứa pca và columns")
    pca = pca_bundle["pca"]
    numeric_names = list(linear.get_numeric_feature_names_out())
    if list(pca_bundle["columns"]) != numeric_names:
        raise ValueError("Thứ tự cột số của PCA không khớp preprocessor linear")
    if pca.n_samples_ != len(train) or pca.n_features_in_ != len(numeric_names):
        raise ValueError("PCA không được fit trên đúng train hoặc sai số feature")

    tree_names = list(tree.get_feature_names_out())
    linear_names = list(linear.get_feature_names_out())
    if any(any(token in name for token in FORBIDDEN) for name in tree_names + linear_names):
        raise ValueError("Feature sau biến đổi chứa cột cấm")
    if len(tree_names) != 49 or len(linear_names) != 49:
        raise ValueError("Schema đã fit không có 49 feature như mong đợi")

    arrays = {}
    for split_name, frame in (("train", train), ("dev", dev)):
        raw = np.asarray(tree.transform(frame[FEATURES]), dtype=float)
        linear_all = np.asarray(linear.transform(frame[FEATURES]), dtype=float)
        numeric_indices = [linear_names.index(name) for name in numeric_names]
        other_indices = [i for i in range(len(linear_names)) if i not in set(numeric_indices)]
        numeric = linear_all[:, numeric_indices]
        pca_matrix = np.column_stack([linear_all[:, other_indices], pca.transform(numeric)])
        if not np.isfinite(raw).all() or not np.isfinite(pca_matrix).all():
            raise ValueError(f"{split_name} có NaN/Inf sau biến đổi")
        arrays[split_name] = {"raw": raw, "pca": pca_matrix}
    pca_names = [linear_names[i] for i in other_indices] + [f"PC{i+1}" for i in range(pca.n_components_)]
    return train, dev, arrays, tree_names, pca_names


def raw_feature_name(encoded_name):
    prefix, name = encoded_name.split("__", 1)
    if prefix == "nom":
        matches = [f for f in FEATURES if name.startswith(f + "_")]
    else:
        matches = [name] if name in FEATURES else []
    if len(matches) != 1:
        raise ValueError(f"Không xác định được cột nguồn: {encoded_name}")
    return matches[0]


def ablation_indices(tree_names):
    metadata = load_column_specs()["raw_spec"]["columns"]
    groups = {feature: metadata[feature]["group"] for feature in FEATURES}
    if set(groups.values()) != {"context", "demographic", "behavior", "assessment"}:
        raise ValueError("Nhóm feature không khớp đặc tả ablation")
    source = [raw_feature_name(name) for name in tree_names]
    plans = {
        "full_original": set(groups.values()),
        "demographic_only": {"context", "demographic"},
        "behavior_only": {"context", "behavior", "assessment"},
    }
    indices = {label: [i for i, feature in enumerate(source) if groups[feature] in allowed]
               for label, allowed in plans.items()}
    if len(indices["full_original"]) != len(tree_names) or not all(indices.values()):
        raise ValueError("Ablation bỏ sót feature")
    return indices


def _score_branch(model_name, config, x_train, y_train, x_dev, y_dev):
    model = _make_model(model_name, CONFIGS[model_name][config], y_train)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_dev)[:, 1]
    return model, probability, metrics_positive(y_dev, probability, 0.5)


def run_dev(dataset_dir, results_dir, cv_selected_path):
    """Use train for fitting and dev for branch/threshold selection; write a lock."""
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    train, dev, arrays, raw_names, pca_names = load_locked_inputs(dataset_dir)
    selected_cv = pd.read_csv(cv_selected_path)
    if len(selected_cv) != 2 or set(selected_cv.model) != set(CONFIGS):
        raise ValueError("Cần một cấu hình CV cho mỗi mô hình")
    chosen_configs = dict(zip(selected_cv.model, selected_cv.config))
    for family, name in chosen_configs.items():
        if name not in CONFIGS[family]:
            raise ValueError(f"Cấu hình CV không được định nghĩa: {name}")
    y_train = train.label_withdrawn.astype(int).to_numpy()
    y_dev = dev.label_withdrawn.astype(int).to_numpy()
    ablation = ablation_indices(raw_names)
    options = []
    cached = {}
    for family, config in chosen_configs.items():
        # Raw and PCA use identical train/dev rows and the same model parameters.
        for branch in ("raw", "pca"):
            model, prob, metric = _score_branch(
                family, config, arrays["train"][branch], y_train,
                arrays["dev"][branch], y_dev
            )
            cached[(family, branch)] = (model, prob)
            options.append({"model": family, "config": config, "experiment": "representation",
                            "branch": branch, "n_features": arrays["train"][branch].shape[1],
                            **metric})
        for group, idx in ablation.items():
            if group == "full_original":
                metric = metrics_positive(y_dev, cached[(family, "raw")][1], 0.5)
            else:
                _, _, metric = _score_branch(
                    family, config, arrays["train"]["raw"][:, idx], y_train,
                    arrays["dev"]["raw"][:, idx], y_dev
                )
            options.append({"model": family, "config": config, "experiment": "ablation",
                            "branch": group, "n_features": len(idx), **metric})

    option_table = pd.DataFrame(options)
    threshold_rows, final_rows, lock_models = [], [], {}
    for family, config in chosen_configs.items():
        candidates = option_table.loc[(option_table.model == family) &
                                      (option_table.experiment == "representation")]
        branch = candidates.sort_values(["pr_auc", "branch"], ascending=[False, True]).iloc[0].branch
        model, prob = cached[(family, branch)]
        threshold_table = pd.DataFrame([
            {"model": family, "config": config, "branch": branch,
             **metrics_positive(y_dev, prob, float(t))} for t in THRESHOLDS
        ])
        threshold_rows.append(threshold_table)
        best = threshold_table.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
        final_rows.append({"model": family, "config": config, "branch": branch,
                           "n_train": len(train), "n_dev": len(dev),
                           "n_features": arrays["train"][branch].shape[1],
                           **{col: best[col] for col in ["threshold", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]}})
        lock_models[family] = {"config": config, "parameters": CONFIGS[family][config],
                               "branch": branch, "threshold": float(best.threshold),
                               "selection": "CV mean PR-AUC -> dev PR-AUC branch -> dev F1 threshold"}

    final = pd.DataFrame(final_rows)
    threshold_table = pd.concat(threshold_rows, ignore_index=True)
    lock = {
        "schema_version": 1,
        "cutoff_day": 28,
        "seed": 42,
        "raw_features": FEATURES,
        "models": lock_models,
        "input_sha256": {name: sha256_file(dataset_dir / name) for name in INPUT_FILES},
        "cv_selected_sha256": sha256_file(cv_selected_path),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    lock_path = results_dir / "rf_xgb_model_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Cấu hình khóa đã tồn tại và khác kết quả mới")
    elif (results_dir / "rf_xgb_test_metrics.csv").exists():
        raise ValueError("Đã có kết quả test nhưng chưa có cấu hình khóa")
    option_table.to_csv(results_dir / "rf_xgb_dev_options.csv", index=False)
    option_table.loc[option_table.experiment == "ablation"].to_csv(
        results_dir / "rf_xgb_ablation_dev.csv", index=False
    )
    threshold_table.to_csv(results_dir / "rf_xgb_dev_thresholds.csv", index=False)
    final.to_csv(results_dir / "rf_xgb_dev_selected.csv", index=False)
    if not lock_path.exists():
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return option_table, threshold_table, final, lock


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    _, _, selected, _ = run_dev(root / "dataset", root / "results", root / "results/rf_xgb_cv_selected.csv")
    print(selected.to_string(index=False))
