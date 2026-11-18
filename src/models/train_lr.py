from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

KEY = "id_student"
# Bộ tiền xử lý của Trường cần code_module (one-hot) nên giữ lại ở đầu vào;
# id_student, code_presentation, nhãn và split bị bỏ.
DROP_COLUMNS = ["id_student", "code_presentation", "label_withdrawn", "split"]
TARGET = "label_withdrawn"


C_GRID = [0.01, 0.1, 1, 10, 100]


def build_model(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, penalty="l2", class_weight="balanced", max_iter=1000, random_state=42)


def load_split_data(feature_path, split_path) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    features = pd.read_csv(feature_path).drop(columns=["split"], errors="ignore")
    split = pd.read_csv(split_path)[[KEY, "split"]]
    merged = features.merge(split, on=KEY, how="inner")  # ghép theo id_student
    data = {}
    for name in ("train", "dev", "test"):
        part = merged[merged["split"] == name]
        data[name] = (part.drop(columns=DROP_COLUMNS), part[TARGET].astype(int))
    return data


def load_preprocessor(path="models/preprocessor.joblib"):
    """Trả về dict của Trường (impute_values + preprocessor_linear, fit trên train)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy bộ tiền xử lý tại {path}")
    return joblib.load(path)


EXCLUDED_OUTPUT = ("cnt_scaled__clicks_pre_start",)  # có thể gồm ngày âm, ngoài cửa sổ 0–27


def transform(pre, X):
    """Điền thiếu bằng giá trị đã lưu (từ train), mã hóa + chuẩn hóa, bỏ clicks_pre_start."""
    ct = pre["preprocessor_linear"]
    keep = [i for i, n in enumerate(ct.get_feature_names_out()) if n not in EXCLUDED_OUTPUT]
    return ct.transform(X.fillna(pre["impute_values"]))[:, keep]


def fit_and_score(model, X_train, y_train, X_eval) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]


def tune_c(Xt_train, y_train, Xt_dev, y_dev) -> pd.DataFrame:
    """Thử lưới C (phạt L2) trên dev. Chưa dùng ở tuần 3; để tuần 4."""
    from src.models.evaluate import metrics_positive

    rows = []
    for C in C_GRID:
        score = fit_and_score(build_model(C), Xt_train, y_train, Xt_dev)
        rows.append({"C": C, **metrics_positive(y_dev, score, 0.5)})
    return pd.DataFrame(rows)


def main(root=".") -> dict:
    from src.models.evaluate import append_metrics_row, metrics_positive

    root = Path(root)
    data = load_split_data(root / "data/processed/feature_28d.csv", root / "data/processed/split.csv")
    pre = load_preprocessor(root / "models/preprocessor.joblib")
    X_train, y_train = data["train"]
    X_dev, y_dev = data["dev"]
    Xt_train, Xt_dev = transform(pre, X_train), transform(pre, X_dev)
    score = fit_and_score(build_model(C=1.0), Xt_train, y_train, Xt_dev)
    m = metrics_positive(y_dev, score, 0.5)  # C=1, L2, ngưỡng 0,5; tinh chỉnh để tuần 4
    append_metrics_row(root / "results/model_metrics.csv", "logistic_regression_C1", "dev", m)
    return m


if __name__ == "__main__":
    print(main())
