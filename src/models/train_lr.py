from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

KEYS = ["id_student", "code_module", "code_presentation"]
DROP_COLUMNS = ["id_student", "code_presentation", "label_withdrawn", "split"]
TARGET = "label_withdrawn"


def build_model() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)


def load_split_data(feature_path, split_path) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    features = pd.read_parquet(feature_path)
    split = pd.read_csv(split_path)
    merged = features.merge(split[KEYS + ["split"]], on=KEYS, how="inner")
    data = {}
    for name in ("train", "dev", "test"):
        part = merged[merged["split"] == name]
        data[name] = (part.drop(columns=DROP_COLUMNS), part[TARGET].astype(int))
    return data


def load_preprocessor(path="models/preprocessor.joblib"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy bộ tiền xử lý tại {path}. Hãy chờ người phụ trách tiền xử lý lưu file này trước khi chạy."
        )
    return joblib.load(path)


def fit_and_score(model, X_train, y_train, X_eval) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]
