"""Huấn luyện baseline XGBoost cho lớp Withdrawn."""

from pandas import DataFrame, Series
from xgboost import XGBClassifier


def train_xgb(X_train: DataFrame, y_train: Series) -> XGBClassifier:
    """Fit cấu hình baseline cố định trên dữ liệu train được truyền vào."""
    n_negative = int(y_train.eq(0).sum())
    n_positive = int(y_train.eq(1).sum())
    if not n_negative or not n_positive or n_negative + n_positive != len(y_train):
        raise ValueError("XGBoost cần đủ hai lớp với nhãn 0 và 1.")

    # OULAD CONTRACT [AI HANDOFF]: tính trọng số chỉ từ y_train.
    # Baseline mẫu chạy số vòng cố định. Khi bổ sung early stopping cho
    # dữ liệu thật, dùng dev của split chung làm eval_set.
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=n_negative / n_positive,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )
    return model.fit(X_train, y_train)
