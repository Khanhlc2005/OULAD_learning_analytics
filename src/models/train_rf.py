"""Huấn luyện baseline Random Forest cho lớp Withdrawn."""

from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier


def train_rf(X_train: DataFrame, y_train: Series) -> RandomForestClassifier:
    """Fit cấu hình baseline cố định trên dữ liệu train được truyền vào."""
    # OULAD CONTRACT [AI HANDOFF]: dùng class_weight thay cho SMOTE.
    # Với dữ liệu thật, caller phải truyền train từ split chung theo id_student.
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    return model.fit(X_train, y_train)
