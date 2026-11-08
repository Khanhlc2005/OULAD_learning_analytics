"""Đọc CSV mẫu và kiểm tra schema đầu vào của baseline RF/XGBoost."""

from pathlib import Path

import numpy as np
import pandas as pd

REGISTRATION_KEYS = ("id_student", "code_module", "code_presentation")
TARGET_COLUMN = "label_withdrawn"

# OULAD CONTRACT [AI HANDOFF]: chỉ đưa các cột hành vi được duyệt vào X.
# Khóa lượt đăng ký dùng để đối chiếu; label_withdrawn chỉ dùng làm y.
# Tạm loại clicks_pre_start vì có thể được tổng hợp từ ngày âm. Chỉ thêm
# lại nếu nhóm cập nhật đặc tả cửa sổ. Không thêm final_result,
# date_unregistration, label_* hoặc split vào danh sách feature.
BEHAVIOR_COLUMNS = (
    "total_clicks",
    "active_days",
    "clicks_per_active_day",
    "clicks_w1",
    "clicks_w2",
    "clicks_w3",
    "clicks_w4",
    "click_trend",
    "n_distinct_sites",
    "n_distinct_activity_types",
    "first_active_day",
    "last_active_day",
    "days_since_last_active",
    "longest_gap_days",
    "no_activity_flag",
    "n_assess_due",
    "n_assess_submitted",
    "assess_submit_rate",
    "avg_score",
    "weighted_score",
    "no_assess_flag",
)


def load_sample(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Trả về X, y và khóa lượt đăng ký theo cùng thứ tự dòng."""
    sample = pd.read_csv(path)
    required = (*REGISTRATION_KEYS, TARGET_COLUMN, *BEHAVIOR_COLUMNS)
    missing = [column for column in required if column not in sample.columns]
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(missing)}")

    keys = sample.loc[:, list(REGISTRATION_KEYS)].copy()
    if keys.isna().any().any() or keys.duplicated().any():
        raise ValueError("Khóa lượt đăng ký bị thiếu hoặc trùng.")

    y = pd.to_numeric(sample[TARGET_COLUMN], errors="raise")
    if set(y.unique()) != {0, 1}:
        raise ValueError("CSV mẫu phải có đủ hai lớp và chỉ dùng nhãn 0, 1.")

    # OULAD CONTRACT [AI HANDOFF]: CSV mẫu được kỳ vọng đã là số hữu hạn.
    # Khi chuyển sang dữ liệu thật, dùng tiền xử lý chung fit trên train.
    # Giữ nguyên thang đo cho cây và giữ khóa ngoài X sau khi ghép split.
    X = sample.loc[:, list(BEHAVIOR_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature có NaN hoặc giá trị vô hạn.")

    return X, y.astype(int), keys
