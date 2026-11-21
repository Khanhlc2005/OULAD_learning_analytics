"""
Module: cohort_labels.py
Mô tả: Tạo bảng enrollment gốc, tách nhóm sinh viên rút lui sớm trước cutoff_day
       và lập Cohort chính thức kèm nhãn mục tiêu label_withdrawn.
"""

import os
import numpy as np
import pandas as pd

KEY_COLS = ['id_student', 'code_module', 'code_presentation']


def build_cohort_and_labels(
    student_info_df: pd.DataFrame,
    student_reg_df: pd.DataFrame,
    cutoff_day: int = 28,
    early_output_path: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ghép studentInfo và studentRegistration, tách nhóm rút sớm và tạo nhãn mục tiêu.
    
    Returns:
        cohort_df: DataFrame chứa các lượt đăng ký còn học tính đến đầu ngày cutoff_day.
        excluded_early_df: DataFrame chứa các lượt rút trước ngày cutoff_day.
    """
    # 1. Ghép bảng bằng đủ bộ 3 khóa chính
    enrollment_raw = pd.merge(
        student_info_df,
        student_reg_df[KEY_COLS + ['date_registration', 'date_unregistration']],
        on=KEY_COLS,
        how='inner'
    )

    # 2. Lọc nhóm rút lui sớm trước cutoff_day
    early_mask = (
        enrollment_raw['date_unregistration'].notna() &
        (enrollment_raw['date_unregistration'] < cutoff_day)
    )
    excluded_early_df = enrollment_raw[early_mask].copy()

    # Loại bỏ cột nhãn tương lai (final_result) để tránh rò rỉ,
    # nhưng GIỮ LẠI date_unregistration phục vụ EDA thống kê rút lui theo ngày (lỗi #8)
    if 'final_result' in excluded_early_df.columns:
        excluded_early_df.drop(columns=['final_result'], inplace=True)

    # Xuất file cục bộ nếu có đường dẫn
    if early_output_path:
        os.makedirs(os.path.dirname(early_output_path), exist_ok=True)
        excluded_early_df.to_csv(early_output_path, index=False)

    # 3. Lọc Cohort chính thức (date_unregistration trống HOẶC >= cutoff_day)
    cohort_mask = (
        enrollment_raw['date_unregistration'].isna() |
        (enrollment_raw['date_unregistration'] >= cutoff_day)
    )
    cohort_df = enrollment_raw[cohort_mask].copy()

    # 4. Tạo nhãn nhị phân: 1 nếu Withdrawn, 0 cho các trường hợp còn lại
    cohort_df['label_withdrawn'] = (cohort_df['final_result'] == 'Withdrawn').astype(np.int8)

    return cohort_df, excluded_early_df