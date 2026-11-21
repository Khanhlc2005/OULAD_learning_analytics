"""
Module: assess_features.py
Mo ta: Trich xuat dac trung bai danh gia (assessments) co han nop < cutoff_day
       va bai nop hop le truoc cutoff_day (loai bo Exam va bai bao luu is_banked == 1).
"""

import numpy as np
import pandas as pd

from config.config import CUTOFF_DAY

KEY_COLS = ["id_student", "code_module", "code_presentation"]

ASSESS_FEATURE_COLS = [
    "n_assess_due",
    "n_assess_submitted",
    "assess_submit_rate",
    "avg_score",
    "weighted_score",
    "no_assess_flag",
]


def extract_assessment_features(
    cohort_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    student_assessment_df: pd.DataFrame,
    cutoff_day: int = CUTOFF_DAY,
) -> pd.DataFrame:
    """Trich xuat 6 dac trung bai danh gia theo moc cutoff_day va ghep vao cohort_df."""
    out_df = cohort_df.copy()

    # 1. Loc bai danh gia co han nop < cutoff_day va khong phai Exam
    valid_due = assessments_df[
        (assessments_df["assessment_type"] != "Exam")
        & (assessments_df["date"].notna())
        & (assessments_df["date"] < cutoff_day)
    ].copy()

    assess_due = (
        valid_due.groupby(["code_module", "code_presentation"], observed=True)["id_assessment"]
        .count()
        .rename("n_assess_due")
        .reset_index()
    )

    # 2. Loc bai nop hop le truoc cutoff_day, khong bao luu (is_banked == 0)
    valid_sub = pd.merge(
        student_assessment_df,
        valid_due[["id_assessment", "code_module", "code_presentation", "weight"]],
        on="id_assessment",
        how="inner",
    )
    valid_sub = valid_sub[
        (valid_sub["is_banked"] == 0) & (valid_sub["date_submitted"] < cutoff_day)
    ].copy()

    valid_sub["score_clean"] = valid_sub["score"].fillna(0.0)
    valid_sub["weight_clean"] = valid_sub["weight"].fillna(0.0)
    valid_sub["weighted_item"] = valid_sub["score_clean"] * valid_sub["weight_clean"]

    student_summary = valid_sub.groupby(KEY_COLS, observed=True).agg(
        n_assess_submitted=("id_assessment", "count"),
        avg_score=("score_clean", "mean"),
        sum_weighted=("weighted_item", "sum"),
    ).reset_index()
    student_summary["weighted_score"] = (student_summary["sum_weighted"] / 100.0).astype(np.float32)
    student_summary["avg_score"] = student_summary["avg_score"].astype(np.float32)
    student_summary.drop(columns=["sum_weighted"], inplace=True)

    # 3. Dong bo kieu str khi merge khoa mon/ky de tranh loi mismatch category
    for df_tmp in (out_df, assess_due, student_summary):
        df_tmp["code_module"] = df_tmp["code_module"].astype(str)
        df_tmp["code_presentation"] = df_tmp["code_presentation"].astype(str)

    out_df = pd.merge(
        out_df,
        assess_due[["code_module", "code_presentation", "n_assess_due"]],
        on=["code_module", "code_presentation"],
        how="left",
    )
    out_df["n_assess_due"] = out_df["n_assess_due"].fillna(0).astype(np.int16)

    out_df = pd.merge(out_df, student_summary, on=KEY_COLS, how="left")

    out_df["no_assess_flag"] = out_df["n_assess_submitted"].isna().astype(np.int8)
    out_df["n_assess_submitted"] = out_df["n_assess_submitted"].fillna(0).astype(np.int16)
    out_df["avg_score"] = out_df["avg_score"].fillna(0.0).astype(np.float32)
    out_df["weighted_score"] = out_df["weighted_score"].fillna(0.0).astype(np.float32)

    out_df["assess_submit_rate"] = np.where(
        out_df["n_assess_due"] > 0,
        np.clip(out_df["n_assess_submitted"] / out_df["n_assess_due"], 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    out_df["code_module"] = out_df["code_module"].astype("category")
    out_df["code_presentation"] = out_df["code_presentation"].astype("category")

    return out_df