"""
Module: vle_features.py
Mo ta: Trich xuat dac trung hanh vi tuong tac tren he thong VLE trong cua so
       thoi gian [0, cutoff_day - 1] theo tung luot dang ky (id_student,
       code_module, code_presentation).
"""

import numpy as np
import pandas as pd

from config.config import CUTOFF_DAY

KEY_COLS = ["id_student", "code_module", "code_presentation"]

VLE_FEATURE_COLS = [
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
]


def compute_longest_gap(dates_series, cutoff: int = CUTOFF_DAY) -> int:
    """Tinh chuoi ngay lien tiep khong co tuong tac click trong khoang [0, cutoff - 1]."""
    active_dates = sorted(set(int(d) for d in dates_series if 0 <= d < cutoff))

    if not active_dates:
        return int(cutoff)

    gaps = [active_dates[0]]
    for i in range(len(active_dates) - 1):
        gaps.append(active_dates[i + 1] - active_dates[i] - 1)
    gaps.append((cutoff - 1) - active_dates[-1])

    return int(max(gaps))


def extract_vle_features(
    cohort_df: pd.DataFrame,
    student_vle_df: pd.DataFrame,
    vle_meta_df: pd.DataFrame,
    cutoff_day: int = CUTOFF_DAY,
) -> pd.DataFrame:
    """Ghep va tinh toan day du cac dac trung VLE cho cohort_df theo moc cutoff_day."""
    out_df = cohort_df.copy()

    # Loc tuong tac trong cua so [0, cutoff_day - 1] truoc de tiet kiem RAM
    vle_in_window = student_vle_df[
        (student_vle_df["date"] >= 0) & (student_vle_df["date"] < cutoff_day)
    ].copy()

    if len(vle_in_window) > 0 and vle_meta_df is not None and "activity_type" in vle_meta_df.columns:
        vle_in_window = pd.merge(
            vle_in_window,
            vle_meta_df[["id_site", "activity_type"]].drop_duplicates(subset=["id_site"]),
            on="id_site",
            how="left",
        )

    # Tong hop cac chi so co ban trong cua so
    summary = vle_in_window.groupby(KEY_COLS, observed=True).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        first_active_day=("date", "min"),
        last_active_day=("date", "max"),
        n_distinct_sites=("id_site", "nunique"),
        n_distinct_activity_types=("activity_type", "nunique"),
    ).reset_index()

    # Tinh clicks theo tung tuan (7 ngay/tuan)
    n_weeks = max(1, cutoff_day // 7)
    week_cols = []
    for w in range(1, n_weeks + 1):
        w_start = (w - 1) * 7
        w_end = min(w * 7 - 1, cutoff_day - 1)
        col_name = f"clicks_w{w}"
        week_cols.append(col_name)

        df_w = (
            vle_in_window[vle_in_window["date"].between(w_start, w_end)]
            .groupby(KEY_COLS, observed=True)["sum_click"]
            .sum()
            .rename(col_name)
            .reset_index()
        )
        summary = pd.merge(summary, df_w, on=KEY_COLS, how="left")

    # Tinh chuoi ngay nghi dai nhat
    gap_df = (
        vle_in_window.groupby(KEY_COLS, observed=True)["date"]
        .apply(lambda s: compute_longest_gap(s, cutoff=cutoff_day))
        .rename("longest_gap_days")
        .reset_index()
    )
    summary = pd.merge(summary, gap_df, on=KEY_COLS, how="left")

    # Tinh xu huong click_trend (chia doi cua so)
    half_day = cutoff_day // 2
    first_half = (
        vle_in_window[vle_in_window["date"] < half_day]
        .groupby(KEY_COLS, observed=True)["sum_click"]
        .sum()
        .rename("click_first_half")
        .reset_index()
    )
    summary = pd.merge(summary, first_half, on=KEY_COLS, how="left")

    # Ghep vao bang cohort chinh
    out_df = pd.merge(out_df, summary, on=KEY_COLS, how="left")

    out_df["total_clicks"] = out_df["total_clicks"].fillna(0).astype(np.int32)
    out_df["active_days"] = out_df["active_days"].fillna(0).astype(np.int16)
    out_df["n_distinct_sites"] = out_df["n_distinct_sites"].fillna(0).astype(np.int16)
    out_df["n_distinct_activity_types"] = out_df["n_distinct_activity_types"].fillna(0).astype(np.int16)

    for col in week_cols:
        out_df[col] = out_df[col].fillna(0).astype(np.int32)

    out_df["first_active_day"] = out_df["first_active_day"].fillna(cutoff_day).astype(np.int16)
    out_df["last_active_day"] = out_df["last_active_day"].fillna(-1).astype(np.int16)
    out_df["longest_gap_days"] = out_df["longest_gap_days"].fillna(cutoff_day).astype(np.int16)

    out_df["clicks_per_active_day"] = np.where(
        out_df["active_days"] > 0,
        out_df["total_clicks"] / out_df["active_days"],
        0.0,
    ).astype(np.float32)

    out_df["days_since_last_active"] = ((cutoff_day - 1) - out_df["last_active_day"]).astype(np.int16)

    first_half_vals = out_df["click_first_half"].fillna(0)
    second_half_vals = out_df["total_clicks"] - first_half_vals
    out_df["click_trend"] = ((second_half_vals - first_half_vals) / (out_df["total_clicks"] + 1)).astype(np.float32)
    out_df.drop(columns=["click_first_half"], inplace=True)

    out_df["no_activity_flag"] = (out_df["total_clicks"] == 0).astype(np.int8)

    return out_df