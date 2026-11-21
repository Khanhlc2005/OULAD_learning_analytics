"""
Module: build_feature_table.py
Mo ta: Dieu phoi toan bo quy trinh ETL tu du lieu tho den bang dac trung dau ra,
       chia tap train/dev/test theo id_student, sinh feature_dictionary.csv tu
       config/columns.yaml va ho tro chay theo tung moc cutoff_day.
"""

import argparse
import os
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    COLUMNS_YAML,
    CUTOFF_DAY,
    EXCLUDED_PATH,
    FEATURE_DICT_PATH,
    FEATURE_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SPLIT_PATH,
    SPLIT_RATIOS,
)
from src.etl.load_raw import load_all_raw_tables
from src.etl.cohort_labels import build_cohort_and_labels
from src.etl.vle_features import extract_vle_features, VLE_FEATURE_COLS
from src.etl.assess_features import extract_assessment_features, ASSESS_FEATURE_COLS

KEY_COLS = ["id_student", "code_module", "code_presentation"]

DEMO_COLS = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "date_registration",
]

DEMO_FLAGS = ["imd_missing", "date_registration_missing"]

LEAKAGE_COLS = ["final_result", "date_unregistration"]


def export_feature_dictionary(
    feature_cols: list = None,
    yaml_path=COLUMNS_YAML,
    output_path=FEATURE_DICT_PATH,
) -> pd.DataFrame:
    """
    Sinh tu dong file data/processed/feature_dictionary.csv tu config/columns.yaml.
    Doi chieu truc tiep voi danh sach cot thuc te cua bang dac trung de dam bao
    moi cot co dung 1 dong mo ta, khong thua, khong thieu.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    cols_meta = spec.get("columns", {})

    if feature_cols is None:
        if os.path.exists(output_path):
            feature_cols = list(pd.read_csv(output_path, nrows=0).columns)
        elif os.path.exists(FEATURE_PATH):
            feature_cols = list(pd.read_csv(FEATURE_PATH, nrows=0).columns)
        else:
            feature_cols = [c for c in cols_meta.keys() if c != "split"]

    rows = []
    for col in feature_cols:
        if col not in cols_meta:
            raise KeyError(f"Cot '{col}' co trong du lieu nhung chua duoc khai bao trong {yaml_path}")

        meta = cols_meta[col]
        used_by_list = meta.get("used_by", [])
        used_by_str = (
            ";".join(str(m) for m in used_by_list)
            if isinstance(used_by_list, list)
            else str(used_by_list)
        )

        description = str(meta.get("description", "")).strip()
        source = str(meta.get("source", "")).strip()

        assert description, f"LOI: Cot '{col}' bi trong truong description trong columns.yaml!"
        assert source, f"LOI: Cot '{col}' bi trong truong source trong columns.yaml!"

        rows.append({
            "column": col,
            "group": meta.get("group", ""),
            "role": meta.get("role", ""),
            "dtype": meta.get("dtype", ""),
            "description": description,
            "source": source,
            "window": meta.get("window", ""),
            "formula": str(meta.get("formula", "")).strip(),
            "missing_rule": str(meta.get("missing", "")).strip(),
            "used_by": used_by_str,
        })

    dict_df = pd.DataFrame(rows)

    out_dir = os.path.dirname(str(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dict_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Da sinh thanh cong {output_path} ({len(dict_df)} cot mo ta).")
    return dict_df


def create_student_split(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Chia tap train/dev/test theo id_student (70/15/15) chong ro ri du lieu."""
    _, dev_ratio, test_ratio = SPLIT_RATIOS
    val_test_total = round(dev_ratio + test_ratio, 6)
    test_share_of_val = test_ratio / val_test_total if val_test_total > 0 else 0.5

    student_labels = df.groupby("id_student")["label_withdrawn"].max().reset_index()

    train_ids, temp_ids = train_test_split(
        student_labels["id_student"],
        test_size=val_test_total,
        random_state=random_state,
        stratify=student_labels["label_withdrawn"],
    )

    temp_labels = student_labels[student_labels["id_student"].isin(temp_ids)]["label_withdrawn"]
    dev_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_share_of_val,
        random_state=random_state,
        stratify=temp_labels,
    )

    split_records = []
    for s_id in train_ids:
        split_records.append({"id_student": s_id, "split": "train"})
    for s_id in dev_ids:
        split_records.append({"id_student": s_id, "split": "dev"})
    for s_id in test_ids:
        split_records.append({"id_student": s_id, "split": "test"})

    return pd.DataFrame(split_records)


def build_feature_table(cutoff_day: int = CUTOFF_DAY, save_outputs: bool = True) -> pd.DataFrame:
    """Chay tron bo ETL cho moc cutoff_day va tra ve bang dac trung hoan chinh."""
    tables = load_all_raw_tables()

    cohort_df, excluded_df = build_cohort_and_labels(
        tables["studentInfo"],
        tables["studentRegistration"],
        cutoff_day=cutoff_day,
    )

    cohort_df = extract_vle_features(
        cohort_df,
        tables["studentVle"],
        tables["vle"],
        cutoff_day=cutoff_day,
    )

    cohort_df = extract_assessment_features(
        cohort_df,
        tables["assessments"],
        tables["studentAssessment"],
        cutoff_day=cutoff_day,
    )

    # Cap nhat co thieu nhan khau hoc
    cohort_df["imd_missing"] = cohort_df["imd_band"].isna().astype("int8")
    cohort_df["date_registration_missing"] = cohort_df["date_registration"].isna().astype("int8")

    # Tao bang split theo id_student va ghep vao bang dac trung
    split_df = create_student_split(cohort_df, random_state=RANDOM_STATE)
    full_df = pd.merge(cohort_df, split_df, on="id_student", how="left")

    # Loai bo cot ro ri neu con ton tai
    cols_to_drop = [c for c in LEAKAGE_COLS if c in full_df.columns]
    if cols_to_drop:
        full_df.drop(columns=cols_to_drop, inplace=True)

    ordered_cols = KEY_COLS + ["label_withdrawn", "split"] + DEMO_COLS + DEMO_FLAGS + VLE_FEATURE_COLS + ASSESS_FEATURE_COLS
    available_ordered = [c for c in ordered_cols if c in full_df.columns]
    extra_cols = [c for c in full_df.columns if c not in available_ordered]
    full_df = full_df[available_ordered + extra_cols]

    if save_outputs:
        out_dir = Path(PROCESSED_DATA_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        if cutoff_day == CUTOFF_DAY:
            csv_path = Path(FEATURE_PATH)
            parquet_path = csv_path.with_suffix(".parquet")
            split_path = Path(SPLIT_PATH)
            excluded_path = Path(EXCLUDED_PATH)
        else:
            csv_path = out_dir / f"feature_{cutoff_day}d.csv"
            parquet_path = out_dir / f"feature_{cutoff_day}d.parquet"
            split_path = out_dir / f"split_{cutoff_day}d.csv"
            excluded_path = out_dir / f"excluded_early_withdrawn_{cutoff_day}d.csv"

        full_df.to_csv(csv_path, index=False)
        try:
            full_df.to_parquet(parquet_path, index=False)
        except Exception:
            pass

        split_df.to_csv(split_path, index=False)

        excl_clean = excluded_df.drop(
            columns=[c for c in ["label_withdrawn"] if c in excluded_df.columns]
        )
        excl_clean.to_csv(excluded_path, index=False)

        if cutoff_day == CUTOFF_DAY:
            export_feature_dictionary(list(full_df.columns))

        print(f"[OK] Da xuat {csv_path} ({full_df.shape[0]:,} dong x {full_df.shape[1]} cot).")

    return full_df


def main():
    parser = argparse.ArgumentParser(description="Chay pipeline ETL tao bang dac trung OULAD.")
    parser.add_argument(
        "--cutoff",
        type=int,
        default=CUTOFF_DAY,
        help="Moc ngay cat du lieu (mac dinh lay tu config.CUTOFF_DAY).",
    )
    args = parser.parse_args()
    build_feature_table(cutoff_day=args.cutoff, save_outputs=True)


if __name__ == "__main__":
    main()