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
from src.etl.load_raw import load_raw_tables
from src.etl.cohort_labels import build_cohort_and_labels
from src.etl.features_vle import build_vle_features, VLE_FEATURE_COLS
from src.etl.features_assessment import build_assessment_features, ASSESS_FEATURE_COLS
from src.etl.make_split import make_split

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


def build_feature_table(
    cutoff_day: int = CUTOFF_DAY,
    cohort_cutoff: int = None,
    split_path: Path = None,
    output_dir: Path = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Chay tron bo ETL cho moc cutoff_day va tra ve bang dac trung hoan chinh."""
    tables = load_raw_tables()

    effective_cohort_cutoff = cohort_cutoff if cohort_cutoff is not None else cutoff_day

    cohort_df, excluded_df = build_cohort_and_labels(
        tables["studentInfo"],
        tables["studentRegistration"],
        cutoff_day=effective_cohort_cutoff,
    )

    cohort_df = build_vle_features(
        cohort_df,
        tables["studentVle"],
        tables["vle"],
        cutoff_day=cutoff_day,
    )

    cohort_df = build_assessment_features(
        cohort_df,
        tables["assessments"],
        tables["studentAssessment"],
        cutoff_day=cutoff_day,
    )

    # Cap nhat co thieu nhan khau hoc
    cohort_df["imd_missing"] = cohort_df["imd_band"].isna().astype("int8")
    cohort_df["date_registration_missing"] = cohort_df["date_registration"].isna().astype("int8")

    # Dung chung file split neu da co, nguoc lai tao moi theo id_student
    if split_path is not None and Path(split_path).exists():
        split_df = pd.read_csv(split_path)
    else:
        split_df = make_split(
            cohort_df,
            ratios=SPLIT_RATIOS,
            seed=RANDOM_STATE,
            output_path=split_path,
        )
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
        out_dir = Path(output_dir) if output_dir else Path(PROCESSED_DATA_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        if output_dir is not None:
            csv_path = out_dir / f"feature_{cutoff_day}d.csv"
            parquet_path = out_dir / f"feature_{cutoff_day}d.parquet"
            out_split_path = Path(split_path) if split_path else (out_dir / "split.csv")
            excluded_path = out_dir / f"excluded_early_withdrawn_{effective_cohort_cutoff}d.csv"
        elif cutoff_day == CUTOFF_DAY:
            csv_path = Path(FEATURE_PATH)
            parquet_path = csv_path.with_suffix(".parquet")
            out_split_path = Path(split_path) if split_path else Path(SPLIT_PATH)
            excluded_path = Path(EXCLUDED_PATH)
        else:
            csv_path = out_dir / f"feature_{cutoff_day}d.csv"
            parquet_path = out_dir / f"feature_{cutoff_day}d.parquet"
            out_split_path = Path(split_path) if split_path else (out_dir / f"split_{cutoff_day}d.csv")
            excluded_path = out_dir / f"excluded_early_withdrawn_{cutoff_day}d.csv"

        full_df.to_csv(csv_path, index=False)
        try:
            full_df.to_parquet(parquet_path, index=False)
        except Exception:
            pass

        split_df.to_csv(out_split_path, index=False)

        excl_clean = excluded_df.drop(
            columns=[c for c in ["label_withdrawn"] if c in excluded_df.columns]
        )
        excl_clean.to_csv(excluded_path, index=False)

        if cutoff_day == CUTOFF_DAY and output_dir is None:
            export_feature_dictionary(list(full_df.columns))

        print(f"[OK] Da xuat {csv_path} ({full_df.shape[0]:,} dong x {full_df.shape[1]} cot).")

    return full_df


def main():
    parser = argparse.ArgumentParser(description="Chay pipeline ETL tao bang dac trung OULAD.")
    parser.add_argument(
        "--cutoff",
        type=int,
        default=CUTOFF_DAY,
        help="Moc ngay cat du lieu dac trung (mac dinh lay tu config.CUTOFF_DAY).",
    )
    parser.add_argument(
        "--cohort-cutoff",
        type=int,
        default=None,
        help="Moc ngay loc cohort co dinh (vi du 56 khi chay do nhay cua so).",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Duong dan file split.csv dung chung (neu da co se doc lai, chua co se tao moi).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thu muc luu dau ra rieng (tranh ghi de len data/processed/feature_28d.csv).",
    )
    args = parser.parse_args()
    build_feature_table(
        cutoff_day=args.cutoff,
        cohort_cutoff=args.cohort_cutoff,
        split_path=Path(args.split_path) if args.split_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        save_outputs=True,
    )


if __name__ == "__main__":
    main()