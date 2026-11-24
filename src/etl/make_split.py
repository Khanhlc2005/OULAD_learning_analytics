"""
Module: make_split.py
Mo ta: Chia tap train/dev/test theo id_student (70/15/15) chong ro ri du lieu.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import RANDOM_STATE, SPLIT_RATIOS


def make_split(
    df: pd.DataFrame,
    ratios: tuple = SPLIT_RATIOS,
    seed: int = RANDOM_STATE,
    output_path=None,
) -> pd.DataFrame:
    """Chia tap train/dev/test theo id_student va tuy chon ghi ra file CSV."""
    _, dev_ratio, test_ratio = ratios
    val_test_total = round(dev_ratio + test_ratio, 6)
    test_share_of_val = test_ratio / val_test_total if val_test_total > 0 else 0.5

    student_labels = df.groupby("id_student")["label_withdrawn"].max().reset_index()

    train_ids, temp_ids = train_test_split(
        student_labels["id_student"],
        test_size=val_test_total,
        random_state=seed,
        stratify=student_labels["label_withdrawn"],
    )

    temp_labels = student_labels[student_labels["id_student"].isin(temp_ids)]["label_withdrawn"]
    dev_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_share_of_val,
        random_state=seed,
        stratify=temp_labels,
    )

    split_records = (
        [{"id_student": s_id, "split": "train"} for s_id in train_ids]
        + [{"id_student": s_id, "split": "dev"} for s_id in dev_ids]
        + [{"id_student": s_id, "split": "test"} for s_id in test_ids]
    )
    split_df = pd.DataFrame(split_records)

    if output_path is not None:
        out_dir = os.path.dirname(str(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        split_df.to_csv(output_path, index=False)

    return split_df