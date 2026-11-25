"""Create reproducible student-grouped CV folds within the OULAD train split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build_folds(dataset_dir, results_dir, n_splits=5, random_state=42):
    """Save student fold assignments privately and a public aggregate summary.

    The splitter sees only train registrations. Each student belongs to exactly
    one validation fold, even if they have several registration rows.
    """
    dataset_dir = Path(dataset_dir)
    results_dir = Path(results_dir)
    data = pd.read_csv(
        dataset_dir / "feature_28d.csv",
        usecols=["id_student", "split", "label_withdrawn"],
    )
    manifest = pd.read_csv(dataset_dir / "split.csv", usecols=["id_student", "split"])
    if manifest.id_student.isna().any() or manifest.id_student.duplicated().any():
        raise ValueError("split.csv chứa id_student thiếu hoặc trùng")
    if manifest.split.isna().any() or not manifest.split.isin(["train", "dev", "test"]).all():
        raise ValueError("split.csv chứa split không hợp lệ")

    train = data.loc[data.split == "train"].reset_index(drop=True)
    train_manifest = manifest.loc[manifest.split == "train", "id_student"]
    if train.empty or train[["id_student", "label_withdrawn"]].isna().any().any():
        raise ValueError("Tập train rỗng hoặc thiếu id/nhãn")
    if set(train.id_student) != set(train_manifest):
        raise ValueError("id_student của train không khớp split.csv")
    if set(train.label_withdrawn.unique()) != {0, 1}:
        raise ValueError("Nhãn train phải có đủ hai lớp 0 và 1")
    if train.id_student.nunique() < n_splits:
        raise ValueError("Không đủ sinh viên để tạo số fold yêu cầu")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    y = train.label_withdrawn.astype(int).to_numpy()
    groups = train.id_student.to_numpy()
    validation_count = np.zeros(len(train), dtype=np.int8)
    assignments = []
    summary = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(train)), y, groups), start=1):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = len(train_groups & val_groups)
        if overlap or not ({0, 1} <= set(y[train_idx])) or not ({0, 1} <= set(y[val_idx])):
            raise ValueError(f"Fold {fold} chồng sinh viên hoặc thiếu lớp nhãn")
        validation_count[val_idx] += 1
        assignments.extend((student_id, fold) for student_id in sorted(val_groups))
        summary.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "train_students": len(train_groups),
            "validation_students": len(val_groups),
            "train_positive": int(y[train_idx].sum()),
            "validation_positive": int(y[val_idx].sum()),
            "train_positive_rate": float(y[train_idx].mean()),
            "validation_positive_rate": float(y[val_idx].mean()),
            "student_overlap": overlap,
        })

    fold_table = pd.DataFrame(assignments, columns=["id_student", "fold"])
    if not np.all(validation_count == 1):
        raise ValueError("Mỗi lượt đăng ký train phải nằm trong đúng một validation fold")
    if fold_table.id_student.duplicated().any() or set(fold_table.id_student) != set(train.id_student):
        raise ValueError("Mỗi sinh viên train phải có đúng một fold")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    fold_table.sort_values("id_student").to_csv(dataset_dir / "train_cv_folds.csv", index=False)
    fold_summary = pd.DataFrame(summary)
    fold_summary.to_csv(results_dir / "train_cv_fold_summary.csv", index=False)
    return fold_summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(build_folds(root / "dataset", root / "results").to_string(index=False))
