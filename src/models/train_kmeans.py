from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.models.train_lr import load_preprocessor, load_split_data, transform

NUMERIC_PREFIXES = ("cnt_scaled__", "num_scaled__")  # cột số liên tục/đếm, bỏ cờ 0/1
STUDENT_INFO = "data/raw/studentInfo.csv"
KEYS = ["id_student", "code_module", "code_presentation"]


def load_scaled_numeric(root="."):
    """Cột số đã chuẩn hóa (fit trên train, bỏ clicks_pre_start) cho train/dev/test.
    Trả về {split: (ma trận, y, tên cột)}; hàng cùng thứ tự với load_split_data."""
    root = Path(root)
    data = load_split_data(root / "data/processed/feature_28d.csv", root / "data/processed/split.csv")
    pre = load_preprocessor(root / "models/preprocessor.joblib")
    names = [n for n in pre["preprocessor_linear"].get_feature_names_out() if n != "cnt_scaled__clicks_pre_start"]
    keep = [i for i, n in enumerate(names) if n.startswith(NUMERIC_PREFIXES)]
    cols = [names[i] for i in keep]
    return {k: (transform(pre, X)[:, keep], y, cols) for k, (X, y) in data.items()}


def choose_k(scores: pd.DataFrame) -> int:
    """Kết hợp elbow và silhouette: elbow = điểm xa đường nối hai đầu nhất (inertia),
    cộng với silhouette; cả hai min-max về [0,1], chọn K có tổng cao nhất."""
    k, inertia, sil = scores.k.to_numpy(float), scores.inertia.to_numpy(), scores.silhouette.to_numpy()
    kn, inn = (k - k[0]) / (k[-1] - k[0]), (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    elbow = np.abs(kn + inn - 1) / np.sqrt(2)  # khoảng cách tới đường nối (0,1)-(1,0)
    norm = lambda v: (v - v.min()) / (v.max() - v.min())
    return int(k[np.argmax(norm(elbow) + norm(sil))])


def run(root=".", k=None):
    """Fit K-Means trên train với K=2..8. k=None: tự gợi ý K; k=số: dùng K đó."""
    root = Path(root)
    data = load_scaled_numeric(root)
    X_train, y_train, _ = data["train"]
    rows = []
    for kk in range(2, 9):
        km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(X_train)
        sil = silhouette_score(X_train, km.labels_, sample_size=5000, random_state=42)
        rows.append({"k": kk, "inertia": km.inertia_, "silhouette": sil})
    scores = pd.DataFrame(rows)
    suggested = choose_k(scores)
    best_k = k or suggested

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(scores.k, scores.inertia, "o-"); ax[0].set(title="Elbow", xlabel="K", ylabel="Inertia")
    ax[1].plot(scores.k, scores.silhouette, "o-"); ax[1].set(title="Silhouette", xlabel="K")
    for a in ax:
        a.axvline(best_k, ls="--", c="gray")
    fig.tight_layout()
    (root / "results").mkdir(exist_ok=True)
    fig.savefig(root / "results/kmeans_elbow_silhouette.png", dpi=120)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_train)
    summary = cluster_summary(root, km, data)
    summary.to_csv(root / "results/cluster_summary.csv", index=False)
    return scores, best_k, summary, fig


def split_keys(root, name):
    """Khóa (id_student, code_module, code_presentation) của một split, cùng thứ tự với load_split_data."""
    feats = pd.read_csv(root / "data/processed/feature_28d.csv", usecols=KEYS)
    sp = pd.read_csv(root / "data/processed/split.csv")
    return feats.merge(sp, on="id_student").query("split == @name")[KEYS].reset_index(drop=True)


def label_rates(df):
    """Số sinh viên và tỉ lệ nhãn theo cụm. final_result chỉ dùng ở đây để thống kê."""
    return df.groupby("cluster").agg(
        n_students=("withdrawn", "size"),
        withdrawn_rate=("withdrawn", "mean"),
        fail_rate=("final_result", lambda s: (s == "Fail").mean()),
    )


def cluster_summary(root, km, data):
    """Thống kê theo cụm trên train; thêm cột dev (gán dev vào tâm cụm của train)."""
    root = Path(root)
    info = pd.read_csv(root / STUDENT_INFO, usecols=KEYS + ["final_result"])
    out = []
    for name in ("train", "dev"):
        X, y, _ = data[name]
        keys = split_keys(root, name)
        df = keys.merge(info, on=KEYS, how="left")
        assert len(df) == len(keys) and df.final_result.notna().all(), "ghép studentInfo bị lệch hàng"
        df["cluster"] = km.predict(X)  # train: trùng km.labels_
        df["withdrawn"] = y.to_numpy()
        assert ((df.final_result == "Withdrawn") == (df.withdrawn == 1)).all(), "final_result lệch label_withdrawn"
        out.append(label_rates(df).add_prefix(name + "_"))
    return pd.concat(out, axis=1).reset_index()


if __name__ == "__main__":
    scores, best_k, summary, _ = run()
    print(scores, f"\nK = {best_k}\n", summary.to_string())
