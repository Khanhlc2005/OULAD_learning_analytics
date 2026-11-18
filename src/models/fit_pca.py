from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.models.train_kmeans import load_scaled_numeric


def run(root=".", n_components=None, target=0.95, top=5, save=False):
    """n_components=None: chọn theo target. save=True: lưu models/pca.joblib (PCA đã fit trên train)."""
    X_train, _, cols = load_scaled_numeric(root)["train"]  # chỉ fit trên train
    pca = PCA(random_state=42).fit(X_train)
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    n = n_components or int(np.argmax(cum >= target) + 1)
    table = pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(ratio))],
                          "eigenvalue": pca.explained_variance_, "explained": ratio, "cumulative": cum})
    load = pd.DataFrame(pca.components_, columns=cols, index=table.component)
    tops = {pc: load.loc[pc].reindex(load.loc[pc].abs().sort_values(ascending=False).index)[:top].round(3)
            for pc in table.component[:n]}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(ratio) + 1), ratio, label="Từng thành phần")
    ax.plot(range(1, len(ratio) + 1), cum, "o-", c="C1", label="Tích lũy")
    ax.axhline(target, ls="--", c="gray"); ax.set(title="Scree plot", xlabel="Thành phần", ylabel="Tỉ lệ phương sai")
    ax.legend(); fig.tight_layout()
    fig.savefig(Path(root) / "results/pca_scree.png", dpi=120)
    if save:
        final = PCA(n_components=n, random_state=42).fit(X_train)
        joblib.dump({"pca": final, "columns": list(cols)}, Path(root) / "models/pca.joblib")
    return table, n, tops, fig


if __name__ == "__main__":
    table, n, tops, _ = run()
    print(table.round(3).to_string(index=False), f"\nSố thành phần đạt 95%: {n}")
    for pc, t in tops.items():
        print(pc, t.to_dict())
