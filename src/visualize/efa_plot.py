import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(ROOT, "outputs", "results", "q2_efa")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ['#104E8B', '#376B9E', '#5F89B1', '#AFC3D8', '#C5E9E3',
          '#D7E1EB', '#F2DADA', '#E5B5B5', '#D89090', '#B22222']

FACTORS = ["F1", "F2", "F3", "F4"]
FACTOR_LABELS = ["身心健康与认知", "情绪与社会关系", "典型环境噪声源", "生活场景噪声源"]


def main():
    d = pd.read_csv(os.path.join(RESULT_DIR, "factor_loadings.csv"))
    m = d[FACTORS].to_numpy()
    dom = m.argmax(axis=1)
    order = np.argsort(dom, kind="stable")  # 按主载荷因子归组，形成分块结构
    m, dom = m[order], dom[order]
    items = d["题项"].to_numpy()[order]

    cmap = LinearSegmentedColormap.from_list("load", ["#FFFFFF", COLORS[5], COLORS[2], COLORS[0]])
    fig, ax = plt.subplots(figsize=(6.6, 7.8))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FACTORS)), [f"{f}\n{n}" for f, n in zip(FACTORS, FACTOR_LABELS)],
                  fontsize=10)
    ax.set_yticks(range(len(items)), items, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(items)):
        for j in range(len(FACTORS)):
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > 0.55 else "#333333",
                    fontweight="bold" if dom[i] == j else "normal")
    for k in np.where(np.diff(dom) != 0)[0]:
        ax.axhline(k + 0.5, color="white", linewidth=1.5)

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03, ticks=np.arange(0, 1.01, 0.2))
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9, colors="#7F7F7F")
    cb.set_label("因子载荷", fontsize=10, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "efa_loading_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("efa heatmap saved to outputs/figures/")


if __name__ == "__main__":
    main()
