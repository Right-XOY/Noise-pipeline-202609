import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(ROOT, "outputs", "results", "q9_env_noise")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ['#104E8B', '#376B9E', '#5F89B1', '#AFC3D8', '#C5E9E3',
          '#D7E1EB', '#F2DADA', '#E5B5B5', '#D89090', '#B22222']


def gradient_bar(ax, x, h, width, base, bottom_c, top_c):
    cmap = LinearSegmentedColormap.from_list("bar", [bottom_c, top_c])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    for xi, hi in zip(x, h):
        ax.imshow(grad, extent=[xi - width / 2, xi + width / 2, base, hi], origin="lower",
                  aspect="auto", cmap=cmap, zorder=2, interpolation="bicubic")


def bare(ax, base, grid=True):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis="y", color="#DCE3EC", linewidth=0.9)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    ax.set_yticklabels([f"{t:.0f}{base}" for t in ax.get_yticks()], color="#7F7F7F")


def plot_period(per, out):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(per))
    width = 0.5
    gradient_bar(ax, x, per["平均等效声级"], width, 40, COLORS[0], COLORS[3])
    ax.bar(x, per["平均等效声级"], width=width, color="none", label="平均等效声级 dB(A)")
    for xi, v in zip(x, per["平均等效声级"]):
        ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=10, color="#404040")

    ax.set_xticks(x, per["监测时段"])
    ax.set_xlim(-0.6, len(per) - 0.4)
    ax.set_ylim(40, 56)
    ax.set_yticks(np.arange(40, 57, 2))
    bare(ax, "")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, handlelength=1.6)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_year(year, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(year))
    width = 0.42
    gradient_bar(ax, x, year["平均等效声级"], width, 48, COLORS[0], COLORS[3])
    ax.bar(x, year["平均等效声级"], width=width, color="none", label="平均等效声级")

    ax.set_xticks(x, year["监测年份"].astype(int))
    ax.set_xlim(-0.6, len(year) - 0.4)
    ax.set_ylim(48, 55)
    ax.set_yticks(np.arange(48, 56))
    bare(ax, "")

    ax2 = ax.twinx()
    ax2.plot(x, year["达标率"], color="#C0504D", linewidth=3, zorder=3, label="声环境质量达标率")
    ax2.set_ylim(65, 75)
    ax2.set_yticks(np.arange(65, 76))
    bare(ax2, "%", grid=False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, frameon=False, handlelength=1.6, columnspacing=2.5)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    per = pd.read_csv(os.path.join(RESULT_DIR, "period_mean.csv"))
    year = pd.read_csv(os.path.join(RESULT_DIR, "year_summary.csv"))
    plot_period(per, os.path.join(FIG_DIR, "env_noise_period.png"))
    plot_year(year, os.path.join(FIG_DIR, "env_noise_year.png"))
    print("env noise figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
