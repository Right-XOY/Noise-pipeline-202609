import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import model.preprocess as pp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 虚拟变量 -> 原始变量的分组（列索引），SHAP 时按组加总合并
GROUPS = {
    "年龄": [0, 1, 2, 3],
    "学历": [4, 5, 6, 7],
    "城市": [8, 9],
    "居住区域": [10, 11, 12],
    "隔音墙": [13],
    "F1身心认知": [14],
    "F2情绪关系": [15],
    "F3环境噪声": [16],
    "F4生活噪声": [17],
}

# 贝叶斯调参（q6_tune）得到的最优 XGBoost 分类器参数
XGB_PARAMS = dict(n_estimators=254, learning_rate=0.274, max_depth=5,
                  subsample=0.601, colsample_bytree=0.825,
                  reg_lambda=0.244, reg_alpha=2.290, min_child_weight=1,
                  objective="multi:softprob", num_class=5, random_state=42)

MODEL_LABELS = {"Logistic": "有序Logistic", "XGBoost": "XGBoost", "RandomForest": "随机森林"}
MODEL_COLORS = {"Logistic": "#1f77b4", "XGBoost": "#ff7f0e", "RandomForest": "#2ca02c"}
MODEL_ORDER = ["Logistic", "XGBoost", "RandomForest"]


def _data():
    df = pp.clean(pp.load_raw())
    scores = pd.read_csv(os.path.join(pp.OUT_DIR, "q2_efa", "factor_scores.csv"))
    x = pp.build_features(df, scores)
    y = df["satisfaction"].astype(int)
    return x, y


def _models(xtr, ytr):
    po = OrderedModel(ytr, xtr, distr="logit").fit(method="bfgs", maxiter=1000, disp=False)
    xgc = xgb.XGBClassifier(**XGB_PARAMS).fit(xtr, ytr - 1)
    rfc = RandomForestClassifier(n_estimators=500, random_state=42).fit(xtr, ytr - 1)
    return po, xgc, rfc


def _predict(po, xgc, rfc, xte):
    return {
        "Logistic": np.argmax(po.predict(xte), axis=1) + 1,
        "XGBoost": xgc.predict(xte) + 1,
        "RandomForest": rfc.predict(xte) + 1,
    }


def _proba(po, xgc, rfc, xte):
    return {
        "Logistic": po.predict(xte).to_numpy(),
        "XGBoost": xgc.predict_proba(xte),
        "RandomForest": rfc.predict_proba(xte),
    }


def plot_radar(out):
    x, y = _data()
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    po, xgc, rfc = _models(xtr, ytr)
    preds = _predict(po, xgc, rfc, xte)

    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score

    def specificity(y_true, y_pred):
        return np.mean([recall_score((y_true == k).astype(int), (y_pred == k).astype(int),
                                     pos_label=0, zero_division=0) for k in [1, 2, 3, 4, 5]])

    labels = ["Accuracy", "Precision", "Recall", "Specificity", "Kappa"]
    metrics = {}
    for name in MODEL_ORDER:
        p = preds[name]
        metrics[name] = [accuracy_score(yte, p),
                         precision_score(yte, p, average="macro", zero_division=0),
                         recall_score(yte, p, average="macro", zero_division=0),
                         specificity(yte, p),
                         cohen_kappa_score(yte, p)]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.grid(False)
    for r in np.arange(0.2, 1.01, 0.2):
        ax.plot(angles, [r] * len(angles), color="lightgray", linewidth=0.5)
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1], color="lightgray", linewidth=0.5)
    for name in MODEL_ORDER:
        vals = metrics[name] + [metrics[name][0]]
        ax.plot(angles, vals, color=MODEL_COLORS[name], linewidth=2, label=MODEL_LABELS[name])
        ax.fill(angles, vals, color=MODEL_COLORS[name], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1))
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion():
    x, y = _data()
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    po, xgc, rfc = _models(xtr, ytr)
    preds = _predict(po, xgc, rfc, xte)
    labels = [1, 2, 3, 4, 5]
    for name in MODEL_ORDER:
        cm = confusion_matrix(yte, preds[name], labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(5), labels)
        ax.set_yticks(range(5), labels)
        ax.set_xlabel("预测等级")
        ax.set_ylabel("真实等级")
        for r in range(5):
            for c in range(5):
                ax.text(c, r, cm[r, c], ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"compare_confusion_{name.lower()}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()


def plot_shap(out):
    x, y = _data()
    xtr, _, ytr, _ = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    po, xgc, rfc = _models(xtr, ytr)

    shap_xg_raw = shap.TreeExplainer(xgc).shap_values(xtr)
    shap_xg = np.abs(np.array(shap_xg_raw)).mean(axis=0) if isinstance(shap_xg_raw, list) \
        else np.abs(shap_xg_raw).mean(axis=2)
    shap_rf_raw = shap.TreeExplainer(rfc).shap_values(xtr)
    shap_rf = np.abs(np.array(shap_rf_raw)).mean(axis=0) if isinstance(shap_rf_raw, list) \
        else np.abs(shap_rf_raw).mean(axis=2)
    beta = po.params[:-4].to_numpy()
    shap_po = np.abs((xtr.to_numpy() - xtr.to_numpy().mean(axis=0)) * beta)

    imps = {"Logistic": [], "XGBoost": [], "RandomForest": []}
    for cols in GROUPS.values():
        imps["Logistic"].append(shap_po[:, cols].sum(axis=1).mean())
        imps["XGBoost"].append(shap_xg[:, cols].sum(axis=1).mean())
        imps["RandomForest"].append(shap_rf[:, cols].sum(axis=1).mean())
    names = list(GROUPS.keys())
    total = sum(np.array(imps[k]) for k in MODEL_ORDER)
    idx = np.argsort(total)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(names))
    for i, name in enumerate(MODEL_ORDER):
        off = 0.22 - 0.22 * i
        ax.barh(y_pos + off, np.array(imps[name])[idx], height=0.2,
                color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    ax.set_yticks(y_pos, [names[i] for i in idx])
    ax.set_xlabel("mean |SHAP|")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc(out):
    x, y = _data()
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    po, xgc, rfc = _models(xtr, ytr)
    probs = _proba(po, xgc, rfc, xte)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name in MODEL_ORDER:
        prob = probs[name]
        fpr_grid = np.linspace(0, 1, 100)
        tprs, aucs = [], []
        for k in range(5):
            yb = (yte == k + 1).astype(int)
            fpr, tpr, _ = roc_curve(yb, prob[:, k])
            tprs.append(np.interp(fpr_grid, fpr, tpr))
            aucs.append(auc(fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        ax.plot(fpr_grid, mean_tpr, color=MODEL_COLORS[name], linewidth=2,
                label=f"{MODEL_LABELS[name]} (宏平均AUC={mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_radar(os.path.join(FIG_DIR, "compare_radar.png"))
    plot_confusion()
    plot_shap(os.path.join(FIG_DIR, "compare_shap.png"))
    plot_roc(os.path.join(FIG_DIR, "compare_roc.png"))
    print("compare figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
