import os

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

import preprocess as pp

# 边际效应变量：中文标签
AME_VARS = {
    "Wall": "隔音墙(无→有)",
    "CityHZ": "杭州(vs温州)",
    "CityNB": "宁波(vs温州)",
    "Area1": "主城区(vs乡镇农村)",
    "Area2": "郊区(vs乡镇农村)",
    "Area3": "县城(vs乡镇农村)",
}


def main():
    df = pp.clean(pp.load_raw())
    scores = pd.read_csv(os.path.join(pp.OUT_DIR, "q2_efa", "factor_scores.csv"))
    x = pp.build_features(df, scores)
    y = df["satisfaction"].astype(int)

    po = OrderedModel(y, x, distr="logit").fit(method="bfgs", maxiter=1000, disp=False)

    def probs(xx):
        p = po.predict(xx).to_numpy()  # (n, 5) = P(Y=1..5)
        le2 = p[:, :2].sum(axis=1)     # P(Y<=2) 不满意概率
        ge4 = p[:, 3:].sum(axis=1)     # P(Y>=4) 满意概率
        ey = (p * np.arange(1, 6)).sum(axis=1)  # E[Y]
        return le2, ge4, ey

    le2, ge4, ey = probs(x)
    pp.save_json({
        "P(Y<=2)_不满意概率": round(float(le2.mean()), 4),
        "P(Y>=4)_满意概率": round(float(ge4.mean()), 4),
        "E[Y]_期望满意度": round(float(ey.mean()), 4),
    }, "q8_decision", "baseline.json")

    # 边际效应（平均边际效应 AME）
    ame_rows = []
    for var, label in AME_VARS.items():
        x0 = x.copy()
        x0[var] = 0
        x1 = x.copy()
        x1[var] = 1
        l0, g0, e0 = probs(x0)
        l1, g1, e1 = probs(x1)
        ame_rows.append({
            "变量": label,
            "ΔP(Y<=2)不满意概率": round(float((l1 - l0).mean()), 4),
            "ΔP(Y>=4)满意概率": round(float((g1 - g0).mean()), 4),
            "ΔE[Y]": round(float((e1 - e0).mean()), 4),
        })
    pp.save_csv(pd.DataFrame(ame_rows), "q8_decision", "marginal_effects.csv")

    # 群体概率
    group_rows = []
    for name, col, mapping in [
        ("城市", df["city"], {1: "杭州", 2: "宁波", 3: "温州"}),
        ("隔音墙", (df["soundproof_wall"] == 1).astype(int), {0: "无/不清楚", 1: "有"}),
        ("年龄", df["age"], {1: "18岁以下", 2: "19-30岁", 3: "31-45岁", 4: "46-60岁", 5: "61岁及以上"}),
    ]:
        for val, label in mapping.items():
            mask = (col == val).to_numpy()
            group_rows.append({
                "变量": name, "类别": label,
                "样本数": int(mask.sum()),
                "P(Y<=2)不满意概率": round(float(le2[mask].mean()), 4),
                "P(Y>=4)满意概率": round(float(ge4[mask].mean()), 4),
                "E[Y]期望满意度": round(float(ey[mask].mean()), 4),
            })
    pp.save_csv(pd.DataFrame(group_rows), "q8_decision", "group_probabilities.csv")

    print("q8 results saved to outputs/results/q8_decision/")
    print(pd.DataFrame(ame_rows).to_string(index=False))


if __name__ == "__main__":
    main()
