import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel

import preprocess as pp

# 贝叶斯调参（q6_tune）得到的最优 XGBoost 分类器参数
XGB_PARAMS = dict(n_estimators=254, learning_rate=0.274, max_depth=5,
                  subsample=0.601, colsample_bytree=0.825,
                  reg_lambda=0.244, reg_alpha=2.290, min_child_weight=1,
                  objective="multi:softprob", num_class=5, random_state=42)


def main():
    df = pp.clean(pp.load_raw())
    scores = pd.read_csv(os.path.join(pp.OUT_DIR, "q2_efa", "factor_scores.csv"))
    x = pp.build_features(df, scores)
    y = df["satisfaction"].astype(int)

    # 独立测试集（与 q6_tune 同划分），调参已只在训练集上完成
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    po = OrderedModel(ytr, xtr, distr="logit").fit(method="bfgs", maxiter=1000, disp=False)
    pred_po = np.argmax(po.predict(xte), axis=1) + 1

    xgbm = xgb.XGBClassifier(**XGB_PARAMS).fit(xtr, ytr - 1)
    pred_xgb = xgbm.predict(xte) + 1

    rfm = RandomForestClassifier(n_estimators=500, random_state=42).fit(xtr, ytr - 1)
    pred_rf = rfm.predict(xte) + 1

    rows = []
    for name, pred in [("Logistic", pred_po), ("XGBoost", pred_xgb), ("RandomForest", pred_rf)]:
        rows.append({
            "模型": name,
            "Accuracy": round(accuracy_score(yte, pred), 4),
            "MacroF1": round(f1_score(yte, pred, average="macro"), 4),
            "Kappa": round(cohen_kappa_score(yte, pred), 4),
        })

    result = pd.DataFrame(rows)
    pp.save_csv(result, "q5_compare", "comparison.csv")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
