import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, "data", "城市区域环境噪声信息.xlsx")
OUT_DIR = os.path.join(ROOT, "outputs", "results", "q9_env_noise")

PERIODS = ["上午", "下午", "晚上", "夜间"]
YEARS = [2020, 2021, 2022, 2023, 2024]


def load():
    df = pd.read_excel(DATA_PATH)
    return df[df["等效声级LEQ"].notna()]


def period_mean(df):
    g = df.groupby("监测时段")["等效声级LEQ"].agg(平均等效声级="mean", 监测样本数="size")
    return g.reindex(PERIODS).round(2).reset_index()


def year_summary(df):
    g = df.groupby("监测年份").agg(平均等效声级=("等效声级LEQ", "mean"),
                                   达标率=("是否达标", "mean"),
                                   监测样本数=("是否达标", "size"))
    g = g.reindex(YEARS)
    g["达标率"] = (g["达标率"] * 100).round(2)
    return g.round(2).reset_index()


def main():
    df = load()
    os.makedirs(OUT_DIR, exist_ok=True)
    period_mean(df).to_csv(os.path.join(OUT_DIR, "period_mean.csv"), index=False, encoding="utf-8-sig")
    year_summary(df).to_csv(os.path.join(OUT_DIR, "year_summary.csv"), index=False, encoding="utf-8-sig")
    print("q9 results saved to outputs/results/q9_env_noise/")


if __name__ == "__main__":
    main()
