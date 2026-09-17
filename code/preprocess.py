# -*- coding: utf-8 -*-
"""
数据预处理模块
==============
对应论文第 3 章「数据预处理」：数据清洗、缺失值处理、数据编码。

功能
----
1. 读取原始问卷数据（493 份回收问卷）。
2. 解析作答时长、将 54 列中文题项重命名为简洁英文变量名。
3. 剔除无效问卷，得到论文所述 467 份有效样本。
4. 统计并说明缺失值情况（仅「居家办公」列存在高缺失）。

关键说明
--------
论文正文表述的无效问卷剔除标准为「作答时长不足 100 秒、所有题项选择同一答案、
量表题项存在明显作答矛盾」。但据数据实际分布，作答时长最短的 26 份问卷
（时长 <= 50 秒）剔除后，可精确复现论文 467 份有效样本的性别、年龄、城市、
居住区域、职业、文化程度全部人口学分布，故此处以「时长 <= 50 秒」作为可复现的
阈值（参数 threshold 可调）。论文正文的「100 秒」与实际数据分布不一致。

编码方向说明
------------
「受噪声影响程度」（noise_impact）在原始数据中为**反向编码**：取值 1=影响非常大、
2=影响较大、3=一般、4=偶尔有影响、5=几乎没有影响（数值越大、影响越小），与论文
编码规则表（1=几乎没有影响，5=严重影响生活）方向相反。其余量表题项（噪声源干扰、
噪声负面影响、满意度等）均为正向编码。

因此：
- 相关性等需要「高分=影响更大」方向的分析，需先对 noise_impact 反向计分
  （``reverse_scale(noise_impact)``，即 6 - 原值），方可复现论文的正相关结论；
- 论文第 4 章信度分析（Alpha=0.707）则是基于未反向的原始 noise_impact 计算得到，
  若正确反向计分，Alpha 会升至 0.746。
"""

import os

import numpy as np
import pandas as pd

# 数据文件路径（基于本文件所在目录定位，可任意工作目录运行）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "浙江省杭甬温环境噪声污染调查问卷.xlsx")

# 原始 54 列 -> 简洁英文变量名（按列顺序一一对应）
SHORT_COLUMNS = [
    "id", "duration_raw",
    "gender", "age", "city", "area", "occupation", "education",
    "soundproof_wall", "home_office",
    "noise_impact", "insulation",
    "noise_traffic", "noise_life", "noise_construction",
    "noise_business", "noise_public", "noise_animal",
    "distance_source", "time_period",
    "eff_sleep_onset", "eff_sleep_interrupt", "eff_attention",
    "eff_relationship", "eff_mental", "eff_physical",
    "eff_hearing", "eff_emotion",
    "satisfaction", "governance_satisfaction",
    "coping_endure", "coping_negotiate", "coping_community", "coping_protect",
    "coping_official", "coping_legal", "coping_other",
    "awareness_channel", "awareness_policy", "awareness_necessity",
    "resp_gov", "resp_property", "resp_business", "resp_resident", "resp_other",
    "meas_enforce", "meas_law", "meas_facility", "meas_time",
    "meas_publicity", "meas_complaint", "meas_planning", "meas_other",
    "open_text",
]

# 6 类噪声源干扰题项（李克特 5 级）
NOISE_SOURCES = [
    "noise_traffic", "noise_life", "noise_construction",
    "noise_business", "noise_public", "noise_animal",
]

# 8 类噪声负面影响题项（李克特 5 级）
NOISE_EFFECTS = [
    "eff_sleep_onset", "eff_sleep_interrupt", "eff_attention",
    "eff_relationship", "eff_mental", "eff_physical",
    "eff_hearing", "eff_emotion",
]

# 探索性因子分析（EFA）的 14 个噪声影响题项 = 6 噪声源 + 8 负面影响
EFA_ITEMS = NOISE_SOURCES + NOISE_EFFECTS

# 信度分析的 22 个量表题项（EFA_ITEMS + 整体感知/隔音/声源距离/两类满意度/三类认知）
RELIABILITY_ITEMS = EFA_ITEMS + [
    "noise_impact", "insulation", "distance_source",
    "satisfaction", "governance_satisfaction",
    "awareness_channel", "awareness_policy", "awareness_necessity",
]

# 人口统计学变量（游程检验对象）
DEMOGRAPHIC_VARS = ["gender", "age", "city", "area", "occupation", "education"]


def load_raw_data(path=DATA_PATH):
    """读取原始问卷数据，解析作答时长并重命名列。

    返回:
        DataFrame —— 493 份回收问卷，列为简洁英文变量名，
        其中 ``duration`` 为作答秒数（数值型）。
    """
    df = pd.read_excel(path)
    df.columns = SHORT_COLUMNS
    df["duration"] = (
        df["duration_raw"].astype(str).str.extract(r"(\d+)").astype(float)
    )
    return df


def clean_data(df, threshold=50):
    """剔除无效问卷，得到有效样本。

    参数:
        df: load_raw_data 返回的原始数据。
        threshold: 作答时长阈值（秒），时长 <= 该值视为无效问卷，默认 50。

    返回:
        DataFrame —— 剔除无效问卷后的 467 份有效样本。
    """
    df = df[df["duration"] > threshold].copy().reset_index(drop=True)
    return df


def impute_missing(df):
    """缺失值处理（对应论文 3.2 节）。

    本数据仅「居家办公」（home_office）列存在缺失（约 48%），按论文规则不插补、
    单独保留并说明；其余进入统计分析的量表题与人口变量无缺失。

    「隔音墙 = 不清楚」（取值 3）按论文第 7 章在二分类建模时并入「无隔音墙」，
    此处仅记录数量、不修改原始取值，供后续有序 Logistic 回归阶段使用。

    返回:
        (df, report) —— 处理后的数据副本，以及缺失情况说明字典。
    """
    df = df.copy()
    report = {
        "home_office_missing": int(df["home_office"].isna().sum()),
        "home_office_missing_rate": round(float(df["home_office"].isna().mean()), 4),
        "soundproof_wall_unclear": int((df["soundproof_wall"] == 3).sum()),
    }
    return df, report


def reverse_scale(series, max_val=5, min_val=1):
    """对李克特量表进行反向计分。

    新值 = max_val + min_val - 原值。用于将反向编码的题项（如 noise_impact）
    转换为「高分 = 程度更强」的正向方向。
    """
    return max_val + min_val - series


if __name__ == "__main__":
    # 快速自检：打印清洗前后样本量与缺失情况
    raw = load_raw_data()
    clean = clean_data(raw)
    clean, report = impute_missing(clean)

    print("原始回收问卷数:", len(raw))
    print("有效样本数:", len(clean))
    print("缺失情况:", report)
    print("信度分析题项数:", len(RELIABILITY_ITEMS))
    print("EFA 题项数:", len(EFA_ITEMS))
