# -*- coding: utf-8 -*-
"""
问卷质量分析
============
对应论文第 4 章「问卷质量分析」：
    1. 随机性检验 —— 游程检验（Runs test）
    2. 相关性分析 —— Spearman 秩相关
    3. 信度分析   —— 克隆巴赫 Alpha 系数
    4. 效度检验   —— KMO 取样适切性量数 + 巴特利特球形检验

运行方式：
    python quality_analysis.py

输出：与论文第 4 章各表格对应的检验结果（467 份有效样本）。

说明
----
- Spearman 相关对「受噪声影响程度」做反向计分（noise_impact 原始为反向编码），
  以复现论文的正相关结论。
- 游程数依赖数据的行序（问卷回收顺序），当前数据文件行序与论文 SPSS 中的
  随机顺序不同，故游程数 / Z / 显著性无法与论文精确对应，但个案数与检验值一致。
"""

import numpy as np
import pandas as pd
from scipy import stats

import preprocess as pp

# 中文题项名（用于结果展示，便于对照论文表格）
ITEM_LABELS = {
    "noise_traffic": "交通噪声影响",
    "noise_life": "周围人生活噪声影响",
    "noise_construction": "装修/施工噪声影响",
    "noise_business": "商业经营噪声影响",
    "noise_public": "公共活动噪声影响",
    "noise_animal": "动物噪声影响",
    "eff_sleep_onset": "噪声入睡情况影响",
    "eff_sleep_interrupt": "噪声中断睡眠影响",
    "eff_attention": "噪声注意力影响",
    "eff_relationship": "噪声身边关系影响",
    "eff_mental": "噪声精神状态影响",
    "eff_physical": "噪声对身体健康影响",
    "eff_hearing": "噪声对听觉影响",
    "eff_emotion": "噪声对极端情绪影响",
}

VAR_LABELS = {
    "gender": "性别",
    "age": "年龄",
    "city": "现居住地",
    "area": "现居住区域",
    "occupation": "职业类型",
    "education": "文化程度",
}


def _fmt_p(p):
    """将 p 值格式化为论文样式（<0.001 或保留三位小数）。"""
    return "<0.001" if p < 0.001 else "{:.3f}".format(p)


# ---------------------------- 游程检验（Runs test） ----------------------------
def runs_test(x, cut="median"):
    """单样本游程检验（二分类：< 检验值 与 >= 检验值 两组）。

    参数:
        x: 一维样本。
        cut: 检验值；"median" 表示取中位数，也可传入具体数值。

    返回:
        (runs, z, p_value) —— 游程数、Z 统计量、渐近双尾显著性。
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if isinstance(cut, str) and cut == "median":
        cut = np.median(x)

    group = (x >= cut).astype(int)          # 0 = <cut, 1 = >=cut
    n1 = int((group == 0).sum())            # 个案数 < 检验值
    n2 = int((group == 1).sum())            # 个案数 >= 检验值
    runs = 1 + int((group[1:] != group[:-1]).sum())

    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))    # 双尾
    return runs, z, p


def runs_test_table(df):
    """对 6 项人口统计学变量开展游程检验，返回与论文表 4.1 对应的结果表。

    检验值：除「性别」取 2（区分男/女，二分变量）外，其余变量取中位数。
    """
    cut_map = {
        "gender": 2,
        "age": "median",
        "city": "median",
        "area": "median",
        "occupation": "median",
        "education": "median",
    }
    rows = []
    for var in pp.DEMOGRAPHIC_VARS:
        x = df[var]
        cut = cut_map[var]
        cut_value = cut if not isinstance(cut, str) else np.median(x.dropna())
        runs, z, p = runs_test(x, cut=cut_value)
        n_lt = int((x < cut_value).sum())
        n_ge = int((x >= cut_value).sum())
        rows.append({
            "变量": VAR_LABELS[var],
            "检验值": int(cut_value),
            "个案数<检验值": n_lt,
            "个案数>=检验值": n_ge,
            "总个案数": n_lt + n_ge,
            "游程数": runs,
            "Z": round(z, 3),
            "渐近显著性(双尾)": round(p, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------- Spearman 秩相关 ----------------------------
def spearman_table(df, x, items=None):
    """整体噪声感知与各细分题项的 Spearman 秩相关，返回论文表 4.2 对应结果。

    参数:
        x: 整体噪声感知序列（应已做方向校正，高分 = 影响更大）。
        items: 参与相关的题项（默认 EFA_ITEMS，14 项）。
    """
    if items is None:
        items = pp.EFA_ITEMS
    rows = []
    for c in items:
        r, p = stats.spearmanr(x, df[c])
        rows.append({
            "题项": ITEM_LABELS[c],
            "相关系数": round(r, 3),
            "显著性": _fmt_p(p),
        })
    return pd.DataFrame(rows)


# ---------------------------- 信度分析（克隆巴赫 Alpha） ----------------------------
def cronbach_alpha(df, items):
    """计算克隆巴赫 Alpha 系数。

    alpha = k/(k-1) * (1 - Σ σ²_item / σ²_total)
    """
    x = df[items].astype(float).to_numpy()
    k = x.shape[1]
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    alpha = k / (k - 1) * (1 - item_var / total_var)
    return alpha

def kmo_bartlett(df, items):
    """计算 KMO 取样适切性量数与巴特利特球形检验。

    返回:
        (kmo, chi2, dof, p_value)
    """
    x = df[items].astype(float).to_numpy()
    r = np.corrcoef(x, rowvar=False)
    inv = np.linalg.inv(r)
    d = np.diag(inv)

    # 偏相关矩阵
    p = -inv / np.sqrt(np.outer(d, d))
    np.fill_diagonal(p, 0)

    r2 = (r ** 2)[~np.eye(r.shape[0], dtype=bool)].sum()
    p2 = (p ** 2)[~np.eye(p.shape[0], dtype=bool)].sum()
    kmo = r2 / (r2 + p2)

    n, k = x.shape
    chi2 = -(n - 1 - (2 * k + 5) / 6) * np.log(np.linalg.det(r))
    dof = k * (k - 1) // 2
    p_val = 1 - stats.chi2.cdf(chi2, dof)
    return kmo, chi2, dof, p_val

def main():
    # 1. 数据预处理
    raw = pp.load_raw_data()
    df = pp.clean_data(raw)
    df, missing = pp.impute_missing(df)

    print("=" * 68)
    print("数据预处理：回收 {} 份 -> 有效 {} 份".format(len(raw), len(df)))
    print("缺失情况：居家办公缺失 {} 人（{:.1%}），隔音墙「不清楚」 {} 人".format(
        missing["home_office_missing"],
        missing["home_office_missing_rate"],
        missing["soundproof_wall_unclear"],
    ))
    print("=" * 68)

    # 2. 游程检验
    print("\n【1】随机性检验（游程检验）—— 论文表 4.1")
    print(runs_test_table(df).to_string(index=False))
    print("注：个案数 / 检验值与论文表 4.1 完全一致；游程数依赖数据行序，")
    print("    当前文件行序为整理后顺序（同类样本聚集），与论文 SPSS 随机顺序")
    print("    不同，故游程数 / Z / 显著性无法精确对应（论文游程数接近随机期望）。")

    # 3. Spearman 秩相关（对 noise_impact 反向计分）
    noise_impact_pos = pp.reverse_scale(df["noise_impact"])
    print("\n【2】相关性分析（Spearman 秩相关）—— 论文表 4.2")
    print(spearman_table(df, noise_impact_pos).to_string(index=False))

    # 4. 信度分析
    alpha = cronbach_alpha(df, pp.RELIABILITY_ITEMS)
    alpha_rev = cronbach_alpha(
        df.assign(noise_impact=pp.reverse_scale(df["noise_impact"])),
        pp.RELIABILITY_ITEMS,
    )
    print("\n【3】信度分析（克隆巴赫 Alpha）—— 论文表 4.3")
    print("克隆巴赫 Alpha = {:.3f}，项数 = {}".format(alpha, len(pp.RELIABILITY_ITEMS)))
    print("注：论文 Alpha=0.707 基于未反向的 noise_impact；若反向计分则 Alpha={:.3f}。".format(alpha_rev))

    # 5. 效度检验
    kmo, chi2, dof, p_val = kmo_bartlett(df, pp.EFA_ITEMS)
    print("\n【4】效度检验（KMO + 巴特利特球形检验）—— 论文表 4.4")
    print("KMO 取样适切性量数 = {:.3f}".format(kmo))
    print("巴特利特球形检验近似卡方 = {:.3f}，自由度 = {}，显著性 = {}".format(
        chi2, dof, _fmt_p(p_val)))
    print("=" * 68)


if __name__ == "__main__":
    main()
