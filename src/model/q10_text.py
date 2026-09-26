import os
import re
from collections import Counter

import jieba
import pandas as pd

import preprocess as pp

DATA_PATH = os.path.join(pp.ROOT, "data", "第30题答案.xlsx")

# 无实质内容的作答（"无/没有/不知道" 一类）
NO_MEANING = {
    "无", "没有", "暂无", "没意见", "没建议", "没有建议", "无建议", "无意见", "没想法",
    "不知道", "不清楚", "说不清楚", "不中", "没问题", "没有意见", "没有补充", "无补充",
    "暂无建议", "暂无意见", "好", "很好", "好的", "好了", "好滴", "好呀", "挺好", "不错",
    "嗯", "是", "是的", "对", "可以", "行", "ok", "合格", "略", "无所谓", "没啥", "差不多",
    "还好", "一般", "一般般", "都行", "随便", "支持", "赞同", "满意", "加油", "哈喽", "哈",
    "整理下", "少打瓦", "失去", "多多",
}

# 主题词与泛化修饰词：出现在词云中不提供信息
STOPWORDS = {
    "噪声", "噪音", "声音", "声环境", "城市", "浙江", "浙江省", "杭州", "宁波", "温州", "全省",
    "我们", "他们", "自己", "大家", "人们", "个人", "政府", "官方", "部门", "国家",
    "加强", "加大", "增大", "提高", "提升", "完善", "推进", "开展", "实行", "进行",
    "严格", "应该", "建议", "希望", "需要", "可以", "能够", "必须", "及时", "有效",
    "进一步", "切实", "大力", "积极", "全面", "全民", "强化", "注重", "做好", "搞好",
    "相关", "有关", "各类", "各种", "以及", "等等", "通过", "对于", "针对", "方面",
    "情况", "问题", "措施", "办法", "内容", "事项", "工作", "力度", "意识", "行为",
    "现在", "目前", "已经", "可能", "还是", "就是", "这样", "那样", "一些", "很多",
    "非常", "比较", "真的", "觉得", "认为", "没有", "不能", "不要", "不会", "的时候",
    "地方", "方面", "同时", "另外", "此外", "如果", "因为", "所以", "但是", "而且",
    "减少", "避免", "远离", "定期", "优化", "建立", "设置", "使用", "采用", "形成",
    "处理", "控制", "专业", "统一", "时间", "环境", "活动", "社会", "生活", "政策",
    "技术", "推广", "增加", "注意", "总体", "整体", "一般", "尽量", "尽可能",
}


def is_meaningless(text):
    core = re.sub(r"[\W\d_]+", "", text)
    if not core:
        return True
    if core.lower() in NO_MEANING:
        return True
    return set(core) <= set("无没有暂不")


def tokenize(text):
    words = []
    for token in jieba.cut(text):
        token = token.strip()
        if len(token) < 2 or token.isdigit() or token in STOPWORDS:
            continue
        if re.search(r"[a-zA-Z0-9]", token):
            continue
        words.append(token)
    return words


def main():
    raw = pd.read_excel(DATA_PATH)
    text = raw["答案文本"].astype(str).str.strip()

    blank = text.str.replace(r"[\W\d_]+", "", regex=True).eq("")
    junk = (~blank) & text.map(is_meaningless)
    valid = text[~blank & ~junk]

    counter = Counter()
    for t in valid:
        counter.update(tokenize(t))

    freq = pd.DataFrame(counter.most_common(), columns=["词语", "词频"])
    pp.save_csv(freq, "q10_text", "word_freq.csv")
    pp.save_csv(pd.DataFrame({"答案文本": valid}), "q10_text", "cleaned_replies.csv")
    pp.save_json({
        "回收总数": int(len(raw)),
        "空白或纯符号": int(blank.sum()),
        "无实质内容": int(junk.sum()),
        "有效作答": int(len(valid)),
        "有效作答占比": round(float(len(valid) / len(raw)), 3),
        "有效词数": int(freq["词频"].sum()),
        "不同词语数": int(len(freq)),
    }, "q10_text", "summary.json")
    print(freq.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
