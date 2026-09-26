import os
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from wordcloud import WordCloud

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(ROOT, "outputs", "results", "q10_text")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")

COLORS = ['#104E8B', '#376B9E', '#5F89B1', '#AFC3D8', '#C5E9E3',
          '#D7E1EB', '#F2DADA', '#E5B5B5', '#D89090', '#B22222']

FONT = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
WORD_COLORS = [COLORS[0], COLORS[1], COLORS[2], COLORS[8], COLORS[9]]
MASK_SIZE = (1500, 950)


def cloud_mask():
    img = Image.new("L", MASK_SIZE, 255)
    d = ImageDraw.Draw(img)
    # 云朵轮廓：底部宽扁主体 + 上方高低错落的圆弧，整体居中于画布
    d.ellipse([190, 498, 1340, 788], fill=0)
    for cx, cy, r in [(315, 483, 155), (615, 363, 200), (920, 428, 170),
                      (1190, 518, 130)]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
    return np.array(img)


def color_func(word, **kwargs):
    return WORD_COLORS[zlib.crc32(word.encode()) % len(WORD_COLORS)]


def main():
    freq = pd.read_csv(os.path.join(RESULT_DIR, "word_freq.csv"))
    weights = dict(zip(freq["词语"], freq["词频"]))

    wc = WordCloud(font_path=FONT, mask=cloud_mask(), background_color="white",
                   margin=12, max_words=110, prefer_horizontal=0.92, min_font_size=10,
                   relative_scaling=0.5, collocations=False, random_state=42,
                   contour_width=2, contour_color=COLORS[3],
                   color_func=color_func).generate_from_frequencies(weights)

    img = np.array(wc.to_image().convert("RGBA"))
    img[(img[..., :3] >= 250).all(axis=2), 3] = 0  # 白底转透明

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "text_wordcloud.png"), dpi=300,
                bbox_inches="tight", transparent=True)
    plt.close()
    print("wordcloud saved to outputs/figures/")


if __name__ == "__main__":
    main()
