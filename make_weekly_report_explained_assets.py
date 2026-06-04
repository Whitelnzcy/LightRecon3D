from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "outputs" / "weekly_report_2026-06-05" / "assets"


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(32, True)
F_H = font(23, True)
F_BODY = font(18)
F_SMALL = font(15)


def wrap(draw, text, width, fnt):
    lines = []
    current = ""
    for ch in text:
        trial = current + ch
        if draw.textlength(trial, font=fnt) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def rounded_box(draw, box, title, body, fill, outline="#dbe1ea"):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)
    draw.text((x1 + 22, y1 + 18), title, fill="#172033", font=F_H)
    yy = y1 + 58
    for line in wrap(draw, body, x2 - x1 - 44, F_BODY):
        draw.text((x1 + 22, yy), line, fill="#4b5563", font=F_BODY)
        yy += 27


def arrow(draw, start, end, color="#64748b"):
    draw.line((start, end), fill=color, width=5)
    ex, ey = end
    sx, sy = start
    if ex > sx:
        pts = [(ex, ey), (ex - 16, ey - 10), (ex - 16, ey + 10)]
    else:
        pts = [(ex, ey), (ex + 16, ey - 10), (ex + 16, ey + 10)]
    draw.polygon(pts, fill=color)


def make_pipeline():
    img = Image.new("RGB", (1500, 760), "#f7f8fb")
    d = ImageDraw.Draw(img)
    d.text((50, 36), "这周到底做了什么？", fill="#172033", font=F_TITLE)
    d.text((50, 82), "一句话：不是直接改点云，而是把点云变成可编辑的结构参数，再用学习头把参数算得更准。", fill="#4b5563", font=F_BODY)

    boxes = [
        ((55, 150, 325, 330), "1. 输入", "DUSt3R/MASt3R 生成的稠密点云，包含 xyz 和 rgb。", "#ffffff"),
        ((405, 150, 675, 330), "2. 粗 proposal", "RANSAC 先从点云里找可能的墙面、地面等平面。", "#ffffff"),
        ((755, 150, 1025, 330), "3. 学习归属", "网络学习每个点属于某个平面的概率，不再只是硬阈值。", "#eaf2ff"),
        ((1105, 150, 1375, 330), "4. 重拟合方程", "用 learned support 重新拟合 n*x+d=0，得到可编辑平面参数。", "#eaf8f0"),
    ]
    for box, title, body, fill in boxes:
        rounded_box(d, box, title, body, fill)
    for x in [345, 695, 1045]:
        arrow(d, (x, 240), (x + 40, 240))

    rounded_box(
        d,
        (150, 430, 660, 645),
        "为什么这不是简单 RANSAC？",
        "RANSAC 只负责给初始候选。最终哪些点属于平面、以及用哪些点重新估计平面方程，是学习头参与决定的。",
        "#fff7ed",
        "#fed7aa",
    )
    rounded_box(
        d,
        (840, 430, 1350, 645),
        "这周的新增创新点",
        "v6 把 learned soft assignment 反馈到 plane refit；同时由平面方程求交得到 line primitive，为后续 line head 做准备。",
        "#f0fdf4",
        "#bbf7d0",
    )
    img.save(ASSETS / "fig0_method_overview.png", quality=95)


def make_reading_guide():
    img = Image.new("RGB", (1500, 700), "#f7f8fb")
    d = ImageDraw.Draw(img)
    d.text((50, 36), "老师看结果时应该看什么？", fill="#172033", font=F_TITLE)
    items = [
        ("看指标图", "v6 的 Plane residual 和 Binding residual 都比 v3/v4 好，说明 soft assignment + refit 有效。", "#2563eb"),
        ("看绑定图", "蓝色点表示网络新认为应该属于该平面的点；红色点表示被移除的点。重点看归属关系是否更合理。", "#16a34a"),
        ("看线结构图", "彩色线不是手画的，而是由两个 refined plane equations 求交得到，说明平面参数能继续产生线 primitive。", "#e11d48"),
        ("看局限", "当前结果是初步验证，不是最终模型。下一步要训练 line head，并加入 line-plane consistency。", "#f59e0b"),
    ]
    y = 125
    for title, body, color in items:
        d.rounded_rectangle((70, y, 1430, y + 110), radius=16, fill="#ffffff", outline="#dbe1ea")
        d.rounded_rectangle((95, y + 26, 125, y + 84), radius=8, fill=color)
        d.text((150, y + 20), title, fill="#172033", font=F_H)
        d.text((150, y + 58), body, fill="#4b5563", font=F_BODY)
        y += 130
    img.save(ASSETS / "fig4_reading_guide.png", quality=95)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_pipeline()
    make_reading_guide()
    print(ASSETS / "fig0_method_overview.png")
    print(ASSETS / "fig4_reading_guide.png")


if __name__ == "__main__":
    main()
