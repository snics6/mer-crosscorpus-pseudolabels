"""Overview figures (2026-11-16, researcher's request: Ishii/Okada-style figures that show the
setting, the two parts of the study, the proposed fusion and what it is compared with).

Fig. A  fig_overview_study      : setting + Q1-Q2 (retraining on the product pseudo-labels, analysis tools) + Q3 (proposed fusion)
Fig. B  fig_fusion_map          : where each fusion method takes its weights from (schematic map; no result values)

Both are schematic. Text only, no measured values. Japanese versions (suffix _ja) are written for the slides.
Output: ../figures/<name>{,_ja}.pdf and .png (300 dpi), width 174 mm.
Run:  python make_figs_overview.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
MM = 1 / 25.4
W = 174
W_EN = 372 / 72 * 25.4   # text width of the manuscript (sn-jnl, 372 pt = 131.2 mm): the English figures are drawn at this width
FS = 7.6

ARIAL_DIR = Path("/mnt/c/Windows/Fonts")
if (ARIAL_DIR / "arial.ttf").exists():
    for f in ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]:
        fm.fontManager.addfont(str(ARIAL_DIR / f))
    EN_FONT = "Arial"
else:
    EN_FONT = "Nimbus Sans"
fm.fontManager.addfont("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
JA_FONT = "IPAexGothic"

MOD = {"A": ("#E69F00", "o"), "V": ("#0072B2", "s"), "T": ("#009E73", "^")}
C_K = "#1F78B4"        # target K labels
C_U = "#7F7F7F"        # unlabelled utterances
C_SRC = "#6A3D9A"      # source corpora
C_P1 = "#882255"       # Part I
C_P2 = "#8C510A"       # Part II
C_PROP = "#D55E00"     # proposed fusion
GREY = "#555555"


def setup(lang):
    plt.rcParams.update({"font.family": EN_FONT if lang == "en" else JA_FONT, "font.size": FS,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def canvas(h, w=W):
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc="white", ec="black", lw=0.7, ls="-", r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, p0, p1, color="black", lw=0.9, head=7, z=3, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))


def txt(ax, x, y, s, ha="left", va="top", size=FS, color="black", weight="normal", style="normal", z=6, ls=1.2):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, weight=weight, style=style, zorder=z, linespacing=ls)


def mark(ax, x, y, m, size=4.0):
    c, mk = MOD[m]
    ax.plot([x], [y], marker=mk, ms=size, mfc=c, mec="black", mew=0.4, zorder=7, clip_on=False)


def header(ax, x, y, w, h, title, color):
    box(ax, x, y, w, h, fc=color, ec=color, lw=0.7, r=1.0, z=2)
    txt(ax, x + 2, y + h / 2, title, va="center", color="white", weight="bold", size=FS + 0.4)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ------------------------------------------------------------------ texts
T = {
    "en": dict(
        target="Target corpus (one of AFEW, IEMOCAP impro., MOSEI, MELD, CREMA-D)",
        k="K labels per emotion\n(K = 5, 10, 25)",
        unl="Unlabelled\nutterances (many)",
        test="Test utterances\n(speakers held out)",
        src="Source corpora\n(the other corpora,\nall utterances labelled)",
        enc="Frozen encoders → one linear classifier per modality",
        mods=["audio (WavLM)", "visual (video)", "text (RoBERTa)"],
        avt="+ AVT: the three\nfeatures concatenated",
        p1="RQ1 (Sec. 5–6): what each source corrects; when UAR rises",
        p2="RQ2 (Sec. 7): how to fuse the answers into pseudo-labels",
        naive_t="Retraining on the product pseudo-labels (Sec. 4.4, 6)",
        naive="Multiply the three classifiers’ probabilities\n(same weight) → pseudo-labels on the unlabelled\nutterances → retrain each classifier on\nthe K labels + pseudo-labels",
        tool_t="Two analysis tools (Sec. 5, 6.5)",
        tool1="(1) Split the error: bias towards some emotions\n      + remaining error (weight direction)",
        tool2="(2) Split the gain into three quantities: room,\n      margin of the pseudo-labels over the classifier,\n      part of the pseudo-labels the classifier can imitate",
        q1="RQ1(a): Which part of the error does each of the 4 sources correct?\nRQ1(b): For which classifiers, and when, do pseudo-labels raise UAR?",
        prop_t="Proposed fusion",
        prop1="Strength of each classifier  ←  K target labels (3 numbers)",
        prop2="Per-emotion profile  ←  source corpora",
        prop3="Reliability-weighted late fusion → pseudo-labels",
        cmp_t="Compared with (F1–F10)",
        cmp="• no labels used (F1 product, F2 mean, F4 confidence)\n• exponents chosen on other corpora (F3)\n• 3 weights fitted on the K labels (F5, F8, F9)\n• weight tables measured or searched on the K labels (F6, F7, F10)",
        q2="RQ2: On which corpora is it above the product, and why?\nWhere are the other fusions below it?",
        eval="Evaluation: UAR on the test utterances, leave-one-corpus-out",
    ),
    "ja": dict(
        target="ターゲット（AFEW，IEMOCAP 即興，MOSEI，MELD，CREMA-D のどれか 1 つ）",
        k="感情ごとに\nK 本のラベル\n（K = 5，10，25）",
        unl="ラベル無しの\n発話（多数）",
        test="テストの発話\n（話者を分ける）",
        src="ソース\n（残りのコーパス．\n全部にラベル）",
        enc="凍結した\nエンコーダ\n→ モダリティ\nごとの線形\n分類器",
        mods=["声（WavLM）", "顔（動画）", "言葉（RoBERTa）"],
        avt="＋ AVT：3 つの特徴を\nつないだ分類器",
        p1="問い 1（5・6 節）：情報源が直す誤りと，UAR が上がる場合",
        p2="問い 2（7 節）：疑似ラベルのために答えをどうまとめるか",
        naive_t="積の疑似ラベルによる学び直し（4.4 節，6 節）",
        naive="3 つの分類器の確率を同じ重みで掛ける\n→ ラベル無しの発話の疑似ラベル\n→ K 本と疑似ラベルで各分類器を学び直す",
        tool_t="2 つの解析の道具（5 節，6.5 節）",
        tool1="① 誤りを分ける：答えの偏り ＋ 残りの誤り（重みの向き）",
        tool2="② 得を 3 つの量に分ける：余地，\n     疑似ラベルの当たる幅，分類器がまねできる分",
        q1="問い 1(a)：4 つの情報源は誤りのどの部分を直すか？\n問い 1(b)：疑似ラベルで，どの分類器の UAR がいつ上がるか？",
        prop_t="提案のまとめ方",
        prop1="分類器ごとの強さ  ←  ターゲットの K 本（3 つの数）",
        prop2="感情ごとの得意・不得意  ←  ソース",
        prop3="信頼度で重みづけた late fusion → 疑似ラベル",
        cmp_t="比べる方法（F1〜F10）",
        cmp="・ラベルを使わない（F1 積，F2 平均，F4 確信度）\n・別のコーパスで選んだ指数（F3）\n・K 本で合わせる 3 つの重み（F5，F8，F9）\n・K 本で測る・探す重みの表（F6，F7，F10）",
        q2="問い 2：どのコーパスで提案が積より高く，なぜか？\nほかの方法はどこで提案より低いか？",
        eval="評価：テストの発話の UAR（1 つのコーパスを新しいコーパスにする）",
    ),
}


def fig_overview(lang):
    setup(lang)
    t = T[lang]
    H = 112
    fig, ax = canvas(H)
    # ---------------- setting band
    y0 = 74
    box(ax, 1, y0, 104, 36, fc="#FAFAFA", ec="black", lw=0.8)
    txt(ax, 3, y0 + 34, t["target"], weight="bold")
    cw = 32
    for i, (key, col, ls) in enumerate([("k", C_K, "-"), ("unl", C_U, "-"), ("test", "black", "--")]):
        x = 3 + i * (cw + 2)
        box(ax, x, y0 + 9, cw, 19, fc="white", ec=col, lw=1.0, ls=ls, z=2)
        txt(ax, x + cw / 2, y0 + 25.5, t[key], ha="center", color=col if key != "test" else "black")
        for j, m in enumerate("AVT"):
            mark(ax, x + cw / 2 - 5 + j * 5, y0 + 12.5, m)
    box(ax, 108, y0 + 8, 30, 28, fc="#F4F0F8", ec=C_SRC, lw=1.0)
    txt(ax, 123, y0 + 33, t["src"], ha="center", color=C_SRC)
    for j, m in enumerate("AVT"):
        mark(ax, 118 + j * 5, y0 + 12, m)
    # encoders/classifiers
    box(ax, 141, y0 - 6, 32, 42, fc="#EEEEEE", ec="black", lw=0.8)
    txt(ax, 157, y0 + 34, t["enc"] if lang == "ja" else "Frozen encoders\n→ one linear\nclassifier per\nmodality", ha="center", ls=1.1)
    for j, (m, lab) in enumerate(zip("AVT", t["mods"])):
        yy = y0 + 14 - j * 5
        mark(ax, 144, yy, m)
        txt(ax, 146.5, yy, lab, va="center", size=FS - 0.4)
    txt(ax, 157, y0 - 2.0, t["avt"], ha="center", va="center", size=FS - 0.8, color=GREY, ls=1.1)
    ax.plot([105, 139], [y0 + 3, y0 + 3], color="black", lw=0.8, zorder=3)
    arrow(ax, (139, y0 + 3), (141, y0 + 3), lw=0.8)
    arrow(ax, (138, y0 + 18), (141, y0 + 18), lw=0.8)
    if lang == "ja":
        # the encoder caption is long in Japanese; replace the multi-line label
        pass
    # ---------------- two parts
    yb, hb = 2, 64
    xl, wl = 1, 84
    xr, wr = 89, 84
    box(ax, xl, yb, wl, hb, fc="white", ec=C_P1, lw=1.0)
    header(ax, xl, yb + hb - 6, wl, 6, t["p1"], C_P1)
    box(ax, xr, yb, wr, hb, fc="white", ec=C_P2, lw=1.0)
    header(ax, xr, yb + hb - 6, wr, 6, t["p2"], C_P2)
    # Part I contents
    yy = yb + hb - 9
    txt(ax, xl + 2, yy, t["naive_t"], weight="bold", color=C_P1)
    box(ax, xl + 2, yy - 18, wl - 4, 14, fc="#FBF3F7", ec=C_P1, lw=0.6, z=2)
    txt(ax, xl + 4, yy - 5.5, t["naive"], ls=1.15)
    yy -= 21
    txt(ax, xl + 2, yy, t["tool_t"], weight="bold", color=C_P1)
    txt(ax, xl + 4, yy - 4.5, t["tool1"], ls=1.15)
    txt(ax, xl + 4, yy - 11.5, t["tool2"], ls=1.15)
    yy -= 22
    box(ax, xl + 2, yb + 2, wl - 4, 11, fc="white", ec=C_P1, lw=0.6, ls="--", z=2)
    txt(ax, xl + 4, yb + 11, t["q1"], style="italic" if lang == "en" else "normal", color=C_P1, ls=1.15)
    # Part II contents
    yy = yb + hb - 9
    txt(ax, xr + 2, yy, t["prop_t"], weight="bold", color=C_PROP)
    box(ax, xr + 2, yy - 19, wr - 4, 16, fc="#FDF0E8", ec=C_PROP, lw=0.9, z=2)
    txt(ax, xr + 4, yy - 4.5, t["prop1"], color=C_K)
    txt(ax, xr + 4, yy - 9.5, t["prop2"], color=C_SRC)
    txt(ax, xr + 4, yy - 14.5, t["prop3"], weight="bold")
    yy -= 23
    txt(ax, xr + 2, yy, t["cmp_t"], weight="bold", color=C_P2)
    txt(ax, xr + 4, yy - 4.5, t["cmp"], ls=1.15)
    box(ax, xr + 2, yb + 2, wr - 4, 11, fc="white", ec=C_P2, lw=0.6, ls="--", z=2)
    txt(ax, xr + 4, yb + 11, t["q2"], style="italic" if lang == "en" else "normal", color=C_P2, ls=1.15)
    # arrows from the setting to the parts
    arrow(ax, (157, y0 - 6), (157, yb + hb), lw=0.8)
    arrow(ax, (43, y0), (43, yb + hb), lw=0.8)
    txt(ax, 3, y0 + 4.5, t["eval"], ha="left", va="center", size=FS - 0.4, color=GREY)
    # slide versions: the setting band and the two parts as separate files (the full figure is too tall for a 16:9 slide)
    from matplotlib.transforms import Bbox
    sfx = "_ja" if lang == "ja" else ""
    for part, (ya, yb_) in {"setting": (y0 - 6.9, H), "parts": (0, yb + hb + 0.6)}.items():
        bb = Bbox([[0, ya * MM], [W * MM, yb_ * MM]])
        fig.savefig(OUT / f"fig_overview_study_{part}{sfx}.pdf", bbox_inches=bb)
        fig.savefig(OUT / f"fig_overview_study_{part}{sfx}.png", dpi=300, bbox_inches=bb)
    save(fig, "fig_overview_study" + sfx)


# ------------------------------------------------------------------ Fig. B
TB = {
    "en": dict(
        xlab="Numbers set on the K target labels  →",
        ylab="Per-emotion weights  →",
        cols=["none\n(fixed, or chosen on other corpora)", "one weight per classifier\n(3 numbers)", "classifier × emotion\n(12 numbers, or 1 of 1,001 tables)"],
        rows=["no\n(same weight for\nall emotions)", "yes, from the\nsource corpora", "yes, from the\nK target labels"],
        c00="Product (Sec. 5)\nMean of probabilities\nConfidence-weighted mean\nExponents chosen on\nother corpora",
        c10="Temperature scaling\nWeights fitted on the K labels\nLevel-only weights",
        c11="Proposed fusion\nstrength ← K labels\nprofile ← source corpora",
        c22="All-K table (12 reliabilities)\nConfusion re-reading\nSearch over 1,000 random\nweight tables",
        w00="below the proposed fusion on AFEW;\nexponents from other corpora: below it\non IEMOCAP and MELD",
        w10="above the proposed fusion on AFEW (K = 5);\nbelow it on IEMOCAP and MOSEI",
        w22="table search: below the proposed fusion on\nIEMOCAP, MOSEI, MELD; the other two:\nbelow it on MOSEI",
        w11="",
        note="Schematic. Where each fusion method takes its weights from.",
    ),
    "ja": dict(
        xlab="ターゲットの K 本で決める数  →",
        ylab="感情ごとの重み  →",
        cols=["なし\n（固定，またはほかのコーパスで選ぶ）", "分類器ごとに 1 つ\n（3 つの数）", "分類器 × 感情\n（12 個，または 1,001 個の表から 1 つ）"],
        rows=["なし\n（感情によらず\n同じ重み）", "あり\n（ソースで測る）", "あり\n（K 本で測る）"],
        c00="積（5 節）\n確率の平均\n確信度で重みづける平均\nほかのコーパスで選んだ\n指数の積",
        c10="温度の較正\nK 本で合わせた重み\n水準だけの重み",
        c11="提案\n強さ ← K 本\n得意・不得意 ← ソース",
        c22="全部を K 本で測った表\n取り違えの表で読み替え\n1,000 個の重みの表から\nK 本で選ぶ",
        w00="AFEW で提案より低い（指数の積は\nIEMOCAP と MELD でも低い）",
        w10="AFEW（K = 5）で提案より高く，\nIEMOCAP と MOSEI で低い",
        w22="表を選ぶ版は IEMOCAP，MOSEI，\nMELD で，ほかの 2 つは MOSEI\nで提案より低い",
        w11="",
        note="模式図．各まとめ方が重みをどこから決めるか",
    ),
}


def fig_map(lang):
    setup(lang)
    t = TB[lang]
    H = 104
    fig, ax = canvas(H)
    x0, y0 = 30, 8
    cw, rh = 46, 26
    # grid
    for i in range(3):
        for j in range(3):
            fc = "#F7F7F7"
            box(ax, x0 + i * cw + 0.8, y0 + j * rh + 0.8, cw - 1.6, rh - 1.6, fc=fc, ec="#DDDDDD", lw=0.5, r=1.0, z=1)
    for i, lab in enumerate(t["cols"]):
        txt(ax, x0 + i * cw + cw / 2, y0 + 3 * rh + 1.5, lab, ha="center", va="bottom", size=FS - 0.4, ls=1.1)
    for j, lab in enumerate(t["rows"]):
        txt(ax, x0 - 2, y0 + j * rh + rh / 2, lab, ha="right", va="center", size=FS - 0.4, ls=1.1)
    arrow(ax, (x0, y0 + 3 * rh + 11), (x0 + 3 * cw, y0 + 3 * rh + 11), lw=0.7, head=6)
    txt(ax, x0 + 3 * cw / 2, y0 + 3 * rh + 12.5, t["xlab"], ha="center", va="bottom", weight="bold")
    arrow(ax, (3, y0), (3, y0 + 3 * rh), lw=0.7, head=6)
    ax.text(1.2, y0 + 1.5 * rh, t["ylab"], rotation=90, ha="center", va="center", fontsize=FS, weight="bold")

    def cell(i, j, body, weak, fc, ec, bold_first=False):
        x, y = x0 + i * cw + 1.6, y0 + j * rh + 1.6
        box(ax, x, y, cw - 3.2, rh - 3.2, fc=fc, ec=ec, lw=0.9, r=1.0, z=2)
        lines = body.split("\n")
        if bold_first:
            txt(ax, x + 1.5, y + rh - 4.6, lines[0], weight="bold", color=ec)
            txt(ax, x + 1.5, y + rh - 8.6, "\n".join(lines[1:]), ls=1.1)
        else:
            txt(ax, x + 1.5, y + rh - 4.6, body, ls=1.1)
        if weak:
            txt(ax, x + 1.5, y + 1.0, weak, va="bottom", size=FS - 0.9, color=GREY, ls=1.05,
                style="italic" if lang == "en" else "normal")

    cell(0, 0, t["c00"], "", "white", "#444444")
    cell(1, 0, t["c10"], "", "white", "#444444")
    cell(1, 1, t["c11"], "", "#FDF0E8", C_PROP, bold_first=True)
    cell(2, 2, t["c22"], "", "white", "#444444")
    txt(ax, x0 + 3 * cw, 2.5, t["note"], ha="right", va="bottom", size=FS - 0.9, color=GREY,
        style="italic" if lang == "en" else "normal")
    save(fig, "fig_fusion_map" + ("_ja" if lang == "ja" else ""))


if __name__ == "__main__":
    for lang in ("en", "ja"):
        fig_overview(lang)
        fig_map(lang)
