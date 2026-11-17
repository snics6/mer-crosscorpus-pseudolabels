"""Figures written for the second version of the paper (ja_v2, then the English version), 2026-11-17.

fig_v2_questions_<lang> : overview of the three research questions on the processing flow (Ishii et al. 2022 Fig. 1 style:
                          the questions are written on the arrows they ask about). No values. 2026-11-17 (researcher): the box of
                          the previous way of making pseudo-labels (product of probabilities) and its dashed connections
                          were removed from both language versions; only the proposed fusion is drawn.
fig_v2_settings_<lang>  : the three settings (single-source, all-source, LOCO), one column each. No values.
The English versions are drawn at the text width of the manuscript (372 pt = 131.2 mm; included at width=\textwidth) with
all text at 8 pt or larger (Discover AI artwork rules); their layout is separate (fig_questions_en, fig_settings_en).
The Japanese versions are drawn as in the commit 9ea5e8c.
Output: ../figures/<name>.pdf and .png
Run:  python make_figs_v2.py [ja|en]
"""
import sys

from make_figs_overview import FS, GREY, W, W_EN, arrow, box, canvas, save, setup, txt

Q1A, Q1B, Q2 = "#8E2457", "#3B2F8F", "#8A5A00"
SRC, TGT = "#6A3D9A", "#1F78B4"

T = {
    "ja": dict(
        srcs=["ソースのコーパス（ラベルあり）", "ターゲットの K 本のラベル", "ターゲットのラベル無しの発話", "同じ発話のほかのモダリティ"],
        srchead="情報源",
        clf="ターゲットの線形分類器\n（声・顔・言葉・AVT）",
        err="誤り ＝ 答えの偏り ＋ 残りの誤り",
        q1a_h="問い 1(a)（4 節）", q1a="4 つの情報源は，答えの偏りと\n残りの誤りのどちらを直すか",
        teacher="ほかのモダリティを含む\n分類器の予測",
        pl="疑似ラベル",
        retr="疑似ラベルで学び直す\n（蒸留）",
        q1b_h="問い 1(b)（5 節）", q1b="どの分類器の UAR が，\nどんな場合に上がるか",
        comp="声・顔・言葉の\n分類器の答え",
        fuse_p="提案：感情ごとの信頼度で\nまとめる（late fusion）",
        q2_h="問い 2（6 節）", q2="疑似ラベルを作るために，\n答えをどうまとめるか",
    ),
    "en": dict(
        srcs=["Labelled source corpora", "K target labels per emotion", "Unlabelled target utterances",
              "Other modalities of\nthe same utterance"],
        srchead="Information sources",
        clf="Linear classifiers on\nthe target (audio,\nvisual, text, AVT)",
        err="Error = answer bias\n+ remaining error",
        q1a_h="RQ1(a) (Sec. 4)", q1a="Which part of the error\ndoes each information\nsource correct?",
        teacher="Predictions of classifiers\nincluding other modalities",
        pl="Pseudo-labels",
        retr="Retraining on the\npseudo-labels\n(distillation)",
        q1b_h="RQ1(b) (Sec. 5)", q1b="Which classifiers gain\nUAR, and when?",
        comp="Answers of the audio,\nvisual and text classifiers",
        fuse_p="Proposed: emotion-wise\nreliability-weighted\nlate fusion",
        q2_h="RQ2 (Sec. 6)", q2="How should the answers be\ncombined into pseudo-labels?",
    ),
}


def qbubble(ax, x, y, w, h, head, body, col, fs):
    box(ax, x, y, w, h, fc="white", ec=col, lw=1.1, r=1.4, z=3)
    txt(ax, x + 1.5, y + h - 1.2, head, size=fs, weight="bold", color=col, va="top", z=4)
    txt(ax, x + 1.5, y + h - 5.2, body, size=fs - 0.6, color=col, va="top", ls=1.2, style="italic", z=4)


def fig_questions(lang):
    if lang == "en":
        return fig_questions_en()
    t = T[lang]
    setup(lang)
    H = 84
    fig, ax = canvas(H)
    fs = FS + 0.4
    # sources column
    txt(ax, 2, H - 2, t["srchead"], size=fs, weight="bold", va="top")
    for i, s in enumerate(t["srcs"]):
        y = H - 12 - i * 7.6
        col = SRC if i == 0 else (TGT if i < 3 else "#555555")
        box(ax, 2, y, 46, 6.2, fc="white", ec=col, lw=0.9, r=1.0)
        txt(ax, 25, y + 3.1, s, ha="center", va="center", size=fs - 0.8, color=col)
    ax.plot([49.5, 52], [H - 8.9, H - 8.9], color=GREY, lw=0.8)
    ax.plot([49.5, 52], [H - 31.7, H - 31.7], color=GREY, lw=0.8)
    ax.plot([52, 52], [H - 31.7, H - 8.9], color=GREY, lw=0.8)
    # classifier
    box(ax, 112, H - 25.5, 58, 17.5, fc="#F2F2F2", ec="black", lw=1.0)
    txt(ax, 141, H - 14.5, t["clf"], ha="center", va="center", size=fs, ls=1.2)
    txt(ax, 141, H - 22.5, t["err"], ha="center", va="center", size=fs - 1.2, color=GREY)
    arrow(ax, (52, H - 20), (111.5, H - 20), lw=1.6, head=10, color=Q1A)
    qbubble(ax, 57, H - 17, 50, 14.5, t["q1a_h"], t["q1a"], Q1A, fs)
    # row 2: distillation
    y2 = H - 52
    box(ax, 2, y2, 46, 12, fc="white", ec="#555555", lw=0.9)
    txt(ax, 25, y2 + 6, t["teacher"], ha="center", va="center", size=fs - 0.8, ls=1.2)
    box(ax, 62, y2 + 2, 26, 8, fc="#FDF0E8", ec="#D55E00", lw=1.0)
    txt(ax, 75, y2 + 6, t["pl"], ha="center", va="center", size=fs - 0.6, color="#D55E00", weight="bold")
    arrow(ax, (48.5, y2 + 6), (61.5, y2 + 6), lw=1.1, head=8, color="#555555")
    box(ax, 112, y2, 58, 12, fc="#F2F2F2", ec="black", lw=1.0)
    txt(ax, 141, y2 + 6, t["retr"], ha="center", va="center", size=fs - 0.4, ls=1.2)
    arrow(ax, (88.5, y2 + 6), (111.5, y2 + 6), lw=1.6, head=10, color=Q1B)
    qbubble(ax, 88, y2 + 13.5, 36, 12.5, t["q1b_h"], t["q1b"], Q1B, fs - 0.4)
    arrow(ax, (141, y2 + 12.5), (141, H - 26), lw=0.8, head=6, color=GREY)
    # row 3: fusion (2026-11-17, researcher: the product box and its dashed connections are removed; only the proposal)
    y3 = -4
    box(ax, 2, y3 + 6, 46, 12, fc="white", ec="#555555", lw=0.9)
    txt(ax, 25, y3 + 12, t["comp"], ha="center", va="center", size=fs - 0.8, ls=1.2)
    box(ax, 56, y3 + 7, 40, 10, fc="#FDF0E8", ec=Q2, lw=1.3)
    txt(ax, 76, y3 + 12, t["fuse_p"], ha="center", va="center", size=fs - 1.0, color=Q2, ls=1.2, weight="bold")
    arrow(ax, (48.5, y3 + 12), (55.5, y3 + 12), lw=1.0, head=7, color="#555555")
    arrow(ax, (75.5, y3 + 17.5), (75, y2 + 1.5), lw=1.6, head=10, color=Q2)
    qbubble(ax, 4, y3 + 21, 48, 13.5, t["q2_h"], t["q2"], Q2, fs)
    save(fig, f"fig_v2_questions_{lang}")


TS = {
    "ja": dict(
        cols=[("単一ソース設定", "4 節（問い 1(a)）", "ソース：1 つの群", "ターゲット：1 つの群",
               "ソースの全発話と K 本をまとめて学ぶ", "合併の分類器\nターゲットだけの分類器\nソースだけの分類器\n切片だけの直し，平均を引く方法", "K = 1〜全件"),
              ("全ソース設定", "4 節（問い 1(a)）", "ソース：ターゲット以外の\n全部の群", "ターゲット：1 つの群",
               "全部の群のアンカーを K 本で学び直す", "K 本の基準の分類器\n疑似ラベルで学び直し\n同じモダリティの 8 手法", "K = 0，5，25"),
              ("LOCO 設定", "5 節，6 節（問い 1(b)，2）", "ソース：残る 3 コーパス", "ターゲット：4 コーパスの 1 つ\n（6 節は CREMA-D も．ソースは 4 つ）",
               "3 つのソースの平均をアンカーにし，\nK 本で学び直す", "K 本の基準の分類器\n蒸留（疑似ラベルで学び直し）\nほかの 11 手法（5 節），提案（6 節）", "K = 0，5，10，25")],
        note="群：IEMOCAP の即興と台本，MOSEI，MELD，AFEW，CREMA-D，RAVDESS の 7 つ"),
    "en": dict(
        cols=[("Single-source setting", "Sec. 4 (RQ1(a))", "Source: one group", "Target: one group\nK = 1 to all",
               "Train on all source utterances\nand the K target labels",
               "Merged, target-only and\nsource-only classifiers;\nintercept-only fix;\nmean subtraction"),
              ("All-source setting", "Sec. 4 (RQ1(a))", "Sources: all groups\nexcept the target", "Target: one group\nK = 0, 5, 25",
               "Retrain an anchor from all\ngroups on the K labels",
               "K-label baseline\nPseudo-label retraining\nEight same-modality\nmethods"),
              ("LOCO setting", "Secs. 5, 6 (RQ1(b), RQ2)", "Sources: the other\nthree corpora",
               "Target: one of four corpora\n(+ CREMA-D in Sec. 6)\nK = 0, 5, 10, 25",
               "Anchor = mean of the three\nsources, retrained on K labels",
               "K-label baseline\nDistillation\n11 other methods (Sec. 5)\nProposal (Sec. 6)")],
        note=None),   # the list of groups is left to the caption
}


def fig_settings(lang):
    if lang == "en":
        return fig_settings_en()
    t = TS[lang]
    setup(lang)
    H = 76
    fig, ax = canvas(H)
    fs = FS + 0.6
    for k, (name, sec, src, tgt, how, clf, ks) in enumerate(t["cols"]):
        x, w = 2 + k * 57.5, 55
        box(ax, x, 5, w, H - 7, fc="#FAFAFA", ec="black", lw=1.0)
        txt(ax, x + 2, H - 4, name, size=fs + 1.0, weight="bold", va="top")
        txt(ax, x + w - 1.5, H - 4.6, sec, ha="right", size=fs - 1.6, color=GREY, va="top")
        box(ax, x + 2, H - 22, w - 4, 10, fc="#F0EAF5", ec=SRC, lw=0.9)
        txt(ax, x + 3.5, H - 17, src, size=fs - 0.6, color=SRC, va="center", ls=1.15)
        box(ax, x + 2, H - 35, w - 4, 11, fc="white", ec=TGT, lw=0.9)
        txt(ax, x + 3.5, H - 29.5, tgt, size=fs - 0.6, color=TGT, va="center", ls=1.15)
        txt(ax, x + 2.5, H - 37.5, how, size=fs - 1.0, va="top", ls=1.15)
        box(ax, x + 2, 10, w - 4, 19, fc="white", ec="black", lw=0.8)
        txt(ax, x + 3.5, 27.5, clf, size=fs - 1.2, va="top", ls=1.25)
        txt(ax, x + 2.5, 7.5, ks, size=fs - 0.6, va="center")
    txt(ax, W - 1, 0.6, t["note"], ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, f"fig_v2_settings_{lang}")


def qbubble_en(ax, x, y, w, h, head, body, col):
    box(ax, x, y, w, h, fc="white", ec=col, lw=1.0, r=1.2, z=3)
    txt(ax, x + 1.4, y + h - 1.0, head, size=8.5, weight="bold", color=col, va="top", z=4)
    txt(ax, x + 1.4, y + h - 4.6, body, size=8, color=col, va="top", ls=1.15, style="italic", z=4)


def fig_questions_en():
    """English version at the text width (131.2 mm), all text 8 pt or larger."""
    t = T["en"]
    setup("en")
    WE, H = W_EN, 63.5
    fig, ax = canvas(H, WE)
    # sources column
    txt(ax, 1, H - 1.0, t["srchead"], size=8.5, weight="bold", va="top")
    ys = [H - 11.0, H - 17.6, H - 24.2, H - 33.6]
    hs = [5.6, 5.6, 5.6, 8.2]
    for i, (s, y, h) in enumerate(zip(t["srcs"], ys, hs)):
        col = SRC if i == 0 else (TGT if i < 3 else "#555555")
        box(ax, 1, y, 41, h, fc="white", ec=col, lw=0.9, r=1.0)
        txt(ax, 21.5, y + h / 2, s, ha="center", va="center", size=8, color=col, ls=1.1)
    yt, yb = ys[0] + hs[0] / 2, ys[3] + hs[3] / 2
    ax.plot([42.5, 44.5], [yt, yt], color=GREY, lw=0.8)
    ax.plot([42.5, 44.5], [yb, yb], color=GREY, lw=0.8)
    ax.plot([44.5, 44.5], [yb, yt], color=GREY, lw=0.8)
    # classifier
    cx0, cw = 86, WE - 87
    box(ax, cx0, H - 27, cw, 20, fc="#F2F2F2", ec="black", lw=1.0)
    txt(ax, cx0 + cw / 2, H - 13.2, t["clf"], ha="center", va="center", size=8, ls=1.15)
    txt(ax, cx0 + cw / 2, H - 22.6, t["err"], ha="center", va="center", size=8, color=GREY, ls=1.1)
    ya = H - 21
    arrow(ax, (44.5, ya), (cx0 - 0.5, ya), lw=1.6, head=9, color=Q1A)
    qbubble_en(ax, 46.5, H - 19.6, 38, 16.4, t["q1a_h"], t["q1a"], Q1A)
    # row 2: distillation
    y2 = H - 42
    box(ax, 1, y2 - 5, 41, 10, fc="white", ec="#555555", lw=0.9)
    txt(ax, 21.5, y2, t["teacher"], ha="center", va="center", size=8, ls=1.15)
    box(ax, 50, y2 - 4, 24, 8, fc="#FDF0E8", ec="#D55E00", lw=1.0)
    txt(ax, 62, y2, t["pl"], ha="center", va="center", size=8, color="#D55E00", weight="bold")
    arrow(ax, (42.5, y2), (49.5, y2), lw=1.1, head=7, color="#555555")
    box(ax, cx0, y2 - 6, cw, 12, fc="#F2F2F2", ec="black", lw=1.0)
    txt(ax, cx0 + cw / 2, y2, t["retr"], ha="center", va="center", size=8, ls=1.1)
    arrow(ax, (74.5, y2), (cx0 - 0.5, y2), lw=1.6, head=9, color=Q1B)
    qbubble_en(ax, 46.5, y2 + 6.5, 38, 12.8, t["q1b_h"], t["q1b"], Q1B)
    arrow(ax, (cx0 + cw / 2, y2 + 6.5), (cx0 + cw / 2, H - 27.5), lw=0.8, head=6, color=GREY)
    # row 3: fusion (2026-11-17, researcher: the product box and its dashed connections are removed; only the proposal)
    box(ax, 1, 2.6, 41, 9, fc="white", ec="#555555", lw=0.9)
    txt(ax, 21.5, 7.1, t["comp"], ha="center", va="center", size=8, ls=1.15)
    box(ax, 47, 1.35, 37.5, 11.5, fc="#FDF0E8", ec=Q2, lw=1.3)
    txt(ax, 65.75, 7.1, t["fuse_p"], ha="center", va="center", size=8, color=Q2, ls=1.1, weight="bold")
    arrow(ax, (42.5, 7.1), (46.5, 7.1), lw=1.0, head=7, color="#555555")
    arrow(ax, (62, 13.15), (62, y2 - 4.4), lw=1.6, head=9, color=Q2)
    qbubble_en(ax, 89.5, 1.0, WE - 90.5, 12.8, t["q2_h"], t["q2"], Q2)
    save(fig, "fig_v2_questions_en")


def fig_settings_en():
    """English version at the text width (131.2 mm) and at most 145 pt tall, all text 8 pt or larger."""
    t = TS["en"]
    setup("en")
    WE, H = W_EN, 51
    fig, ax = canvas(H, WE)
    w = 42.2
    for k, (name, sec, src, tgt, how, clf) in enumerate(t["cols"]):
        x = 0.5 + k * (w + 2.2)
        box(ax, x, 0.5, w, H - 1.0, fc="#FAFAFA", ec="black", lw=1.0)
        txt(ax, x + 1.5, H - 1.4, name, size=9, weight="bold", va="top")
        txt(ax, x + 1.5, H - 5.4, sec, size=8, color=GREY, va="top")
        box(ax, x + 1.2, 33.0, w - 2.4, 7.8, fc="#F0EAF5", ec=SRC, lw=0.9)
        txt(ax, x + 2.4, 36.9, src, size=8, color=SRC, va="center", ls=1.1)
        box(ax, x + 1.2, 21.6, w - 2.4, 10.6, fc="white", ec=TGT, lw=0.9)
        txt(ax, x + 2.4, 26.9, tgt, size=8, color=TGT, va="center", ls=1.1)
        txt(ax, x + 1.6, 20.6, how, size=8, va="top", ls=1.1)
        box(ax, x + 1.2, 1.4, w - 2.4, 12.9, fc="white", ec="black", lw=0.8)
        txt(ax, x + 2.4, 7.85, clf, size=8, va="center", ls=1.08)
    save(fig, "fig_v2_settings_en")


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["ja", "en"]):
        fig_questions(lang)
        fig_settings(lang)
