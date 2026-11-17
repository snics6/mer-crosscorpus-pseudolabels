"""Slide figures (2026-11-16 night, researcher: "p15について，もう少し詳しく図を書いて伝わるようにして"):

fig_margin_toy_ja   : how the margin (当たる幅) is computed, on 8 made-up candidate utterances
                      (true labels, judgments of the pseudo-label source, judgments of the classifier before retraining,
                      per-emotion recall, UAR, margin), and how it is paired with the gain on the test utterances.
fig_margin_real_ja  : the two measured examples with the two candidate UARs that make the margin
                      (teacher_uar_pool = (a), self_uar_pool = (b)) and the gain with its 95% interval.
Values of the real examples: results/dai1/stats/teacher_margin.csv
  (MOSEI, student A, K = 25, teacher pair_VT; AFEW, student V, K = 5, teacher pair_AT).
English versions fig_margin_toy_en and fig_margin_real_en (2026-11-17) for the manuscript: same values, text 8-12 pt at
the final width of 174 mm, ⓐ/ⓑ written out as "pseudo-label source" / "before retraining" (Arial has no circled letters).
The Japanese versions are drawn exactly as before.
Output: ../figures/<name>.pdf and .png
Run:  python make_fig_margin_detail.py [ja|en]
"""
import sys
from pathlib import Path

import matplotlib.patches as mp
import pandas as pd

from make_figs_overview import C_PROP, C_U, FS, GREY, W, W_EN, arrow, box, canvas, save, setup, txt

ROOT = Path(__file__).resolve().parents[2]
C_OK, C_NG = "#009E73", "#D55E00"
C_UP, C_DOWN = "#0072B2", "#D55E00"
EMO = ["N", "H", "S", "A"]


def okmark(ax, x, y, ok, color, ms=3.6):
    """English figures: an open circle (same as the reference) or a cross (different), drawn as a marker because the
    Arial circle glyph is very small at 8 pt."""
    ax.plot([x], [y], marker="o" if ok else "x", ms=ms, mfc="none", mec=color, mew=1.0, zorder=7, clip_on=False)


TTOY = {
    "ja": dict(
        rows=["正解（解析のときだけ使う）", "ⓐ 疑似ラベルの出所の判定", "ⓑ 学び直す前の分類器（K 本の基準）の判定"],
        head="① ターゲットの候補の発話（K 本を除く．ここでは 8 本の例）",
        utt="発話 {i}", utt_left=None,
        rec="感情ごとの再現率 → UAR", sep="，",
        legend="○ は正解と同じ判定，× は違う判定．N：Neutral，H：Happy，S：Sad，A：Angry．UAR は 4 感情の再現率の平均",
        m_head="② 当たる幅 ＝ ⓐ の UAR − ⓑ の UAR",
        m_body="この例では 87.5 − 50.0 ＝ +37.5 点．\n疑似ラベルのほうが，学び直す前の分類器より\n37.5 点多く当たる．負なら，疑似ラベルのほうが当たらない",
        g_head="③ gain：この疑似ラベルで学び直して比べる",
        g_body="学び直す前の分類器を，ⓐ の判定の確率を\n疑似ラベルにして学び直す．テストの発話で\ngain ＝ 学び直した分類器 − 学び直す前の分類器",
        step4="④ 疑似ラベルの出所を替えて，②（当たる幅）と ③（gain）の組を K = 5 と 25 で 360 組ずつ測り，2 つの関係を調べる",
        note="8 本の発話と判定は説明のための例で，実験の値ではない",
    ),
    "en": dict(
        rows=["Truth (used only\nin the analysis)", "Pseudo-label\nprovider", "Classifier before\nretraining (K-label\nbaseline)"],
        head="(1) Decisions on target candidate utterances (K labels excluded; 8 shown as an example)",
        utt="{i}", utt_left="Utterance",
        rec="Recall per emotion →\nUAR (mean of the 4)", sep=", ",
        legend=("same as the truth,", "different. N: Neutral, H: Happy, S: Sad, A: Angry."),
        m_head="(2) Advantage = UAR of the provider\n      − UAR before retraining",
        m_body="Here 87.5 − 50.0 = +37.5 points: the\nprovider’s UAR is 37.5 points higher.\n"
               "A negative advantage means the\npseudo-labels have lower UAR.",
        g_head="(3) Gain: retrain on these\n      pseudo-labels and compare",
        g_body="The classifier before retraining is\nretrained on the provider’s probabilities\n"
               "as pseudo-labels. On the test utterances,\ngain = retrained − before retraining.",
        step4="(4) Vary the provider: 360 (advantage, gain) pairs at each of K = 5 and 25.",
        note=None,   # the illustration note goes to the caption
    ),
}


def fig_toy(lang="ja"):
    """The Japanese layout is the one drawn before the English version was added (first argument of L)."""
    setup(lang)
    t = TTOY[lang]
    L = (lambda ja, en: ja) if lang == "ja" else (lambda ja, en: en)
    H = L(84, 72)
    fig, ax = canvas(H, L(W, W_EN))
    fs = FS + 0.5
    true = ["N", "N", "H", "H", "S", "S", "A", "A"]
    rows = [(t["rows"][0], true, "black"),
            (t["rows"][1], ["N", "N", "H", "H", "S", "A", "A", "A"], C_PROP),
            (t["rows"][2], ["N", "H", "H", "N", "N", "S", "A", "N"], "black")]
    lx = L(2, 1)
    txt(ax, lx, H - L(3, 1), t["head"], size=L(fs, 8.5), weight="bold")
    x0, cw, ch = L((66, 8.2, 7.5), (37, 7.0, 7.0))
    for i in range(8):
        txt(ax, x0 + (i + 0.5) * cw, H - L(9.5, 7.4), t["utt"].format(i=i + 1), ha="center", va="center", size=L(fs - 1.8, 8),
            color=GREY)
    if t["utt_left"]:
        txt(ax, lx, H - 7.4, t["utt_left"], va="center", size=8, color=GREY)
    ys = [H - L(20, 16.6), H - L(30, 25.6), H - L(40, 34.6)]
    for k, ((name, pred, col), y) in enumerate(zip(rows, ys)):
        txt(ax, lx, y + ch / 2, name, va="center", size=L(fs - 0.3, 8), color=col, weight="bold" if col != "black" else "normal",
            ls=L(1.2, 1.1))
        for i, p in enumerate(pred):
            ok = p == true[i]
            fc = "#F2F2F2" if k == 0 else ("#E5F5EF" if ok else "#FBE3D6")
            box(ax, x0 + i * cw + 0.4, y, cw - 0.8, ch, fc=fc, ec="#999999", lw=0.5, r=0.6, z=2)
            c = "black" if k == 0 else (C_OK if ok else C_NG)
            if lang == "ja":
                mark = "" if k == 0 else (" ○" if ok else " ×")
                txt(ax, x0 + (i + 0.5) * cw, y + ch / 2, p + mark, ha="center", va="center", size=fs - 0.3, color=c, weight="bold")
            elif k == 0:
                txt(ax, x0 + (i + 0.5) * cw, y + ch / 2, p, ha="center", va="center", size=8.5, color=c, weight="bold")
            else:
                txt(ax, x0 + (i + 0.5) * cw - 1.1, y + ch / 2, p, ha="center", va="center", size=8.5, color=c, weight="bold")
                okmark(ax, x0 + (i + 0.5) * cw + 1.7, y + ch / 2, ok, c, ms=3.3)
    # per-emotion recall and UAR
    xr = x0 + 8 * cw + L(3, 2)
    txt(ax, xr, H - L(9.5, 4.9), t["rec"], size=L(fs - 1.0, 8), color=GREY, va="top", ls=L(1.2, 1.1))
    for (name, pred, col), y in zip(rows[1:], ys[1:]):
        rec = [sum(1 for tr, p in zip(true, pred) if tr == e and p == e) / 2 for e in EMO]
        uar = 100 * sum(rec) / 4
        s = t["sep"].join(f"{e} {r:g}" for e, r in zip(EMO, rec))
        txt(ax, xr, y + ch / 2 + 1.6, s, va="center", size=L(fs - 1.2, 8))
        txt(ax, xr, y + ch / 2 - 2.0, f"UAR {uar:.1f}", va="center", size=L(fs, 9), weight="bold", color=col)
    if lang == "ja":
        txt(ax, 2, ys[2] - 4, t["legend"], size=fs - 1.2, color=GREY)
    else:   # marker, text, marker, text; the second marker is placed after the measured end of the first text
        yl = ys[2] - 3.9
        okmark(ax, 2.0, yl, True, GREY, ms=3.2)
        a = ax.text(3.6, yl, t["legend"][0], va="center", fontsize=8, color=GREY)
        x1 = ax.transData.inverted().transform(a.get_window_extent(fig.canvas.get_renderer()).p1)[0]
        okmark(ax, x1 + 2.2, yl, False, GREY, ms=3.2)
        txt(ax, x1 + 3.8, yl, t["legend"][1], va="center", size=8, color=GREY)
    # margin
    by, bh = L(11, 6), L(22, 23.6)
    bx1, bx2, bw = L((2, 92, 80), (0.6, 67.6, 63))
    box(ax, bx1, by, bw, bh, fc="#FDF0E8", ec=C_PROP, lw=1.1)
    txt(ax, bx1 + L(3, 1.6), by + bh - L(3.5, 1.2), t["m_head"], size=L(fs, 8.5), weight="bold", color=C_PROP, ls=L(1.2, 1.1))
    txt(ax, bx1 + L(3, 1.6), by + bh - L(8.5, 9.0), t["m_body"], size=L(fs - 0.6, 8), ls=L(1.3, 1.2), va="top")
    arrow(ax, (bx1 + bw, by + L(11, bh / 2)), (bx2, by + L(11, bh / 2)), lw=1.0, head=L(8, 6))
    box(ax, bx2, by, bw, bh, fc="#E8F0F8", ec=C_UP, lw=1.1)
    txt(ax, bx2 + L(3, 1.6), by + bh - L(3.5, 1.2), t["g_head"], size=L(fs, 8.5), weight="bold", color=C_UP, ls=L(1.2, 1.1))
    txt(ax, bx2 + L(3, 1.6), by + bh - L(8.5, 9.0), t["g_body"], size=L(fs - 0.6, 8), ls=L(1.3, 1.2), va="top")
    txt(ax, lx, L(8.0, 4.4), t["step4"], size=L(fs - 0.6, 8))
    if t["note"]:
        txt(ax, 2, 3.6, t["note"], size=fs - 1.4, color=GREY)
    save(fig, f"fig_margin_toy_{lang}")


TREAL = {
    "ja": dict(
        ex=[("例 1：MOSEI，K = 25", "声の分類器を，顔と言葉をつないだ分類器の\n疑似ラベルで学び直す", "上がった"),
            ("例 2：AFEW，K = 5", "顔の分類器（AFEW で最も強い）を，声と言葉を\nつないだ分類器の疑似ラベルで学び直す", "下がった")],
        cand="候補の上の UAR（K 本を除く）",
        bars=["ⓐ 疑似ラベル", "ⓑ 学び直す前"],
        margin="当たる幅 ＝ ⓐ − ⓑ ＝ {a:.1f} − {b:.1f} ＝ {m:+.1f} 点",
        gain="テストの UAR の差（gain．学び直し − 学び直す前）",
        ci="95% 区間 [{lo:+.2f}, {hi:+.2f}] は 0 を含まない → UAR が{word}",
    ),
    "en": dict(
        ex=[("(a) MOSEI, K = 25", "Audio classifier retrained on the pseudo-labels of the\nclassifier on the visual and text features concatenated",
             "rose"),
            ("(b) AFEW, K = 5", "Visual classifier (the strongest on AFEW) retrained on\nthe pseudo-labels of the classifier on the audio and\n"
             "text features concatenated", "fell")],
        cand="UAR on the candidates (K labels excluded)",
        bars=["Pseudo-labels", "Before retraining"],
        margin="Margin = {a:.1f} − {b:.1f} = {m:+.1f} points",
        gain="Gain: test UAR, retrained − before retraining",
        ci="95% interval [{lo:+.2f}, {hi:+.2f}] excludes 0 → UAR {word}",
    ),
}


def fig_real(lang="ja"):
    """The Japanese layout is the one drawn before the English version was added (first argument of L)."""
    setup(lang)
    t = TREAL[lang]
    L = (lambda ja, en: ja) if lang == "ja" else (lambda ja, en: en)
    d = pd.read_csv(ROOT / "results" / "dai1" / "stats" / "teacher_margin.csv", comment="#")
    keys = [("MOSEI", "A", 25, "pair_VT", C_UP), ("AFEW", "V", 5, "pair_AT", C_DOWN)]
    ex = [(head, desc, T, st, K, tea, col, word) for (head, desc, word), (T, st, K, tea, col) in zip(t["ex"], keys)]
    H = L(74, 74)
    fig, ax = canvas(H)
    fs = FS + 0.5
    for j, (head, desc, T, st, K, tea, col, word) in enumerate(ex):
        r = d[(d["T"] == T) & (d.student == st) & (d.K == K) & (d.teacher == tea)].iloc[0]
        x = 2 + j * 86
        box(ax, x, 2, 84, H - 4, fc="white", ec=col, lw=1.2)
        txt(ax, x + 2, H - 5.5, head, size=L(fs + 0.4, 9), weight="bold", color=col)
        txt(ax, x + 2, H - L(9, 10), desc, size=L(fs - 0.8, 8), va="top", ls=1.2)
        # candidate UARs
        txt(ax, x + 2, H - L(20, 22.2), t["cand"], size=L(fs - 0.6, 8), weight="bold")
        sc = 0.62
        bx = x + L(26, 28)
        for k, (lab, v, c) in enumerate([(t["bars"][0], r.teacher_uar_pool, C_PROP), (t["bars"][1], r.self_uar_pool, "#555555")]):
            y = H - L(26.5, 29) - k * 6.5
            txt(ax, x + 3, y, lab, va="center", size=L(fs - 0.6, 8), color=c)
            ax.add_patch(mp.Rectangle((bx, y - 2.2), v * sc, 4.4, fc=c, ec="none", alpha=0.85, zorder=3))
            txt(ax, bx + v * sc + 1, y, f"{v:.1f}", va="center", size=L(fs - 0.4, 8.5), weight="bold")
        txt(ax, x + 3, H - L(41, 42), t["margin"].format(a=r.teacher_uar_pool, b=r.self_uar_pool, m=r.margin).replace("-", "−").replace("＝ −", "＝ −"),
            size=L(fs - 0.2, 8.5), weight="bold", color=C_PROP)
        # gain
        txt(ax, x + 2, H - L(49, 50), t["gain"], size=L(fs - 0.6, 8), weight="bold")
        x0 = x + 42
        gs = 4.0
        ax.plot([x0, x0], [10, 20], color="black", lw=0.6)
        ax.add_patch(mp.Rectangle((min(x0, x0 + r.gain * gs), 14), abs(r.gain) * gs, 4, fc=col, ec="none", zorder=3))
        lab = f"{r.gain:+.2f}".replace("-", "−")
        txt(ax, x0 + r.gain * gs + (1 if r.gain > 0 else -1), 16, lab, ha="left" if r.gain > 0 else "right", va="center",
            size=L(fs, 8.5), weight="bold")
        ax.plot([x0 + r.gain_lo * gs, x0 + r.gain_hi * gs], [11.8, 11.8], color=col, lw=0.9)
        for e in (r.gain_lo, r.gain_hi):
            ax.plot([x0 + e * gs] * 2, [11.0, 12.6], color=col, lw=0.9)
        txt(ax, x0, 9.6, "0", ha="center", va="top", size=L(fs - 1.2, 8), color=GREY)
        txt(ax, x + 3, L(6.2, 5.6), t["ci"].format(lo=r.gain_lo, hi=r.gain_hi, word=word).replace("-", "−"),
            size=L(fs - 0.9, 8), color=col, weight="bold")
    save(fig, f"fig_margin_real_{lang}")


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["ja", "en"]):
        fig_toy(lang)
        fig_real(lang)
