"""Figures for the Japanese draft ver4 (2026-11-17 night; researcher: "図を作って", one figure = one conclusion,
following Mensh & Kording 2017 rules 7 and 9 and the MIT CommLab results formula).

fig_v4_overview_ja : Fig. 1, overview: pseudo-labels made from the other modalities of the same utterance, combined
                     with equal weights (product) or with the proposal's per-emotion weights, versus the 11 methods
                     that use no pseudo-labels including other modalities (conceptual, no data).
fig_v4_methods_ja  : Fig. 3, audio classifier, distillation - each of the 11 methods (4 corpora x K = 5, 10, 25 =
                     132 comparisons) with 95% intervals. Supports Sect. 4.1.
fig_v4_prop_ja     : Fig. 5, (a) proposal - product (fused predictions) and (b) AVT retrained on the proposal's
                     pseudo-labels - AVT retrained on the product's pseudo-labels, 12 conditions each. Supports
                     Sects. 4.4 and 4.6.

Values:
  Fig. 3: results/dai2/stats/prop_minus_method.csv (student = A, K in 5/10/25, the 11 methods of Table 2;
          columns effect, lo, hi). The script checks 132 rows, 132 positive, 122 with lo > 0, 10 not above
          zero all on IEMOCAP (Sect. 4.1).
  Fig. 5: results/dai20_eval/summary/effects.csv (a = prop, b = prod, seeds = all) and
          results/dai21_eval5/summary/effects.csv (part = 4corpora, a = stud_prop, b = stud_prod, seeds = all).
          The script checks 11 of 12 positive in each panel, and the counts of intervals above zero (7 and 5).
The values drawn are written to ../figures/data/fig_v4.csv.
Run:  python make_figs_v4.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from make_figs_overview import C_K, C_PROP, C_SRC, C_U, FS, GREY, MM, MOD, W, W_EN, arrow, box, canvas, save, setup, txt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "figures" / "data"
CORPORA = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
KS = [5, 10, 25]
KMARK = {5: "o", 10: "s", 25: "^"}
METHODS = [("self", "自己学習"), ("entmin", "エントロピー最小化"), ("im", "IM"), ("t3a", "T3A"),
           ("lp", "ラベル伝播"), ("own", "自身の確率"), ("st", "S+T"), ("coral", "CORAL"),
           ("dann", "DANN"), ("mme", "MME"), ("xmuda", "xMUDA 型")]
METHODS_EN = {"self": "Self-training", "entmin": "Entropy min.", "im": "IM", "t3a": "T3A", "lp": "Label propagation",
              "own": "Own probabilities", "st": "S+T", "coral": "CORAL", "dann": "DANN", "mme": "MME",
              "xmuda": "xMUDA-style"}
TL = {"ja": dict(above="区間が 0 より上", incl="区間が 0 を含む", xl3="蒸留 − 比べる手法（声の分類器の UAR の差，点）",
                 iem="IEMOCAP（即興）", iem2="IEMOCAP\n（即興）", yl5="UAR の差（点）",
                 t5=["(a) まとめた予測：提案 − 積", "(b) 学び直した AVT：提案の疑似ラベル − 積の疑似ラベル"]),
      "en": dict(above="interval above 0", incl="interval includes 0", xl3="Distillation $-$ method (audio UAR, points)",
                 iem="IEMOCAP (impro.)", iem2="IEMOCAP\n(impro.)", yl5="Difference in UAR (points)",
                 t5=["(a) Fused predictions: proposal $-$ product", "(b) Retrained AVT: proposal $-$ product pseudo-labels"])}
rows = []


def k_legend(ax, color, loc, ncol=5, extra=True, lang="ja", fs=FS):
    h = [Line2D([], [], ls="", marker=KMARK[k], ms=4.2, mfc=color, mec=color, label=f"K = {k}") for k in KS]
    if extra:
        h += [Line2D([], [], ls="", marker="o", ms=4.2, mfc=color, mec=color, label=TL[lang]["above"]),
              Line2D([], [], ls="", marker="o", ms=4.2, mfc="white", mec=color, label=TL[lang]["incl"])]
    ax.legend(handles=h, loc=loc, ncol=ncol, frameon=False, fontsize=fs - 0.4, handletextpad=0.2,
              columnspacing=1.0, borderaxespad=0.0)


# ------------------------------------------------------------------ Fig. 3
def fig_methods(lang="ja"):
    d = pd.read_csv(ROOT / "results/dai2/stats/prop_minus_method.csv")
    keys = [m for m, _ in METHODS]
    s = d[(d.student == "A") & (d.K.isin(KS)) & (d.method.isin(keys))].copy()
    assert len(s) == 132 and (s.effect > 0).sum() == 132 and (s.lo > 0).sum() == 122
    assert set(s[~(s.lo > 0)]["T"]) == {"IEMOCAP"} and (~(s.lo > 0)).sum() == 10
    setup(lang)
    fs = FS if lang == "ja" else 6.6
    xmax, xmin = 16.0, -5.0
    Wf = W if lang == "ja" else W_EN
    fig, axes = plt.subplots(1, 4, figsize=(Wf * MM, (92 if lang == "ja" else 84) * MM), sharey=True)
    fig.subplots_adjust(left=0.165 if lang == "ja" else 0.17, right=0.985, top=0.86, bottom=0.12, wspace=0.08)
    col = MOD["A"][0]
    for ax, c in zip(axes, CORPORA):
        ax.axvline(0, color="black", lw=0.6, zorder=1)
        for i, (m, _) in enumerate(METHODS):
            for j, K in enumerate(KS):
                r = s[(s["T"] == c) & (s.method == m) & (s.K == K)].iloc[0]
                y = i + (j - 1) * 0.26
                lo, hi, e = r.lo, r.hi, r.effect
                if lang == "ja":
                    rows.append(dict(fig="fig3", corpus=c, method=m, K=K, effect=e, lo=lo, hi=hi))
                ax.plot([max(lo, xmin), min(hi, xmax)], [y, y], color=col, lw=0.7, zorder=2, solid_capstyle="butt")
                filled = lo > 0
                if e <= xmax:
                    ax.plot(e, y, ls="", marker=KMARK[K], ms=3.6, mfc=col if filled else "white", mec=col,
                            mew=0.8, zorder=3)
                else:
                    ax.plot(xmax, y, ls="", marker=">", ms=3.6, mfc=col if filled else "white", mec=col, mew=0.8,
                            zorder=3, clip_on=False)
            big = s[(s["T"] == c) & (s.method == m)].effect
            if big.max() > xmax:
                ax.text(xmax - 0.4, i - 0.55, f"+{big.min():.0f}" + ("〜" if lang == "ja" else "–") + f"+{big.max():.0f}",
                        ha="right", va="center", fontsize=fs - 1.4, color=GREY)
        ax.set_xlim(xmin, xmax)
        ax.set_title(c if c != "IEMOCAP" else TL[lang]["iem"], fontsize=fs + 0.4, pad=3)
        ax.set_xticks([0, 5, 10, 15])
        ax.tick_params(labelsize=fs - 0.6, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    axes[0].set_yticks(range(len(METHODS)))
    axes[0].set_yticklabels([n if lang == "ja" else METHODS_EN[m] for m, n in METHODS], fontsize=fs - 0.2)
    axes[0].invert_yaxis()
    fig.text(0.575, 0.02, TL[lang]["xl3"], ha="center", fontsize=fs)
    k_legend(axes[0], col, loc="lower left", lang=lang, fs=fs)
    axes[0].get_legend().set_bbox_to_anchor((0.0 if lang == "ja" else -0.55, 1.10), transform=axes[0].transAxes)
    save(fig, f"fig_v4_methods_{lang}")


# ------------------------------------------------------------------ Fig. 5
def fig_prop(lang="ja"):
    a = pd.read_csv(ROOT / "results/dai20_eval/summary/effects.csv")
    a = a[(a.a == "prop") & (a.b == "prod") & (a.seeds == "all")]
    b = pd.read_csv(ROOT / "results/dai21_eval5/summary/effects.csv")
    b = b[(b.part == "4corpora") & (b.a == "stud_prop") & (b.b == "stud_prod") & (b.seeds == "all")]
    for x, n_above in ((a, 7), (b, 5)):
        assert len(x) == 12 and (x.est > 0).sum() == 11 and (x.lo > 0).sum() == n_above, (len(x), (x.lo > 0).sum())
    setup(lang)
    fs = FS if lang == "ja" else 6.6
    Wf = W if lang == "ja" else W_EN
    fig, axes = plt.subplots(1, 2, figsize=(Wf * MM, (62 if lang == "ja" else 58) * MM), sharey=True)
    fig.subplots_adjust(left=0.08 if lang == "ja" else 0.10, right=0.99, top=0.80, bottom=0.14, wspace=0.06)
    titles = TL[lang]["t5"]
    for ax, x, t, tag in zip(axes, (a, b), titles, ("fig5a", "fig5b")):
        ax.axhline(0, color="black", lw=0.6, zorder=1)
        for ci, c in enumerate(CORPORA):
            for j, K in enumerate(KS):
                r = x[(x.c == c) & (x.K == K)].iloc[0]
                xx = ci + (j - 1) * 0.24
                if lang == "ja":
                    rows.append(dict(fig=tag, corpus=c, K=K, effect=r.est, lo=r.lo, hi=r.hi))
                ax.plot([xx, xx], [r.lo, r.hi], color=C_PROP, lw=0.8, zorder=2)
                ax.plot(xx, r.est, ls="", marker=KMARK[K], ms=4.0, mfc=C_PROP if r.lo > 0 else "white", mec=C_PROP,
                        mew=0.8, zorder=3)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["AFEW", TL[lang]["iem2"], "MOSEI", "MELD"], fontsize=fs - 0.2)
        ax.set_xlim(-0.6, 3.6)
        ax.set_title(t, fontsize=fs + 0.2, loc="left", pad=3)
        ax.tick_params(labelsize=fs - 0.6, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)
    axes[0].set_ylabel(TL[lang]["yl5"], fontsize=fs)
    k_legend(axes[0], C_PROP, loc="lower left", lang=lang, fs=fs)
    axes[0].get_legend().set_bbox_to_anchor((0.0, 1.13), transform=axes[0].transAxes)
    save(fig, f"fig_v4_prop_{lang}")


# ------------------------------------------------------------------ Fig. 1
TO = {
    "ja": dict(src="ソース（ほかのコーパス）\nのラベル付きの発話", unl="ターゲットの\nラベル無しの発話", k="ターゲットの\n感情ごとに K 本のラベル",
               clf="ソースの分類器を\nK 本で学び直した分類器", prob="ラベル無しの発話ごとに\n4 感情の確率を付ける", name={"A": "声", "V": "顔", "T": "言葉"},
               head="提案：分類器 × 感情の重みでまとめる", strength="強さ：ターゲットの K 本で測る", profile="感情ごとの形：ソースで測る",
               ex="重みは説明のための例", pl="重みつきでまとめた\n確率を疑似ラベルにする", rt="K 本と疑似ラベルで\n分類器を学び直す"),
    "en": dict(src="Labelled utterances\nof the sources", unl="Unlabelled utterances\nof the target", k="K labels per emotion\nof the target",
               clf="Source classifiers\nretrained on K labels", prob="Four-emotion\nprobabilities for each\nunlabelled utterance",
               name={"A": "Audio", "V": "Visual", "T": "Text"},
               head="Proposal: weights per\nclassifier and emotion", strength="Strengths: measured on\nthe K target labels",
               profile="Per-emotion profiles:\nmeasured on the sources", ex="illustrative weights",
               pl="Weighted fused\nprobabilities become\npseudo-labels", rt="Retrain a classifier\non the K labels and\nthe pseudo-labels"),
}


def fig_overview(lang="ja"):
    setup(lang)
    t = TO[lang]
    Wf = W if lang == "ja" else W_EN
    k = Wf / W                      # horizontal scale of the English version
    X = lambda x: x * k
    H = 74
    fig, ax = canvas(H, Wf)
    fs = FS if lang == "ja" else 7.0
    NAME = t["name"]
    # inputs
    box(ax, X(1), 56, X(36), 14, fc="#EDE7F3", ec=C_SRC, lw=0.9)
    txt(ax, X(3), 67.5, t["src"], size=fs - 0.3, color=C_SRC, ls=1.15)
    box(ax, X(1), 24, X(36), 28, fc="white", ec=C_U, lw=0.8, ls="--")
    txt(ax, X(3), 49.5, t["unl"], size=fs - 0.3, color="#444444", ls=1.15)
    for i, m in enumerate("AVT"):
        y = 38 - i * 4.6
        ax.plot(X(6), y, marker=MOD[m][1], ms=3.6, color=MOD[m][0], ls="")
        txt(ax, X(6) + 2.4, y, NAME[m], va="center", size=fs - 0.8, color=MOD[m][0])
    box(ax, X(1), 4, X(36), 14, fc="white", ec=C_K, lw=0.9)
    txt(ax, X(3), 15.5, t["k"], size=fs - 0.3, color=C_K, ls=1.15)
    # classifiers trained on the sources
    box(ax, X(43), 22, X(32), 42, fc="white", ec="#9A9A9A", lw=0.7)
    txt(ax, X(45), 62.2, t["clf"], size=fs - (0.6 if lang == "ja" else 1.0), color=GREY, ls=1.1)
    ys = [51, 43.5, 36]
    for y, m in zip(ys, "AVT"):
        box(ax, X(46), y - 3, X(26), 6.2, fc="white", ec=MOD[m][0], lw=0.9)
        txt(ax, X(59), y + 0.1, NAME[m], ha="center", va="center", size=fs - 0.4, color=MOD[m][0])
    txt(ax, X(45), 30.5, t["prob"], size=fs - 1.4, color=GREY, ls=1.1)
    arrow(ax, (X(37), 61), (X(43), 57), color=C_SRC, lw=0.8, head=6)
    arrow(ax, (X(37), 38), (X(43), 38), color=C_U, lw=0.8, head=6, ls="--")
    arrow(ax, (X(30), 18), (X(52), 22), color=C_K, lw=0.8, head=6)
    # proposal: the classifier-by-emotion weights
    box(ax, X(81), 8, X(48), 56, fc="#FFF4EC", ec=C_PROP, lw=1.3)
    txt(ax, X(83), 62, t["head"], size=fs - 0.2, weight="bold", color=C_PROP, ls=1.1)
    for y, m in zip(ys, "AVT"):
        arrow(ax, (X(72), y), (X(81), 40 - (ys.index(y) - 1) * 2.5), color=MOD[m][0], lw=0.8, head=5)
    gx, gy, cw, chh = X(94), 27, X(7.4), 4.4
    vals = np.array([[0.9, 1.1, 1.0, 0.8], [1.3, 1.0, 0.9, 1.2], [0.8, 0.9, 1.1, 1.0]])
    for r_, m in enumerate("AVT"):
        txt(ax, gx - X(1.2), gy + (2 - r_) * chh + chh / 2 - 0.2, NAME[m], ha="right", va="center", size=fs - 1.2,
            color=MOD[m][0])
        for c_ in range(4):
            v = vals[r_, c_]
            ax.add_patch(plt.Rectangle((gx + c_ * cw, gy + (2 - r_) * chh), cw - X(0.5), chh - 0.5,
                                       fc=plt.cm.Oranges(0.15 + 0.55 * (v - 0.8) / 0.5), ec="none"))
            ax.text(gx + c_ * cw + (cw - X(0.5)) / 2, gy + (2 - r_) * chh + (chh - 0.5) / 2, f"{v:.1f}", ha="center",
                    va="center", fontsize=fs - 2.2, color="black" if v < 1.15 else "white")
    for c_, e in enumerate(["N", "H", "S", "A"]):
        txt(ax, gx + c_ * cw + (cw - X(0.5)) / 2, gy - 0.4, e, ha="center", size=fs - 1.6, color=GREY)
    txt(ax, gx + 4 * cw - X(0.5), gy - 4.2, t["ex"], ha="right", size=fs - 2.0, color=GREY)
    yp = 52.5 if lang == "ja" else 54.0
    txt(ax, X(83), yp, t["profile"], size=fs - 1.0, color=C_SRC, ls=1.1)
    txt(ax, X(83), 17.0, t["strength"], size=fs - 1.0, color=C_K, ls=1.1)
    # where the two parts of the weights are measured (routed around the classifier box)
    ax.plot([X(37), X(78), X(78)], [66, 66, yp - 1.6], color=C_SRC, lw=0.8)
    arrow(ax, (X(78), yp - 1.6), (X(81), yp - 1.6), color=C_SRC, lw=0.8, head=6)
    ax.plot([X(37), X(78), X(78)], [11, 11, 15.6], color=C_K, lw=0.8)
    arrow(ax, (X(78), 15.6), (X(81), 15.6), color=C_K, lw=0.8, head=6)
    # pseudo-labels and retraining
    box(ax, X(135), 40, X(37), 16, fc="white", ec=C_PROP, lw=1.0)
    txt(ax, X(137), 53.5, t["pl"], size=fs - 0.6, color=C_PROP, ls=1.15)
    box(ax, X(135), 14, X(37), 18, fc="white", ec="black", lw=1.0)
    txt(ax, X(137), 29.5, t["rt"], size=fs - 0.6, ls=1.15)
    arrow(ax, (X(129), 44), (X(135), 48), color=C_PROP, lw=1.1, head=7)
    arrow(ax, (X(153.5), 40), (X(153.5), 32), color=C_PROP, lw=1.1, head=7)
    save(fig, f"fig_v4_overview_{lang}")


if __name__ == "__main__":
    for lang in ("ja", "en"):
        fig_methods(lang)
        fig_prop(lang)
    fig_overview('ja')
    fig_overview('en')
    pd.DataFrame(rows).to_csv(DATA / "fig_v4.csv", index=False)
    print("wrote", DATA / "fig_v4.csv", len(rows))
