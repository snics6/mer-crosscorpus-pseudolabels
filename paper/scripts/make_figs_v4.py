"""Figures for the Japanese draft ver4 (2026-11-17 night; researcher: "図を作って", one figure = one conclusion,
following Mensh & Kording 2017 rules 7 and 9 and the MIT CommLab results formula).

fig_v4_overview_ja : Fig. 1, overview: pseudo-labels made from the other modalities of the same utterance, combined
                     with equal weights (product) or with the proposal's per-emotion weights, versus the 11 methods
                     that use no pseudo-labels including other modalities (conceptual, no data).
fig_v4_methods_ja  : Fig. 3, audio classifier, distillation - each of the 11 methods (4 corpora x K = 5, 10, 25 =
                     132 comparisons) with 95% intervals. Supports Sect. 4.1.
fig_v4_pipeline_ja : Fig. 2, pipeline of the LOCO experiments (corpora, input, frozen encoders, linear classifiers,
                     retraining on the target); drawn, no data. Redrawn from the slide of 2026-11-16 (p. 13).
fig_v4_prop_flow_ja: Fig. 5, how the proposal makes the pseudo-labels: what the K labels and the other candidates are
                     used for, the three classifiers' probabilities, the weight table, fusion into q, retraining.
                     Illustrative values; q from Eq. (5) with the intercept shift c = 0 (written to fig_v4.csv).
fig_v4_prop_ja     : Fig. 8 (Fig. 5 before 2026-11-17 night), (a) proposal - product (fused predictions) and (b) AVT retrained on the proposal's
                     pseudo-labels - AVT retrained on the product's pseudo-labels, 12 conditions each. Supports
                     Sects. 4.4 and 4.6.

Values:
  Fig. 6: results/dai2/stats/prop_minus_method.csv (student = A, K in 5/10/25, the 11 methods of Table 2;
          columns effect, lo, hi). The script checks 132 rows, 132 positive, 122 with lo > 0, 10 not above
          zero all on IEMOCAP (Sect. 4.1).
  Fig. 8: results/dai20_eval/summary/effects.csv (a = prop, b = prod, seeds = all) and
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

from make_figs_overview import C_K, C_PROP, C_SRC, C_U, FS, GREY, MM, MOD, W, W_EN, arrow, box, canvas, mark, save, setup, txt

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


# ------------------------------------------------------------------ Fig. 2: pipeline of the LOCO experiments (2026-11-17
# night; researcher: "スライドの 13 ページにあるような図はほしい". Redrawn from make_figs_explain.fig_pipeline at the
# text width with the slide notes removed and the paper's terms: anchor, baseline, product, proposal).
TPP = {
    "ja": dict(heads=["コーパス\n（LOCO）", "入力", "凍結した\nエンコーダ", "線形分類器", "ターゲットで\n学び直す"],
               corp_note="1 つを\nターゲット，\n残る 3 つを\nソースにする\n（4 回）",
               mods=["声", "顔", "言葉"],
               cut=["16 kHz の波形，\n先頭 60 秒", "最も大きい顔，\nYOLOv8n-face，\n1 秒に 5〜6 枚", "書き起こし\n（AFEW は\nWhisper large-v3）"],
               enc=["WavLM-large，\n最終層，\n発話で平均\n→ 1,024 次元", "ResNet-50\n（AffectNet），\n発話で平均\n→ 512 次元",
                    "RoBERTa-base，\n最終層，\n発話で平均\n→ 768 次元"],
               avt="AVT：3 つを\nつなぐ → 2,304 次元",
               clfs=["声の\n分類器", "顔の\n分類器", "言葉の\n分類器", "AVT の\n分類器"],
               clf_note="ソースごとに\n学び，3 つの\n平均をアンカー\nにする",
               tgt=["基準の分類器：\nアンカーを K 本で\n学び直す", "学び直し：K 本と，\n候補の疑似ラベル\n（積か提案）",
                    "テストの発話の UAR\n（話者かクラスタで\n分けた 5 分割 × K 本\nの 10 回 = 50 反復）"]),
    "en": dict(heads=["Corpora\n(LOCO)", "Input", "Frozen\nencoders", "Linear\nclassifiers", "Retraining on\nthe target"],
               corp_note="One target,\nthree sources\n(4 runs)",
               mods=["Audio", "Visual", "Text"],
               cut=["16 kHz\nwaveform,\nfirst 60 s", "largest face,\nYOLOv8n-face,\n5–6 frames/s", "transcript\n(AFEW: Whisper\nlarge-v3)"],
               enc=["WavLM-large,\nlast layer,\nutterance mean\n→ 1,024 dims", "ResNet-50\n(AffectNet),\nutterance mean\n→ 512 dims",
                    "RoBERTa-base,\nlast layer,\nutterance mean\n→ 768 dims"],
               avt="AVT: concatenated\n→ 2,304 dims",
               clfs=["Audio", "Visual", "Text", "AVT"],
               clf_note="One per source;\ntheir mean is\nthe anchor",
               tgt=["Baseline: the\nanchor retrained\non K labels", "Retraining: K labels\n+ pseudo-labels on\nthe candidates\n(product or proposal)",
                    "Test UAR (speaker-\nor cluster-disjoint\nfolds; 5 folds × 10\nK-label draws)"]),
}


def fig_pipeline(lang="ja"):
    setup(lang)
    t = TPP[lang]
    Wf = W if lang == "ja" else W_EN
    k = Wf / W_EN
    X = lambda x: x * k
    fs = FS if lang == "ja" else 7.6
    fsm = fs - (1.0 if lang == "ja" else 0.6)
    H = 92
    fig, ax = canvas(H, Wf)
    xs = [0.4, 20.6, 47.6, 75.0, 99.4]
    ws = [19.0, 25.8, 26.2, 23.2, 31.4]
    hcol = ["#EEEEEE", "#FCEFD9", "#E3EEF7", "#ECE6F2", "#FDF0E8"]
    ecol = ["#777777", "#C98A1B", "#2F6FA6", C_SRC, C_PROP]
    top, hh = H - 0.5, 9.0
    for x, w, h, fc, ec in zip(xs, ws, t["heads"], hcol, ecol):
        box(ax, X(x), 0.5, X(w), top - 0.5, fc=fc, ec=ec, lw=0.9, r=1.2)
        box(ax, X(x), top - hh, X(w), hh, fc=ec, ec=ec, lw=0.9, r=1.0, z=2)
        txt(ax, X(x + w / 2), top - hh / 2, h, ha="center", va="center", size=fs, color="white", weight="bold", ls=1.05)
    rows = [top - hh - 10.5, top - hh - 28.0, top - hh - 45.5]     # centres of the audio, visual and text rows
    ya = top - hh - 60.0                                              # centre of the AVT row
    # corpora
    for i, c in enumerate(["AFEW", "IEMOCAP", "MOSEI", "MELD"]):
        y = top - hh - 9.0 - i * 8.5
        tg = i == 0
        box(ax, X(xs[0] + 1.4), y, X(ws[0] - 2.8), 6.4, fc="white" if tg else "#EDE7F3", ec="black" if tg else C_SRC,
            lw=1.1 if tg else 0.6, z=3)
        txt(ax, X(xs[0] + ws[0] / 2), y + 3.2, c, ha="center", va="center", size=fsm, weight="bold" if tg else "normal",
            color="black" if tg else C_SRC)
    txt(ax, X(xs[0] + 1.4), top - hh - 47.0, t["corp_note"], size=fsm, color=GREY, ls=1.1)
    # input, encoders and classifiers per modality
    for r, m in enumerate("AVT"):
        y, col = rows[r], MOD[m][0]
        box(ax, X(xs[1] + 1.2), y - 7.6, X(ws[1] - 2.4), 15.2, fc="white", ec=col, lw=1.0, z=3)
        mark(ax, X(xs[1] + 3.6), y + 5.0, m, size=3.4)
        txt(ax, X(xs[1] + 5.6), y + 5.0, t["mods"][r], va="center", size=fs, weight="bold", color=col)
        txt(ax, X(xs[1] + 2.6), y + 2.6, t["cut"][r], size=fsm, ls=1.1)
        box(ax, X(xs[2] + 1.2), y - 7.6, X(ws[2] - 2.4), 15.2, fc="white", ec=col, lw=1.0, z=3)
        txt(ax, X(xs[2] + 2.6), y + 6.2, t["enc"][r], size=fsm, ls=1.1)
        box(ax, X(xs[3] + 1.6), y - 4.2, X(ws[3] - 3.2), 8.4, fc="white", ec=col, lw=1.0, z=3)
        txt(ax, X(xs[3] + ws[3] / 2), y, t["clfs"][r], ha="center", va="center", size=fs, color=col, ls=1.05)
        arrow(ax, (X(xs[0] + ws[0] - 0.2), y), (X(xs[1] + 1.2), y), lw=0.6, head=5, color=GREY)
        arrow(ax, (X(xs[1] + ws[1] - 1.2), y), (X(xs[2] + 1.2), y), lw=0.8, head=5, color=col)
        arrow(ax, (X(xs[2] + ws[2] - 1.2), y), (X(xs[3] + 1.6), y), lw=0.8, head=5, color=col)
    # AVT: concatenation and its classifier
    box(ax, X(xs[2] + 1.2), ya - 4.4, X(ws[2] - 2.4), 8.8, fc="#F5F5F5", ec="black", lw=0.8, z=3)
    txt(ax, X(xs[2] + ws[2] / 2), ya, t["avt"], ha="center", va="center", size=fsm, ls=1.1)
    for r in range(3):
        ax.plot([X(xs[2] + ws[2] / 2 + (r - 1) * 5)] * 2, [rows[2] - 7.6, ya + 4.4], color=MOD["AVT"[r]][0], lw=0.6,
                zorder=2)
    box(ax, X(xs[3] + 1.6), ya - 4.2, X(ws[3] - 3.2), 8.4, fc="white", ec="black", lw=1.0, z=3)
    txt(ax, X(xs[3] + ws[3] / 2), ya, t["clfs"][3], ha="center", va="center", size=fs, ls=1.05)
    arrow(ax, (X(xs[2] + ws[2] - 1.2), ya), (X(xs[3] + 1.6), ya), lw=0.8, head=5)
    txt(ax, X(xs[3] + 1.6), ya - 6.8, t["clf_note"], size=fsm, color=C_SRC, ls=1.1)
    # classifiers -> target retraining
    xr = xs[3] + ws[3] - 0.6
    for yy in rows + [ya]:
        ax.plot([X(xs[3] + ws[3] - 1.6), X(xr)], [yy, yy], color=GREY, lw=0.6, zorder=2)
    ax.plot([X(xr), X(xr)], [ya, rows[0]], color=GREY, lw=0.6, zorder=2)
    # target boxes
    hs = [12.0, 15.5, 16.0] if lang == "en" else [12.0, 12.5, 16.0]
    ecs = [C_K, C_PROP, "black"]
    y = top - hh - 2.0
    for i, (lab, h) in enumerate(zip(t["tgt"], hs)):
        box(ax, X(xs[4] + 1.4), y - h, X(ws[4] - 2.8), h, fc="white", ec=ecs[i], lw=1.0, z=3)
        txt(ax, X(xs[4] + 2.6), y - 1.4, lab, size=fsm, ls=1.12)
        if i == 0:
            arrow(ax, (X(xr), y - h / 2), (X(xs[4] + 1.4), y - h / 2), lw=0.9, head=6, color=GREY)
        else:
            arrow(ax, (X(xs[4] + ws[4] / 2), y + 3.0), (X(xs[4] + ws[4] / 2), y), lw=0.8, head=5, color=GREY)
        y -= h + 3.0
    save(fig, f"fig_v4_pipeline_{lang}")


# ------------------------------------------------------------------ Fig. 5: how the proposal makes the pseudo-labels
# (2026-11-17 night; researcher: Fig. 3 should explain the proposal, show that the pseudo-labels are made for the other
# candidates, and show what the K labels and the other candidates are each used for). Illustrative values: the three
# classifiers' probabilities of the former product figure, the weight table of Fig. 1, and q from Eq. (5) with c = 0.
TPF = {
    "ja": dict(heads=["① ターゲット\nの候補", "② 3 つの\n分類器", "③ 重みの表", "④ 提案で\nまとめる", "⑤ 学び直して\n比べる"],
               k="K 本（感情ごとに K 本．\nラベルを使う）\n→ ② の分類器を学び直す\n→ ③ の強さを測る\n→ ⑤ の正解にする",
               other="ほかの候補（ラベル無し）\n→ ② の分類器を学び直す\n　（基準の AVT の確率で）\n→ ② で確率を付ける\n→ ⑤ で疑似ラベル q を付ける",
               cand="候補 ＝ テスト以外の\nターゲットの発話．\n② の確率と ⑤ の q は\nK 本を含む候補すべて\nに付ける．テストの発話\nは ⑤ の UAR にだけ使う",
               p2="積を作る声・顔・言葉の\n分類器（3.4 節）が，\n候補の発話ごとに 4 感情\nの確率を付ける",
               clf={"A": "声", "V": "顔", "T": "言葉"},
               p3="分類器 × 感情の\n12 個の重み\n＝ 強さ ＋ 感情\nごとの形（全体の\n平均で割る）",
               strength="強さ：K 本で測る", profile="感情ごとの形：\nソースのコーパスで測る",
               p4="感情ごとに，3 つの分類器\nのスコアに重みを掛けて\n足し，ラベルを使わずに\n切片をずらして確率にする",
               q="この候補の\n疑似ラベル q",
               retr="AVT の分類器をアンカーから，\nK 本の正解と，候補の q で\n学び直す",
               cmp="テストの UAR を比べる\ngain ＝ 学び直し\n　　− 基準の分類器"),
    "en": dict(heads=["(1) Target\ncandidates", "(2) Three\nclassifiers", "(3) Weight\ntable", "(4) Fusion by\nthe proposal", "(5) Retrain\nand compare"],
               k="K labels per emotion\n(labels used)\n→ classifiers of (2)\n→ strength in (3)\n→ true labels in (5)",
               other="Other candidates\n(no labels)\n→ classifiers of (2), with\n   baseline AVT targets\n→ probabilities in (2)\n→ pseudo-labels q in (5)",
               cand="Candidates = target\nutterances except the\ntest set. (2) and (5)\ncover all candidates,\nK labels included. Test\nutterances: UAR in (5)",
               p2="The classifiers of\nthe product (Sect.\n3.4) give each\ncandidate probabilities\nof the 4 emotions",
               clf={"A": "Audio", "V": "Visual", "T": "Text"},
               p3="One weight per\nclassifier and emotion\n= strength + per-\nemotion profile,\ndivided by the mean",
               strength="Strength: measured\non the K labels", profile="Per-emotion profile:\nmeasured on the\nsource corpora",
               p4="Per emotion, add the\nweighted scores of the\n3 classifiers, shift the\nintercepts without\nlabels, apply softmax",
               q="Pseudo-label q\nof this candidate",
               retr="The AVT classifier is retrained from\nthe anchor on the K labels and on q\nof the candidates",
               cmp="Test UAR: gain =\nretrained − baseline"),
}
PF_PROBS = {"A": [0.40, 0.35, 0.15, 0.10], "V": [0.20, 0.60, 0.10, 0.10], "T": [0.30, 0.50, 0.10, 0.10]}
PF_OMEGA = np.array([[0.9, 1.1, 1.0, 0.8], [1.3, 1.0, 0.9, 1.2], [0.8, 0.9, 1.1, 1.0]])   # same example as Fig. 1


def pf_q():
    """q of Eq. (5) for the illustrative candidate, with the label-free intercept shift c set to 0."""
    s_ = np.zeros(4)
    for r, m in enumerate("AVT"):
        lp = np.log(np.array(PF_PROBS[m]))
        s_ += PF_OMEGA[r] * (lp - lp.mean())
    e = np.exp(s_ - s_.max())
    return e / e.sum()


def pf_bars(ax, x, y, w, h, vals, color, fs, X):
    n = len(vals)
    for i, v in enumerate(vals):
        cx = x + (i + 0.5) * w / n
        bw = w / n * 0.6
        ax.add_patch(plt.Rectangle((cx - bw / 2, y), bw, h * v, fc=color, ec="none", zorder=4))
        txt(ax, cx, y - 0.5, "NHSA"[i], ha="center", va="top", size=fs, color=GREY)
        txt(ax, cx, y + h * v + 0.4, f"{v:.2f}", ha="center", va="bottom", size=fs)
    ax.plot([x, x + w], [y, y], color=GREY, lw=0.4, zorder=4)


def fig_prop_flow(lang="ja"):
    setup(lang)
    t = TPF[lang]
    Wf = W if lang == "ja" else W_EN
    k = Wf / W_EN
    X = lambda x: x * k
    fs = FS if lang == "ja" else 7.6
    fv = fs - 0.6                                  # values and axis letters
    hb = [23.0, 21.0, 22.0, 18.5, 13.5]
    gap, x0, hw = 1.2, 0.6, 21.0
    H = sum(hb) + gap * 4 + 1.2
    fig, ax = canvas(H, Wf)
    hcol = ["#EEEEEE", "#F2F2F2", "#FDF0E8", "#FDF0E8", "#ECE6F2"]
    ecol = ["#777777", "#444444", C_PROP, C_PROP, C_SRC]
    bands, y = [], H - 0.6
    for h in hb:
        bands.append((y - h, y))
        y -= h + gap
    for i, ((yb, yt), fc, ec) in enumerate(zip(bands, hcol, ecol)):
        box(ax, X(x0), yb, Wf - 2 * X(x0), yt - yb, fc=fc, ec=ec, lw=0.9, r=1.0)
        box(ax, X(x0), yb, X(hw), yt - yb, fc=ec, ec=ec, lw=0.9, r=1.0, z=2)
        txt(ax, X(x0 + hw / 2), (yb + yt) / 2, t["heads"][i], ha="center", va="center", size=fs, color="white",
            weight="bold", ls=1.1)
        if i:
            arrow(ax, (X(x0 + hw / 2), bands[i - 1][0] + 1.6), (X(x0 + hw / 2), yt - 1.6), lw=1.0, head=6, color="white", z=4)
    cx = x0 + hw + 2.0
    R = (Wf - X(0.6) - X(1.0)) / k                 # right edge of the contents, in English units
    # (1) the K labels and the other candidates, and what each is used for
    yb, yt = bands[0]
    box(ax, X(cx), yb + 0.8, X(34.5), yt - yb - 1.6, fc="white", ec=C_K, lw=1.2, z=3)
    txt(ax, X(cx + 1.2), (yb + yt) / 2, t["k"], va="center", size=fs, color=C_K, ls=1.12)
    box(ax, X(cx + 36.0), yb + 0.8, X(36.5), yt - yb - 1.6, fc="white", ec=C_U, lw=1.1, ls="--", z=3)
    txt(ax, X(cx + 37.2), (yb + yt) / 2, t["other"], va="center", size=fs, color="#333333", ls=1.12)
    txt(ax, X(cx + 74.0), (yb + yt) / 2, t["cand"], va="center", size=fv, color=GREY, ls=1.12)
    # (2) probabilities of the three classifiers on one other candidate
    yb, yt = bands[1]
    txt(ax, X(cx), (yb + yt) / 2, t["p2"], va="center", size=fs, ls=1.12)
    x2 = cx + 30.0
    cw = (R - x2) / 3
    for r, m in enumerate("AVT"):
        bx, col = x2 + r * cw, MOD[m][0]
        box(ax, X(bx + 0.5), yb + 0.8, X(cw - 1.0), yt - yb - 1.6, fc="white", ec=col, lw=1.0, z=3)
        mark(ax, X(bx + 2.8), yt - 3.0, m, size=3.6)
        txt(ax, X(bx + 4.6), yt - 3.0, t["clf"][m], va="center", size=fs, weight="bold", color=col)
        pf_bars(ax, X(bx + 1.0), yb + 4.6, X(cw - 2.0), 8.6, PF_PROBS[m], col, fv, X)
    # (3) the weight table: strength (K labels) + per-emotion profile (sources)
    yb, yt = bands[2]
    txt(ax, X(cx), (yb + yt) / 2, t["p3"], va="center", size=fs, ls=1.12)
    gx, cwh, chh = cx + 40.0, 7.2, 4.9
    gy = yb + 4.2
    for r, m in enumerate("AVT"):
        txt(ax, X(gx - 1.0), gy + (2 - r) * chh + chh / 2, t["clf"][m], ha="right", va="center", size=fv, color=MOD[m][0])
        for c_ in range(4):
            v = PF_OMEGA[r, c_]
            ax.add_patch(plt.Rectangle((X(gx + c_ * cwh), gy + (2 - r) * chh), X(cwh - 0.5), chh - 0.5,
                                       fc=plt.cm.Oranges(0.15 + 0.55 * (v - 0.8) / 0.5), ec="none", zorder=3))
            ax.text(X(gx + c_ * cwh + (cwh - 0.5) / 2), gy + (2 - r) * chh + (chh - 0.5) / 2, f"{v:.1f}", ha="center",
                    va="center", fontsize=fv, color="black" if v < 1.15 else "white", zorder=5)
    for c_ in range(4):
        txt(ax, X(gx + c_ * cwh + (cwh - 0.5) / 2), gy - 0.4, "NHSA"[c_], ha="center", size=fv, color=GREY)
    xs_ = gx + 4 * cwh + 3.0
    txt(ax, X(xs_), yt - 1.8, t["strength"], size=fs, color=C_K, ls=1.12)
    txt(ax, X(xs_), yt - (10.2 if lang == "en" else 8.0), t["profile"], size=fs, color=C_SRC, ls=1.12)
    # (4) fusion by the proposal -> q
    yb, yt = bands[3]
    txt(ax, X(cx), (yb + yt) / 2, t["p4"], va="center", size=fs, ls=1.12)
    q = pf_q()
    qx = cx + 44.0
    box(ax, X(qx), yb + 0.8, X(30.0), yt - yb - 1.6, fc="white", ec=C_PROP, lw=1.2, z=3)
    pf_bars(ax, X(qx + 1.0), yb + 4.4, X(28.0), 9.0, list(q), C_PROP, fv, X)
    txt(ax, X(qx + 32.0), (yb + yt) / 2, t["q"], va="center", size=fs, color=C_PROP, ls=1.12)
    # (5) retrain the AVT classifier and compare
    yb, yt = bands[4]
    box(ax, X(cx), yb + 0.9, X(60.0), yt - yb - 1.8, fc="white", ec=C_SRC, lw=1.1, z=3)
    txt(ax, X(cx + 1.2), (yb + yt) / 2, t["retr"], va="center", size=fs, ls=1.12)
    box(ax, X(cx + 64.0), yb + 0.9, X(R - cx - 64.0), yt - yb - 1.8, fc="white", ec="black", lw=1.0, z=3)
    txt(ax, X(cx + 65.2), (yb + yt) / 2, t["cmp"], va="center", size=fs, ls=1.12)
    arrow(ax, (X(cx + 60.3), (yb + yt) / 2), (X(cx + 63.7), (yb + yt) / 2), lw=0.8, head=6)
    rows.append(dict(figure="fig_prop_flow", lang=lang, q=";".join(f"{v:.4f}" for v in q)))
    save(fig, f"fig_v4_prop_flow_{lang}")


if __name__ == "__main__":
    for lang in ("ja", "en"):
        fig_methods(lang)
        fig_prop(lang)
    fig_overview('ja')
    fig_overview('en')
    fig_pipeline('ja')
    fig_pipeline('en')
    fig_prop_flow('ja')
    fig_prop_flow('en')
    pd.DataFrame(rows).to_csv(DATA / "fig_v4.csv", index=False)
    print("wrote", DATA / "fig_v4.csv", len(rows))
