"""Slide figure (2026-11-16 night, researcher: "p6の前に，二つの誤りの説明が必要"):

fig_two_errors_ja : the two parts of a linear classifier's error, drawn in a 2-D feature space.
  (a) a classifier whose boundary separates the two emotions,
  (b) answer bias (intercept shift): the boundary has the right direction but is shifted;
      moving it in parallel (changing only the intercepts) removes this error,
  (c) remaining error (direction of the weights): the boundary is tilted; no parallel shift removes it,
      only changing the weights (the direction) does.
Illustration only: the points are drawn from two made-up Gaussians (seed 0), not from the experiments.
fig_two_errors_en (2026-11-17): the English version for the manuscript, drawn at the text width (372 pt = 131.2 mm;
  included at width=\textwidth) with all text at 8 pt or larger (Discover AI artwork rules: no title or note line inside
  the figure; the illustration note goes to the caption). The Japanese version is drawn exactly as before.
Output: ../figures/fig_two_errors_{ja,en}.pdf and .png
Run:  python make_fig_two_errors.py [ja|en]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from make_figs_overview import FS, GREY, MM, W, W_EN, save, setup

C1, C2 = "#0072B2", "#D55E00"   # emotion k (circles), emotion l (triangles)


TXT = {
    "ja": dict(
        a=("(a) 境目が 2 つの感情を分けている", "線（境目）の右を感情 k，左を感情 l と\n答える．塗りつぶした印が誤り"),
        b=("(b) 答えの偏り（切片のずれ）",
           "線の向きは正しいが平行にずれ，答えが\n感情 l に偏る．切片だけを直して線を\n戻す（破線）と，(a) と同じ誤りに戻る"),
        c=("(c) 残りの誤り（重みの向きのずれ）",
           "線が傾き，線の近くで 2 つの感情が混ざる．\n平行にずらしても (a) の数までは減らない．\n重み（線の向き）を直す（破線）必要がある"),
        k="感情 k ○", l="△ 感情 l",
        note="2 次元の特徴で描いた模式図．点は説明のために作った例で，実験の値ではない",
    ),
    "en": dict(
        a=("(a) The boundary separates\n      the two emotions",
           "Right of the line (boundary):\nemotion k; left: emotion l.\nFilled markers are errors."),
        b=("(b) Answer bias\n      (intercept shift)",
           "Right direction, but shifted:\nthe answers lean to emotion l.\nCorrecting only the intercepts\nmoves the line back (dashed)\nto the errors of (a)."),
        c=("(c) Remaining error\n      (weight direction)",
           "Tilted line: the emotions mix\nnear it. No parallel shift\nreaches the errors of (a); the\nweights (direction of the line)\nmust change (dashed)."),
        k="emotion k", l="emotion l",   # markers drawn as markers (Arial has no white triangle)
        note=None,   # the illustration note goes to the caption
    ),
}
# sizes and placement: the Japanese values are the ones used before the English version was added
GEO = {
    "ja": dict(w=W, h=70, rect=lambda i: [0.02 + i * 0.335, 0.27, 0.30, 0.60], title=FS + 1.2, body=FS - 0.2, leg=FS - 0.4,
               note=FS - 0.6, note_xy=(0.02, 0.955), nls=1.35),
    "en": dict(w=W_EN, h=61, rect=lambda i: [0.004 + i * 0.3365, 0.322, 0.322, 0.533], title=8.5, body=8.0, leg=8.0,
               note=None, note_xy=None, nls=1.2),
}


def panel(ax, pts1, pts2, w, b, title, note, good_b=None, good_w=None, tsize=FS + 1.2, nsize=FS - 0.2, nls=1.35):
    ax.scatter(pts1[:, 0], pts1[:, 1], s=7, marker="o", fc="none", ec=C1, lw=0.6)
    ax.scatter(pts2[:, 0], pts2[:, 1], s=8, marker="^", fc="none", ec=C2, lw=0.6)
    def line(bb, **kw):   # boundary w.x + bb = 0, w = (w0, w1)
        if abs(w[1]) < 1e-9:
            ax.plot([-bb / w[0]] * 2, [-3.2, 3.2], **kw)
        else:
            xs = np.linspace(-4, 4, 50)
            ax.plot(xs, -(w[0] * xs + bb) / w[1], **kw)
    line(b, color="black", lw=1.4)
    if good_b is not None:
        line(good_b, color="black", lw=0.9, ls="--")
    if good_w is not None:
        ax.plot([0, 0], [-3.2, 3.2], color="black", lw=0.9, ls="--")
    # count errors
    s1 = pts1 @ w + b
    s2 = pts2 @ w + b
    e1 = (s1 < 0)
    e2 = (s2 > 0)
    ax.scatter(pts1[e1, 0], pts1[e1, 1], s=7, marker="o", fc=C1, ec=C1, lw=0.6)
    ax.scatter(pts2[e2, 0], pts2[e2, 1], s=8, marker="^", fc=C2, ec=C2, lw=0.6)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999999")
        s.set_linewidth(0.5)
    ax.set_title(title, fontsize=tsize, loc="left", pad=3)
    ax.text(0.0, -0.05, note, transform=ax.transAxes, ha="left", va="top", fontsize=nsize, linespacing=nls)
    return int(e1.sum() + e2.sum())


def main(lang="ja"):
    setup(lang)
    t, g = TXT[lang], GEO[lang]
    rng = np.random.default_rng(0)
    n = 60
    cov = np.array([[0.55, 0.0], [0.0, 1.1]])
    p1 = rng.multivariate_normal([1.3, 0.0], cov, n)    # emotion k
    p2 = rng.multivariate_normal([-1.3, 0.0], cov, n)   # emotion l
    fig = plt.figure(figsize=(g["w"] * MM, g["h"] * MM))
    axs = [fig.add_axes(g["rect"](i)) for i in range(3)]
    sz = dict(tsize=g["title"], nsize=g["body"], nls=g["nls"])
    w_good = np.array([1.0, 0.0])
    panel(axs[0], p1, p2, w_good, 0.0, *t["a"], **sz)
    panel(axs[1], p1, p2, w_good, -1.25, *t["b"], good_b=0.0, **sz)
    th = np.deg2rad(62)
    w_tilt = np.array([np.cos(th), np.sin(th)])
    panel(axs[2], p1, p2, w_tilt, 0.0, *t["c"], good_w=True, **sz)
    for ax in axs:
        if lang == "ja":
            ax.text(3.8, 2.9, t["k"], ha="right", va="top", fontsize=g["leg"], color=C1)
            ax.text(-3.8, 2.9, t["l"], ha="left", va="top", fontsize=g["leg"], color=C2)
        else:
            bg = dict(boxstyle="square,pad=0.15", fc="white", ec="none")   # the boundary of (b) passes behind the label
            ax.scatter([3.65], [2.5], s=12, marker="o", fc="none", ec=C1, lw=0.8, zorder=5)
            ax.text(3.35, 2.5, t["k"], ha="right", va="center", fontsize=g["leg"], color=C1, bbox=bg, zorder=6)
            ax.scatter([-3.65], [2.5], s=14, marker="^", fc="none", ec=C2, lw=0.8, zorder=5)
            ax.text(-3.35, 2.5, t["l"], ha="left", va="center", fontsize=g["leg"], color=C2, bbox=bg, zorder=6)
    if t["note"]:
        x, y = g["note_xy"]
        fig.text(x, y, t["note"], fontsize=g["note"], color=GREY)
    save(fig, f"fig_two_errors_{lang}")


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["ja", "en"]):
        main(lang)
