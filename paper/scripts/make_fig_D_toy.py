"""Slide figure (2026-11-16 night, researcher: "これぐらいの説明がないとわからんね" for the D page):

fig_D_toy_ja : how R, R' and D are computed, on 8 made-up utterances in two (emotion, speaker) groups.
  J  = the text classifier's answer (the target), predicted from the voice features -> R (agreement with J)
  J' = J shuffled within each (true emotion, speaker cluster) group, predicted again -> R'
  D  = R - R' : agreement that the true emotion and the speaker cluster do not explain.
For readability the toy counts the agreement rate; the experiment uses the UAR with J (or J') as the truth.
Illustration only: not experimental values.
fig_D_toy_en (2026-11-17): the English version for the manuscript. The aim box at the top of the Japanese version is left
  out (it belongs to the caption); circles and crosses are drawn as markers; text 8-12 pt at the final width of 174 mm.
  The Japanese version is drawn exactly as before.
Output: ../figures/fig_D_toy_{ja,en}.pdf and .png
Run:  python make_fig_D_toy.py [ja|en]
"""
import sys

from make_figs_overview import FS, GREY, W, arrow, box, canvas, save, setup, txt

C_OK, C_NG = "#009E73", "#D55E00"
C_A, C_T = "#E69F00", "#009E73"


T = {
    "ja": dict(
        aim=("ねらい：積の疑似ラベルには言葉の分類器の答えが入っている．声の特徴から言葉の分類器の答えを真似できるなら，",
             "声の分類器はその分を疑似ラベルから学べるはずである．真似できる度合いを，声と顔の特徴で測って比べる"),
        group=["感情 N・話者 a", "感情 A・話者 b"],
        utt="発話 {i}", utt_left=None,
        group_left="本当の感情・話者のまとまり",
        r1="① 言葉の分類器の答え J（これを目標にする）", r2="② 声の特徴だけで J を当てた答え",
        r3="③ J を同じまとまりの中で入れ替えた J′", r4="　 声の特徴だけで J′ を当てた答え",
        R="R ＝ 7/8\n＝ 87.5", Rp="R′ ＝ 5/8\n＝ 62.5",
        expl="J′ は感情と話者のまとまりを保つので，R′ は「同じ感情・同じ話者だから一致する」分だけを表す．○ は目標と同じ，× は違う",
        D="④ D ＝ R − R′ ＝ 87.5 − 62.5 ＝ 25 点",
        Dbody="感情と話者だけでは説明できない一致の大きさ．顔の特徴でも同じ D を測り，声と比べる",
        side="例では数えやすいように一致の割合で数えた．\n実験では J を正解とみなした UAR で数え，\n当てる分類器は別の発話で学んで予測する．\n発話と答えは説明のための例で，実験の値ではない",
    ),
    "en": dict(
        aim=None,   # the aim box of the Japanese version is left to the caption
        group=["Emotion N, speaker a", "Emotion A, speaker b"],
        utt="{i}", utt_left="Utterance",
        group_left="True emotion and speaker group",
        r1="(1) Text classifier’s answers J (the target)", r2="(2) J predicted from the audio features only",
        r3="(3) J′: J shuffled within each group", r4="      J′ predicted from the audio features only",
        R="R = 7/8\n   = 87.5", Rp="R′ = 5/8\n    = 62.5",
        expl=("J′ keeps the emotion and speaker groups, so R′ counts only the agreement that comes from the same emotion and",
              "the same speaker.", "same as the target,", "different"),
        D="(4) D = R − R′ = 87.5 − 62.5 = 25 points",
        Dbody="Agreement that emotion and speaker do not explain. The same D is\nmeasured with the visual features and compared with audio.",
        side="For easy counting, the example counts\nthe agreement rate; the experiment uses\nthe UAR with J as the truth, and the\n"
             "predicting classifier is trained on other\nutterances. Illustrative utterances and\nanswers, not experimental values.",
    ),
}


def okmark(ax, x, y, ok, color, ms=3.6):
    """English figure: open circle (same as the target) or cross (different), drawn as a marker."""
    ax.plot([x], [y], marker="o" if ok else "x", ms=ms, mfc="none", mec=color, mew=1.0, zorder=7, clip_on=False)


def main(lang="ja"):
    """The Japanese layout is the one drawn before the English version was added (first argument of L)."""
    setup(lang)
    t = T[lang]
    L = (lambda ja, en: ja) if lang == "ja" else (lambda ja, en: en)
    H = L(92, 84)
    fig, ax = canvas(H)
    fs = FS + 0.5
    if t["aim"]:
        box(ax, 2, H - 14, W - 4, 12, fc="#F5F5F5", ec="#999999", lw=0.6)
        txt(ax, 4, H - 4.2, t["aim"][0], size=fs - 0.6)
        txt(ax, 4, H - 9.6, t["aim"][1], size=fs - 0.6)
    group = [t["group"][0]] * 4 + [t["group"][1]] * 4
    J = ["N", "H", "N", "S", "A", "A", "S", "A"]
    PJ = ["N", "H", "N", "N", "A", "A", "S", "A"]
    Js = ["H", "N", "S", "N", "A", "S", "A", "A"]
    PJs = ["N", "N", "N", "N", "A", "A", "A", "A"]
    x0, cw, ch = 70, 9.2, 6.6
    yt = H - L(21, 4)
    for i in range(8):
        txt(ax, x0 + (i + 0.5) * cw, yt, t["utt"].format(i=i + 1), ha="center", va="center", size=L(fs - 1.8, 8), color=GREY)
    if t["utt_left"]:
        txt(ax, 2, yt, t["utt_left"], va="center", size=8, color=GREY)
    for g, gx in ((0, 0), (1, 4)):
        box(ax, x0 + gx * cw + 0.3, yt - 9, 4 * cw - 0.6, 5.6, fc="#EEEEEE", ec="#999999", lw=0.5, r=0.6)
        txt(ax, x0 + (gx + 2) * cw, yt - 6.2, group[gx], ha="center", va="center", size=L(fs - 0.6, 8))
    txt(ax, 2, yt - 6.2, t["group_left"], va="center", size=L(fs - 0.4, 8))

    def row(y, name, vals, ref, col, bold=False):
        txt(ax, 2, y + ch / 2, name, va="center", size=L(fs - 0.5, 8), color=col, weight="bold" if bold else "normal")
        for i, v in enumerate(vals):
            if ref is None:
                fc, s, c = "white", v, col
            else:
                ok = v == ref[i]
                fc, s, c = ("#E5F5EF", v + " ○", C_OK) if ok else ("#FBE3D6", v + " ×", C_NG)
            box(ax, x0 + i * cw + 0.4, y, cw - 0.8, ch, fc=fc, ec="#999999", lw=0.5, r=0.6, z=2)
            if lang == "ja" or ref is None:
                txt(ax, x0 + (i + 0.5) * cw, y + ch / 2, s, ha="center", va="center", size=L(fs - 0.4, 8.5), color=c, weight="bold")
            else:
                txt(ax, x0 + (i + 0.5) * cw - 1.4, y + ch / 2, v, ha="center", va="center", size=8.5, color=c, weight="bold")
                okmark(ax, x0 + (i + 0.5) * cw + 1.9, y + ch / 2, ok, c)

    y1 = yt - 18
    row(y1, t["r1"], J, None, C_T, True)
    row(y1 - 8, t["r2"], PJ, J, C_A, True)
    txt(ax, x0 + 8 * cw + 2, y1 - 8 + ch / 2, t["R"], va="center", size=L(fs, 9), weight="bold", ls=1.15)
    row(y1 - 19, t["r3"], Js, None, "#555555")
    row(y1 - 27, t["r4"], PJs, Js, C_A)
    txt(ax, x0 + 8 * cw + 2, y1 - 27 + ch / 2, t["Rp"], va="center", size=L(fs, 9), weight="bold", ls=1.15)
    if lang == "ja":
        txt(ax, 2, y1 - 31.5, t["expl"], size=fs - 1.3, color=GREY)
        box(ax, 2, 3, 112, 14, fc="#FDF0E8", ec="#D55E00", lw=1.1)
        txt(ax, 5, 13.6, t["D"], size=fs + 0.4, weight="bold", color="#D55E00")
        txt(ax, 5, 7.6, t["Dbody"], size=fs - 0.6)
        txt(ax, 117, 13.2, t["side"], size=fs - 1.5, color=GREY, va="top", ls=1.2)
    else:
        e1, e2, e3, e4 = t["expl"]
        ye = y1 - 32.6
        txt(ax, 2, ye, e1, va="center", size=8, color=GREY)
        a = ax.text(2, ye - 3.6, e2, va="center", fontsize=8, color=GREY)
        xe = ax.transData.inverted().transform(a.get_window_extent(fig.canvas.get_renderer()).p1)[0]
        okmark(ax, xe + 2.4, ye - 3.6, True, GREY, ms=3.2)
        b = ax.text(xe + 4.0, ye - 3.6, e3, va="center", fontsize=8, color=GREY)
        xe = ax.transData.inverted().transform(b.get_window_extent(fig.canvas.get_renderer()).p1)[0]
        okmark(ax, xe + 2.2, ye - 3.6, False, GREY, ms=3.2)
        txt(ax, xe + 3.8, ye - 3.6, e4, va="center", size=8, color=GREY)
        box(ax, 2, 3, 112, 18, fc="#FDF0E8", ec="#D55E00", lw=1.1)
        txt(ax, 5, 18.6, t["D"], size=10, weight="bold", color="#D55E00")
        txt(ax, 5, 12.4, t["Dbody"], size=8, ls=1.25)
        txt(ax, 117, 21.0, t["side"], size=8, color=GREY, va="top", ls=1.2)
    save(fig, f"fig_D_toy_{lang}")


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["ja", "en"]):
        main(lang)
