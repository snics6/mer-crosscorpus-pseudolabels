"""Slide figures for the proposed fusion, second try (2026-11-16 night, researcher: "提案は各コーパスごとに感情ごとに
強いモダリティが違うっていう考えからじゃん？それを説明したいしそれを学ぶためのメタ学習ちっくなことをやった" / "文ばっか過ぎ"):

fig_prop_idea_ja    : for each corpus and emotion, the modality with the highest reliability d (design-time values,
                      ../figures/data/fig7_strength_profile.csv: provisional-target runs, full candidate labels, K = 5, 30 reps).
fig_prop_decomp_ja  : the 12-cell weight table = strength (one number per classifier, from the K labels)
                      + per-emotion shape (from the source corpora). Made-up numbers.
fig_prop_episode_ja : practising on the source corpora (target AFEW): each source becomes a provisional target in turn,
                      the settings are chosen there and used once on AFEW; the shape's transfer (eta) is measured
                      source -> source.
fig_prop_calc_ja    : how the strength (K labels) and the per-emotion shape (source corpora) are computed. Made-up numbers.
English versions fig_prop_{idea,decomp,calc,episode}_en (2026-11-17) for the manuscript: same values, text 8-12 pt at the
final width of 174 mm, the heading line of the episode figure replaced by panel labels (a)-(c). The Japanese versions are
drawn exactly as before.
Output: ../figures/<name>.pdf and .png
Run:  python make_figs_prop_v2.py [ja|en]
"""
import sys
from pathlib import Path

import matplotlib.patches as mp
import numpy as np
import pandas as pd

from make_figs_overview import C_K, C_PROP, C_SRC, FS, GREY, MOD, W, W_EN, arrow, box, canvas, save, setup, txt

DATA = Path(__file__).resolve().parents[1] / "figures" / "data"
EMO = ["Neutral", "Happy", "Sad", "Angry"]
MODN = {"A": "声", "V": "顔", "T": "言葉"}
MODN_LANG = {"ja": MODN, "en": {"A": "Audio", "V": "Visual", "T": "Text"}}


def picker(lang):
    """L(ja, en): the Japanese value is the one used before the English version was added."""
    return (lambda ja, en: ja) if lang == "ja" else (lambda ja, en: en)
HATCH = {"A": "", "V": "///", "T": "..."}
CORP = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
LIGHT = {"A": "#F6D78F", "V": "#8FC1E3", "T": "#8FD3B9"}


TIDEA = {
    "ja": dict(iemocap="IEMOCAP\n（即興）", legend="マスの字：その感情を\n最もよく見分けた\nモダリティ",
               foot="値は手法を決めるときに使った信頼度（仮のターゲットの回，候補の全ラベル，K = 5，30 反復の平均）．評価の回の値ではない"),
    "en": dict(iemocap="IEMOCAP\n(improvised)", legend="Cell label: the\nmodality that best\ndistinguished the\nemotion",
               foot=None),   # the source of the reliabilities goes to the caption
}


def fig_idea(lang="ja"):
    setup(lang)
    t, L, modn = TIDEA[lang], picker(lang), MODN_LANG[lang]
    d = pd.read_csv(DATA / "fig7_strength_profile.csv")
    H = L(70, 45)
    fig, ax = canvas(H, L(W, W_EN))
    fs = FS + 1.0
    x0, y0, cw, ch = L((32, 8, 25, 12), (20.5, 0.8, 19, 9.6))
    for j, e in enumerate(EMO):
        txt(ax, x0 + (j + 0.5) * cw, y0 + 4 * ch + L(3, 1.4), e, ha="center", size=L(fs, 9), weight="bold", va=L("top", "baseline"))
    for i, c in enumerate(CORP):
        y = y0 + (3 - i) * ch
        txt(ax, x0 - L(3, 1.6), y + ch / 2, c if c != "IEMOCAP" else t["iemocap"], ha="right", va="center", size=L(fs, 8.5), ls=1.1)
        for j, e in enumerate(EMO):
            vals = {m: d[(d.modality == m) & (d.corpus == c)][f"d_{e}"].values[0] for m in "AVT"}
            best = max(vals, key=vals.get)
            ax.add_patch(mp.Rectangle((x0 + j * cw + 0.4, y + 0.4), cw - 0.8, ch - 0.8, fc=LIGHT[best], ec="white",
                                      hatch=HATCH[best], lw=0.0, zorder=2))
            ax.add_patch(mp.Rectangle((x0 + j * cw + 0.4, y + 0.4), cw - 0.8, ch - 0.8, fc="none", ec="#777777", lw=0.5, zorder=3))
            bw_ = L(12, 13.6)
            box(ax, x0 + (j + 0.5) * cw - bw_ / 2, y + ch / 2 - 3, bw_, 6, fc="white", ec="none", r=0.8, z=4)
            txt(ax, x0 + (j + 0.5) * cw, y + ch / 2, modn[best], ha="center", va="center", size=L(fs + 1.5, 9), weight="bold",
                color=MOD[best][0])
    # legend
    lx = x0 + 4 * cw + L(5, 3.4)
    txt(ax, lx, y0 + 4 * ch + L(3, 3.4), t["legend"], size=L(fs - 1.0, 8), va="top", ls=L(1.25, 1.15))
    for k, m in enumerate("AVT"):
        yy = y0 + L(2.6, 1.85) * ch - k * L(8, 7.4)
        ax.add_patch(mp.Rectangle((lx, yy), 8, 5.5, fc=LIGHT[m], hatch=HATCH[m], ec="#777777", lw=0.5))
        txt(ax, lx + 10, yy + 2.75, modn[m], va="center", size=L(fs, 9), color=MOD[m][0], weight="bold")
    if t["foot"]:
        txt(ax, W - 1, 0.5, t["foot"], ha="right", va="bottom", size=FS - 1.6, color=GREY)
    save(fig, f"fig_prop_idea_{lang}")


def heat(ax, x, y, vals, cw, ch, cmap, vmin, vmax, fmt, fs, labels_row=None, labels_col=None, col_y=2, col_va="top",
         col_fs=None):
    import matplotlib.cm as cm
    cmap = __import__("matplotlib").colormaps[cmap]
    vals = np.asarray(vals)
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            col = cmap((v - vmin) / (vmax - vmin))
            ax.add_patch(mp.Rectangle((x + c * cw, y - (r + 1) * ch), cw - 0.5, ch - 0.5, fc=col, ec="#999999", lw=0.4))
            lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
            txt(ax, x + c * cw + cw / 2, y - (r + 0.5) * ch, fmt.format(v).replace("-", "−"), ha="center", va="center",
                size=fs, color="white" if lum < 0.5 else "black")
    if labels_row:
        for r, l in enumerate(labels_row):
            txt(ax, x - 1.5, y - (r + 0.5) * ch, l, ha="right", va="center", size=fs, color=MOD["AVT"[r]][0], weight="bold")
    if labels_col:
        for c, l in enumerate(labels_col):
            txt(ax, x + c * cw + cw / 2, y + col_y, l, ha="center", va=col_va, size=fs - 0.6 if col_fs is None else col_fs,
                color=GREY)


TDEC = {
    "ja": dict(table="分類器 × 感情の重みの表（12 マス）", level="① 強さ", level_col="強さ", prof="② 感情ごとの形（行の平均は 0）",
               kh="ターゲットの K 本で測る", kb="コーパスごとに入れ替わるので，\nターゲットで測る．K 本から\n決める数は 3 つだけ",
               sh="ソースのコーパスで測る", sb="12 マスは K 本では測れない．\nラベルの多いソースで測る",
               ph="この表の重みで", pb="声・顔・言葉の\n分類器の答えを\n感情ごとに\n重みづけて足す\n\n→ 疑似ラベル\n→ AVT を\n　学び直す",
               note="数値は説明のための例"),
    "en": dict(table="Weight table (12 cells)", level="(1) Strength", level_col="Strength",
               prof="(2) Per-emotion profile\n      (row mean 0)",
               kh="From the K target labels", kb="Strength changes between\ncorpora, so it is measured\non the target; only 3\nnumbers come from K labels.",
               sh="From the source corpora", sb="The 12 cells cannot be\nmeasured from K labels;\nthey come from the source\ncorpora (many labels).",
               ph="With these weights", pb="the audio, visual and text answers\nare weighted per emotion and\nsummed → pseudo-labels\n→ retrain AVT",
               note=None),   # the illustration note goes to the caption
}


def fig_decomp(lang="ja"):
    """Japanese layout: as drawn before the English version was added. English: fig_decomp_en."""
    if lang == "en":
        return fig_decomp_en()
    setup(lang)
    t, modn = TDEC[lang], MODN_LANG[lang]
    H = 64
    fig, ax = canvas(H)
    fs = FS + 0.6
    level = np.array([[1.2], [1.9], [1.0]])
    prof = np.array([[-0.25, -0.10, 0.05, 0.30], [-0.45, 0.55, -0.15, 0.05], [-0.20, 0.10, 0.05, 0.05]])
    D = np.maximum(0, level + prof)
    E = ["N", "H", "S", "A"]
    rows = [modn[m] for m in "AVT"]
    ytop = H - 14
    hf = fs - 0.8
    # weight table
    txt(ax, 2, H - 3, t["table"], size=fs + 0.4, weight="bold", va="top")
    heat(ax, 10, ytop, D, 9, 7, "Oranges", 0, 2.6, "{:.2f}", hf, rows, E)
    txt(ax, 50, ytop - 10.5, "＝", size=fs + 8, va="center", ha="left")
    # strength
    txt(ax, 60, H - 3, t["level"], size=fs + 0.2, weight="bold", va="top", color=C_K)
    heat(ax, 68, ytop, level, 11, 7, "Blues", 0, 2.6, "{:.1f}", hf, rows, [t["level_col"]])
    txt(ax, 84, ytop - 10.5, "＋", size=fs + 8, va="center", ha="left")
    # profile
    txt(ax, 92, H - 3, t["prof"], size=fs + 0.2, weight="bold", va="top", color=C_SRC, ls=1.2)
    heat(ax, 98, ytop, prof, 9, 7, "PuOr_r", -0.8, 0.8, "{:+.2f}", hf, rows, E)
    # where from
    kx, kw, sx, sw = (52, 40, 95, 41)
    bh = 22
    box(ax, kx, 3, kw, bh, fc="#E8F0F8", ec=C_K, lw=1.1)
    txt(ax, kx + 2, 3 + bh - 2.5, t["kh"], size=fs - 0.2, weight="bold", color=C_K, va="top")
    txt(ax, kx + 2, 3 + bh - 8.5, t["kb"], size=fs - 1.3, va="top", ls=1.3)
    box(ax, sx, 3, sw, bh, fc="#F0EAF5", ec=C_SRC, lw=1.1)
    txt(ax, sx + 2, 3 + bh - 2.5, t["sh"], size=fs - 0.2, weight="bold", color=C_SRC, va="top")
    txt(ax, sx + 2, 3 + bh - 8.5, t["sb"], size=fs - 1.3, va="top", ls=1.3)
    box(ax, 140, 3, 32, 47, fc="#FDF0E8", ec=C_PROP, lw=1.1)
    txt(ax, 142, 47.5, t["ph"], size=fs - 0.2, weight="bold", color=C_PROP, va="top", ls=1.2)
    txt(ax, 142, 41.5, t["pb"], size=fs - 1.0, va="top", ls=1.35)
    arrow(ax, (137, 30), (139.5, 30), lw=0.9, head=7)
    txt(ax, 2, 3.5, t["note"], size=FS - 1.6, color=GREY)
    save(fig, f"fig_prop_decomp_{lang}")


def fig_decomp_en():
    """English version at the text width (131.2 mm), all text 8 pt or larger: the three tables in the top row and the
    three explanation boxes below them (the weight-table box under the weight table). Same illustrative numbers."""
    setup("en")
    t, modn = TDEC["en"], MODN_LANG["en"]
    WE, H = W_EN, 50
    fig, ax = canvas(H, WE)
    level = np.array([[1.2], [1.9], [1.0]])
    prof = np.array([[-0.25, -0.10, 0.05, 0.30], [-0.45, 0.55, -0.15, 0.05], [-0.20, 0.10, 0.05, 0.05]])
    D = np.maximum(0, level + prof)
    E = ["N", "H", "S", "A"]
    rows = [modn[m] for m in "AVT"]
    ytop, ch = H - 10.8, 6.0
    cl = dict(col_y=0.5, col_va="bottom", col_fs=8.0)
    txt(ax, 0.6, H - 0.8, t["table"], size=8.5, weight="bold", va="top")
    heat(ax, 12.8, ytop, D, 9.5, ch, "Oranges", 0, 2.6, "{:.2f}", 8.4, rows, E, **cl)
    txt(ax, 54.6, ytop - 1.5 * ch, "=", size=12, va="center", ha="center")
    txt(ax, 57.6, H - 0.8, t["level"], size=8.5, weight="bold", va="top", color=C_K)
    heat(ax, 58.6, ytop, level, 10.5, ch, "Blues", 0, 2.6, "{:.1f}", 8.4, None, [t["level_col"]], **cl)
    txt(ax, 73.2, ytop - 1.5 * ch, "+", size=12, va="center", ha="center")
    txt(ax, 76.6, H - 0.8, t["prof"], size=8.5, weight="bold", va="top", color=C_SRC, ls=1.1)
    heat(ax, 77.0, ytop, prof, 10.0, ch, "PuOr_r", -0.8, 0.8, "{:+.2f}", 8.4, None, E, **cl)
    # boxes: the use of the table (under the weight table), strength (K labels), profile (source corpora)
    by, bh = 0.6, 18.2
    for x, w, head, body, fc, ec in [(0.6, 50.4, t["ph"], t["pb"], "#FDF0E8", C_PROP),
                                     (52.8, 38.4, t["kh"], t["kb"], "#E8F0F8", C_K),
                                     (93.0, WE - 93.6, t["sh"], t["sb"], "#F0EAF5", C_SRC)]:
        box(ax, x, by, w, bh, fc=fc, ec=ec, lw=1.1)
        txt(ax, x + 1.4, by + bh - 1.0, head, size=8, weight="bold", color=ec, va="top")
        txt(ax, x + 1.4, by + bh - 4.6, body, size=8, va="top", ls=1.1)
    arrow(ax, (31.8, ytop - 3 * ch - 0.3), (31.8, by + bh + 0.2), lw=0.9, head=6, color=C_PROP)
    save(fig, "fig_prop_decomp_en")


def corp_box(ax, x, y, w, h, name, role, fs):
    col = C_K if role == "t" else C_SRC
    fc = "#E8F0F8" if role == "t" else "#F0EAF5"
    box(ax, x, y, w, h, fc=fc, ec=col, lw=1.0, z=3)
    txt(ax, x + w / 2, y + h / 2, name, ha="center", va="center", size=fs, color=col, weight="bold", z=4)


TEPI = {
    "ja": dict(
        head="ハイパーパラメータの選択：ソースを 1 つずつ仮のターゲットにした 3 つのエピソード（ターゲットが AFEW のとき．AFEW は使わない）",
        ep="エピソード {k}", pt="仮のターゲット", src="ソース",
        ep_body="強さ：K 本で　形：ソースで\n→ ハイパーパラメータの候補\n　ごとにテストの UAR",
        sel="選ぶハイパーパラメータ", sel_body="3 つのエピソード（30 反復）の UAR の平均が\n最大の組を選ぶ（16 通りから 1 つ）",
        items=[("・n0：K 本の強さに既定の強さをどれだけ混ぜるか", "ラベル何本ぶん混ぜるか（0，20，80，∞）．\n0 なら K 本の値だけ，∞ なら既定の値だけ"),
               ("・戻し先：混ぜる既定の強さは何か", "3 つの分類器の強さの平均（3 つを同じ強さ\nに近づける）か，ソースで測った強さか"),
               ("・β：感情ごとの形を使うか", "1 なら使う，0 なら使わない（感情に共通の重み）")],
        chosen="選んだ値で",
        eta="形がほかのコーパスでどれだけ残るか（η）", din="同じコーパスの中で\n測る信頼度 d_in", dout="別のコーパスに当てた\n信頼度 d_out",
        eta_f="η ＝ Σ d_out ÷ Σ d_in を形に掛ける（ソースどうしで測る）",
        ev="評価：AFEW",
        ev_body="強さ：AFEW の K 本で\n形：3 つのソースで\nハイパーパラメータ：\n　エピソードで選んだ値\n→ 疑似ラベルで AVT を学び\n　直し，テストで 1 回だけ評価",
    ),
    "en": dict(
        head="(a) Hyperparameter selection: each source corpus in turn is a provisional\n      target (3 episodes; target AFEW, which is not used here)",
        ep="Episode {k}", pt="provisional target", src="sources",
        ep_body="In each episode: strength from the K labels, profile\nfrom the sources → UAR on the provisional target’s\n"
                "test folds for each hyperparameter candidate",
        sel="Hyperparameters to select",
        sel_body="The combination (1 of 16) with the highest\nmean UAR over the 3 episodes (30 repetitions)",
        items=[("• n0: default strength mixed into the\n   K-label strength", "In labels’ worth (0, 20, 80, ∞);\n0: K labels only, ∞: default only"),
               ("• Default strength (shrinkage target)", "Mean strength of the 3 classifiers (pulls\nthem to equal strength) or the strength\nmeasured on the sources"),
               ("• β: use the per-emotion profile?", "1: yes; 0: no (one weight for all emotions)")],
        chosen="with the selected values",
        eta="(b) How much of the profile carries\n      over to another corpus (η)", din="d_in: reliability within\nthe same corpus",
        dout="d_out: reliability on\nanother corpus",
        eta_f="η = Σ d_out ÷ Σ d_in multiplies the profile\n(measured between source corpora)",
        ev="(c) Evaluation on AFEW",
        ev_body="Strength: AFEW’s K labels; profile: the\n3 sources; hyperparameters: the values\nchosen in the episodes → retrain AVT on\n"
                "the pseudo-labels; evaluate once on the\ntest set",
    ),
}


def fig_episode(lang="ja"):
    """Japanese layout: as drawn before the English version was added. English: fig_episode_en."""
    if lang == "en":
        return fig_episode_en()
    setup(lang)
    t = TEPI[lang]
    H = 96
    fig, ax = canvas(H)
    fs = FS + 0.4
    txt(ax, 2, H - 2, t["head"], size=fs - 0.2, weight="bold", va="top", ls=1.2)
    eps = [("IEMOCAP", ["MOSEI", "MELD"]), ("MOSEI", ["IEMOCAP", "MELD"]), ("MELD", ["IEMOCAP", "MOSEI"])]
    eb, et = 42, 85        # bottom and top of the episode boxes
    sh = 0                         # shift of the contents of the episode boxes
    for k, (pt, ss) in enumerate(eps):
        x = 2 + k * 37
        box(ax, x, eb, 35, et - eb, fc="white", ec="#888888", lw=0.8)
        txt(ax, x + 2, et - 2, t["ep"].format(k=k + 1), size=fs, weight="bold", va="top")
        corp_box(ax, x + 1.5, 63 + sh, 15.5, 9, pt, "t", fs - 1.4)
        txt(ax, x + 9.2, 61.5 + sh, t["pt"], ha="center", va="top", size=fs - 2.2, color=C_K, ls=1.2)
        for j, src in enumerate(ss):
            corp_box(ax, x + 19.5, 68 + sh - j * 10, 14, 8, src, "s", fs - 1.8)
        txt(ax, x + 26.5, 55.5 + sh, t["src"], ha="center", va="top", size=fs - 2.2, color=C_SRC)
        txt(ax, x + 2, 51, t["ep_body"], size=fs - 1.9, va="top", ls=1.3)
    arrow(ax, (113, 63), (116.5, 63), lw=0.9, head=7, color="#555555")
    box(ax, 117, eb, 55, et - eb, fc="#FDF0E8", ec=C_PROP, lw=1.1)
    txt(ax, 119, et - 2, t["sel"], size=fs, weight="bold", color=C_PROP, va="top")
    txt(ax, 119, et - 7.5, t["sel_body"], size=fs - 1.7, va="top", ls=1.3)
    y = 70.5
    for name, body in t["items"]:
        txt(ax, 119, y, name, size=fs - 1.8, weight="bold", va="top", color=C_PROP, ls=1.2)
        txt(ax, 121.5, y - 3.6, body, size=fs - 2.1, va="top", ls=1.25)
        y -= 9.8
    arrow(ax, (147, 41.5), (147, 36.5), lw=1.0, head=8, color=C_PROP)
    txt(ax, 149, 39, t["chosen"], size=fs - 1.6, color=C_PROP, va="center")
    # eta inset
    bt = 34    # top of the two lower boxes
    ew = 74
    box(ax, 2, 3, ew, bt - 3, fc="#F7F7F7", ec="#999999", lw=0.6)
    txt(ax, 4, bt - 2.5, t["eta"], size=fs - 0.2, weight="bold", va="top", ls=1.2)
    corp_box(ax, 6, 8, 18, 9, "IEMOCAP", "s", fs - 1.0)
    corp_box(ax, 54, 8, 18, 9, "MELD", "s", fs - 1.0)
    txt(ax, 15, 22.5, t["din"], ha="center", va="center", size=fs - 1.8, color=C_SRC, ls=1.15)
    arrow(ax, (24.5, 12.5), (53.5, 12.5), lw=0.9, head=7, color=C_SRC)
    txt(ax, 39, 16.5, t["dout"], ha="center", va="center", size=fs - 1.8, color=C_SRC, ls=1.15)
    txt(ax, 4, 5.5, t["eta_f"], size=fs - 1.8, ls=1.2)
    # final
    fx = 82
    box(ax, fx, 3, W - 2 - fx, bt - 3, fc="#E8F0F8", ec=C_K, lw=1.1)
    txt(ax, fx + 2, bt - 2.5, t["ev"], size=fs, weight="bold", color=C_K, va="top")
    corp_box(ax, 86, 10, 20, 10, "AFEW", "t", fs - 0.4)
    for j, src in enumerate(["IEMOCAP", "MOSEI", "MELD"]):
        corp_box(ax, 112, 21.5 - j * 6.6, 18, 5.6, src, "s", fs - 1.8)
    txt(ax, 134, 26, t["ev_body"], size=fs - 1.7, va="top", ls=1.3)
    save(fig, f"fig_prop_episode_{lang}")


def fig_episode_en():
    """English version at the text width (131.2 mm), all text 8 pt or larger: (a) the three episodes as rows with the
    selection box beside them, (b) the transfer of the profile (eta), (c) the evaluation on AFEW."""
    setup("en")
    t = TEPI["en"]
    WE, H = W_EN, 95
    fig, ax = canvas(H, WE)
    txt(ax, 0.6, H - 0.8, t["head"], size=8.5, weight="bold", va="top", ls=1.1)
    eb, et = 39.0, H - 9.0                 # bottom and top of the two upper boxes
    box(ax, 0.6, eb, 65.4, et - eb, fc="white", ec="#888888", lw=0.8)
    txt(ax, 24.6, et - 1.4, t["pt"], ha="center", va="top", size=8, color=C_K)
    txt(ax, 49.6, et - 1.4, t["src"], ha="center", va="top", size=8, color=C_SRC)
    eps = [("IEMOCAP", ["MOSEI", "MELD"]), ("MOSEI", ["IEMOCAP", "MELD"]), ("MELD", ["IEMOCAP", "MOSEI"])]
    for k, (pt, ss) in enumerate(eps):
        yc = et - 8.6 - k * 8.4
        txt(ax, 1.8, yc, t["ep"].format(k=k + 1), size=8, weight="bold", va="center")
        corp_box(ax, 17.0, yc - 3.2, 15.4, 6.4, pt, "t", 8)
        for j, src in enumerate(ss):
            corp_box(ax, 34.2 + j * 15.8, yc - 3.2, 15.2, 6.4, src, "s", 8)
    txt(ax, 1.8, et - 33.0, t["ep_body"], size=8, va="top", ls=1.1)
    arrow(ax, (66.2, (eb + et) / 2), (68.2, (eb + et) / 2), lw=0.9, head=6, color="#555555")
    sx = 68.4
    box(ax, sx, eb, WE - 0.6 - sx, et - eb, fc="#FDF0E8", ec=C_PROP, lw=1.1)
    txt(ax, sx + 1.4, et - 1.2, t["sel"], size=8.5, weight="bold", color=C_PROP, va="top")
    txt(ax, sx + 1.4, et - 5.0, t["sel_body"], size=8, va="top", ls=1.1)
    y = et - 5.0 - 3.1 * (t["sel_body"].count("\n") + 1) - 1.2
    for name, bd in t["items"]:
        nl, bl = name.count("\n") + 1, bd.count("\n") + 1
        txt(ax, sx + 1.4, y, name, size=8, weight="bold", va="top", color=C_PROP, ls=1.1)
        txt(ax, sx + 3.6, y - 3.1 * nl, bd, size=8, va="top", ls=1.1)
        y -= 3.1 * (nl + bl) + 1.0
    ax_ = 99.0
    arrow(ax, (ax_, eb - 0.4), (ax_, 33.6), lw=1.0, head=7, color=C_PROP)
    txt(ax, ax_ - 1.6, (eb + 33.6) / 2, t["chosen"], size=8, color=C_PROP, va="center", ha="right")
    # (b) eta
    bt, pw = 33.0, 64.3
    box(ax, 0.6, 0.6, pw, bt - 0.6, fc="#F7F7F7", ec="#999999", lw=0.6)
    txt(ax, 1.8, bt - 1.0, t["eta"], size=8.5, weight="bold", va="top", ls=1.1)
    txt(ax, 2.0, bt - 8.6, t["din"], size=8, color=C_SRC, va="top", ls=1.1)
    txt(ax, 35.4, bt - 8.6, t["dout"], size=8, color=C_SRC, va="top", ls=1.1)
    corp_box(ax, 2.0, 10.0, 18, 7, "IEMOCAP", "s", 8)
    corp_box(ax, 45.0, 10.0, 18, 7, "MELD", "s", 8)
    arrow(ax, (20.4, 13.5), (44.6, 13.5), lw=0.9, head=7, color=C_SRC)
    txt(ax, 2.0, 8.4, t["eta_f"], size=8, va="top", ls=1.1)
    # (c) evaluation
    fx = 66.3
    box(ax, fx, 0.6, pw, bt - 0.6, fc="#E8F0F8", ec=C_K, lw=1.1)
    txt(ax, fx + 1.4, bt - 1.0, t["ev"], size=8.5, weight="bold", color=C_K, va="top")
    corp_box(ax, fx + 1.4, 20.0, 14.0, 6.4, "AFEW", "t", 8)
    for j, (src, w) in enumerate([("IEMOCAP", 15.2), ("MOSEI", 12.6), ("MELD", 11.8)]):
        corp_box(ax, fx + 18.0 + [0, 15.8, 29.0][j], 20.0, w, 6.4, src, "s", 8)
    txt(ax, fx + 1.4, 18.4, t["ev_body"], size=8, va="top", ls=1.1)
    save(fig, "fig_prop_episode_en")


def bars(ax, x0, y0, vals, col, bw=5.0, gap=1.6, scale=8.0, fs=FS, signed=False, hatch="", lbl_bg=None):
    for i, v in enumerate(vals):
        x = x0 + i * (bw + gap)
        if signed:
            ax.add_patch(mp.Rectangle((x, y0 if v >= 0 else y0 + v * scale), bw, abs(v) * scale, fc=col, ec="black", lw=0.3, hatch=hatch))
            txt(ax, x + bw / 2, y0 + v * scale + (1.0 if v >= 0 else -1.0), f"{v:+.2f}".replace("-", "−"), ha="center",
                va="bottom" if v >= 0 else "top", size=fs - 1.6)
        else:
            ax.add_patch(mp.Rectangle((x, y0), bw, v * scale, fc=col, ec="black", lw=0.3, hatch=hatch))
            if lbl_bg is None:
                txt(ax, x + bw / 2, y0 + v * scale + 0.8, f"{v:.2f}", ha="center", va="bottom", size=fs - 1.6)
            else:
                ax.text(x + bw / 2, y0 + v * scale + 0.8, f"{v:.2f}", ha="center", va="bottom", fontsize=fs - 1.6, zorder=6,
                        bbox=dict(boxstyle="square,pad=0.1", fc=lbl_bg, ec="none"))
        txt(ax, x + bw / 2, y0 - (0.8 if not signed else 7.5), "NHSA"[i], ha="center", va="top", size=fs - 1.2, color=GREY)
    ax.plot([x0 - 1, x0 + 4 * (bw + gap) - gap + 1], [y0, y0], color="black", lw=0.5)


TCALC = {
    "ja": dict(
        lhead="① 強さ：ターゲットの K 本だけで計算（例：顔の分類器，K = 5）", grp="組{g}", grp_left=None,
        kcount="K 本 ＝ 4 感情 × 5 本 ＝ 20 本",
        steps="1. 20 本を 5 組に分ける（各組に\n　 各感情 1 本）\n2. 4 組で分類器を学び直し，\n　 残る 1 組に答えを付ける．\n　 これを 5 回くり返す\n3. 20 本すべてに，その発話を\n　 学びに使っていない答えが付く",
        step4="4. その答えで感情ごとに d を求める（前のページの方法）", mean_k="平均 ＝ 強さ {m:.2f}",
        lmean="意味：このターゲットで，\n顔の分類器が感情全体を\nどれだけよく見分けるか", lnote="K 本が少ないとぶれるので，\nn0 本ぶん既定の値に寄せる",
        rhead="② 感情ごとの形：ソースのコーパスで計算（K 本は使わない）",
        rstep="1. 各ソースの中で，話者を分けた交差検証で\n　 顔の分類器の 4 感情の d を求め，ソースどうしで\n　 平均する（ほかのコーパスで残る割合 η も掛ける）",
        mean_s="平均 {m:.2f}", sub="2. 平均を引く", prof="＝ 感情ごとの形",
        rmean="意味：顔は自分の平均\nより Happy が得意で，\nNeutral が苦手．\n強さの情報は持たない", note="数値は説明のための例",
    ),
    "en": dict(
        lhead="(a) Strength: from the K target\n      labels only (visual, K = 5)", grp="{g}", grp_left="Subset",
        kcount="K labels = 4 emotions × 5 = 20",
        steps="1. Split the 20 into\n    5 subsets (one per\n    emotion)\n2. Retrain on 4 sub-\n    sets, answer the\n"
              "    5th; repeat 5 times\n3. All 20 get answers\n    from classifiers not\n    trained on them",
        step4="4. From these\n    answers, compute\n    the reliability d\n    per emotion", mean_k="mean = strength {m:.2f}",
        lmean="Meaning: how well the\nvisual classifier tells\nthe emotions apart on\nthis target",
        lnote="Few K labels give noisy\nvalues, so the value is\nshrunk towards a default\nby n0 labels’ worth",
        rhead="(b) Per-emotion profile: from the\n      sources (K labels not used)",
        rstep="1. In each source corpus, compute the\n    visual classifier’s d for the 4 emotions\n    by speaker-independent cross-\n"
              "    validation; average over the sources\n    (× η, the share carried over to\n    other corpora)",
        mean_s="mean {m:.2f}", sub="2. Subtract the mean", prof="= per-emotion\n   profile",
        rmean="Meaning: relative to its\nown mean, the visual\nclassifier is better at\nHappy, worse at Neutral;\nno strength information",
        note=None,   # the illustration note goes to the caption
    ),
}


def fig_calc(lang="ja"):
    """Japanese layout: as drawn before the English version was added. English: fig_calc_en."""
    if lang == "en":
        return fig_calc_en()
    setup(lang)
    t = TCALC[lang]
    H = 84
    fig, ax = canvas(H)
    fs = FS + 0.4
    colV = MOD["V"][0]
    # left: strength from K labels
    box(ax, 2, 2, 84, H - 4, fc="#F4F8FC", ec=C_K, lw=1.1)
    txt(ax, 4, H - 4.5, t["lhead"], size=fs - 0.2, weight="bold", color=C_K, va="top", ls=1.2)
    # 4 x 5 grid of K-labelled utterances coloured by group
    gcol = ["#CFE3F3", "#F9E0C7", "#D8EFD6", "#EADCF0", "#F3D3D8"]
    gx, gy, cw, ch = 6, H - 34, 5.2, 4.6
    for e in range(4):
        txt(ax, gx - 1, gy + (3 - e) * ch + ch / 2, "NHSA"[e], ha="right", va="center", size=fs - 1.2, color=GREY)
        for g in range(5):
            ax.add_patch(mp.Rectangle((gx + g * cw, gy + (3 - e) * ch), cw - 0.6, ch - 0.6, fc=gcol[g], ec="#888888", lw=0.4))
    for g in range(5):
        txt(ax, gx + g * cw + cw / 2 - 0, gy - 1.2, t["grp"].format(g=g + 1), ha="center", va="top", size=fs - 2.2, color=GREY)
    txt(ax, gx, gy + 4 * ch + 2, t["kcount"], size=fs - 1.4)
    txt(ax, 40, H - 13, t["steps"], size=fs - 1.6, va="top", ls=1.3)
    dK = [1.40, 2.55, 1.60, 2.05]
    txt(ax, 6, 33, t["step4"], size=fs - 1.4, ls=1.2)
    by = 9
    bars(ax, 10, by, dK, colV, scale=7.0, fs=fs, hatch="///")
    m = float(np.mean(dK))
    mx1 = 37.5
    ax.plot([8.5, mx1], [by + m * 7.0] * 2, color=C_K, lw=1.2, ls="--")
    txt(ax, mx1 + 1.5, by + m * 7.0, t["mean_k"].format(m=m), va="center", size=fs - 0.6, color=C_K, weight="bold")
    txt(ax, 40, 18.5, t["lmean"], size=fs - 1.6, va="top", ls=1.3)
    txt(ax, 40, 8.5, t["lnote"], size=fs - 1.8, va="top", ls=1.3, color=GREY)
    # right: shape from sources
    box(ax, 90, 2, 82, H - 4, fc="#F8F4FA", ec=C_SRC, lw=1.1)
    txt(ax, 92, H - 4.5, t["rhead"], size=fs - 0.2, weight="bold", color=C_SRC, va="top", ls=1.2)
    txt(ax, 93, H - 12, t["rstep"], size=fs - 1.6, va="top", ls=1.3)
    dS = [0.95, 1.95, 1.25, 1.45]
    sy = 45
    bars(ax, 98, sy, dS, colV, scale=7.0, fs=fs, hatch="///")
    ms = float(np.mean(dS))
    sx1 = 125.5
    ax.plot([96.5, sx1], [sy + ms * 7.0] * 2, color=C_SRC, lw=1.0, ls="--")
    txt(ax, sx1 + 1.5, sy + ms * 7.0, t["mean_s"].format(m=ms), va="center", size=fs - 0.8, color=C_SRC)
    arrow(ax, (112, 42), (112, 35), lw=0.9, head=7, color=C_SRC)
    txt(ax, 114, 38.5, t["sub"], size=fs - 1.2, color=C_SRC, va="center")
    r = [v - ms for v in dS]
    ry = 22
    bars(ax, 98, ry, r, "#C9B8DA", scale=7.0, fs=fs, signed=True)
    txt(ax, 128, 28, t["prof"], size=fs - 0.4, color=C_SRC, weight="bold", va="center")
    txt(ax, 128, 21, t["rmean"], size=fs - 1.6, va="top", ls=1.3)
    txt(ax, W - 4, 5.0, t["note"], ha="right", size=FS - 1.6, color=GREY)
    save(fig, f"fig_prop_calc_{lang}")


def fig_calc_en():
    """English version at the text width (131.2 mm), all text 8 pt or larger. Same illustrative numbers as Japanese."""
    setup("en")
    t = TCALC["en"]
    WE, H = W_EN, 81
    fig, ax = canvas(H, WE)
    colV = MOD["V"][0]
    bw = dict(bw=5.0, gap=2.0)
    pw = 64.3
    # (a) strength from the K labels
    box(ax, 0.6, 0.6, pw, H - 1.2, fc="#F4F8FC", ec=C_K, lw=1.1)
    txt(ax, 1.8, H - 1.6, t["lhead"], size=8.5, weight="bold", color=C_K, va="top", ls=1.1)
    txt(ax, 2.0, H - 9.8, t["kcount"], size=8, va="top")
    gcol = ["#CFE3F3", "#F9E0C7", "#D8EFD6", "#EADCF0", "#F3D3D8"]
    gx, cw, ch = 13.0, 4.4, 3.9
    gy = H - 14.0 - 4 * ch
    for e in range(4):
        txt(ax, gx - 1, gy + (3 - e) * ch + ch / 2, "NHSA"[e], ha="right", va="center", size=8, color=GREY)
        for g in range(5):
            ax.add_patch(mp.Rectangle((gx + g * cw, gy + (3 - e) * ch), cw - 0.6, ch - 0.6, fc=gcol[g], ec="#888888", lw=0.4))
    for g in range(5):
        txt(ax, gx + g * cw + (cw - 0.6) / 2, gy - 0.8, t["grp"].format(g=g + 1), ha="center", va="top", size=8, color=GREY)
    txt(ax, gx - 1, gy - 0.8, t["grp_left"], ha="right", va="top", size=8, color=GREY)
    txt(ax, 37.0, H - 14.0, t["steps"], size=8, va="top", ls=1.1)
    txt(ax, 2.0, gy - 4.8, t["step4"], size=8, va="top", ls=1.1)
    dK = [1.40, 2.55, 1.60, 2.05]
    by, sc = 6.0, 7.0
    bars(ax, 4.0, by, dK, colV, scale=sc, fs=9.6, hatch="///", lbl_bg="#F4F8FC", **bw)
    m = float(np.mean(dK))
    ax.plot([2.5, 31.5], [by + m * sc] * 2, color=C_K, lw=1.2, ls="--")
    txt(ax, 33.0, by + m * sc, t["mean_k"].format(m=m), va="center", size=8, color=C_K, weight="bold")
    txt(ax, 33.5, 33.5, t["lmean"], size=8, va="top", ls=1.1)
    txt(ax, 33.5, 16.6, t["lnote"], size=8, va="top", ls=1.1, color=GREY)
    # (b) per-emotion profile from the source corpora
    rx = 66.3
    box(ax, rx, 0.6, pw, H - 1.2, fc="#F8F4FA", ec=C_SRC, lw=1.1)
    txt(ax, rx + 1.2, H - 1.6, t["rhead"], size=8.5, weight="bold", color=C_SRC, va="top", ls=1.1)
    txt(ax, rx + 1.4, H - 9.8, t["rstep"], size=8, va="top", ls=1.1)
    dS = [0.95, 1.95, 1.25, 1.45]
    sy, ss = 36.0, 5.5
    bars(ax, rx + 4.0, sy, dS, colV, scale=ss, fs=9.6, hatch="///", lbl_bg="#F8F4FA", **bw)
    ms = float(np.mean(dS))
    ax.plot([rx + 2.5, rx + 31.5], [sy + ms * ss] * 2, color=C_SRC, lw=1.0, ls="--")
    txt(ax, rx + 33.0, sy + ms * ss, t["mean_s"].format(m=ms), va="center", size=8, color=C_SRC)
    arrow(ax, (rx + 16.5, 31.6), (rx + 16.5, 26.6), lw=0.9, head=7, color=C_SRC)
    txt(ax, rx + 19.0, 29.1, t["sub"], size=8, color=C_SRC, va="center")
    r = [v - ms for v in dS]
    bars(ax, rx + 4.0, 16.0, r, "#C9B8DA", scale=ss, fs=9.6, signed=True, **bw)
    txt(ax, rx + 33.0, 23.0, t["prof"], size=8.5, color=C_SRC, weight="bold", va="center", ls=1.1)
    txt(ax, rx + 33.0, 17.8, t["rmean"], size=8, va="top", ls=1.1)
    save(fig, "fig_prop_calc_en")


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["ja", "en"]):
        fig_calc(lang)
        fig_idea(lang)
        fig_decomp(lang)
        fig_episode(lang)
