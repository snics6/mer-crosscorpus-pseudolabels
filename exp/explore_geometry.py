"""【探索的】良いソース・良い組み合わせを，特徴空間の形で説明できるか（E9）。

コーパスごとに，4 つの感情の発話の平均（重心）を求め，コーパス全体の重心（4 つの重心の平均）から
各感情の重心へ引いた矢印を「感情の向き」と呼ぶ。直線で感情を分ける学習器がソースからターゲットへ
移せるのは，両者の感情の向きがそろっているときだと考え，次の量を測る。

**空間**：その種類の全群を等しい重みで合わせた平均と標準偏差で，次元ごとに標準化した特徴
（群の大きさで偏らないように群ごとの平均・分散から作る）。

コーパスごとの量（`e9_corpus.csv`）
- `reliability`：話者で 2 つに分けた半分どうしの，感情の向き（4 本の矢印をまとめたもの）のコサイン。
  話者が変わっても同じ向きが出るか
- `arrow_power`：矢印の長さの二乗和を，半分どうしの内積で見積もった値（雑音で長く見えるのを除く）
- `dprime`：2 つの感情の重心を結ぶ線の上で，2 つの雲の中心が標準偏差の何倍離れているか
  （線の向きは片方の半分，離れ方はもう片方の半分で測り，入れ替えて平均．6 組の感情の対の平均）
- `eff_dim`：感情ごとに中心をそろえた雲の実効次元数（固有値の参加率 (Σλ)²/Σλ²．標本数による偏りを補正）
- `spread`：感情ごとに中心をそろえた雲の，次元あたりの分散
- `typicality`：ほかの群との `align` の平均

組ごとの量（`e9_pair.csv`．対称なものと向きのあるものがある）
- `align`：ターゲットとソースの感情の向きのコサイン．分母は `arrow_power`（雑音の偏りを除いた長さ）を
  使う（主に使う版）。`align_raw` は全データの矢印でそのまま求めたコサイン（確かめ用），
  `cos_<感情>` は感情ごとの矢印のコサイン（記述用）
- `shift_emo`：ターゲットの重心がソースの重心からずれた向きを，ソースの感情の矢印に射影し，
  矢印の長さを 1 とした量 β_k = ⟨δ, d_k⟩/‖d_k‖² の二乗平均の平方根（δ = c_t − c_s）。
  ソースの目でみて，ターゲット全体がどれかの感情の側へ寄って見える大きさ
- `shift_total`：‖δ‖ をソースの矢印の長さの二乗平均の平方根で割った量（ずれ全体の大きさ．比べる用）
- `src_uar`：ソースのみの分類器のターゲットでの UAR（`results/shift_<種類>/labeling.csv` の `uar_t`）

解析（いずれもターゲットの全ラベルを使う量であり，新しいターゲットでソースを選ぶ方法には使えない）
1. 組に固有の部分：E2 と同じく，順位に直した得を μ + a_t + b_s + β·量 に当てはめ，β を残りの自由度で検定
   （`e9_partial.csv`．種類ごと・K ごと．Holm は種類ごとに量 × K の全検定にかける）。`src_uar` も応答に入れる
2. ソースの効果 b_s・ターゲットの効果 a_t と，コーパスごとの量の Spearman の ρ（`e9_effects.csv`．n = 6〜7）
3. 音声と映像：同じ 6 群の 15 対で `align` を比べる（`e9_modality.csv`．Wilcoxon の符号順位検定）

    ./.venv/bin/python exp/explore_geometry.py
    ./.venv/bin/python exp/explore_geometry.py --kinds audio_wavlm video fuse-audio-video text_roberta \
        fuse-audio-text fuse-video-text fuse-audio-video-text      # 言葉系も（2026-10-18）

`--kinds` に既定の 3 種類（`KINDS`）以外を足すと，その種類ごとに全群を 1 つの設定（名前は種類の名前）として
解析 1・2 に加え，解析 3 の後ろに，選んだ全種類に共通の群（言葉系を含めると IEMOCAP-impro，IEMOCAP-script，
MOSEI，MELD の 4 群・6 対）で `align` を比べる行（`groups` の列に群の数）を足す。既定の 3 種類の行は
`--kinds` を付けないときと同じ値になる。`KINDS` は E10（`explore_shape.py`）も読むので変えない。

出力：`results/explore_relation/e9_corpus.csv`・`e9_pair.csv`・`e9_partial.csv`・`e9_effects.csv`・
`e9_modality.csv`・`fig_e9.png`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for, load_any  # noqa: E402
from pair_table import OUT, SIX, fit, gains, holm, partial_rank  # noqa: E402
from split import speaker_folds  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

KINDS = ("audio_wavlm", "video", "fuse-audio-video")
K_ORDER = ["1", "5", "10", "25", "50", "all"]
EMO = ["neutral", "happy", "sad", "angry"]
PAIR_Q = ["align", "align_raw", "shift_emo", "shift_total"]
CORPUS_Q = ["reliability", "dprime", "eff_dim", "spread", "typicality"]


def arrows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """感情ごとの重心 m_k と，4 つの重心の平均 c から引いた矢印 d_k = m_k − c（4 × 次元）。"""
    m = np.stack([X[y == k].mean(axis=0) for k in range(4)])
    c = m.mean(axis=0)
    return c, m - c


def dprime(XA, yA, XB, yB) -> float:
    """方向は A，離れ方は B で測る。6 組の感情の対の平均。"""
    out = []
    for k in range(4):
        for j in range(k + 1, 4):
            u = XA[yA == k].mean(0) - XA[yA == j].mean(0)
            u /= np.linalg.norm(u)
            pk, pj = XB[yB == k] @ u, XB[yB == j] @ u
            out.append(abs(pk.mean() - pj.mean()) / np.sqrt((pk.var(ddof=1) + pj.var(ddof=1)) / 2))
    return float(np.mean(out))


def corpus(df: pd.DataFrame, X: np.ndarray) -> dict:
    y = df.label_4.to_numpy()
    half = speaker_folds(df.speaker_group, k=2, seed=0).to_numpy()
    A, B = half == 0, half == 1
    c, D = arrows(X, y)
    _, DA = arrows(X[A], y[A])
    _, DB = arrows(X[B], y[B])
    power = float((DA * DB).sum())
    rel = power / np.sqrt((DA ** 2).sum() * (DB ** 2).sum())
    W = X - np.stack([X[y == k].mean(0) for k in range(4)])[y]       # 感情ごとに中心をそろえる
    n, p = W.shape
    S = (W.T @ W) / (n - 4)
    tr, tr2 = float(np.trace(S)), float((S ** 2).sum())
    tr2_u = tr2 - tr ** 2 / (n - 4)                                    # 標本数による偏りを除く
    return {"c": c, "D": D, "n": n, "n_A": int(A.sum()), "n_B": int(B.sum()),
            "reliability": rel, "arrow_power": power, "arrow_power_raw": float((D ** 2).sum()),
            "dprime": (dprime(X[A], y[A], X[B], y[B]) + dprime(X[B], y[B], X[A], y[A])) / 2,
            "eff_dim": tr ** 2 / tr2_u, "spread": tr / p}


def standardize(data: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> dict:
    mus = np.stack([X.mean(0) for _, X in data.values()])
    vs = np.stack([X.var(0) for _, X in data.values()])
    mu = mus.mean(0)
    sd = np.sqrt(vs.mean(0) + mus.var(0))                              # 等しい重みの混合の分散
    sd = np.where(sd > 1e-12, sd, 1.0)
    return {g: (df, (X - mu) / sd) for g, (df, X) in data.items()}


def geometry(kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = groups_for(kind)
    data = standardize({g: load_any(g, kind) for g in groups})
    C = {g: corpus(df, X) for g, (df, X) in data.items()}
    lab = pd.read_csv(ROOT / "results" / f"shift_{kind}" / "labeling.csv", comment="#")
    src_uar = lab.set_index(["target", "source"]).uar_t
    rows = []
    for t in groups:
        for s in groups:
            if t == s:
                continue
            Dt, Ds = C[t]["D"], C[s]["D"]
            inner = float((Dt * Ds).sum())
            pw_t, pw_s = C[t]["arrow_power"], C[s]["arrow_power"]
            delta = C[t]["c"] - C[s]["c"]
            beta = (Ds @ delta) / (Ds ** 2).sum(axis=1)
            rows.append({"kind": kind, "target": t, "source": s,
                         "align": inner / np.sqrt(pw_t * pw_s) if min(pw_t, pw_s) > 0 else np.nan,
                         "align_raw": inner / np.sqrt((Dt ** 2).sum() * (Ds ** 2).sum()),
                         **{f"cos_{e}": float(Dt[k] @ Ds[k] / np.linalg.norm(Dt[k]) / np.linalg.norm(Ds[k]))
                            for k, e in enumerate(EMO)},
                         "shift_emo": float(np.sqrt((beta ** 2).mean())),
                         "shift_total": float(np.linalg.norm(delta) / np.sqrt((Ds ** 2).sum(1).mean())),
                         "src_uar": float(src_uar.get((t, s), np.nan))})
    pair = pd.DataFrame(rows)
    sym = pair.groupby("source")["align"].mean()                         # align は対称なので向きは問わない
    corp = pd.DataFrame([{"kind": kind, "group": g, **{k: v for k, v in C[g].items() if k not in ("c", "D")},
                          "typicality": float(sym[g])} for g in groups])
    return corp, pair


def common_reps(G: pd.DataFrame) -> list[int]:
    return sorted(set.intersection(*[set(G[G.target == t].rep) for t in G.target.unique()]))


def settings_for(kinds, G: dict) -> list[tuple[str, str, list[str]]]:
    """既定の 3 種類は 4 つの設定（これまでと同じ）。それ以外の種類は，全群で 1 つの設定（名前は種類の名前）。"""
    out = []
    if "audio_wavlm" in kinds:
        seven = sorted(G["audio_wavlm"].target.unique())
        out += [("audio 7 groups", "audio_wavlm", seven), ("audio 6 groups", "audio_wavlm", SIX)]
    if "video" in kinds:
        out.append(("video", "video", SIX))
    if "fuse-audio-video" in kinds:
        out.append(("audio+video", "fuse-audio-video", SIX))
    out += [(k, k, sorted(G[k].target.unique())) for k in kinds if k not in KINDS]
    return out


def analyse(corp: pd.DataFrame, pair: pd.DataFrame, kinds=KINDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    G = {k: gains(k) for k in kinds}
    settings = settings_for(kinds, G)
    partial, effects = [], []
    for name, kind, groups in settings:
        reps = common_reps(G[kind])
        pq = pair[(pair.kind == kind) & pair.target.isin(groups) & pair.source.isin(groups)]
        cq = corp[(corp.kind == kind) & corp.group.isin(groups)].set_index("group").copy()
        cq["typicality"] = pq.groupby("source")["align"].mean()             # その設定の群だけで数え直す
        rows = []
        for K in K_ORDER + ["src"]:
            if K == "src":
                d = pq.rename(columns={"src_uar": "y"})
            else:
                h = G[kind]
                h = h[(h.K == K) & h.target.isin(groups) & h.source.isin(groups) & h.rep.isin(reps)]
                d = h.groupby(["target", "source"], as_index=False).gain.mean().rename(columns={"gain": "y"})
                d = d.merge(pq, on=["target", "source"])
                f = fit(d, "y")
                for side, eff in (("source", f["b"]), ("target", f["a"])):
                    for q in CORPUS_Q:
                        x = cq[q].reindex(eff.index)
                        effects.append({"setting": name, "K": K, "effect": side, "quantity": q,
                                        "rho": stats.spearmanr(eff, x)[0], "n": len(eff)})
            for q in PAIR_Q:
                pr = partial_rank(d, "y", q)
                rows.append({"setting": name, "reps": len(reps), "K": K, "quantity": q,
                             "rho_raw": stats.spearmanr(d.y, d[q])[0], **pr})
        r = pd.DataFrame(rows)
        main_q = r.quantity.isin(["align", "shift_emo", "shift_total"])
        r.loc[main_q, "p_holm"] = holm(r.loc[main_q, "p"].to_numpy())
        partial.append(r)
    return pd.concat(partial, ignore_index=True), pd.DataFrame(effects)


def modality(pair: pd.DataFrame, kinds=KINDS) -> pd.DataFrame:
    rows = []
    if all(k in kinds for k in KINDS):                                    # これまでの比べ（同じ 6 群の 15 対）
        pairs = [(a, b) for i, a in enumerate(SIX) for b in SIX[i + 1:]]
        A = {k: pair[pair.kind == k].set_index(["target", "source"])["align"] for k in KINDS}
        v = {k: np.array([A[k][(a, b)] for a, b in pairs]) for k in KINDS}
        rows += [{"kind": k, "n_pairs": len(pairs), "mean": v[k].mean(), "median": np.median(v[k]),
                  "min": v[k].min(), "max": v[k].max(), "sd": v[k].std(ddof=1)} for k in KINDS]
        for a, b in (("video", "audio_wavlm"), ("fuse-audio-video", "audio_wavlm"), ("video", "fuse-audio-video")):
            rows.append({"kind": f"{a} - {b}", "n_pairs": len(pairs), "mean": (v[a] - v[b]).mean(),
                         "n_greater": int((v[a] > v[b]).sum()), "p_wilcoxon": stats.wilcoxon(v[a], v[b]).pvalue,
                         "rho_between": stats.spearmanr(v[a], v[b])[0]})
    extra = [k for k in kinds if k not in KINDS]
    if extra:                                                             # 言葉系など：選んだ全種類に共通の群で比べる
        common = [g for g in groups_for(extra[0]) if all(g in set(pair[pair.kind == k].target) for k in kinds)]
        pairs = [(a, b) for i, a in enumerate(common) for b in common[i + 1:]]
        A = {k: pair[pair.kind == k].set_index(["target", "source"])["align"] for k in kinds}
        v = {k: np.array([A[k][(a, b)] for a, b in pairs]) for k in kinds}
        rows += [{"kind": k, "groups": len(common), "n_pairs": len(pairs), "mean": v[k].mean(),
                  "median": np.median(v[k]), "min": v[k].min(), "max": v[k].max(), "sd": v[k].std(ddof=1)}
                 for k in kinds]
        for j, a in enumerate(kinds):                                     # 後ろの種類 − 前の種類（言葉系を含む対だけ）
            for b in kinds[:j]:
                if a in KINDS and b in KINDS:
                    continue
                rows.append({"kind": f"{a} - {b}", "groups": len(common), "n_pairs": len(pairs),
                             "mean": (v[a] - v[b]).mean(), "n_greater": int((v[a] > v[b]).sum()),
                             "p_wilcoxon": stats.wilcoxon(v[a], v[b]).pvalue,
                             "rho_between": stats.spearmanr(v[a], v[b])[0]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="E9：特徴空間の形")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS),
                    help="種類（既定は音声・映像・音声＋映像．言葉系を足すとその種類の設定と比べを足す）")
    kinds = list(dict.fromkeys(ap.parse_args().kinds))
    corps, pairs = zip(*[geometry(k) for k in kinds])
    corp, pair = pd.concat(corps, ignore_index=True), pd.concat(pairs, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    corp.to_csv(OUT / "e9_corpus.csv", index=False)
    pair.to_csv(OUT / "e9_pair.csv", index=False)
    partial, effects = analyse(corp, pair, kinds)
    partial.to_csv(OUT / "e9_partial.csv", index=False)
    effects.to_csv(OUT / "e9_effects.csv", index=False)
    mod = modality(pair, kinds)
    mod.to_csv(OUT / "e9_modality.csv", index=False)
    if all(k in kinds for k in KINDS):                                    # 図はこれまでの 3 種類だけを描く
        fig(pair, partial, effects, corp, OUT / "fig_e9.png")
    print(f"書いた: {OUT}")
    return 0


def fig(pair, partial, effects, corp, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), facecolor=SURFACE)
    col = {"audio_wavlm": C_MERGED, "video": C_TGT, "fuse-audio-video": C_SRC}
    ax = axes[0]
    _style(ax)
    for i, k in enumerate(KINDS):
        h = pair[(pair.kind == k) & pair.target.isin(SIX) & pair.source.isin(SIX) & (pair.target < pair.source)]
        ax.scatter(np.full(len(h), i) + np.linspace(-0.15, 0.15, len(h)), h["align"], s=22, color=col[k], zorder=3)
    ax.set_xticks(range(len(KINDS)), KINDS, fontsize=8)
    ax.set_title("emotion-direction alignment, same 15 pairs", fontsize=8.5, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    for k in KINDS:
        h = pair[pair.kind == k]
        ax.scatter(h["align"], h.src_uar, s=18, color=col[k], label=k, zorder=3)
    ax.axhline(0.25, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("alignment of emotion directions", fontsize=8, color=INK2)
    ax.set_ylabel("source-only UAR on target", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("source-only accuracy vs alignment (all ordered pairs)", fontsize=8.5, color=INK, loc="left")
    ax = axes[2]
    _style(ax)
    x = np.arange(len(K_ORDER))
    for s, c in zip(partial.setting.unique(), (MUTED, C_MERGED, C_TGT, C_SRC)):
        h = partial[(partial.setting == s) & (partial.quantity == "align")].set_index("K").reindex(K_ORDER)
        ax.plot(x, h.r_partial_rank, color=c, linewidth=2, marker="o", markersize=5, label=s)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, K_ORDER)
    ax.set_xlabel("K", fontsize=8, color=INK2)
    ax.set_title("pair-specific gain vs alignment (rank partial r)", fontsize=8.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[3]
    _style(ax)
    h = effects[(effects.K == "5") & (effects.effect == "source")]
    piv = h.pivot(index="quantity", columns="setting", values="rho").reindex(CORPUS_Q)
    for j, (s, c) in enumerate(zip(partial.setting.unique(), (MUTED, C_MERGED, C_TGT, C_SRC))):
        ax.barh(np.arange(len(CORPUS_Q)) + (j - 1.5) * 0.2, piv[s], height=0.2, color=c, label=s)
    ax.set_yticks(range(len(CORPUS_Q)), CORPUS_Q, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("K=5: source effect vs corpus property (Spearman, n=6-7)", fontsize=8.5, color=INK, loc="left")
    fig.suptitle("EXPLORATORY E9: feature-space geometry of good sources and pairs", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
