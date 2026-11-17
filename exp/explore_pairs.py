"""【探索的】いい組・悪い組で何が起きているか：ソースのみの分類器の d と o を感情の対ごとに直接測る（E13）。

ソースのみの分類器がターゲットの全発話に出した 4 つの点数（`results/shift_<kind>/logits/<ソース>.npz` の
`t_<ターゲット>`．同じコーパスの群どうしは人物を除いて学び直した分類器）を使う。新しい学習はしない。

感情の対 (k, l) ごとに，ターゲットの k と l の発話について z = 点数_k − 点数_l を並べる（0 が境目）。
- d = (z の k の平均 − l の平均) / s，o = −(2 つの平均の中点) / s（s は 2 つの山の標準偏差の二乗平均の平方根）
- 2 つの山のモデルの正解率 ½[Φ(d/2 − o) + Φ(d/2 + o)] と，実際の正解率 ½[P(z > 0 | k) + P(z < 0 | l)]
- 位置による損 = モデルの正解率(d, 0) − モデルの正解率(d, o)
- 向きによる損 = モデルの正解率(Δ, 0) − モデルの正解率(d, 0)．Δ はターゲット自身の最良の向き（E10 と同じ：
  話者の半分で決めた LDA の向きで，もう半分を測る）での離れ方

解析
1. 2 つの山のモデルが，実際の分類器の正解率をどれだけ再現するか
2. 組ごとに 6 対で平均した位置による損・向きによる損と，ソースのみの UAR の組に固有の部分の関係
3. いい組・悪い組（ソースのみの UAR の組に固有の部分の上位 1/3・下位 1/3）で，どの感情の対に損が集まるか
4. 位置による損が，E12 で中心をそろえて取り戻せた分（`results/mech_<kind>/`）と一致するか
5. 音声で，演技の録音（RAVDESS，CREMA-D）と自然な会話（MELD，IEMOCAP-即興，MOSEI）の間の組の損の内訳

    ./.venv/bin/python exp/explore_pairs.py

出力：`results/explore_relation/e13_*.csv`・`fig_e13.png`
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for, load_any  # noqa: E402
from explore_geometry import standardize  # noqa: E402
from explore_mechanism import partial_rank  # noqa: E402
from pair_table import OUT, SIX, fit, map_dir  # noqa: E402
from split import speaker_folds  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

KINDS = ("audio_wavlm", "video", "text_roberta", "fuse-audio-video", "fuse-audio-text",
         "fuse-video-text", "fuse-audio-video-text")
EMO = ["neutral", "happy", "sad", "angry"]
PAIRS = [(k, j) for k in range(4) for j in range(k + 1, 4)]
PAIR_NAMES = [f"{EMO[k]}-{EMO[j]}" for k, j in PAIRS]
ACTED = {"RAVDESS", "CREMA-D"}
NATURAL = {"MELD", "IEMOCAP-impro", "MOSEI"}


def ba(d, o):
    return 0.5 * (stats.norm.cdf(d / 2 - o) + stats.norm.cdf(d / 2 + o))


def own_separation(X: np.ndarray, y: np.ndarray, half: np.ndarray) -> np.ndarray:
    """ターゲット自身の最良の向き（LDA．話者の半分で決め，もう半分で測る）での 6 対の離れ方。"""
    def dirs(Xa, ya):
        m = np.stack([Xa[ya == k].mean(0) for k in range(4)])
        S = LedoitWolf(assume_centered=True).fit(Xa - m[ya]).covariance_
        return np.linalg.solve(S, np.stack([m[k] - m[j] for k, j in PAIRS]).T).T

    def sep(U, Xb, yb):
        out = []
        for u, (k, j) in zip(U, PAIRS):
            pk, pj = Xb[yb == k] @ u, Xb[yb == j] @ u
            out.append((pk.mean() - pj.mean()) / np.sqrt((pk.var(ddof=1) + pj.var(ddof=1)) / 2))
        return np.array(out)

    A, B = half == 0, half == 1
    return (sep(dirs(X[A], y[A]), X[B], y[B]) + sep(dirs(X[B], y[B]), X[A], y[A])) / 2


def measure(kind: str) -> pd.DataFrame:
    groups = groups_for(kind)
    data = standardize({g: load_any(g, kind) for g in groups})
    own = {}
    for g, (df, X) in data.items():
        half = speaker_folds(df.speaker_group, k=2, seed=0).to_numpy()
        own[g] = own_separation(X, df.label_4.to_numpy(), half)
    lab = pd.read_csv(ROOT / "results" / f"shift_{kind}" / "labeling.csv", comment="#")
    src_uar = lab.set_index(["target", "source"]).uar_t
    rows = []
    for s in groups:
        z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
        for t in groups:
            if t == s:
                continue
            L = z[f"t_{t}"]
            y = data[t][0].label_4.to_numpy()
            if len(L) != len(y):
                raise SystemExit(f"{kind} {s}→{t}：点数 {len(L)} 行と発話 {len(y)} 本が合わない")
            for (k, j), name, D in zip(PAIRS, PAIR_NAMES, own[t]):
                zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
                sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
                d = (zk.mean() - zj.mean()) / sd
                o = -(zk.mean() + zj.mean()) / 2 / sd
                rows.append({"kind": kind, "target": t, "source": s, "pair": name,
                             "d": d, "o": o, "own_d": D, "acc_actual": 0.5 * ((zk > 0).mean() + (zj < 0).mean()),
                             "acc_model": ba(d, o), "loss_offset": ba(d, 0) - ba(d, o),
                             "loss_direction": ba(D, 0) - ba(d, 0),
                             "rate_k": (zk > 0).mean(), "rate_l": (zj < 0).mean(),
                             "src_uar": float(src_uar.get((t, s), np.nan))})
    return pd.DataFrame(rows)


def mech_gain(kind: str) -> pd.DataFrame | None:
    fs = sorted((ROOT / "results" / f"mech_{kind}").glob("*.csv"))
    mp = sorted(map_dir(kind).glob("*.csv"))
    if not fs:
        return None
    m = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str}) for f in fs])
    s = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str},
                               usecols=["target", "source", "method", "rep", "uar"]) for f in mp])
    s = s[s.method == "src_only"][["target", "source", "rep", "uar"]].rename(columns={"uar": "s"})
    c = m[(m.op == "center") & (m.method == "src_only")][["target", "source", "rep", "uar"]].merge(s)
    d1 = m[(m.op == "none") & (m.method == "src_only+D1") & (m.K == "5")][["target", "source", "rep", "uar"]]
    d1 = d1.rename(columns={"uar": "d1"}).merge(s)
    g = c.assign(center_gain=c.uar - c.s).groupby(["target", "source"]).center_gain.mean().reset_index()
    g2 = d1.assign(d1_gain=d1.d1 - d1.s).groupby(["target", "source"]).d1_gain.mean().reset_index()
    return g.merge(g2, how="outer")


def main() -> int:
    P = pd.concat([measure(k) for k in KINDS if (ROOT / "results" / f"shift_{k}" / "logits").exists()],
                  ignore_index=True)
    P.to_csv(OUT / "e13_pairs.csv", index=False)
    fitq, summ, prof, mech, ax_rows = [], [], [], [], []
    for kind, h in P.groupby("kind", sort=False):
        settings = [(kind, sorted(h.target.unique()))]
        if kind == "audio_wavlm":
            settings.append(("audio_wavlm 6 groups", SIX))
        for name, gs in settings:
            x = h[h.target.isin(gs) & h.source.isin(gs)]
            # 1. モデルの再現
            fitq.append({"setting": name, "n_cells": len(x),
                         "r_model_actual": float(np.corrcoef(x.acc_model, x.acc_actual)[0, 1]),
                         "mean_abs_err": float((x.acc_model - x.acc_actual).abs().mean()),
                         "max_abs_err": float((x.acc_model - x.acc_actual).abs().max()),
                         "share_reversed": float((x.d < 0).mean())})
            # 2. 組ごとの損と，ソースのみの UAR
            pp = x.groupby(["target", "source"], as_index=False).agg(
                loss_offset=("loss_offset", "mean"), loss_direction=("loss_direction", "mean"),
                d=("d", "mean"), abs_o=("o", lambda v: np.abs(v).mean()), own_d=("own_d", "mean"),
                src_uar=("src_uar", "first"))
            row = {"setting": name, "n_pairs": len(pp),
                   "mean_loss_offset": float(pp.loss_offset.mean()),
                   "mean_loss_direction": float(pp.loss_direction.mean())}
            for q in ("loss_offset", "loss_direction"):
                pr = partial_rank(pp, "src_uar", q)
                row[f"rho_srcuar_{q}"] = float(stats.spearmanr(pp.src_uar, pp[q])[0])
                row[f"r_partial_srcuar_{q}"] = pr["r_partial_rank"] if pr else np.nan
                row[f"p_partial_srcuar_{q}"] = pr["p"] if pr else np.nan
            f = fit(pp, "src_uar")
            row["share_pair_specific_srcuar"] = f["pair"]
            summ.append(row)
            # 3. いい組・悪い組で，どの感情の対に損が集まるか
            res = f["resid"]
            lo, hi = res.quantile(1 / 3), res.quantile(2 / 3)
            grade = {ix: ("good" if v >= hi else "bad" if v <= lo else "middle") for ix, v in res.items()}
            x = x.assign(grade=[grade[(t, s)] for t, s in zip(x.target, x.source)])
            for (g, pr_name), hh in x.groupby(["grade", "pair"]):
                prof.append({"setting": name, "grade": g, "pair": pr_name, "n": len(hh),
                             "loss_offset": float(hh.loss_offset.mean()),
                             "loss_direction": float(hh.loss_direction.mean()),
                             "d": float(hh.d.mean()), "own_d": float(hh.own_d.mean()),
                             "abs_o": float(hh.o.abs().mean()), "acc_actual": float(hh.acc_actual.mean())})
            # 4. E12 で取り戻せた分
            mg = mech_gain(kind)
            if mg is not None:
                j = pp.merge(mg, on=["target", "source"])
                for col in ("center_gain", "d1_gain"):
                    jj = j.dropna(subset=[col])
                    if len(jj) >= 6:
                        pr = partial_rank(jj, col, "loss_offset")
                        mech.append({"setting": name, "gain": col, "n_pairs": len(jj),
                                     "rho_loss_offset": float(stats.spearmanr(jj[col], jj.loss_offset)[0]),
                                     "rho_loss_direction": float(stats.spearmanr(jj[col], jj.loss_direction)[0]),
                                     "r_partial_loss_offset": pr["r_partial_rank"] if pr else np.nan,
                                     "p_partial_loss_offset": pr["p"] if pr else np.nan,
                                     "mean_gain": float(jj[col].mean()),
                                     "mean_loss_offset": float(jj.loss_offset.mean())})
            # 5. 演技と自然な会話（音声 7 群）
            if name == "audio_wavlm":
                def cls(g):
                    return "acted" if g in ACTED else "natural" if g in NATURAL else "other"
                x2 = x.assign(direction=[f"{cls(s)}->{cls(t)}" for t, s in zip(x.target, x.source)])
                for (dr, pr_name), hh in x2.groupby(["direction", "pair"]):
                    ax_rows.append({"direction": dr, "pair": pr_name, "n": len(hh),
                                    "loss_offset": float(hh.loss_offset.mean()),
                                    "loss_direction": float(hh.loss_direction.mean()),
                                    "rate_k": float(hh.rate_k.mean()), "rate_l": float(hh.rate_l.mean()),
                                    "acc_actual": float(hh.acc_actual.mean())})
    outs = {"e13_fit.csv": fitq, "e13_summary.csv": summ, "e13_profile.csv": prof,
            "e13_mech.csv": mech, "e13_acted_natural.csv": ax_rows}
    for fname, rows in outs.items():
        pd.DataFrame(rows).to_csv(OUT / fname, index=False)
    fig(P, pd.DataFrame(prof), OUT / "fig_e13.png")
    print(f"書いた: {OUT}")
    return 0


def fig(P: pd.DataFrame, prof: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), facecolor=SURFACE)
    col = {"audio_wavlm": C_MERGED, "video": C_TGT, "fuse-audio-video": C_SRC}
    ax = axes[0]
    _style(ax)
    for k, c in col.items():
        h = P[P.kind == k]
        ax.scatter(h.acc_model, h.acc_actual, s=6, color=c, alpha=0.6, label=k)
    ax.plot([0.3, 1], [0.3, 1], color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("two-bell model accuracy", fontsize=8, color=INK2)
    ax.set_ylabel("actual pairwise accuracy", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("model vs actual (source-only, per emotion pair)", fontsize=8.5, color=INK, loc="left")
    for ax, setting in zip(axes[1:], ("audio_wavlm", "video")):
        _style(ax)
        h = prof[(prof.setting == setting) & prof.grade.isin(["good", "bad"])]
        x = np.arange(len(PAIR_NAMES))
        for i, (g, c) in enumerate((("good", C_SRC), ("bad", C_TGT))):
            hh = h[h.grade == g].set_index("pair").reindex(PAIR_NAMES)
            ax.bar(x + (i - 0.5) * 0.38, hh.loss_direction, width=0.36, color=c, label=f"{g}: direction")
            ax.bar(x + (i - 0.5) * 0.38, hh.loss_offset, width=0.36, bottom=hh.loss_direction,
                   color=c, alpha=0.45, label=f"{g}: offset")
        ax.set_xticks(x, PAIR_NAMES, fontsize=7, rotation=25)
        ax.set_ylabel("accuracy lost (pairwise)", fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=6)
        ax.set_title(f"{setting}: loss by emotion pair, good vs bad pairs", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
