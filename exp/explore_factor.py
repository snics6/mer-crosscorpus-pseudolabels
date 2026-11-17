"""【探索的】得の足し算の分け方（得 = μ + ターゲットの効果 + ソースの効果 + 残り）は妥当か（E8）。

1. **掛け算の形の決まり**：残り e(t, s) の表（対角は無い）を，1 本の軸 u_t × v_s（と 2 本目）で
   近似し，残りの二乗和の何割を説明するかを出す（観測した升だけで交互に最小二乗）。
   でたらめでも 1 本の軸はある程度説明するので，残りの値を升の間でかき混ぜ，足し算を当てはめ直して
   同じ割合を出す（500 回）。Tukey の検定（残りが a_t × b_s に比例するか）も行う
2. **尺度**：得を「伸びしろに対する割合」(m − t)/(1 − t) と，ロジットの差 logit(m) − logit(t) に直して
   内訳を出し直す（m は合併，t はターゲットのみの UAR．反復ごとに直してから平均する）。
   ターゲットの効果 a_t と，ターゲットのみの UAR の平均との相関も見る
3. **向き**：組の対 {A, B} について，残り e(A, B) と e(B, A) の相関をとる

    ./.venv/bin/python exp/explore_factor.py

出力：`results/explore_relation/e8_lowrank.csv`・`e8_axis.csv`・`e8_scale.csv`・`e8_symmetry.csv`・`fig_e8.png`
"""

from __future__ import annotations

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
from pair_table import OUT, SIX, fit, map_dir  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

K_ORDER = ["1", "5", "10", "25", "50", "all"]
N_PERM = 500
EPS = 1e-3


def per_rep(kind: str) -> pd.DataFrame:
    """反復ごとの合併とターゲットのみの UAR。列：target, source, K, rep, m, t。"""
    rows = []
    for p in sorted(map_dir(kind).glob("*.csv")):
        d = pd.read_csv(p, comment="#", dtype={"K": str, "source": str},
                        usecols=["target", "source", "method", "K", "rep", "uar"])
        t = d[d.method == "tgt_only"].set_index(["K", "rep"])["uar"]
        m = d[d.method == "merged"].copy()
        m["t"] = t.loc[list(zip(m.K, m.rep))].to_numpy()
        rows.append(m.rename(columns={"uar": "m"})[["target", "source", "K", "rep", "m", "t"]])
    return pd.concat(rows, ignore_index=True)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def means(R: pd.DataFrame, groups, reps, K) -> pd.DataFrame:
    h = R[(R.K == K) & R.target.isin(groups) & R.source.isin(groups) & R.rep.isin(reps)].copy()
    h["gain"] = h.m - h.t
    h["rel"] = (h.m - h.t) / (1 - h.t)
    h["logit"] = logit(h.m) - logit(h.t)
    return h.groupby(["target", "source"], as_index=False)[["gain", "rel", "logit", "t"]].mean()


def rank1(E: pd.Series, groups: list[str], iters: int = 200) -> tuple[float, pd.Series, pd.Series]:
    """観測した升（対角以外）だけで e ≈ u_t v_s を当てはめる。説明した二乗和の割合と u, v を返す。"""
    M = E.unstack().reindex(index=groups, columns=groups).to_numpy(float)
    obs = np.isfinite(M)
    X = np.where(obs, M, 0.0)
    v = np.linalg.svd(X)[2][0]
    u = np.zeros(len(groups))
    for _ in range(iters):
        u = np.array([(X[i, obs[i]] * v[obs[i]]).sum() / max((v[obs[i]] ** 2).sum(), 1e-12)
                      for i in range(len(groups))])
        v = np.array([(X[obs[:, j], j] * u[obs[:, j]]).sum() / max((u[obs[:, j]] ** 2).sum(), 1e-12)
                      for j in range(len(groups))])
    fitted = np.outer(u, v)
    ss = float((X[obs] ** 2).sum())
    share = 1 - float(((X - fitted)[obs] ** 2).sum()) / ss if ss > 0 else np.nan
    scale = np.linalg.norm(v) or 1.0                       # 位置の向きと大きさをそろえる
    return share, pd.Series(u * scale, index=groups), pd.Series(v / scale, index=groups)


def lowrank(d: pd.DataFrame, groups: list[str], col: str, rng) -> dict:
    f = fit(d, col)
    e = f["resid"]
    s1, u, v = rank1(e, groups)
    e2 = e - pd.Series(np.outer(u, v)[[groups.index(t) for t, _ in e.index],
                                       [groups.index(s) for _, s in e.index]], index=e.index)
    s2, _, _ = rank1(e2, groups)
    two = s1 + (1 - s1) * s2                               # 2 本の軸で説明した割合
    null = []
    for _ in range(N_PERM):
        dp = d.copy()
        dp[col] = rng.permutation(dp[col].to_numpy())
        null.append(rank1(fit(dp, col)["resid"], groups)[0])
    # Tukey：残りを a_t × b_s で回帰
    ab = np.array([f["a"].get(t, 0.0) * f["b"].get(s, 0.0) for t, s in e.index])
    X = np.column_stack([np.ones(len(ab)), ab])
    beta, *_ = np.linalg.lstsq(X, e.to_numpy(), rcond=None)
    res = e.to_numpy() - X @ beta
    df = len(ab) - (len(set(d.target)) + len(set(d.source)) - 1) - 1
    se = np.sqrt((res @ res) / df * np.linalg.inv(X.T @ X)[1, 1]) if df > 0 else np.nan
    tukey_share = 1 - float(res @ res) / float(e @ e)
    return {"pair_share": f["pair"], "rank1_share": s1, "rank2_share": two,
            "perm_mean": float(np.mean(null)), "perm_p": float(np.mean(np.array(null) >= s1)),
            "tukey_share": tukey_share,
            "tukey_p": float(2 * stats.t.sf(abs(beta[1] / se), df)) if df > 0 else np.nan,
            "u": u, "v": v, "resid": e}


def main() -> int:
    rng = np.random.default_rng(0)
    R = {k: per_rep(k) for k in ("audio_wavlm", "video", "fuse-audio-video")}
    # 全ターゲットでそろった反復をすべて使う（2026-10-09 まで音声は 0〜9，映像と音声＋映像は 0〜4 に固定していた）
    reps = {k: sorted(set.intersection(*[set(g.rep) for _, g in r.groupby("target")])) for k, r in R.items()}
    tag = {k: f"reps {v[0]}-{v[-1]}" for k, v in reps.items()}
    seven = sorted(R["audio_wavlm"].target.unique())
    settings = [(f"audio 7 groups ({tag['audio_wavlm']})", "audio_wavlm", seven, reps["audio_wavlm"]),
                (f"audio 6 groups ({tag['audio_wavlm']})", "audio_wavlm", SIX, reps["audio_wavlm"]),
                (f"video ({tag['video']})", "video", SIX, reps["video"]),
                (f"audio+video ({tag['fuse-audio-video']})", "fuse-audio-video", SIX, reps["fuse-audio-video"])]
    low, axis, scale, sym = [], [], [], []
    for name, kind, groups, reps in settings:
        groups = sorted(groups)
        for K in K_ORDER:
            d = means(R[kind], groups, reps, K)
            r = lowrank(d, groups, "gain", rng)
            low.append({"setting": name, "K": K, **{k: v for k, v in r.items()
                                                    if k not in ("u", "v", "resid")}})
            if K in ("1", "5", "all"):
                for g in groups:
                    axis.append({"setting": name, "K": K, "group": g,
                                 "u_target": r["u"][g], "v_source": r["v"][g]})
            for col in ("gain", "rel", "logit"):
                f = fit(d, col)
                tm = d.groupby("target").t.mean()
                scale.append({"setting": name, "K": K, "scale": col, "target": f["target"],
                              "source": f["source"], "pair": f["pair"],
                              "rho_a_vs_tgtonly": stats.spearmanr(f["a"], tm.reindex(f["a"].index))[0]})
            e = r["resid"]
            pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
            x = np.array([e[(a, b)] for a, b in pairs])
            y = np.array([e[(b, a)] for a, b in pairs])
            gx = d.set_index(["target", "source"]).gain
            sym.append({"setting": name, "K": K, "n_pairs": len(pairs),
                        "rho_resid": stats.spearmanr(x, y)[0],
                        "rho_raw_gain": stats.spearmanr([gx[(a, b)] for a, b in pairs],
                                                        [gx[(b, a)] for a, b in pairs])[0]})
    low, axis, scale, sym = map(pd.DataFrame, (low, axis, scale, sym))
    low.to_csv(OUT / "e8_lowrank.csv", index=False)
    axis.to_csv(OUT / "e8_axis.csv", index=False)
    scale.to_csv(OUT / "e8_scale.csv", index=False)
    sym.to_csv(OUT / "e8_symmetry.csv", index=False)
    fig(low, axis, OUT / "fig_e8.png")
    print(f"書いた: {OUT}")
    return 0


def fig(low: pd.DataFrame, axis: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)
    x = np.arange(len(K_ORDER))
    ax = axes[0]
    _style(ax)
    for s, c in zip(low.setting.unique(), (MUTED, C_MERGED, C_TGT, C_SRC)):
        h = low[low.setting == s].set_index("K").reindex(K_ORDER)
        ax.plot(x, h.rank1_share, color=c, linewidth=2, marker="o", markersize=5, label=s)
        ax.plot(x, h.perm_mean, color=c, linewidth=1, linestyle=":")
    ax.set_xticks(x, K_ORDER)
    ax.set_ylim(0, 1)
    ax.set_title("share of residual SS explained by one axis (dotted: shuffled)", fontsize=8.5,
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    for ax, s in zip(axes[1:], ("audio 7 groups (reps 0-9)", "video (reps 0-4)")):
        _style(ax)
        h = axis[(axis.setting == s) & (axis.K == "5")]
        ax.scatter(h.u_target, h.v_source, s=40, color=C_MERGED, zorder=3)
        for r in h.itertuples():
            ax.annotate(r.group, (r.u_target, r.v_source), fontsize=7.5, color=INK,
                        xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel("position as target (u)", fontsize=8, color=INK2)
        ax.set_ylabel("position as source (v)", fontsize=8, color=INK2)
        ax.set_title(f"{s}, K=5: first axis of pair-specific gain", fontsize=8.5, color=INK, loc="left")
    fig.suptitle("EXPLORATORY E8: is the additive split adequate?", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
