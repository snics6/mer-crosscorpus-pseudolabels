"""【探索的】得の仕組みの仮説 MH1〜MH5 の予言を，今ある地図と特徴空間の量で確かめる（E11）。

仮説と式は `docs/exploratory-analyses.md` E11 節。共通のモデルは，2 つの感情を直線で分ける場面で，
ターゲットでの成績が「向きの良さ d」と「位置のずれ o」で決まるというもの。4 感情の UAR は，
4 つの感情の重心が正四面体の頂点にあり（どの 2 つも距離 d），散らばりが等方の正規分布のときの
正解率 `uar4(d)` で d に換算する（実効の離れ方 d_eff = uar4⁻¹(UAR)）。

予言と確かめ方（いずれも学び直しはしない）
- **MH1（雑音の削減）**：K=1 の得 ≈ ソースのみ − ターゲットのみ（K=1）。組ごとに回帰し，傾きと決定係数を見る。
  反復をまたいだ UAR の標準偏差を，ターゲットのみと合併で比べる
- **MH2（向きのずれ）**：ターゲットのみの学習曲線 d_eff(K) = Δ_t/√(1 + q_t/K) をターゲットごとに当てはめ，
  向きの効率 e（`e10_pair.csv` の `eff_lda`，または地図のソースのみから e_emp = d_eff(ソースのみ)/Δ_t）から
  交わる K を K_x = q_t e²/(1 − e²) と予言する。観測した交わる K（反復平均の得の符号が変わる K を
  log K で線形補間）と順位相関をとる
- **MH3（中心のずれ）**：ソースのみの分類器の答えの偏り（`labeling.csv` の混同行列の列ごとの平均と
  0.25 の差の総変動距離 TV）と `shift_emo` の組に固有の関係。偏る感情が，ずれ δ = c_t − c_s を
  ソースの感情の矢印に射影した β_k の最大の感情と一致するか（偶然なら 1/4）
- **MH3c（少数の例が位置のずれを直す）**：合併 − ソースのみ（K 本を足して良くなった幅）が，ソースのみの
  答えの偏り TV と，ターゲットがソースからどれだけ離れているか（PAD，フレシェ距離）と組に固有に関係するか
- **MH4（ターゲットが埋もれる）**：ターゲットとソースのみの間で合併がどこにいるか
  φ = (合併 − ソースのみ)/(ターゲットのみ − ソースのみ) が，同じターゲット・同じ K の中で，
  ターゲットの割合 α = n_tgt/(n_src + n_tgt) の大きいソースほど大きいか。K ≥ 25 のソースの効果と log n_src
- **MH5（表し方の軸）**：E8 の 1 本の軸の位置と，コーパスの感情の分かれ方（`dprime`，`own_lda`）の順位相関。
  組の得の組に固有の部分と −|x_t − x_s| の偏相関
- **まとめのモデル**：d_merged(K) = Δ_t (e + (1 − e) α_K^γ) として，γ だけを種類ごとに当てはめ，
  全組 × 全 K の得をどれだけ再現するかを見る

    ./.venv/bin/python exp/explore_mechanism.py

出力：`results/explore_relation/e11_*.csv`・`fig_e11.png`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import optimize, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import groups_for, load_any  # noqa: E402
from explore_factor import rank1  # noqa: E402
from explore_geometry import corpus, standardize  # noqa: E402
from pair_table import OUT, SIX, fit, holm, map_dir, partial_rank as _partial_rank  # noqa: E402
from summarize_map import C_MERGED, C_SRC, C_TGT, INK, INK2, MUTED, SURFACE, _style  # noqa: E402

KINDS = ("audio_wavlm", "video", "text_roberta", "fuse-audio-video", "fuse-audio-text",
         "fuse-video-text", "fuse-audio-video-text")
GEOM_KINDS = ("audio_wavlm", "video", "fuse-audio-video")     # E9・E10 の量がある種類
K_ORDER = ["1", "5", "10", "25", "50", "all"]
MIN_GAP = 0.03            # φ を求める最小の |ターゲットのみ − ソースのみ|


def partial_rank(d: pd.DataFrame, resp: str, met: str) -> dict | None:
    """`explore_decomp.partial_rank`。値が 1 種類しかない・自由度が無いなどで解けなければ None。"""
    x = d[["target", "source", resp, met]].dropna()
    if x[resp].nunique() < 2 or x[met].nunique() < 2:
        return None
    try:
        r = _partial_rank(x, resp, met)
    except np.linalg.LinAlgError:
        return None
    return r if r is not None and r["df"] > 0 and np.isfinite(r["p"]) else None


def one_sided(pr: dict | None) -> tuple[float, float]:
    if pr is None:
        return np.nan, np.nan
    r, p = pr["r_partial_rank"], pr["p"]
    return r, (p / 2 if r > 0 else 1 - p / 2)


# ---------------------------------------------------------------- UAR と d の換算

def uar4_table(n: int = 400_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """正四面体の頂点に重心を置き（どの 2 つも距離 d），等方の正規分布で最も近い重心を答えたときの正解率。"""
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / (2 * np.sqrt(2))
    Z = np.random.default_rng(seed).standard_normal((n, 3))
    grid = np.linspace(0, 8, 401)
    acc = [0.25]                                     # d = 0 は全部同点なので解析的に 1/4
    for d in grid[1:]:
        P = Z + d * V[0]
        dist = ((P[:, None, :] - d * V[None]) ** 2).sum(2)
        acc.append(float((dist.argmin(1) == 0).mean()))
    return grid, np.maximum.accumulate(np.array(acc))


GRID_D, GRID_U = uar4_table()


def uar4(d):
    return np.interp(d, GRID_D, GRID_U)


def d_eff(u):
    return np.interp(u, GRID_U, GRID_D)


# ---------------------------------------------------------------- 地図の読み込み

def load_map(kind: str) -> tuple[pd.DataFrame, list[int]]:
    """組 × K ごとの反復平均（全ターゲットでそろった反復だけ）。"""
    rows = []
    for p in sorted(map_dir(kind).glob("*.csv")):
        rows.append(pd.read_csv(p, comment="#", dtype={"K": str, "source": str},
                                usecols=["target", "source", "method", "K", "rep", "uar",
                                         "n_src", "n_tgt"]))
    d = pd.concat(rows, ignore_index=True)
    cnt = d.groupby(["target", "rep"]).size()
    full = cnt[cnt == cnt.groupby(level=0).transform("max")].reset_index()
    reps = sorted(set.intersection(*[set(full[full.target == t].rep) for t in full.target.unique()]))
    d = d[d.rep.isin(reps)]
    t = d[d.method == "tgt_only"].rename(columns={"uar": "t"})
    s = d[d.method == "src_only"].rename(columns={"uar": "s"})
    m = d[d.method == "merged"].rename(columns={"uar": "m"})
    x = m.merge(t[["target", "K", "rep", "t"]], on=["target", "K", "rep"]).merge(
        s[["target", "source", "rep", "s"]], on=["target", "source", "rep"])
    x["gain"] = x.m - x.t
    g = x.groupby(["target", "source", "K"]).agg(
        m=("m", "mean"), t=("t", "mean"), s=("s", "mean"), gain=("gain", "mean"),
        sd_m=("m", "std"), sd_t=("t", "std"), sd_gain=("gain", "std"), n=("gain", "size"),
        n_src=("n_src", "mean"), n_tgt=("n_tgt", "mean")).reset_index()
    g["alpha"] = g.n_tgt / (g.n_src + g.n_tgt)
    g["kind"] = kind
    return g, reps


def kstar(g: pd.DataFrame) -> pd.Series:
    """差の 95% 区間の下端が 0 を上回る最大の K（`summarize_map` と同じ定義．なしは 0 番目）。"""
    half = stats.t.ppf(0.975, g.n - 1) * g.sd_gain / np.sqrt(g.n)
    ok = (g.gain - half) > 0
    out = {}
    for (t, s), h in g.assign(ok=ok).groupby(["target", "source"]):
        ks = [K_ORDER.index(K) + 1 for K, o in zip(h.K, h.ok) if o]
        out[(t, s)] = max(ks) if ks else 0
    return pd.Series(out)


# ---------------------------------------------------------------- MH2：学習曲線

def curve(K, delta, q):
    return delta / np.sqrt(1 + q / K)


def fit_curves(g: pd.DataFrame) -> pd.DataFrame:
    """ターゲットごとに d_eff(K) = Δ/√(1+q/K) を当てはめる（K=all はクラスあたりの本数）。"""
    out = []
    for t, h in g.groupby("target"):
        c = h.groupby("K").agg(t=("t", "first"), n_tgt=("n_tgt", "first")).reindex(K_ORDER)
        Ks = np.array([int(k) if k != "all" else c.loc["all", "n_tgt"] / 4 for k in K_ORDER], float)
        d = d_eff(c.t.to_numpy())
        (delta, q), _ = optimize.curve_fit(curve, Ks, d, p0=[max(d[-1], 0.5), 10.0],
                                           bounds=([1e-6, 1e-6], [50, 1e5]))
        pred = uar4(curve(Ks, delta, q))
        ss = ((c.t - c.t.mean()) ** 2).sum()
        out.append({"target": t, "delta": delta, "q": q, "K_all": Ks[-1],
                    "r2_uar": 1 - float(((c.t.to_numpy() - pred) ** 2).sum()) / ss,
                    "max_abs_err": float(np.abs(c.t.to_numpy() - pred).max())})
    return pd.DataFrame(out).set_index("target")


def crossing(h: pd.DataFrame, K_all: float) -> float:
    """反復平均の得が正から負に変わる K（log K で線形補間）。ずっと負なら 0.5，ずっと正なら 2·K_all。"""
    h = h.set_index("K").reindex(K_ORDER)
    Ks = np.array([1, 5, 10, 25, 50, K_all], float)
    y = h.gain.to_numpy()
    if y[0] <= 0:
        return 0.5
    for i in range(1, len(y)):
        if y[i] <= 0:
            a, b = np.log(Ks[i - 1]), np.log(Ks[i])
            return float(np.exp(a + (b - a) * y[i - 1] / (y[i - 1] - y[i])))
    return 2 * K_all


def kx_pred(q: float, e: float, K_all: float) -> float:
    if not np.isfinite(e):
        return np.nan
    if e >= 1:
        return 2 * K_all
    return q * e ** 2 / (1 - e ** 2)


# ---------------------------------------------------------------- MH3：ずれの向き

def beta_table(kind: str) -> pd.DataFrame:
    """β_k = ⟨δ, d_k^s⟩/‖d_k^s‖²（E9 と同じ空間）。"""
    groups = groups_for(kind)
    data = standardize({g: load_any(g, kind) for g in groups})
    C = {g: corpus(df, X) for g, (df, X) in data.items()}
    rows = []
    for t in groups:
        for s in groups:
            if t != s:
                delta = C[t]["c"] - C[s]["c"]
                Ds = C[s]["D"]
                beta = (Ds @ delta) / (Ds ** 2).sum(axis=1)
                rows.append({"target": t, "source": s, **{f"beta_{k}": float(b) for k, b in enumerate(beta)}})
    return pd.DataFrame(rows)


def bias_table(kind: str) -> pd.DataFrame:
    lab = pd.read_csv(ROOT / "results" / f"shift_{kind}" / "labeling.csv", comment="#")
    rows = []
    for r in lab.itertuples():
        C = np.array([[getattr(r, f"conf_t_{a}{b}") for b in range(4)] for a in range(4)])
        p = C.mean(0)                                        # 感情ごとの正解率と同じ重みでみた答えの割合
        rows.append({"target": r.target, "source": r.source, "tv": 0.5 * float(np.abs(p - 0.25).sum()),
                     "top_pred": int(p.argmax()), **{f"pred_{k}": float(p[k]) for k in range(4)},
                     "src_uar": r.uar_t})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 本体

def setting_list(maps: dict) -> list[tuple[str, str, list[str]]]:
    out = []
    for k in KINDS:
        groups = sorted(maps[k].target.unique())
        out.append((k, k, groups))
        if k == "audio_wavlm":
            out.append(("audio_wavlm 6 groups", k, SIX))
    return out


def sub(g: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return g[g.target.isin(groups) & g.source.isin(groups)]


def main() -> int:
    maps, reps = {}, {}
    for k in KINDS:
        maps[k], reps[k] = load_map(k)
    settings = setting_list(maps)
    e9 = pd.read_csv(OUT / "e9_pair.csv")
    e9c = pd.read_csv(OUT / "e9_corpus.csv")
    e10 = pd.read_csv(OUT / "e10_pair.csv")
    e10c = pd.read_csv(OUT / "e10_corpus.csv")

    per = pd.read_csv(OUT / "per_pair.csv")
    h1, h2, h2pair, h3, h3c, h4, h4src, h5, model, curves_out, primary = ([] for _ in range(11))
    for name, kind, groups in settings:
        g = sub(maps[kind], groups)
        geom = kind in GEOM_KINDS

        # ---- MH1
        for K in ("1", "5"):
            h = g[g.K == K]
            x, y = (h.s - h.t).to_numpy(), h.gain.to_numpy()
            lr = stats.linregress(x, y)
            h1.append({"setting": name, "K": K, "n_pairs": len(h), "slope": lr.slope,
                       "slope_lo": lr.slope - 1.96 * lr.stderr, "slope_hi": lr.slope + 1.96 * lr.stderr,
                       "intercept": lr.intercept, "r2": lr.rvalue ** 2,
                       "mean_m_minus_s": float((h.m - h.s).mean()),
                       "mean_abs_m_minus_s": float((h.m - h.s).abs().mean())})
        for K in K_ORDER:
            h = g[g.K == K]
            h1.append({"setting": name, "K": K, "n_pairs": len(h), "sd_t": float(h.sd_t.mean()),
                       "sd_m": float(h.sd_m.mean()), "sd_ratio": float((h.sd_t / h.sd_m).median()),
                       "mean_alpha": float(h.alpha.mean())})

        # ---- MH2
        cv = fit_curves(g)
        curves_out.append(cv.assign(setting=name).reset_index())
        P = g[g.K == "1"][["target", "source", "s"]].copy()
        P["delta"] = P.target.map(cv.delta)
        P["q"] = P.target.map(cv.q)
        P["e_emp"] = d_eff(P.s.to_numpy()) / P.delta
        if geom:
            e = e10[e10.kind == kind].set_index(["target", "source"]).eff_lda
            P["e_geom"] = [e.get((t, s), np.nan) for t, s in zip(P.target, P.source)]
            own = e10c[e10c.kind == kind].set_index("group").own_lda
            P["own_lda"] = P.target.map(own)
        P["kx_obs"] = [crossing(g[(g.target == t) & (g.source == s)], cv.K_all[t])
                       for t, s in zip(P.target, P.source)]
        ks = kstar(g)
        P["kstar_idx"] = [ks[(t, s)] for t, s in zip(P.target, P.source)]
        for ecol in ["e_emp"] + (["e_geom"] if geom else []):
            P[f"kx_{ecol}"] = [kx_pred(q, e, cv.K_all[t]) for q, e, t in zip(P.q, P[ecol], P.target)]
            P[f"log_e2_{ecol}"] = np.log(np.clip(P[ecol], 1e-3, 0.999) ** 2 /
                                         (1 - np.clip(P[ecol], 1e-3, 0.999) ** 2))
            for obs in ("kx_obs", "kstar_idx"):
                rho, p = stats.spearmanr(P[f"kx_{ecol}"], P[obs], nan_policy="omit")
                pr = partial_rank(P.dropna(subset=[f"kx_{ecol}"]), obs, f"log_e2_{ecol}")
                h2.append({"setting": name, "e": ecol, "observed": obs, "n_pairs": int(P[f"kx_{ecol}"].notna().sum()),
                           "rho_pred_obs": rho, "p": p, "r_partial_e": pr["r_partial_rank"] if pr else np.nan,
                           "p_partial_e": pr["p"] if pr else np.nan})
        # q_t はターゲットの効果を説明するか
        tf = fit(P.assign(v=P.kx_obs.apply(np.log)), "v")
        h2.append({"setting": name, "e": "target effect of log K_x vs log q_t", "observed": "kx_obs",
                   "n_pairs": len(cv), "rho_pred_obs": stats.spearmanr(tf["a"].reindex(cv.index), np.log(cv.q))[0],
                   "p": np.nan})
        if geom:
            for a, b in (("e_geom", "e_emp"), ("own_lda", "delta")):
                rho, p = stats.spearmanr(P[a], P[b], nan_policy="omit")
                h2.append({"setting": name, "e": f"{a} vs {b}", "observed": "", "n_pairs": len(P),
                           "rho_pred_obs": rho, "p": p})
            r, p1 = one_sided(partial_rank(P, "kx_obs", "log_e2_e_geom"))
            primary.append({"setting": name, "test": "MH2 K_x ~ e_geom (partial)", "stat": r, "p": p1})
        h2pair.append(P.assign(setting=name))

        # ---- MH3（幾何のある種類）
        if geom:
            B = bias_table(kind).merge(beta_table(kind), on=["target", "source"])
            B = B.merge(e9[e9.kind == kind][["target", "source", "shift_emo", "shift_total"]],
                        on=["target", "source"])
            B = sub(B, groups)
            beta = B[[f"beta_{k}" for k in range(4)]].to_numpy()
            pred = B[[f"pred_{k}" for k in range(4)]].to_numpy()
            match = (beta.argmax(1) == pred.argmax(1))
            within = [r for r in (stats.spearmanr(b, p)[0] for b, p in zip(beta, pred)) if np.isfinite(r)]
            bt = stats.binomtest(int(match.sum()), len(match), 0.25, alternative="greater")
            wt = stats.wilcoxon(within, alternative="greater")
            row = {"setting": name, "n_pairs": len(B), "match_rate": float(match.mean()),
                   "match_p": bt.pvalue, "within_rho_mean": float(np.mean(within)), "within_p": wt.pvalue}
            for q in ("shift_emo", "shift_total"):
                pr = partial_rank(B, "tv", q)
                row[f"r_tv_{q}"], row[f"p_tv_{q}"] = (pr["r_partial_rank"], pr["p"]) if pr else (np.nan, np.nan)
            pr = partial_rank(B, "src_uar", "tv")
            row["r_srcuar_tv"], row["p_srcuar_tv"] = (pr["r_partial_rank"], pr["p"]) if pr else (np.nan, np.nan)
            row["tv_mean"], row["tv_max"] = float(B.tv.mean()), float(B.tv.max())
            h3.append(row)
            primary.append({"setting": name, "test": "MH3 top predicted = top beta", "stat": row["match_rate"], "p": bt.pvalue})
            r, p1 = one_sided(partial_rank(B, "tv", "shift_emo"))
            primary.append({"setting": name, "test": "MH3 TV ~ shift_emo (partial)", "stat": r, "p": p1})

            # ---- MH3c：K 本を足して良くなった幅
            D = per[per.kind == kind][["target", "source", "pad", "frechet"]]
            for K in ("1", "5", "10", "25"):
                h = g[g.K == K].assign(m_s=lambda x: x.m - x.s).merge(B, on=["target", "source"]).merge(D)
                for q in ("tv", "pad", "frechet", "shift_emo", "shift_total"):
                    pr = partial_rank(h, "m_s", q)
                    h3c.append({"setting": name, "K": K, "quantity": q, "n_pairs": len(h),
                                "rho": stats.spearmanr(h.m_s, h[q])[0],
                                "r_partial_rank": pr["r_partial_rank"] if pr else np.nan,
                                "p": pr["p"] if pr else np.nan,
                                "mean_m_minus_s": float(h.m_s.mean())})
            r, p1 = one_sided(partial_rank(
                g[g.K == "5"].assign(m_s=lambda x: x.m - x.s).merge(B, on=["target", "source"]), "m_s", "tv"))
            primary.append({"setting": name, "test": "MH3c merged - src at K=5 ~ TV (partial)", "stat": r, "p": p1})

        # ---- MH4
        signs = []
        for K in K_ORDER[1:]:
            rhos = []
            for t, h in g[g.K == K].groupby("target"):
                h = h[(h.t - h.s).abs() >= MIN_GAP]
                if len(h) >= 4 and h.alpha.nunique() >= 4:
                    rhos.append(stats.spearmanr(h.alpha, (h.m - h.s) / (h.t - h.s))[0])
            rhos = np.array(rhos)
            if len(rhos):
                signs += list(rhos)
                h4.append({"setting": name, "K": K, "n_targets": len(rhos), "rho_mean": float(rhos.mean()),
                           "n_positive": int((rhos > 0).sum())})
        if len(signs) >= 6:
            w = stats.wilcoxon(signs, alternative="greater")
            primary.append({"setting": name, "test": "MH4 phi ~ alpha within target", "stat": float(np.mean(signs)), "p": w.pvalue})
        n_src = g.groupby("source").n_src.mean()
        for K in K_ORDER:
            f = fit(g[g.K == K], "gain")
            b = f["b"]
            b = b - b.mean()
            rho, p = stats.spearmanr(b, np.log(n_src.reindex(b.index)))
            h4src.append({"setting": name, "K": K, "n_sources": len(b), "rho_b_logn": rho, "p": p})

        # ---- MH5（幾何のある種類）
        if geom:
            cp = e9c[e9c.kind == kind].set_index("group")
            own = e10c[e10c.kind == kind].set_index("group").own_lda
            for K in ("1", "5", "10", "25"):
                f = fit(g[g.K == K], "gain")
                s1, u, v = rank1(f["resid"], groups)
                for xname, x in (("dprime", cp.dprime), ("own_lda", own), ("arrow_power", cp.arrow_power),
                                 ("eff_dim", cp.eff_dim)):
                    x = x.reindex(groups)
                    pos = u / np.linalg.norm(u) + v / np.linalg.norm(v)  # 位置（軸の符号は任意）
                    D = g[g.K == K][["target", "source", "gain"]].copy()
                    D["neg_absdiff"] = [-abs(x[t] - x[s]) for t, s in zip(D.target, D.source)]
                    pr = partial_rank(D, "gain", "neg_absdiff")
                    E = e10[e10.kind == kind][["target", "source", "eff_lda"]]
                    E = sub(E, groups).copy()
                    E["neg_absdiff"] = [-abs(x[t] - x[s]) for t, s in zip(E.target, E.source)]
                    pe = partial_rank(E, "eff_lda", "neg_absdiff")
                    h5.append({"setting": name, "K": K, "corpus_quantity": xname, "rank1_share": s1,
                               "rho_axis": stats.spearmanr(pos, x)[0], "rho_u": stats.spearmanr(u, x)[0],
                               "rho_v": stats.spearmanr(v, x)[0],
                               "r_gain_negabsdiff": pr["r_partial_rank"] if pr else np.nan,
                               "p_gain_negabsdiff": pr["p"] if pr else np.nan,
                               "r_efflda_negabsdiff": pe["r_partial_rank"] if pe else np.nan,
                               "p_efflda_negabsdiff": pe["p"] if pe else np.nan})
                    if K == "5" and xname == "dprime":
                        r, p1 = one_sided(pr)
                        primary.append({"setting": name, "test": "MH5 gain K=5 ~ -|d' diff| (partial)",
                                        "stat": r, "p": p1})

        # ---- まとめのモデル
        for ecol in ["e_emp"] + (["e_geom"] if geom else []):
            E = P.set_index(["target", "source"])[ecol]
            gg = g.copy()
            gg["e"] = [E.get((t, s), np.nan) for t, s in zip(gg.target, gg.source)]
            gg["delta"] = gg.target.map(cv.delta)
            gg["Kn"] = [int(k) if k != "all" else cv.K_all[t] for k, t in zip(gg.K, gg.target)]
            gg["t_hat"] = uar4(curve(gg.Kn, gg.delta, gg.target.map(cv.q)))
            gg = gg.dropna(subset=["e"])

            def pred(gam):
                return uar4(gg.delta * (gg.e + (1 - gg.e) * gg.alpha ** gam)) - gg.t_hat

            gams = np.exp(np.linspace(np.log(0.05), np.log(5), 200))
            sse = [float(((gg.gain - pred(c)) ** 2).sum()) for c in gams]
            gam = gams[int(np.argmin(sse))]
            gg["gain_hat"] = pred(gam)
            sst = float(((gg.gain - gg.gain.mean()) ** 2).sum())
            row = {"setting": name, "e": ecol, "gamma": gam, "n_cells": len(gg),
                   "r2_all": 1 - min(sse) / sst}
            for K in K_ORDER:
                h = gg[gg.K == K]
                if h.gain_hat.std() < 1e-9:
                    print(f"注意: {name} {ecol} K={K} の予言が全組で同じ（{h.gain_hat.iloc[0]:.3f}）", flush=True)
                    row[f"rho_K{K}"] = row[f"rho_pair_K{K}"] = np.nan
                    continue
                ro = fit(h, "gain")["resid"]
                rp = fit(h.assign(gain=h.gain_hat), "gain")["resid"]
                row[f"rho_K{K}"] = stats.spearmanr(h.gain, h.gain_hat)[0]
                row[f"rho_pair_K{K}"] = stats.spearmanr(ro, rp.reindex(ro.index))[0]
            model.append(row)

    prim = pd.DataFrame(primary)
    prim["p_holm"] = np.nan
    for s, idx in prim.groupby("setting").groups.items():
        prim.loc[idx, "p_holm"] = holm(prim.loc[idx, "p"].to_numpy())

    outs = {"e11_h1.csv": pd.DataFrame(h1), "e11_h2.csv": pd.DataFrame(h2),
            "e11_h2_pair.csv": pd.concat(h2pair, ignore_index=True),
            "e11_curves.csv": pd.concat(curves_out, ignore_index=True),
            "e11_h3.csv": pd.DataFrame(h3), "e11_h3c.csv": pd.DataFrame(h3c), "e11_h4.csv": pd.DataFrame(h4),
            "e11_h4_source.csv": pd.DataFrame(h4src), "e11_h5.csv": pd.DataFrame(h5),
            "e11_model.csv": pd.DataFrame(model), "e11_primary.csv": prim}
    for f, d in outs.items():
        d.to_csv(OUT / f, index=False)
    (OUT / "e11_state.json").write_text(json.dumps({k: [int(r) for r in v] for k, v in reps.items()}))
    fig(maps, outs, OUT / "fig_e11.png")
    print(f"書いた: {OUT}（反復 {', '.join(f'{k} {len(v)}' for k, v in reps.items())}）")
    return 0


def fig(maps, outs, path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.3), facecolor=SURFACE)
    col = {"audio_wavlm": C_MERGED, "video": C_TGT, "fuse-audio-video": C_SRC}
    ax = axes[0]
    _style(ax)
    for k, c in col.items():
        h = maps[k][maps[k].K == "1"]
        ax.scatter(h.s - h.t, h.gain, s=16, color=c, label=k, zorder=3)
    lim = [-0.15, 0.35]
    ax.plot(lim, lim, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("source-only − target-only (K=1)", fontsize=8, color=INK2)
    ax.set_ylabel("gain at K=1", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("MH1: gain at K=1 vs source-only advantage", fontsize=8.5, color=INK, loc="left")
    ax = axes[1]
    _style(ax)
    P = outs["e11_h2_pair.csv"]
    for k, c in col.items():
        h = P[P.setting == k]
        ax.scatter(h.kx_e_geom, h.kx_obs, s=16, color=c, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted crossing K (geometry)", fontsize=8, color=INK2)
    ax.set_ylabel("observed crossing K", fontsize=8, color=INK2)
    ax.set_title("MH2: K where target-only overtakes merged", fontsize=8.5, color=INK, loc="left")
    ax = axes[2]
    _style(ax)
    h = outs["e11_h3.csv"]
    ax.bar(range(len(h)), h.match_rate, color=C_MERGED)
    ax.axhline(0.25, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(h)), h.setting, fontsize=7, rotation=15)
    ax.set_title("MH3: over-predicted emotion = shift direction (chance 0.25)", fontsize=8.5, color=INK, loc="left")
    ax = axes[3]
    _style(ax)
    for k, c in col.items():
        B = bias_table(k)
        h = maps[k][maps[k].K == "5"].merge(B, on=["target", "source"])
        ax.scatter(h.tv, h.m - h.s, s=16, color=c, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("bias of source-only answers (TV)", fontsize=8, color=INK2)
    ax.set_ylabel("merged − source-only at K=5", fontsize=8, color=INK2)
    ax.set_title("MH3c: few target examples fix biased answers", fontsize=8.5, color=INK, loc="left")
    ax = axes[4]
    _style(ax)
    for k, c in col.items():
        h = maps[k][(maps[k].K != "1") & ((maps[k].t - maps[k].s).abs() >= MIN_GAP)]
        ax.scatter(h.alpha, (h.m - h.s) / (h.t - h.s), s=8, color=c, alpha=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel("target share of training data α", fontsize=8, color=INK2)
    ax.set_ylabel("φ = (merged − src)/(tgt − src)", fontsize=8, color=INK2)
    ax.set_title("MH4: merged moves toward target as α grows", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
