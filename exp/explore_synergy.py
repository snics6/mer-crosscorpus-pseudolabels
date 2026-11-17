"""【探索的】E15 の段階 A：合併が位置のずれを直す割合 λ_K の式を実データに当てはめ，予言 P15-2〜P15-6 を調べる。

式と予言は `docs/exploratory-analyses.md` E15（3 節・5 節・6 節）。新しい学習はしない。

    λ_K = ρ/(1+ρ),  ρ = N_t κ_t · [1/(N_s κ_s) + δᵀ (A_s + I/C)⁻¹ δ]

- 特徴は地図と同じ（`measure/data.py` の `load_any`）。ソースの平均と母標準偏差で標準化する
- κ はソースのみの分類器の点数（`results/shift_<kind>/logits/<ソース>.npz`）から：
  ソースは交差検証の当てはめ外の点数 `oof`，ターゲットは `t_<ターゲット>`（全発話．ラベルは使わない）。
  κ = (1 − Σ_k p_k²)/(k − 1)
- N_s, N_t はクラスを均等にする重み（`exp/logreg.py`）の和．ソースのクラス c に n_c 本，ターゲットに K 本ずつ
- C は地図の合併の行の反復ごとの値．同じコーパスの組は，反復ごとにテストの人物を除いたソースで計算する

予言（種類ごとに Holm．p は予言の向きの片側）
- P15-2：K=5 の「合併 − ソースのみ」（地図 30 回）と，式から作った位置の直り
  mean_対 [A(d_s, (1 − λ) o_s) − A(d_s, o_s)]（d_s・o_s は E13）のスピアマン（主）と偏相関（副）
- P15-3：K=5 で上乗せのある組（「合併 − ターゲットのみ」と「合併 − ソースのみ」の 95% t 区間の下端がどちらも正）
  は，ない組より位置の直りが大きい（マン・ホイットニー）
- P15-4：K=5 の式の λ の平均（W7 と V）と，その部分（記述）
- P15-6：E13 の位置による損と見分けやすさ G = N_s κ_s · δᵀ (A_s + I/C)⁻¹ δ のスピアマンと偏相関
- E12 の比（合併 − ソースのみ）÷（D1 − ソースのみ）と式の λ のスピアマン（P15-1 の粗い版．記述）

    ./.venv/bin/python exp/explore_synergy.py [--kinds audio_wavlm video ...]
    ./.venv/bin/python exp/explore_synergy.py --stage b    # 段階 B（`exp/merged_scores.py` の後）

段階 B（P15-1）：保存した合併とソースのみのテストの点数から感情の対ごとに d・o を測り，
組 × K の測った λ = 1 − Σ sign(o_s)·o_m / Σ |o_s|（|o_s| ≥ 0.3）を式の λ（反復 0〜9 の C）と比べる。
近似 (c) の診断 τ（スイッチの向きに沿ったターゲットとソースの散らばりの比）は段階 A で計算する。

出力：`results/explore_relation/e15_lambda.csv`（組 × K），`e15_pairs.csv`（組ごとの予言の量），
`e15_tests.csv`，`e15_e12.csv`・`e15_e12_summary.csv`，`e15_parts.csv`，`fig_e15.png`．
段階 B は `e15b_rows.csv`・`e15b_pairs.csv`・`e15b_tests.csv`・`e15b_decomp.csv`・`fig_e15b.png`
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_pairs import ba, measure  # noqa: E402
from pair_table import OUT, OUT_ROOT, SIX, holm, map_dir, partial_rank  # noqa: E402
from split import speaker_folds  # noqa: E402

KINDS = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
TEXT_KINDS = {"text_roberta": "T", "fuse-audio-text": "AT", "fuse-video-text": "VT", "fuse-audio-video-text": "AVT"}
ALL_KINDS = {**KINDS, **TEXT_KINDS}       # 名前の表．既定で流すのは KINDS だけ（言葉を含む 4 種類は E21 で --kinds を指定）
SIX_NAME = {"audio_wavlm": "W6", "audio_hubert": "Hu6"}
KS = ["1", "5", "10"]
C_GRID = (0.01, 0.1, 1.0, 10.0)
N_CLASSES = 4
N_TEST_FOLDS = 5
E12_MIN = 0.01


# ---------------------------------------------------------------- 地図

def load_map(kind: str) -> pd.DataFrame:
    """反復ごとの m・t・s と合併の C（K ∈ KS）。"""
    fs = sorted(map_dir(kind).glob("*.csv"))
    d = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str},
                               usecols=["target", "source", "method", "K", "rep", "uar", "C", "n_src"])
                   for f in fs], ignore_index=True)
    m = d[(d.method == "merged") & d.K.isin(KS)].rename(columns={"uar": "m", "C": "C_m"})
    t = d[(d.method == "tgt_only") & d.K.isin(KS)][["target", "K", "rep", "uar"]].rename(columns={"uar": "t"})
    s = d[d.method == "src_only"][["target", "source", "rep", "uar"]].rename(columns={"uar": "s"})
    return (m[["target", "source", "K", "rep", "m", "C_m", "n_src"]]
            .merge(t, on=["target", "K", "rep"]).merge(s, on=["target", "source", "rep"]))


def t_lower(x: np.ndarray) -> float:
    n = len(x)
    return float(x.mean() - stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------- 式の λ

def kappa(L: np.ndarray) -> np.ndarray:
    p = softmax(L, axis=1)
    return (1 - (p ** 2).sum(1)) / (N_CLASSES - 1)


def src_parts(Zs: np.ndarray, ys: np.ndarray, ks: np.ndarray, K: int) -> dict:
    """ソースの行と K だけで決まる部分（ターゲットによらない）。"""
    n_c = np.bincount(ys, minlength=N_CLASSES) + K
    n = n_c.sum()
    s_src = n / (N_CLASSES * n_c[ys])
    w = s_src * ks
    a = float(w.sum())
    zbar = (w[:, None] * Zs).sum(0) / a
    D = Zs - zbar
    ev, V = np.linalg.eigh((D * w[:, None]).T @ D)
    return {"N_t": float((n / (N_CLASSES * n_c) * K).sum()), "a": a,
            "kappa_s": float(np.average(ks, weights=s_src)), "zbar": zbar, "ev": np.clip(ev, 0, None), "V": V}


def parts(sp: dict, zt_mean: np.ndarray, kt: float) -> dict:
    return {**sp, "kappa_t": kt, "proj2": (sp["V"].T @ (zt_mean - sp["zbar"])) ** 2}


def lam_at(p: dict, C: float) -> dict:
    route_w = float((p["proj2"] / (p["ev"] + 1.0 / C)).sum())
    rho = p["N_t"] * p["kappa_t"] * (1.0 / p["a"] + route_w)
    return {"lam": rho / (1 + rho), "rho": rho, "route_b": 1.0 / p["a"], "route_w": route_w,
            "G": p["a"] * route_w}


def lambda_table(kind: str, mp: pd.DataFrame) -> pd.DataFrame:
    groups = groups_for(kind)
    data = {g: load_any(g, kind) for g in groups}
    person = {g: (GROUP_SPEC[g][0] + ":" + df.speaker_group.astype(str)).to_numpy() for g, (df, _) in data.items()}
    rows = []
    for s in groups:
        dfs, Xs = data[s]
        ys = dfs.label_4.to_numpy()
        mu, sd = Xs.mean(0), Xs.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Zs = ((Xs - mu) / sd).astype(np.float64)
        cache = {}
        z = np.load(ROOT / "results" / f"shift_{kind}" / "logits" / f"{s}.npz")
        ks_all = kappa(z["oof"])
        if len(ks_all) != len(ys):
            raise SystemExit(f"{kind} {s}：oof {len(ks_all)} 行と発話 {len(ys)} 本が合わない")
        for t in groups:
            if t == s:
                continue
            dft, Xt = data[t]
            zt_mean = ((Xt - mu) / sd).astype(np.float64).mean(0)
            kt = float(kappa(z[f"t_{t}"]).mean())
            g = mp[(mp.target == t) & (mp.source == s)]
            cross = bool(np.isin(person[t], person[s]).any())
            for K in KS:
                gk = g[g.K == K]
                if cross:                               # 反復ごとにテストの人物を除く（exp/map.py の run_rep）
                    out = []
                    for r in gk.itertuples():
                        f = speaker_folds(dft.speaker_group.astype(str).reset_index(drop=True),
                                          k=N_TEST_FOLDS, seed=int(r.rep)).to_numpy()
                        keep = ~np.isin(person[s], np.unique(person[t][f == int(r.rep) % N_TEST_FOLDS]))
                        if keep.sum() != r.n_src:
                            raise SystemExit(f"{kind} {t}←{s} 反復 {r.rep}：ソース {keep.sum()} 本が地図の {r.n_src} 本と違う")
                        p = parts(src_parts(Zs[keep], ys[keep], ks_all[keep], int(K)), zt_mean, kt)
                        out.append({**lam_at(p, r.C_m), **{q: p[q] for q in ("N_t", "a", "kappa_s", "kappa_t")}})
                else:
                    if (gk.n_src != len(ys)).any():
                        raise SystemExit(f"{kind} {t}←{s}：ソースの本数が地図と違う")
                    if K not in cache:
                        cache[K] = src_parts(Zs, ys, ks_all, int(K))
                    p = parts(cache[K], zt_mean, kt)
                    by_c = {c: lam_at(p, c) for c in C_GRID}
                    out = [{**by_c[float(c)], **{q: p[q] for q in ("N_t", "a", "kappa_s", "kappa_t")}}
                           for c in gk.C_m]
                o = pd.DataFrame(out)
                tau, c_mode = np.nan, np.nan
                if K == "5":                            # 近似 (c) の診断（同じコーパスの組もソース全件で近似する）
                    if K not in cache:
                        cache[K] = src_parts(Zs, ys, ks_all, int(K))
                    sp = cache[K]
                    c_mode = float(gk[gk.rep < 10].C_m.mode().iloc[0])
                    v = sp["V"] @ ((sp["V"].T @ (zt_mean - sp["zbar"])) / (sp["ev"] + 1.0 / c_mode))
                    var_t = float(np.var(((Xt - mu) / sd).astype(np.float64) @ v))
                    var_s = float((sp["ev"] * (sp["V"].T @ v) ** 2).sum() / sp["a"])
                    tau = var_t / var_s
                rows.append({"kind": kind, "target": t, "source": s, "K": K, "n_rep": len(o), "cross": cross,
                             "lam_pred": o.lam.mean(), "rho": o.rho.mean(), "route_b": o.route_b.mean(),
                             "route_w": o.route_w.mean(), "G": o.G.mean(), "N_t": o.N_t.mean(),
                             "N_s_kappa_s": o.a.mean(), "kappa_s": o.kappa_s.mean(), "kappa_t": o.kappa_t.mean(),
                             "C_median": float(gk.C_m.median()),
                             "lam_pred_rep012": o.lam[gk.rep.to_numpy() < 3].mean(),
                             "lam_pred_rep09": o.lam[gk.rep.to_numpy() < 10].mean(),
                             "tau_K5": tau, "C_mode09_K5": c_mode})
        print(f"  {kind} ソース {s} 済み", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 組ごとの量と検定

def pair_table(kind: str, mp: pd.DataFrame, lam: pd.DataFrame, e13: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, s, K), g in mp.groupby(["target", "source", "K"]):
        mt, ms = (g.m - g.t).to_numpy(), (g.m - g.s).to_numpy()
        lr = lam[(lam.target == t) & (lam.source == s) & (lam.K == K)].iloc[0]
        e = e13[(e13.target == t) & (e13.source == s)]
        fix = float(np.mean(ba(e.d, (1 - lr.lam_pred) * e.o) - ba(e.d, e.o)))
        rows.append({"kind": kind, "target": t, "source": s, "K": K, "n_rep": len(g),
                     "m": g.m.mean(), "t": g.t.mean(), "s": g.s.mean(),
                     "m_minus_t": mt.mean(), "m_minus_s": ms.mean(),
                     "synergy": g.m.mean() - max(g.t.mean(), g.s.mean()),
                     "synergy_flag": bool(t_lower(mt) > 0 and t_lower(ms) > 0),
                     "lam_pred": lr.lam_pred, "G": lr.G, "pos_fix_pred": fix,
                     "loss_offset": float(e.loss_offset.mean()), "loss_direction": float(e.loss_direction.mean())})
    return pd.DataFrame(rows)


def one_sided_partial(d: pd.DataFrame, resp: str, met: str) -> tuple[float, float]:
    pr = partial_rank(d, resp, met)
    if pr is None:
        return np.nan, np.nan
    r = pr["r_partial_rank"]
    return r, pr["p"] / 2 if r > 0 else 1 - pr["p"] / 2


def tests(P: pd.DataFrame, setting: str, groups: list[str]) -> list[dict]:
    d = P[(P.K == "5") & P.target.isin(groups) & P.source.isin(groups)]
    out = []
    r, p = stats.spearmanr(d.m_minus_s, d.pos_fix_pred, alternative="greater")
    out.append({"test": "P15-2 spearman", "stat": r, "p": p})
    r, p = one_sided_partial(d, "m_minus_s", "pos_fix_pred")
    out.append({"test": "P15-2 partial", "stat": r, "p": p})
    a, b = d[d.synergy_flag].pos_fix_pred, d[~d.synergy_flag].pos_fix_pred
    if len(a) >= 2 and len(b) >= 2:
        u = stats.mannwhitneyu(a, b, alternative="greater")
        out.append({"test": "P15-3 mannwhitney", "stat": u.statistic / (len(a) * len(b)), "p": u.pvalue,
                    "n_flag": len(a), "median_flag": a.median(), "median_rest": b.median()})
    else:
        out.append({"test": "P15-3 mannwhitney", "stat": np.nan, "p": np.nan, "n_flag": len(a)})
    r, p = stats.spearmanr(d.loss_offset, d.G, alternative="greater")
    out.append({"test": "P15-6 spearman", "stat": r, "p": p})
    r, p = one_sided_partial(d, "loss_offset", "G")
    out.append({"test": "P15-6 partial", "stat": r, "p": p})
    ps = np.array([o["p"] for o in out])
    ok = ~np.isnan(ps)
    adj = np.full(len(ps), np.nan)
    adj[ok] = holm(ps[ok])
    for o, h in zip(out, adj):
        o.update({"setting": setting, "n_pairs": len(d), "p_holm": h})
    return out


def e12_ratio(kind: str, mp: pd.DataFrame, lam: pd.DataFrame) -> pd.DataFrame | None:
    fs = sorted((ROOT / "results" / f"mech_{kind}").glob("*.csv"))
    if not fs:
        return None
    m = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str, "alpha": str},
                               usecols=["op", "target", "source", "method", "K", "rep", "uar"]) for f in fs])
    d1 = m[(m.op == "none") & (m.method == "src_only+D1") & m.K.isin(KS)].rename(columns={"uar": "d1"})
    x = d1.merge(mp, on=["target", "source", "K", "rep"])
    x = x[x.rep < 3]
    g = x.groupby(["target", "source", "K"]).agg(n=("rep", "size"), m=("m", "mean"), s=("s", "mean"),
                                                 d1=("d1", "mean")).reset_index()
    g["d1_minus_s"], g["m_minus_s"] = g.d1 - g.s, g.m - g.s
    g = g.merge(lam[["target", "source", "K", "lam_pred_rep012"]], on=["target", "source", "K"])
    g["ratio"] = np.where(g.d1_minus_s >= E12_MIN, g.m_minus_s / g.d1_minus_s, np.nan)
    g["kind"] = kind
    return g


# ---------------------------------------------------------------- 段階 B：保存した点数から測った λ

PAIRS = [(k, j) for k in range(N_CLASSES) for j in range(k + 1, N_CLASSES)]
O_MIN = 0.3


def d_o(L: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    """E13 と同じ：感情の対 (k, j) ごとに z = 点数_k − 点数_j の 2 つの山の d と o。"""
    out = []
    for k, j in PAIRS:
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        out.append(((zk.mean() - zj.mean()) / sd, -(zk.mean() + zj.mean()) / 2 / sd))
    return out


def stage_b(kinds: list[str]) -> int:
    lam = pd.read_csv(OUT / "e15_lambda.csv", dtype={"K": str})
    rows = []
    for kind in kinds:
        d = OUT_ROOT / f"merged_scores_{kind}"
        for f in sorted(d.glob("*.npz")):
            z, t = np.load(f), f.stem
            reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
            srcs = sorted({k[2:].rsplit("_r", 1)[0] for k in z.files if k.startswith("s_")})
            for r in reps:
                y = z[f"y_test_r{r}"]
                for s in srcs:
                    ds = d_o(z[f"s_{s}_r{r}"], y)
                    for K in KS:
                        dm = d_o(z[f"m_{s}_K{K}_r{r}"], y)
                        for (k, j), (d_s, o_s), (d_m, o_m) in zip(PAIRS, ds, dm):
                            rows.append({"kind": kind, "target": t, "source": s, "K": K, "rep": r,
                                         "pair": f"{k}-{j}", "d_s": d_s, "o_s": o_s, "d_m": d_m, "o_m": o_m})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "e15b_rows.csv", index=False)
    # 2 つの山のモデルで，合併 − ソースのみ（2 つの感情の正解率）を位置の分と向きの分に分ける
    R["pos"] = ba(R.d_s, R.o_m) - ba(R.d_s, R.o_s)
    R["dir"] = ba(R.d_m, R.o_m) - ba(R.d_s, R.o_m)
    R["total"] = R.pos + R.dir
    (R.groupby(["kind", "K"])[["pos", "dir", "total", "d_s", "d_m", "o_s", "o_m"]]
     .agg(lambda x: x.abs().mean() if x.name in ("o_s", "o_m") else x.mean())
     .reset_index().to_csv(OUT / "e15b_decomp.csv", index=False))
    agg = []
    for (kind, t, s, K), g in R.groupby(["kind", "target", "source", "K"]):
        v = g[g.o_s.abs() >= O_MIN]
        lo = 1 - float((np.sign(v.o_s) * v.o_m).sum() / v.o_s.abs().sum()) if len(v) else np.nan
        agg.append({"kind": kind, "target": t, "source": s, "K": K, "n_valid": len(v), "lam_obs": lo,
                    "lam_obs_median": float((1 - v.o_m / v.o_s).median()) if len(v) else np.nan,
                    "abs_o_s": g.o_s.abs().mean(), "abs_o_m": g.o_m.abs().mean(),
                    "d_s": g.d_s.mean(), "d_m": g.d_m.mean(), "d_change": (g.d_m - g.d_s).mean()})
    A = pd.DataFrame(agg).merge(lam[["kind", "target", "source", "K", "lam_pred_rep09", "G", "tau_K5"]],
                                on=["kind", "target", "source", "K"])
    A.to_csv(OUT / "e15b_pairs.csv", index=False)
    tests = []
    for kind, g in A.dropna(subset=["lam_obs"]).groupby("kind", sort=False):
        r, p = stats.spearmanr(g.lam_pred_rep09, g.lam_obs, alternative="greater")
        row = {"setting": ALL_KINDS[kind], "n": len(g), "spearman": r, "p_one_sided": p,
               "mean_lam_obs": g.lam_obs.mean(), "mean_lam_pred": g.lam_pred_rep09.mean(),
               "mae": (g.lam_obs - g.lam_pred_rep09).abs().mean()}
        for K, h in g.groupby("K"):
            row[f"spearman_K{K}"] = stats.spearmanr(h.lam_pred_rep09, h.lam_obs)[0]
            row[f"mean_lam_obs_K{K}"], row[f"mean_lam_pred_K{K}"] = h.lam_obs.mean(), h.lam_pred_rep09.mean()
            row[f"d_change_K{K}"] = h.d_change.mean()
        row["tau_K5_median"] = g[g.K == "5"].tau_K5.median()
        tests.append(row)
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e15b_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).T.to_string())
    f, axes = plt.subplots(1, len(T), figsize=(4 * len(T), 4), squeeze=False)
    for ax, (kind, g) in zip(axes[0], A.groupby("kind", sort=False)):
        for K, h in g.groupby("K"):
            ax.scatter(h.lam_pred_rep09, h.lam_obs, s=12, label=f"K={K}")
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title(ALL_KINDS[kind])
        ax.set_xlabel("predicted lambda"); ax.set_ylabel("observed lambda"); ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(OUT / "fig_e15b.png", dpi=120); plt.close(f)
    return 0


# ---------------------------------------------------------------- 図

def fig(P: pd.DataFrame, path: Path) -> None:
    kinds = list(dict.fromkeys(P.kind))
    f, axes = plt.subplots(2, len(kinds), figsize=(4 * len(kinds), 7.5), squeeze=False)
    for j, k in enumerate(kinds):
        d = P[(P.kind == k) & (P.K == "5")]
        ax = axes[0, j]
        ax.scatter(d.pos_fix_pred * 100, d.m_minus_s * 100, s=14,
                   c=np.where(d.synergy_flag, "tab:red", "tab:gray"))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{ALL_KINDS[k]}  K=5"); ax.set_xlabel("predicted position fix (pt)")
        ax.set_ylabel("merged - source-only (UAR pt)")
        ax = axes[1, j]
        for K, h in P[P.kind == k].groupby("K"):
            ax.scatter(h.G, h.lam_pred, s=10, label=f"K={K}")
        ax.set_xscale("log"); ax.set_xlabel("separability G"); ax.set_ylabel("predicted lambda")
        ax.legend(fontsize=7)
    f.tight_layout(); f.savefig(path, dpi=120); plt.close(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(ALL_KINDS))
    ap.add_argument("--stage", choices=["a", "b"], default="a",
                    help="b：段階 B の点数（`results/merged_scores_<kind>/`）から測った λ を作り，P15-1 を調べる")
    args = ap.parse_args()
    if args.stage == "b":
        return stage_b(args.kinds)
    OUT.mkdir(parents=True, exist_ok=True)
    lams, pairs, tst, e12s = [], [], [], []
    for kind in args.kinds:
        t0 = time.time()
        mp = load_map(kind)
        lam = lambda_table(kind, mp)
        e13 = measure(kind)
        P = pair_table(kind, mp, lam, e13)
        groups = groups_for(kind)
        tst += tests(P, ALL_KINDS[kind], groups)
        if kind in SIX_NAME:
            tst += tests(P, SIX_NAME[kind], SIX)
        e = e12_ratio(kind, mp, lam)
        if e is not None:
            e12s.append(e)
        lams.append(lam); pairs.append(P)
        print(f"{kind}: {time.time() - t0:.0f} 秒", flush=True)
    lam, P, T = pd.concat(lams), pd.concat(pairs), pd.DataFrame(tst)
    lam.to_csv(OUT / "e15_lambda.csv", index=False)
    P.to_csv(OUT / "e15_pairs.csv", index=False)
    T.to_csv(OUT / "e15_tests.csv", index=False)
    parts_ = (lam.groupby(["kind", "K"])[["lam_pred", "rho", "route_b", "route_w", "G", "N_t", "N_s_kappa_s",
                                         "kappa_s", "kappa_t", "C_median"]].mean().reset_index())
    parts_.to_csv(OUT / "e15_parts.csv", index=False)
    if e12s:
        E = pd.concat(e12s)
        E.to_csv(OUT / "e15_e12.csv", index=False)
        rows = []
        for (k, K), g in E.dropna(subset=["ratio"]).groupby(["kind", "K"]):
            r, p = stats.spearmanr(g.lam_pred_rep012, g.ratio) if len(g) >= 5 else (np.nan, np.nan)
            rows.append({"kind": k, "K": K, "n_pairs": len(g), "spearman": r, "p_two_sided": p,
                         "mean_ratio": g.ratio.mean(), "mean_lam_pred": g.lam_pred_rep012.mean()})
        pd.DataFrame(rows).to_csv(OUT / "e15_e12_summary.csv", index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    fig(P, OUT / "fig_e15.png")
    print(T.round(4).to_string(index=False))
    print(parts_.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
