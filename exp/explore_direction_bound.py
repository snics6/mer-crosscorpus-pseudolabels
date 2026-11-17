"""【探索的】E28 の 9 節：K 本で向きをどこまで直せるかの基準モデル d*(K) と，実測（ターゲットのみ，仮説転移，合併）の比較（U1〜U4）。

基準：d*(K) = D/√(1 + t/(1 + r t))，t = q/K，r = e²/(1 − e²)，e = d_s/D（感情の対ごと．D は E13 の own_d）。
q は 2 通り：q_dir（ターゲットのみの d を K=1・5・10 で測って当てる．`results/tgt_scores_<kind>/`），q_t（E11 の学習曲線）。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_direction_bound.py

出力：`OUT/e28t9_targets.csv`（ターゲットごとの q），`OUT/e28t9_pairs.csv`（組 × K），`OUT/e28t9_tests.csv`，`OUT/e28t9_summary.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for  # noqa: E402
from explore_pairs import measure  # noqa: E402
from explore_synergy import KS, PAIRS, d_o, load_map  # noqa: E402
from explore_uar_link import shared  # noqa: E402
from pair_table import OUT  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
NAMES = ["neutral", "happy", "sad", "angry"]
FREE = ["free_C0.001", "free_C0.01", "free_C0.1", "free_C1"]
DEFAULT = "cb8_C0.01"


def d_star(D: np.ndarray, e: np.ndarray, q: float, K: int) -> np.ndarray:
    t = q / K
    r = e ** 2 / np.clip(1 - e ** 2, 1e-9, None)
    return D / np.sqrt(1 + t / (1 + r * t))


def q_fit(dT: pd.DataFrame, D: pd.Series) -> float:
    """d_T(K) = D/√(1 + q/K) を，感情の対 × K の平均で当てる（q を格子で探す）。"""
    grid = np.logspace(-1, 3, 400)
    best, best_err = np.nan, np.inf
    for q in grid:
        err = 0.0
        for (pair, K), v in dT.groupby(["pair", "K"]).d.mean().items():
            err += (v - D[pair] / np.sqrt(1 + q / int(K))) ** 2
        if err < best_err:
            best, best_err = float(q), err
    return best


def uar_from_d(d6: np.ndarray) -> float:
    c = np.zeros((4, 4))
    for (k, l), d in zip(PAIRS, d6):
        c[k, l] = c[l, k] = d / 2
    return shared(c)


def main() -> int:
    e13 = pd.read_csv(OUT / "e13_pairs.csv")
    e13 = pd.concat([e13[e13.kind.isin(KINDS)]] + [measure(k).assign(kind=k) for k in KINDS if k not in set(e13.kind)])
    e13["pair_idx"] = e13.pair.map({f"{NAMES[a]}-{NAMES[b]}": f"{a}-{b}" for a, b in PAIRS})
    qt_tab = pd.concat([pd.read_csv(OUT / "e11_curves.csv"), pd.read_csv(ROOT / "results" / "check_mh" / "curves.csv")])
    H = pd.read_csv(OUT / "e28t1_rows.csv", dtype={"K": str})
    Mg = pd.read_csv(OUT / "e15b_rows.csv", dtype={"K": str})
    Mg = Mg[Mg.rep < 10]
    tg_rows, pair_rows = [], []
    for kind in KINDS:
        groups = groups_for(kind)
        mp = load_map(kind)
        for t in groups:
            z = np.load(ROOT / "results" / f"tgt_scores_{kind}" / f"{t}.npz")
            reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
            rec = []
            for r in reps:
                y = z[f"y_test_r{r}"]
                for K in KS:
                    for (a, b), (d, o) in zip(PAIRS, d_o(z[f"t_K{K}_r{r}"].astype(np.float64), y)):
                        rec.append({"rep": r, "K": K, "pair": f"{a}-{b}", "d": d, "o": o})
            dT = pd.DataFrame(rec)
            # own_d は E13（ソースによらずターゲットだけで決まる．ソースごとの行で同じ値）
            own = e13[(e13.kind == kind) & (e13.target == t)].groupby("pair_idx").own_d.mean()
            q_dir = q_fit(dT, own)
            qt = qt_tab[(qt_tab["setting"] == kind) & (qt_tab["target"] == t)]
            q_t = float(qt.q.iloc[0]) if len(qt) else np.nan
            tg_rows.append({"kind": kind, "target": t, "q_dir": q_dir, "q_t": q_t, "own_d": own.mean(),
                            **{f"dT_K{K}": dT[dT.K == K].d.mean() for K in KS},
                            **{f"dT_fit_K{K}": float((own / np.sqrt(1 + q_dir / int(K))).mean()) for K in KS}})
            for s in groups:
                if s == t:
                    continue
                e13ts = e13[(e13.kind == kind) & (e13.target == t) & (e13.source == s)].set_index("pair_idx")
                if e13ts.empty:
                    continue
                D = own.reindex([f"{a}-{b}" for a, b in PAIRS]).to_numpy()
                ds = e13ts.d.reindex([f"{a}-{b}" for a, b in PAIRS]).to_numpy()
                e = np.clip(ds / D, 0.0, 0.999)
                h = H[(H.kind == kind) & (H.target == t) & (H.source == s)]
                m = Mg[(Mg.kind == kind) & (Mg.target == t) & (Mg.source == s)]
                for K in KS:
                    dstar = {"dir": d_star(D, e, q_dir, int(K)), "t": d_star(D, e, q_t, int(K)) if not np.isnan(q_t) else np.full(6, np.nan)}
                    hK = h[h.K == K]
                    d_free = {cfg: hK[hK.config == cfg].groupby("pair").d_h.mean().reindex([f"{a}-{b}" for a, b in PAIRS]).to_numpy() for cfg in FREE}
                    d_free_best = max(d_free.values(), key=lambda v: np.nanmean(v))
                    d_def = hK[hK.config == DEFAULT].groupby("pair").d_h.mean().reindex([f"{a}-{b}" for a, b in PAIRS]).to_numpy()
                    d_m = m[m.K == K].groupby("pair").d_m.mean().reindex([f"{a}-{b}" for a, b in PAIRS]).to_numpy()
                    d_s_obs = hK.groupby("pair").d_s.mean().reindex([f"{a}-{b}" for a, b in PAIRS]).to_numpy()
                    pair_rows.append({"kind": kind, "target": t, "source": s, "K": K,
                                      "same_corpus": GROUP_SPEC[t][0] == GROUP_SPEC[s][0],
                                      "D": D.mean(), "d_s": ds.mean(), "d_s_obs": d_s_obs.mean(), "e": float(np.mean(e)),
                                      "q_dir": q_dir, "q_t": q_t,
                                      "d_star_dir": dstar["dir"].mean(), "d_star_t": float(np.nanmean(dstar["t"])),
                                      "d_T": float(D.mean() / np.sqrt(1 + q_dir / int(K))),
                                      "d_h_best": float(np.nanmean(d_free_best)), "d_h_default": float(np.nanmean(d_def)),
                                      "d_m": float(np.nanmean(d_m)),
                                      "uar_star_dir": uar_from_d(dstar["dir"]) * 100, "uar_D": uar_from_d(D) * 100,
                                      "uar_h_default": float(hK[hK.config == DEFAULT].uar.mean() * 100) if "uar" in hK else np.nan})
        print(f"  {kind} 済み", flush=True)
    T = pd.DataFrame(tg_rows)
    P = pd.DataFrame(pair_rows)
    # 実測の UAR（既定の仮説転移）と全ラベルの天井を足す
    A = pd.read_csv(OUT / "e28t1_pairs.csv", dtype={"K": str})
    A = A[A.config == DEFAULT][["kind", "target", "source", "K", "uar"]].rename(columns={"uar": "uar_h_default"})
    P = P.drop(columns=["uar_h_default"]).merge(A, on=["kind", "target", "source", "K"], how="left")
    P["uar_h_default"] *= 100
    ceil = []
    for kind in KINDS:
        mp = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str})
                        for f in sorted((ROOT / "results" / "fixc_map" / kind).glob("*.csv"))])
        c = mp[(mp.method == "tgt_only") & (mp.K == "all")].groupby("target").uar.mean() * 100
        ceil.append(c.rename("uar_all").reset_index().assign(kind=kind))
    P = P.merge(pd.concat(ceil), on=["kind", "target"], how="left")
    T.to_csv(OUT / "e28t9_targets.csv", index=False)
    P.to_csv(OUT / "e28t9_pairs.csv", index=False)
    Q = P[~P.same_corpus].copy()
    Q["f_recovered"] = (Q.d_h_best - Q.d_s_obs) / (Q.d_star_dir - Q.d_s_obs)
    Q["share_recoverable"] = (Q.d_star_dir - Q.d_s) / (Q.D - Q.d_s)
    Q["gap_star_minus_best"] = Q.d_star_dir - Q.d_h_best
    tests, summ = [], []
    for kind in KINDS:
        g = Q[Q.kind == kind]
        above = float((g.d_star_dir > g.d_h_best).mean())
        tests.append({"kind": kind, "test": "U1", "frac_dstar_above_best": above, "holds": above >= 0.8, "n": len(g)})
        g10 = g[g.K == "10"]
        f_med = float(g10.f_recovered.median())
        tests.append({"kind": kind, "test": "U2", "median_f_recovered_K10": f_med, "holds": f_med >= 0.5, "n": len(g10),
                      "median_f_default": float(((g10.d_h_default - g10.d_s_obs) / (g10.d_star_dir - g10.d_s_obs)).median()),
                      "median_f_merged": float(((g10.d_m - g10.d_s_obs) / (g10.d_star_dir - g10.d_s_obs)).median())})
        sh = float(g10.share_recoverable.median())
        tests.append({"kind": kind, "test": "U3", "median_share_recoverable_K10": sh, "holds": sh < 0.5, "n": len(g10),
                      **{f"share_K{K}": float(g[g.K == K].share_recoverable.median()) for K in KS}})
        for K in KS:
            gk = g[g.K == K]
            summ.append({"kind": kind, "K": K, "D": gk.D.mean(), "d_s": gk.d_s.mean(), "d_T": gk.d_T.mean(),
                         "d_star_dir": gk.d_star_dir.mean(), "d_star_t": gk.d_star_t.mean(), "d_h_best": gk.d_h_best.mean(),
                         "d_h_default": gk.d_h_default.mean(), "d_m": gk.d_m.mean(),
                         "uar_star_dir": gk.uar_star_dir.mean(), "uar_h_default": gk.uar_h_default.mean(),
                         "uar_D_bridge": gk.uar_D.mean(), "uar_all_label": gk.uar_all.mean(),
                         "q_dir_median": gk.q_dir.median(), "q_t_median": gk.q_t.median()})
    Tst, S = pd.DataFrame(tests), pd.DataFrame(summ)
    Tst.to_csv(OUT / "e28t9_tests.csv", index=False)
    S.to_csv(OUT / "e28t9_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    print(Tst.round(3).to_string(index=False))
    print(S.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
