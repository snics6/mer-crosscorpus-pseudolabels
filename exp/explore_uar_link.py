"""【探索的】E28 の 4 節：感情の対の量（a_k(l)，d・o）から 4 感情の UAR を作る橋の検証（T4-1〜T4-3）。新しい学習はしない。

入力：C を固定した段階 B の点数（`OUT_ROOT/merged_scores_<kind>/`：ソースのみ `s_`，合併 `m_…_K5`）と，
仮説転移（手 7）の点数（`results/transfer_scores_<kind>/`：`h7_…_K5`）。反復 0〜9。

    MERDA_MAP_ROOT=results/fixc_map MERDA_OUT_ROOT=results/fixc_analysis ./.venv/bin/python exp/explore_uar_link.py

出力：`OUT/e28t4_rows.csv`（種類 × 組 × 反復 × 分類器の UAR と 3 つの橋），`OUT/e28t4_tests.csv`，`OUT/e28t4_summary.csv`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from explore_synergy import PAIRS, d_o  # noqa: E402
from pair_table import OUT, OUT_ROOT  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
K = "5"


_U, _W = np.polynomial.hermite_e.hermegauss(40)       # 共通の雑音 u ~ N(0,1) の積分（ガウス・エルミート）
_W = _W / _W.sum()


def shared(c: np.ndarray) -> float:
    """共通の雑音の橋：P(正解 | k) = E_u Π_l Φ(√2 c_kl + u)。c_kl は対 (k, l) で感情 k が勝つ割合のプロビット（Φ⁻¹）。
    各感情の点数が独立で等分散の正規なら，対の差 z_kl と z_km は感情 k の点数を共有して相関 1/2 を持つ。
    **結果を見た後（独立近似が 8〜10 点低く出た）に足した橋で，予言ではない。**"""
    acc = []
    for k in range(4):
        cs = [c[k, l] for l in range(4) if l != k]
        acc.append(float(np.sum(_W * np.prod([stats.norm.cdf(np.sqrt(2) * ck + _U) for ck in cs], axis=0))))
    return float(np.mean(acc))


def bridges(L: np.ndarray, y: np.ndarray) -> dict:
    """実際の UAR と，対ごとの勝ち割合 a_k(l) から作る橋（独立近似，和の上界，正規版，共通の雑音版）。"""
    pred = L.argmax(1)
    uar = float(np.mean([np.mean(pred[y == k] == k) for k in range(4)]))
    a = np.ones((4, 4))                                   # a[k, l] = 感情 k の発話が対 (k, l) で勝つ割合
    for k in range(4):
        for l in range(4):
            if k != l:
                a[k, l] = float(np.mean(L[y == k, k] > L[y == k, l]))
    indep = float(np.mean([np.prod([a[k, l] for l in range(4) if l != k]) for k in range(4)]))
    union = float(np.mean([max(0.0, 1 - sum(1 - a[k, l] for l in range(4) if l != k)) for k in range(4)]))
    g = np.ones((4, 4))
    for (k, l), (d, o) in zip(PAIRS, d_o(L, y)):
        g[k, l], g[l, k] = stats.norm.cdf(d / 2 - o), stats.norm.cdf(d / 2 + o)
    gauss = float(np.mean([np.prod([g[k, l] for l in range(4) if l != k]) for k in range(4)]))
    c_emp = stats.norm.ppf(np.clip(a, 1e-4, 1 - 1e-4))
    c_gau = np.zeros((4, 4))
    for (k, l), (d, o) in zip(PAIRS, d_o(L, y)):
        c_gau[k, l], c_gau[l, k] = d / 2 - o, d / 2 + o
    return {"uar": uar, "indep": indep, "union": union, "gauss": gauss,
            "shared": shared(c_emp), "shared_gauss": shared(c_gau)}


def rows() -> pd.DataFrame:
    out = []
    for kind in KINDS:
        for f in sorted((OUT_ROOT / f"merged_scores_{kind}").glob("*.npz")):
            z, t = np.load(f), f.stem
            h = np.load(ROOT / "results" / f"transfer_scores_{kind}" / f"{t}.npz")
            reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
            srcs = sorted({k[2:].rsplit("_r", 1)[0] for k in z.files if k.startswith("s_")})
            for r in reps:
                y = z[f"y_test_r{r}"]
                assert np.array_equal(z[f"test_idx_r{r}"], h[f"test_idx_r{r}"])
                for s in srcs:
                    for clf, key in (("src", f"s_{s}_r{r}"), ("merged", f"m_{s}_K{K}_r{r}"), ("ht", f"h7_{s}_K{K}_r{r}")):
                        L = (h if clf == "ht" else z)[key].astype(np.float64)
                        out.append({"kind": kind, "target": t, "source": s, "rep": r, "clf": clf, **bridges(L, y)})
        print(f"{kind}：済み", flush=True)
    return pd.DataFrame(out)


def grid_check() -> int:
    """T4-4：E28 の 1 節の設定（罰則と切片を振った仮説転移）× K の反復ごとの UAR を，共通の雑音の橋（正規版）が当てるか。
    d・o は `OUT/e28t1_rows.csv`（感情の対ごと），UAR は `results/transfer_grid_<kind>/<ターゲット>.csv`。"""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "exp"))
    from data import groups_for
    H = pd.read_csv(OUT / "e28t1_rows.csv", dtype={"K": str})
    out, tests = [], []
    for (kind, t, s, K, r, cfg), g in H.groupby(["kind", "target", "source", "K", "rep", "config"]):
        c = np.zeros((4, 4))
        for pr, d, o in zip(g.pair, g.d_h, g.o_h):
            k, l = (int(x) for x in pr.split("-"))
            c[k, l], c[l, k] = d / 2 - o, d / 2 + o
        out.append({"kind": kind, "target": t, "source": s, "K": K, "rep": r, "config": cfg, "shared_gauss": shared(c)})
    P = pd.DataFrame(out)
    U = pd.concat([pd.read_csv(ROOT / "results" / f"transfer_grid_{k}" / f"{t}.csv", dtype={"K": str}).assign(kind=k)
                   for k in KINDS for t in groups_for(k) if (ROOT / "results" / f"transfer_grid_{k}" / f"{t}.csv").exists()])
    U = U[U.K != ""][["kind", "target", "source", "K", "rep", "config", "uar"]]
    P = P.merge(U, on=["kind", "target", "source", "K", "rep", "config"])
    P.to_csv(OUT / "e28t4_grid_rows.csv", index=False)
    for kind in KINDS:
        g = P[P.kind == kind]
        rho = float(stats.spearmanr(g.shared_gauss, g.uar)[0])
        mae = float((g.shared_gauss - g.uar).abs().mean())
        tests.append({"kind": kind, "test": "T4-4", "config": "all", "spearman": rho, "mae": mae, "bias": float((g.shared_gauss - g.uar).mean()),
                      "holds": rho >= 0.95 and mae < 0.02, "n": len(g)})
        for cfg, h in g.groupby("config"):
            tests.append({"kind": kind, "test": "T4-4 設定ごと", "config": cfg, "spearman": float(stats.spearmanr(h.shared_gauss, h.uar)[0]),
                          "mae": float((h.shared_gauss - h.uar).abs().mean()), "bias": float((h.shared_gauss - h.uar).mean()), "n": len(h)})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "e28t4_grid_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(4).to_string(index=False))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--grid":
        return grid_check()
    R = rows()
    R.to_csv(OUT / "e28t4_rows.csv", index=False)
    tests, summ = [], []
    for kind in KINDS:
        r = R[R.kind == kind]
        for clf in ("src", "merged", "ht"):
            g = r[r.clf == clf]
            for br, name, thr_r, thr_e in (("indep", "T4-1", 0.9, 0.03), ("gauss", "T4-3", 0.85, 0.04),
                                           ("shared", "追加（共通の雑音）", 0.9, 0.03), ("shared_gauss", "追加（共通の雑音，正規）", 0.85, 0.04)):
                rho = float(stats.spearmanr(g[br], g.uar)[0])
                mae = float((g[br] - g.uar).abs().mean())
                tests.append({"kind": kind, "test": name, "clf": clf, "bridge": br, "spearman": rho, "mae": mae,
                              "bias": float((g[br] - g.uar).mean()), "holds": rho >= thr_r and mae < thr_e, "n": len(g)})
            summ.append({"kind": kind, "clf": clf, **{c: g[c].mean() for c in ("uar", "indep", "gauss", "union", "shared", "shared_gauss")}})
        P = r.groupby(["target", "source", "clf"])[["uar", "indep", "gauss", "shared", "shared_gauss"]].mean().unstack("clf")
        for clf in ("merged", "ht"):
            da = P[("uar", clf)] - P[("uar", "src")]
            for br in ("indep", "gauss", "shared", "shared_gauss"):
                dp = P[(br, clf)] - P[(br, "src")]
                rho = float(stats.spearmanr(dp, da)[0])
                name = {"indep": "T4-2", "gauss": "T4-2 参考（正規）", "shared": "追加 Δ（共通の雑音）", "shared_gauss": "追加 Δ（共通の雑音，正規）"}[br]
                tests.append({"kind": kind, "test": name, "clf": f"{clf} − src",
                              "bridge": br, "spearman": rho, "mae": float((dp - da).abs().mean()),
                              "bias": float((dp - da).mean()), "holds": rho >= 0.8 if br == "indep" else np.nan, "n": len(da)})
    T, S = pd.DataFrame(tests), pd.DataFrame(summ)
    T.to_csv(OUT / "e28t4_tests.csv", index=False)
    S.to_csv(OUT / "e28t4_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    print(S.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
