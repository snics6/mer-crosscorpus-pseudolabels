"""【探索的】E39（GPU）：適応の 3 つの部分に対応する 5 つの数（γ, ρ_s, C_b, C′, ρ_M）を，ソースコーパスを仮のターゲットにしてまとめてメタ学習する。

規則は `docs/exploratory-analyses.md` E39（設計と予言 F1・F2 は計算の前に書いた）。評価・候補・中心・K 本・学習器の型は E31 の P4 と同じ。
下書き：E35 の追加の段 2（`combine_shrink.py`）と同じ閉じた形の向き W_s(ρ_s) をラベル無しの尺度でそろえ，重み ∝ N_s^γ で足して P4 の下書きの
尺度に合わせる。罰則の計量：E38（`ht_metric.py`）と同じく，ラベル無しのターゲットの散らばりで M_ρM を作り座標を直す。C_b と C′ は 1 回の
計算に束ねる。ターゲット（role=target，反復 0〜9）と，T のソースコーパス c を仮のターゲットにした分（role=pseudo．T と c のコーパスの群を
除くソースで下書きを作り，P4 型の合併の C = 0.01，反復 0〜4）を，243 通りすべてで解く。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/meta5.py --kind audio_wavlm [--target …]
    ./.venv/bin/python exp/meta5.py --analyze

出力：`results/meta5_<kind>/<ターゲット>.csv`（role, meta_for, eval, gamma, rho_s, rho_M, cb, cp, rep, K, uar），`provenance.json`，
`provenance_log.jsonl`。判定は `results/explore_relation/e39_targets.csv`・`e39_tests.csv`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import combine_drafts as CD  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402

GRIDS = {                                              # base：E39．ext：E39 の追加（端に張り付いた候補を広げる）
    "base": {"gamma": (-0.5, 0.0, 0.5), "rho_s": (0.3, 0.5, 0.7), "cb": (2.0, 8.0, None), "cp": (0.01, 0.03, 0.1),
             "rho_M": (1.0, 0.97, 0.9)},
    "ext": {"gamma": (-0.5, 0.0, 0.5), "rho_s": (0.1, 0.3, 0.5, 0.7), "cb": (0.5, 2.0, 8.0, None), "cp": (0.003, 0.01, 0.03, 0.1),
            "rho_M": (1.0, 0.97, 0.9)},
}
OUT_NAME = {"base": "meta5", "ext": "meta5x"}
GAMMAS, RHOS_S, CBS5, CPS5, RHOS_M = (GRIDS["base"][k] for k in ("gamma", "rho_s", "cb", "cp", "rho_M"))


def set_grid(name: str) -> None:
    global GAMMAS, RHOS_S, CBS5, CPS5, RHOS_M
    GAMMAS, RHOS_S, CBS5, CPS5, RHOS_M = (GRIDS[name][k] for k in ("gamma", "rho_s", "cb", "cp", "rho_M"))
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_PSEUDO = 0.01
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"
KEYS = ["gamma", "rho_s", "cb", "cp", "rho_M"]


def cbname(cb) -> str:
    return "free" if cb is None else f"{cb:g}"


def pool_info(data, pool, C, cache):
    """ソースの組ごとに：P4 型の合併の下書き（尺度と標準化の基準），閉じた形の向き（ρ_s ごと．中心合わせの座標の 1 次関数），発話数。"""
    key = (tuple(sorted(pool)), C)
    if key in cache:
        return cache[key]
    Xp = torch.cat([data[s].X for s in pool])
    yp = M.labels(np.concatenate([data[s].y for s in pool]))
    src_pc = logreg.fit(Xp - Xp.mean(0), yp, (C,))
    mu_p, sd_p = Xp.mean(0), Xp.std(0, unbiased=False)
    sd_p = torch.where(sd_p > 1e-12, sd_p, torch.ones_like(sd_p))
    del Xp
    lin = {}
    for s in pool:
        Z = (data[s].X - mu_p) / sd_p
        yt = torch.as_tensor(np.array(data[s].y), device=Z.device)
        m = torch.stack([Z[yt == k].mean(0) for k in range(4)])
        R = Z - m[yt]
        lam, V = torch.linalg.eigh(R.T @ R / (len(yt) - 4))
        lam = lam.clamp(min=0.0)
        mt = m - Z.mean(0)
        P = V.T @ mt.T
        for rho in RHOS_S:
            W = (V @ (P / ((1 - rho) * lam + rho * lam.mean())[:, None])).T
            lin[(s, rho)] = (W / sd_p, -0.5 * (mt * W).sum(1))
        del Z, R, V
    out = (src_pc, lin, {s: len(data[s].y) for s in pool})
    cache[key] = out
    return out


def evaluate(role, meta_for, ev, pool, data, C, reps, ks, rows, cache):
    src_pc, lin, n_of = pool_info(data, pool, C, cache)
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    dev, dt = src_pc.mu.device, src_pc.mu.dtype
    zero = torch.zeros(ev.X.shape[1], dtype=dt, device=dev)
    one = torch.ones(ev.X.shape[1], dtype=dt, device=dev)
    for r in reps:
        nontest = np.flatnonzero(~ctx.te[r])
        Xu = M.rows(ev.X, nontest) - ctx.ct[r]                                 # ラベル無しの発話（中心合わせ後の生の座標）
        ref_scale = CD.pair_scale(src_pc.logits(Xu, C))
        comps = {}
        for (s, rho), (A, a0) in lin.items():
            sc = CD.pair_scale(Xu @ A.T + a0)
            comps[(s, rho)] = (A / sc, a0 / sc)
        Zp = (Xu - src_pc.mu) / src_pc.sd
        Zp = Zp - Zp.mean(0)
        lam, V = torch.linalg.eigh(Zp.T @ Zp / len(nontest))
        lam = lam.clamp(min=0.0)
        tau = lam.mean()
        del Zp
        used = np.unique(np.concatenate([ctx.idx[(r, K)] for K in ctx.ks]))
        Xs_used = (M.rows(ev.X, used) - ctx.ct[r] - src_pc.mu) / src_pc.sd
        yu = M.labels(ev.y[used])
        jobs, masks = CB.jobs_for(ctx, [r], {int(i): k for k, i in enumerate(used)}, len(used), False)
        mats = {}
        for rho_m in RHOS_M:
            e_ = (1 - rho_m) * lam / tau + rho_m
            mats[rho_m] = ((V * e_.rsqrt()) @ V.T, (V * e_.sqrt()) @ V.T)
        for gamma in GAMMAS:
            w = {s: float(n_of[s]) ** gamma for s in pool}
            tot = sum(w.values())
            for rho_s in RHOS_S:
                A = sum(w[s] / tot * comps[(s, rho_s)][0] for s in pool)
                a0 = sum(w[s] / tot * comps[(s, rho_s)][1] for s in pool)
                g = ref_scale / CD.pair_scale(Xu @ A.T + a0)
                A, a0 = g * A, g * a0
                W_std, b_std = A * src_pc.sd, a0 + src_pc.mu @ A.T                # 下書きを標準化した座標で
                for rho_m, (Am, Mh) in mats.items():
                    anchor = logreg.Fitted(zero, one, {C: W_std @ Mh}, {C: b_std}, {C: 0}, {C: 0.0})
                    fits = logreg.fit_batch(Xs_used @ Am, yu, masks * len(CBS5), CPS5, anchor=anchor, anchor_c=C,
                                            stats=(zero, one), Cb=[cb for cb in CBS5 for _ in masks])
                    X_of = lambda r_, ii, Am=Am, r=r: ((M.rows(ev.X, ii) - ctx.ct[r] - src_pc.mu) / src_pc.sd) @ Am  # noqa: E731
                    for k, cb in enumerate(CBS5):
                        def emit(rr, K, c, u, cv, L, gamma=gamma, rho_s=rho_s, rho_m=rho_m, cb=cb, **kw):
                            rows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "gamma": gamma, "rho_s": rho_s,
                                         "rho_M": rho_m, "cb": cbname(cb), "cp": c, "rep": rr, "K": K, "uar": u})
                        CB.collect(ctx, jobs, fits[k * len(masks):(k + 1) * len(masks)], CPS5, X_of, None, emit, {})


def run_target(kind, T, data, reps, ks, out, cache, pseudo=True, pseudo_only=False):
    """pseudo_only（E62．第 2 エンコーダ）：ターゲットの行を解かず，仮のターゲットだけを解く（5 つの値を選ぶのに要るのは仮のターゲットの行だけ．
    ターゲットの行に要る合併の C（multisource_scores）は第 2 エンコーダには無い）。"""
    t0 = time.perf_counter()
    rows = []
    corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731
    pool_T = [s for s in data if corp(s) != corp(T)]
    if not pseudo_only:
        C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
        evaluate("target", T, data[T], pool_T, data, C_T, reps, ks, rows, cache)
    t1 = time.perf_counter()
    for c in (pool_T if pseudo else []):                   # --no-pseudo：反復を足すときは仮のターゲット（反復 0〜4．前回と同じ）を解き直さない
        pool_c = [s for s in pool_T if corp(s) != corp(c)]
        evaluate("pseudo", T, data[c], pool_c, data, C_PSEUDO, PSEUDO_REPS, ks, rows, cache)
    print(f"  {kind} {T}：ターゲット {t1 - t0:.0f} 秒，仮のターゲット {len(pool_T)} 個 {time.perf_counter() - t1:.0f} 秒", flush=True)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def pick(ps: pd.Series) -> tuple:
    """仮のターゲットの UAR が最大の組。最大と 1e-12 以内で並ぶ組の中では γ = 0（等しい重み）を選ぶ（言葉系の 1 ソースの下書きで γ が同点になるため．
    並ぶ組が 1 つなら pandas の idxmax と同じ）。"""
    top = ps[ps >= ps.max() - 1e-12]
    if len(top) > 1:
        g0 = [k for k in top.index if k[0] == 0.0]
        if g0:
            return g0[0]
    return top.index[0]


def analyze_pseudo(kinds, suffix: str = "", results=None) -> int:
    """E62（第 2 エンコーダ）：仮のターゲットの行だけから 5 つの値を選び，`e62{suffix}_targets.csv`（kind, target, γ, ρ_s, C_b, C′, ρ_M, 仮のターゲットの UAR）に書く。
    `geom_e0.settings` が読む（ターゲットの行は無いので E39 の表の meta・P4 などの列は無い）。"""
    results = results or ROOT / "results"
    rows = []
    for kind in kinds:
        for f in sorted((results / f"meta5_{kind}").glob("*.csv")):
            a = pd.read_csv(f, dtype={"K": str, "cb": str})
            ps = a[a.role == "pseudo"].groupby(KEYS).uar.mean()
            star = pick(ps)
            rows.append({"kind": kind, "target": f.stem, **dict(zip(KEYS, star)), "pseudo_uar": float(ps.loc[star]) * 100,
                         "pseudo_targets": int(a[a.role == "pseudo"]["eval"].nunique()), "pseudo_reps": int(a[a.role == "pseudo"].rep.nunique())})
    D = pd.DataFrame(rows)
    D.to_csv(OUT_REL / f"e62{suffix}_targets.csv", index=False)
    print(D.to_string(index=False))
    return 0


def analyze(grid: str = "base", kinds=MAIN, suffix: str = "") -> int:
    tabs, checks = [], []
    sfx = ("" if grid == "base" else "x") + suffix
    rho_tab = pd.read_csv(OUT_REL / "e35s_targets.csv").set_index(["kind", "target"])
    for kind in kinds:
        fs = sorted((ROOT / "results" / f"{OUT_NAME[grid]}_{kind}").glob("*.csv"))
        if not fs:
            continue
        for f in fs:
            T = f.stem
            a = pd.read_csv(f, dtype={"K": str, "cb": str})
            ps = a[a.role == "pseudo"].groupby(KEYS).uar.mean()
            star = pick(ps)
            tg = a[(a.role == "target") & a.rep.isin(EC.REPS)].groupby(KEYS + ["K"]).uar.mean() * 100
            c = pd.read_csv(ROOT / "results" / f"compare_{kind}" / f"{T}.csv", dtype={"K": str, "cb": str})
            p4 = c[(c.method == "HT_ALL") & (c.cen == 1) & (c.cp == 0.01) & (c.cb == "8") & c.rep.isin(EC.REPS)].groupby("K").uar.mean() * 100
            e = pd.read_csv(ROOT / "results" / f"combine_{kind}" / f"{T}.csv", dtype={"K": str})
            d1 = e[(e.source == "equal") & e.rep.isin(EC.REPS)].groupby("K").uar.mean() * 100
            fsh = ROOT / "results" / f"combine_shrink_{kind}" / f"{T}.csv"     # 言葉系には 1 つの数の版（E35 の追加）が無い
            sh = pd.read_csv(fsh, dtype={"K": str}) if fsh.exists() else pd.DataFrame(columns=["source", "rep", "K", "uar"])
            one = sh[(sh.source == "star") & sh.rep.isin(EC.REPS)]
            rs = float(rho_tab.loc[(kind, T), "rho_star"]) if (kind, T) in rho_tab.index else np.nan
            if grid == "ext":                                             # E39 と重なる組は E39 と一致するはず
                b = pd.read_csv(ROOT / "results" / f"meta5_{kind}" / f"{T}.csv", dtype={"K": str, "cb": str})
                b = b[b.role == "target"]
                m = a[a.role == "target"].merge(b, on=["eval"] + KEYS + ["rep", "K"], suffixes=("_x", "_b"))
                checks.append({"kind": kind, "target": T, "rows": len(m), "max_diff": float((m.uar_x - m.uar_b).abs().max())})
            elif rs in RHOS_S:                                            # (0, ρ*, 8, 0.01, 1) は 1 つの数の版と一致するはず
                mine = a[(a.role == "target") & (a.gamma == 0.0) & (a.rho_s == rs) & (a.cb == "8") & (a.cp == 0.01) & (a.rho_M == 1.0)]
                m = mine.merge(one[["rep", "K", "uar"]], on=["rep", "K"], suffixes=("_m", "_s"))
                checks.append({"kind": kind, "target": T, "rows": len(m), "max_diff": float((m.uar_m - m.uar_s).abs().max())})
            for K in ("1", "5", "10", "25"):
                tk = tg.xs(K, level="K")
                row = {"kind": kind, "target": T, "K": K, **dict(zip(KEYS, star)), "meta": float(tk.loc[star]),
                       "oracle": float(tk.max()), "P4": float(p4[K]), "D1": float(d1[K]),
                       "one": float(one[one.K == K].uar.mean() * 100) if len(one) else np.nan}
                tabs.append(row)
    D = pd.DataFrame(tabs)
    rows = []
    for (kind, K), g in D.groupby(["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g), "meta": float(g.meta.mean())}
        for b in ("D1", "one", "P4", "oracle"):
            diff = g.meta - g[b]
            row[f"meta-{b}"] = float(diff.mean())
            row[f"meta>{b}"] = int((diff > 0).sum())
        rows.append(row)
    R = pd.DataFrame(rows)
    for kind in ("audio_wavlm", "audio_hubert"):
        m = (R.kind == kind) & R.K.isin(["5", "10"])
        R.loc[R.kind == kind, "F1"] = bool((R[m]["meta-D1"] > 0).all()) if m.sum() == 2 else np.nan
        R.loc[R.kind == kind, "F2"] = bool((R[m]["meta-one"] > 0).all()) if m.sum() == 2 else np.nan
    Cks = pd.DataFrame(checks)
    if grid == "ext":                                                    # X1：広げた候補の版 − E39 の版
        base = pd.read_csv(OUT_REL / "e39_targets.csv", dtype={"K": str})[["kind", "target", "K", "meta", "oracle"]]
        J = D.merge(base, on=["kind", "target", "K"], suffixes=("", "_base"))
        for (kind, K), g in J.groupby(["kind", "K"]):
            m = (R.kind == kind) & (R.K == K)
            R.loc[m, "meta-base"] = float((g.meta - g.meta_base).mean())
            R.loc[m, "oracle-base_oracle"] = float((g.oracle - g.oracle_base).mean())
        for kind in ("audio_wavlm", "audio_hubert"):
            m = (R.kind == kind) & R.K.isin(["5", "10"])
            R.loc[R.kind == kind, "X1"] = bool((R[m]["meta-base"] >= -0.2).all()) if m.sum() == 2 else np.nan
    D.to_csv(OUT_REL / f"e39{sfx}_targets.csv", index=False)
    R.to_csv(OUT_REL / f"e39{sfx}_tests.csv", index=False)
    Cks.to_csv(OUT_REL / f"e39{sfx}_checks.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print("照合（base は 1 つの数の版，ext は E39 と重なる組）：", Cks.max_diff.max() if len(Cks) else "なし", f"（{len(Cks)} ターゲット）")
    for k in KEYS:
        print(k, D[D.K == "5"].groupby("kind")[k].value_counts().unstack(fill_value=0).to_dict("index"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--no-pseudo", action="store_true", help="仮のターゲットを解かない（ターゲットの反復を足すとき）")
    ap.add_argument("--grid", choices=list(GRIDS), default="base")
    ap.add_argument("--kinds", nargs="+", default=MAIN, help="--analyze の種類（言葉系は --suffix _text で別のファイルに書く）")
    ap.add_argument("--suffix", default="", help="--analyze の出力名の接尾辞")
    ap.add_argument("--pseudo-only", action="store_true", help="E62：仮のターゲットだけを解く（ターゲットの行は解かない）")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, help="試走用．既定は 0〜4")
    ap.add_argument("--analyze-pseudo", action="store_true", help="E62：仮のターゲットの行だけから 5 つの値を選び e62_targets.csv に書く（--kinds，--suffix，--results-dir）")
    ap.add_argument("--results-dir", type=Path, help="試走用．--analyze-pseudo が meta5_* を読む場所")
    args = ap.parse_args()
    set_grid(args.grid)
    global PSEUDO_REPS
    if args.pseudo_reps is not None:
        PSEUDO_REPS = args.pseudo_reps
    if args.analyze_pseudo:
        return analyze_pseudo(args.kinds, args.suffix, args.results_dir)
    if args.analyze:
        return analyze(args.grid, args.kinds, args.suffix)
    out = args.out_dir or ROOT / "results" / f"{OUT_NAME[args.grid]}_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/meta5.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "pseudo_reps": [] if args.no_pseudo else PSEUDO_REPS, "pseudo_only": args.pseudo_only,
            "grid_name": args.grid, "grid": {"gamma": GAMMAS, "rho_s": RHOS_S, "cb": [cbname(c) for c in CBS5], "cp": CPS5, "rho_M": RHOS_M},
            "C_pseudo": C_PSEUDO, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    cache: dict = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, args.ks, out, cache, pseudo=not args.no_pseudo, pseudo_only=args.pseudo_only)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
