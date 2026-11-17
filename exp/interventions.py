"""E25 の手の比較（GPU）。8 つの手を，地図と同じテスト・K 本で，C ＝ 同じ反復のソースのみの C に固定して比べる。

手と予言は `docs/exploratory-analyses.md` E25。

    1 ふつうの合併（C を固定した地図の値をそのまま使う）
    2 中心合わせ＋ソースのみ          3 中心合わせ＋合併
    4 ソースのみ＋切片直し（D1）      5 中心合わせ＋ソースのみ＋切片直し
    6 ターゲット専用の切片つき合併（特徴に「ターゲットの発話か」の列を足す）
    7 仮説転移（ターゲットの K 本だけで，ソースのみの係数からの距離に罰則．C ＝ ソースのみの C）
    8 CORAL＋合併（ソースの散らばりをターゲットに合わせる．平均は動かさない．共分散 ＋ 単位行列）

中心合わせは E12（`exp/mech.py`）と同じく，ターゲットの全発話の平均と，使うソースの行の平均をそれぞれ引く。
確かめとして，学び直したソースのみの UAR を地図の値と並べて残す。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/interventions.py --kind audio_wavlm [--reps 0 … 9]

出力：`results/interventions_<kind>/<ターゲット>.csv`（行＝ソース × K × 反復 × 手）。終わったターゲットは飛ばす。
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
import ablate  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import shift  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402

KS = ("1", "5", "10")


def centered(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(0)


def coral(Xs: torch.Tensor, Xt_all: torch.Tensor) -> torch.Tensor:
    """ソースの散らばりをターゲットの散らばりに合わせる（ソースの標準化した空間で．平均は動かさない）。"""
    mu, sd = Xs.mean(0), Xs.std(0, correction=0)
    sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    Zs, Zt = (Xs - mu) / sd, (Xt_all - mu) / sd
    eye = torch.eye(Zs.shape[1], dtype=Zs.dtype, device=Zs.device)
    def mpow(C, p):
        w, V = torch.linalg.eigh(C)
        return (V * w.clamp_min(1e-8) ** p) @ V.T
    Cs = torch.cov(Zs.T) + eye
    Ct = torch.cov(Zt.T) + eye
    A = mpow(Cs, -0.5) @ mpow(Ct, 0.5)
    ms = Zs.mean(0)
    return ((Zs - ms) @ A + ms) * sd + mu


def map_rows(kind: str, target: str) -> dict:
    d = pd.read_csv(map_dir(kind) / f"{target}.csv", comment="#", dtype={"K": str, "source": str},
                    usecols=["method", "source", "K", "rep", "C", "uar"])
    d["K"], d["source"] = d.K.fillna(""), d.source.fillna("")
    return {(r.method, r.source, r.K, int(r.rep)): (float(r.C), float(r.uar)) for r in d.itertuples()}


def uar_of(L: torch.Tensor | np.ndarray, y: np.ndarray) -> float:
    L = L.cpu().numpy() if isinstance(L, torch.Tensor) else L
    return shift.uar(y, L.argmax(1))


def run_target(kind: str, tgt: M.Group, sources: list[M.Group], reps, out: Path) -> pd.DataFrame:
    """同じ（ターゲット，ソース）で使うソースの行が同じ反復を束ね，手ごとに全反復 × K を一度に解く。"""
    ref = map_rows(kind, tgt.name)
    Xt, Xt_c = tgt.X, centered(tgt.X)
    nt = len(tgt.y)
    yt = M.labels(tgt.y)
    split = {}
    for r in reps:
        te = M.test_mask(tgt, r)
        order = M.kshot_order(tgt, np.flatnonzero(~te), r)
        split[r] = (te, {K: M.take_k(order, K) for K in KS}, np.unique(tgt.person[te]))
    rows = []
    for s in sources:
        t0 = time.perf_counter()
        groups = {}
        for r in reps:                                        # ソースの行と C が同じ反復を束ねる
            keep = ~np.isin(s.person, split[r][2])
            C = ref[("src_only", s.name, "", r)][0]
            groups.setdefault((keep.tobytes(), C), []).append(r)
        for (kb, C), rs in groups.items():
            keep = np.frombuffer(kb, dtype=bool)
            Xs = M.rows(s.X, keep)
            ys_np = s.y[keep]
            ys = M.labels(ys_np)
            Xs_c, Xs_coral = centered(Xs), coral(Xs, Xt)
            src, src_c = logreg.fit(Xs, ys, (C,)), logreg.fit(Xs_c, ys, (C,))
            ns = len(ys_np)
            y_all = M.labels(np.r_[tgt.y, ys_np])
            jobs = [(r, K) for r in rs for K in KS]
            def mmask(r, K):
                m = np.r_[np.zeros(nt, bool), np.ones(ns, bool)]
                m[split[r][1][K]] = True
                return m
            def tmask(r, K):
                m = np.zeros(nt, bool)
                m[split[r][1][K]] = True
                return m
            fit3 = logreg.fit_batch(torch.cat([Xt_c, Xs_c]), y_all, [mmask(r, K) for r, K in jobs], (C,))
            one_t = torch.ones(nt, 1, dtype=Xt.dtype, device=Xt.device)
            zero_s = torch.zeros(ns, 1, dtype=Xt.dtype, device=Xt.device)
            Xa = torch.cat([torch.cat([Xt, one_t], 1), torch.cat([Xs, zero_s], 1)])
            fit6 = logreg.fit_batch(Xa, y_all, [mmask(r, K) for r, K in jobs], (C,))
            fit7 = logreg.fit_batch(Xt, yt, [tmask(r, K) for r, K in jobs], (C,), anchor=src, anchor_c=C)
            fit8 = logreg.fit_batch(torch.cat([Xt, Xs_coral]), y_all, [mmask(r, K) for r, K in jobs], (C,))
            fits = {j: (f3, f6, f7, f8) for j, f3, f6, f7, f8 in zip(jobs, fit3, fit6, fit7, fit8)}
            for r in rs:
                te, idx, _ = split[r]
                y_te = tgt.y[te]
                Xte, Xte_c = M.rows(Xt, te), M.rows(Xt_c, te)
                Xte_a = torch.cat([Xte, torch.ones(len(y_te), 1, dtype=Xt.dtype, device=Xt.device)], 1)
                base = {"target": tgt.name, "source": s.name, "rep": r, "C": C}
                L_src, L_src_c = src.logits(Xte, C).cpu().numpy(), src_c.logits(Xte_c, C).cpu().numpy()
                rows.append({**base, "K": "", "method": "0 ソースのみ（学び直し）", "uar": uar_of(L_src, y_te),
                             "uar_map": ref[("src_only", s.name, "", r)][1]})
                rows.append({**base, "K": "", "method": "2 中心合わせ＋ソースのみ", "uar": uar_of(L_src_c, y_te)})
                for K in KS:
                    i = idx[K]
                    f3, f6, f7, f8 = fits[(r, K)]
                    b = ablate.fit_class_bias(src.logits(M.rows(Xt, i), C).cpu().numpy(), tgt.y[i])
                    bc = ablate.fit_class_bias(src_c.logits(M.rows(Xt_c, i), C).cpu().numpy(), tgt.y[i])
                    for name, u in (("1 ふつうの合併", ref[("merged", s.name, K, r)][1]),
                                    ("3 中心合わせ＋合併", uar_of(f3.logits(Xte_c, C), y_te)),
                                    ("4 ソースのみ＋切片直し", uar_of(L_src + b, y_te)),
                                    ("5 中心合わせ＋ソースのみ＋切片直し", uar_of(L_src_c + bc, y_te)),
                                    ("6 ターゲット専用の切片つき合併", uar_of(f6.logits(Xte_a, C), y_te)),
                                    ("7 仮説転移", uar_of(f7.logits(Xte, C), y_te)),
                                    ("8 CORAL＋合併", uar_of(f8.logits(Xte, C), y_te))):
                        rows.append({**base, "K": K, "method": name, "uar": u})
        print(f"  {kind} {tgt.name} ← {s.name}：{time.perf_counter() - t0:.1f} 秒（束 {len(groups)}）", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--source", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"interventions_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/interventions.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "map_dir": str(map_dir(args.kind))}
    groups = groups_for(args.kind)
    data = {g: M.load(g, args.kind) for g in groups}
    for t in args.target or groups:
        path = out / f"{t}.csv"
        if path.exists():
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t0 = time.perf_counter()
        d = run_target(args.kind, data[t], [data[s] for s in (args.source or groups) if s != t], args.reps, out)
        with open(path, "w") as f:
            f.write("# " + json.dumps(prov, ensure_ascii=False) + "\n")
            d.to_csv(f, index=False)
        chk = d.dropna(subset=["uar_map"])
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒．ソースのみの学び直しと地図の差 最大 "
              f"{(chk.uar - chk.uar_map).abs().max():.2e}", flush=True)


if __name__ == "__main__":
    main()
