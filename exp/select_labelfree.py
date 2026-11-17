"""【探索的】E34（GPU）：ラベル無しで選ぶ herding（現実の手順）。下書きの予測で群を作り，群の中で特徴だけで herding する。

規則は `docs/exploratory-analyses.md` E34（設計と予言 L1・L2 は計算の前に書いた）。評価・候補・中心・下書き・学習器は E33
（`exp/select_kshot.py`）と同じ（P4 の既定）。候補のラベルは選ぶときに使わず，予算は 4K 本，選んだ発話にだけラベルが付く。
学習器の感情の釣り合いの重みは，選んだ中にある感情だけで作る（4 つそろえば E31 と同じ重み）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/select_labelfree.py --kind audio_wavlm [--target …] [--reps …]

出力：`results/select_lf_<kind>/<ターゲット>.csv`（method=HT_LF, source=選び方, rep, K, uar），`.npz`（テストの点数），
`<ターゲット>_sel.csv`（選んだ発話の感情ごとの本数，欠けた感情の数，群の純度），`provenance.json`，`provenance_log.jsonl`。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import compare_b1 as CB  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_herding import herd  # noqa: E402

STRATS = ("herding_pred", "random_pred", "herding_all", "random_true")


def present_weights(y: torch.Tensor) -> torch.Tensor:
    """感情の釣り合いの重みを，そこにある感情だけで作る（4 つそろえば `logreg.balanced_weights` と同じ）。"""
    cnt = torch.bincount(y, minlength=logreg.N_CLASSES).to(logreg.DTYPE)
    n_present = int((cnt > 0).sum())
    return (len(y) / (n_present * cnt[y]))


def ht_block_present(ctx, rs, src, C_s, add):
    """`compare_b1._ht_block` の帰納の中心合わせ（cen = 1），C′ = 0.01，C_b = 8，交差検証なしの版で，重みだけ present_weights にする。"""
    tgt = ctx.tgt
    for part in CB.chunks(ctx, rs):
        blocks, ys, jobs, masks, layout, off = [], [], [], [], [], 0
        for r in part:
            used_r = np.unique(np.concatenate([ctx.idx[(r, K)] for K in ctx.ks]))
            blocks.append(M.rows(tgt.X, used_r) - ctx.ct[r]); ys.append(tgt.y[used_r])
            layout.append((r, used_r, off)); off += len(used_r)
        for r, used_r, o in layout:
            jj, mm = CB.jobs_for(ctx, [r], {int(i): o + k for k, i in enumerate(used_r)}, off, False)
            jobs += jj; masks += mm
        X, yX = torch.cat(blocks), M.labels(np.concatenate(ys))
        weights = []
        for m in masks:
            w = torch.zeros(len(yX), dtype=logreg.DTYPE, device=CB.DEV)
            mt = torch.as_tensor(m, device=CB.DEV)
            w[mt] = present_weights(yX[mt])
            weights.append(w)
        fits = logreg.fit_batch(X, yX, masks, (CB.DEF_CP,), anchor=src, anchor_c=C_s, stats=(src.mu, src.sd),
                                Cb=[CB.DEF_CB] * len(masks), weights=weights)
        CB.collect(ctx, jobs, fits, (CB.DEF_CP,), lambda r, ii: M.rows(tgt.X, ii) - ctx.ct[r], None, add,
                   {"cen": 1, "cb": f"{CB.DEF_CB:g}"})


def fill(groups: list[np.ndarray], picks: list[list[int]], K: int, more) -> list[int]:
    """群ごとに K 本．足りない群の分は，余りのある群で more(群, 既に選んだ) を続けて補う。"""
    out = [p for ps in picks for p in ps]
    need = 4 * K - len(out)
    g = 0
    while need > 0:
        cand = [j for j in range(4) if len(groups[j]) > len(picks[j])]
        if not cand:
            break
        j = cand[g % len(cand)]
        nxt = more(j, picks[j])
        picks[j].append(nxt); out.append(nxt); need -= 1; g += 1
    return out


def run_target(kind, tgt, data, reps, ks, out):
    ctx = CB.Ctx(tgt, reps, ks, center="ind")
    pool_src = [s for s in data if s != tgt.name and GROUP_SPEC[s][0] != GROUP_SPEC[tgt.name][0]]
    rows, saved, sel_rows = [], {}, []
    for r in reps:
        saved[f"test_idx_r{r}"] = np.flatnonzero(ctx.te[r])
        saved[f"y_test_r{r}"] = tgt.y[ctx.te[r]]
    t0 = time.perf_counter()
    Xp = torch.cat([data[s].X for s in pool_src])
    yp = M.labels(np.concatenate([data[s].y for s in pool_src]))
    C_p = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{tgt.name}.csv", dtype={"K": str}).C.iloc[0])
    src_pc = logreg.fit(Xp - Xp.mean(0), yp, (C_p,))
    del Xp
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    idx = {st: {} for st in STRATS}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = tgt.y[pool]
        Xc = M.rows(tgt.X, pool) - ctx.ct[r]
        pred = src_pc.logits(Xc, C_p).argmax(1).cpu().numpy()
        Xt = (Xc - src_pc.mu) / src_pc.sd
        groups = [np.flatnonzero(pred == g) for g in range(4)]
        pos_of = {int(i): j for j, i in enumerate(pool)}
        for K in ks:
            kk = int(K)
            rng = np.random.default_rng(zlib.crc32(f"lf|{kind}|{tgt.name}|{r}|{K}".encode()))
            # herding_pred：群の中で特徴だけの herding（順序つきに全部並べておき，補うときは続きを取る）
            orders = [list(G[herd(Xt[torch.as_tensor(G, device=CB.DEV)], min(len(G), 4 * kk))]) if len(G) else [] for G in groups]
            hp = fill(groups, [o[:kk] for o in orders], kk, lambda j, ps: orders[j][len(ps)])
            perm = [list(rng.permutation(G)) for G in groups]
            rp = fill(groups, [p[:kk] for p in perm], kk, lambda j, ps: perm[j][len(ps)])
            ha = list(np.arange(len(pool))[herd(Xt, 4 * kk)])
            rt = [pos_of[int(i)] for i in ctx.idx[(r, K)]]
            for st, pp in (("herding_pred", hp), ("random_pred", rp), ("herding_all", ha), ("random_true", rt)):
                pp = np.array(pp)
                if len(pp) != 4 * kk or len(np.unique(pp)) != len(pp):
                    raise ValueError(f"選んだ本数がおかしい: {st} r{r} K{K} {len(pp)}")
                idx[st][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                sel_rows.append({"target": tgt.name, "strategy": st, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                                 "min_count": int(cnt.min()), "max_count": int(cnt.max()),
                                 "purity": float((yq[pp] == pred[pp]).mean()),
                                 **{f"n_{c}": int(cnt[c]) for c in range(4)}})
    t_sel = time.perf_counter() - t1

    t2 = time.perf_counter()
    for st in STRATS:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[st]

        def add(r, K, c, u, cv, L, cen=0, cb="", st=st):
            rows.append({"target": tgt.name, "source": st, "method": "HT_LF", "cen": cen, "cp": c, "cb": cb,
                         "rep": r, "K": K, "uar": u})
            saved[f"{st}_K{K}_r{r}"] = L.astype(np.float32)
        ht_block_present(sub, reps, src_pc, C_p, add)
    print(f"  {kind} {tgt.name}：準備 {t_prep:.1f} 秒，選ぶ {t_sel:.1f} 秒，仮説転移 {time.perf_counter() - t2:.1f} 秒", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    pd.DataFrame(rows).to_csv(out / f"{tgt.name}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(out / f"{tgt.name}_sel.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"select_lf_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.npz").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/select_labelfree.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "strategies": STRATS,
            "cp": CB.DEF_CP, "cb": CB.DEF_CB}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, data[t], data, args.reps, args.ks, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
