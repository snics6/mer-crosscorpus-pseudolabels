"""【探索的】E33 の追加（GPU）：勾配を合わせる選び方（herding）の仮説転移と，全選び方の K 本のずれの量（予言 G1〜G3）。

規則は `docs/exploratory-analyses.md` E33「勾配を合わせる選び方」（予言は計算の前に書いた）。テスト・候補・中心・下書き・学習器は
E33（`exp/select_kshot.py`）と同じ。感情 k ごとに，候補のうち真の感情が k の発話について φ = (a·r, r ⊗ x̃)（r は下書きでの残差，
x̃ は下書きの統計で標準化した中心合わせ後の特徴，a² = tr Var(r ⊗ x̃)/tr Var(r)）を作り，K 本の φ の平均が候補の平均に近づくように
1 本ずつ足す。仮説転移は herding と random（E33 と一致するかの確かめ）だけ解き，ずれの量は E33 の 7 通りと herding の全部で測る。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/select_herding.py --kind audio_wavlm [--target …] [--reps …]
        [--feat-only]

`--feat-only`（E33 の追加 2）：残差を使わず x̃ だけで herding する herding_feat と random を解き，random・herding・herding_feat の
勾配のずれの量と特徴のずれの量を記録する（出力は `results/select_herdfeat_<kind>/`）。

出力：`results/select_herd_<kind>/<ターゲット>.csv`（method=HT_SEL, source=herding|random, rep, K, uar），`.npz`（テストの点数），
`<ターゲット>_mis.csv`（選び方 × 反復 × K × 感情のずれの量），`provenance.json`，`provenance_log.jsonl`。
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
import compare_b1 as CB  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from interventions import map_rows  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_kshot import LIN, STRATS, pad_pick, softmax  # noqa: E402

ALL = STRATS + ("herding",)
RUN = ("herding", "random")
ALL_FEAT = ("random", "herding", "herding_feat")
RUN_FEAT = ("herding_feat", "random")


def herd(phi: torch.Tensor, K: int) -> list[int]:
    """‖(Σ_S φ + φ_i)/t − φ̄‖² が最小の行を 1 本ずつ K 回足す。"""
    mean = phi.mean(0)
    norms = (phi * phi).sum(1)
    s = torch.zeros_like(mean)
    taken = torch.zeros(len(phi), dtype=torch.bool, device=phi.device)
    out = []
    for t in range(1, K + 1):
        score = norms + 2 * phi @ (s - t * mean)
        score[taken] = float("inf")
        i = int(torch.argmin(score))
        out.append(i)
        taken[i] = True
        s = s + phi[i]
    return out


def run_target(kind, tgt, data, reps, ks, out, feat=False):
    strats, run = (ALL_FEAT, RUN_FEAT) if feat else (ALL, RUN)
    ctx = CB.Ctx(tgt, reps, ks, center="ind")
    ref = map_rows(kind, tgt.name)
    pool_src = [s for s in data if s != tgt.name and GROUP_SPEC[s][0] != GROUP_SPEC[tgt.name][0]]
    rows, saved, mis = [], {}, []
    for r in reps:
        saved[f"test_idx_r{r}"] = np.flatnonzero(ctx.te[r])
        saved[f"y_test_r{r}"] = tgt.y[ctx.te[r]]
    t0 = time.perf_counter()
    Xp = torch.cat([data[s].X for s in pool_src])
    yp = M.labels(np.concatenate([data[s].y for s in pool_src]))
    C_p = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{tgt.name}.csv", dtype={"K": str}).C.iloc[0])
    src_pc = logreg.fit(Xp - Xp.mean(0), yp, (C_p,))
    del Xp
    d = tgt.X.shape[1]
    zero = torch.zeros(d, dtype=logreg.DTYPE, device=CB.DEV)
    one = torch.ones(d, dtype=logreg.DTYPE, device=CB.DEV)
    lin = []
    for s in pool_src:
        C_s = ref[("src_only", s, "", reps[0])][0]
        if (s, C_s) not in LIN:
            Xs = data[s].X
            LIN[(s, C_s)] = logreg.fit(Xs - Xs.mean(0), M.labels(data[s].y), (C_s,)).moved_to(zero, one, C_s)
        lin.append(LIN[(s, C_s)])
    Ssrc = []
    for s in pool_src:
        Ls = src_pc.logits(data[s].X - data[s].X.mean(0), C_p).cpu().numpy()
        Ssrc.append((Ls - Ls.mean(1, keepdims=True), data[s].y))
    S_by_k = {k: np.concatenate([L[y == k] for L, y in Ssrc]) for k in range(4)}
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    idx = {st: {} for st in run}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = tgt.y[pool]
        Xc = M.rows(tgt.X, pool) - ctx.ct[r]
        Ld_t = src_pc.logits(Xc, C_p)
        Ld = Ld_t.cpu().numpy()
        P = softmax(Ld)
        p_true = P[np.arange(len(pool)), yq]
        Z = Ld - Ld.mean(1, keepdims=True)
        p_src = np.stack([softmax((Xc @ A.T + a0).cpu().numpy())[np.arange(len(pool)), yq] for A, a0 in lin])
        disagree = p_src.std(0)
        Xt = (Xc - src_pc.mu) / src_pc.sd                                  # 仮説転移と同じ標準化
        Rt = torch.nn.functional.one_hot(torch.as_tensor(yq, device=CB.DEV), 4).to(Xt.dtype) - torch.softmax(Ld_t, 1)
        pos_of = {int(i): j for j, i in enumerate(pool)}
        phis = {}
        for k in range(4):
            cand = np.flatnonzero(yq == k)
            ci = torch.as_tensor(cand, device=CB.DEV)
            R, X = Rt[ci], Xt[ci]
            RX = (R[:, :, None] * X[:, None, :]).reshape(len(cand), -1)
            vr, vx = float(R.var(0, unbiased=False).sum()), float(RX.var(0, unbiased=False).sum())
            a = np.sqrt(vx / vr) if vr > 0 else 1.0
            phi = torch.cat([a * R, RX], 1)
            phis[k] = (cand, phi, phi.mean(0), float(phi.var(0, unbiased=False).sum()),
                       X, X.mean(0), float(X.var(0, unbiased=False).sum()))
        for K in ks:
            kk = int(K)
            picks = {st: [] for st in strats}
            picks["random"] = [pos_of[int(i)] for i in ctx.idx[(r, K)]]
            for k in range(4):
                cand, phi, mean, tv, X, _, _ = phis[k]
                picks["herding"] += list(cand[herd(phi, kk)])
                if feat:
                    picks["herding_feat"] += list(cand[herd(X, kk)])
                    continue
                o = np.argsort(p_true[cand])
                picks["hard"] += list(cand[o[:kk]])
                picks["easy"] += list(cand[o[::-1][:kk]])
                picks["committee"] += list(cand[np.argsort(disagree[cand])[::-1][:kk]])
                for name, (pp, _, _, _) in pad_pick(Z[cand], S_by_k[k], kk, r).items():
                    picks[name] += list(cand[pp])
            for st, pp in picks.items():
                pp = np.array(pp)
                if len(pp) != 4 * kk or len(np.unique(pp)) != len(pp):
                    raise ValueError(f"選んだ本数がおかしい: {st} r{r} K{K} {len(pp)}")
                for k in range(4):
                    cand, phi, mean, tv, X, xm, xv = phis[k]
                    loc = {int(c): j for j, c in enumerate(cand)}
                    sel = torch.as_tensor([loc[int(i)] for i in pp if yq[i] == k], device=CB.DEV)
                    m_all = float(((phi[sel].mean(0) - mean) ** 2).sum()) / tv
                    row = {"target": tgt.name, "strategy": st, "rep": r, "K": K, "emotion": k, "mismatch": m_all,
                           "residual": float(1 - p_true[pp[yq[pp] == k]].mean())}
                    if feat:
                        row["mismatch_feat"] = float(((X[sel].mean(0) - xm) ** 2).sum()) / xv
                    mis.append(row)
                if st in run:
                    idx[st][(r, K)] = pool[pp]
    t_sel = time.perf_counter() - t1

    t2 = time.perf_counter()
    for st in run:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[st]

        def add(r, K, c, u, cv, L, cen=0, cb="", st=st):
            rows.append({"target": tgt.name, "source": st, "method": "HT_SEL", "cen": cen, "cp": c, "cb": cb,
                         "rep": r, "K": K, "uar": u})
            saved[f"{st}_K{K}_r{r}"] = L.astype(np.float32)
        CB.ht_block(sub, reps, src_pc, C_p, 1, add, cps=(CB.DEF_CP,), cbs=(CB.DEF_CB,), with_cv=False)
    print(f"  {kind} {tgt.name}：準備 {t_prep:.1f} 秒，選ぶ {t_sel:.1f} 秒，仮説転移 {time.perf_counter() - t2:.1f} 秒", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    pd.DataFrame(rows).to_csv(out / f"{tgt.name}.csv", index=False)
    pd.DataFrame(mis).to_csv(out / f"{tgt.name}_mis.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--feat-only", action="store_true")
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / (f"select_herdfeat_{args.kind}" if args.feat_only else f"select_herd_{args.kind}")
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.npz").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/select_herding.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "run": RUN_FEAT if args.feat_only else RUN, "mismatch_for": ALL_FEAT if args.feat_only else ALL,
            "feat_only": args.feat_only,
            "cp": CB.DEF_CP, "cb": CB.DEF_CB}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, data[t], data, args.reps, args.ks, out, args.feat_only)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
