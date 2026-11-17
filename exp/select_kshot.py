"""【探索的】E33（GPU）：ターゲットの K 本の選び方を変えた仮説転移（P4 の既定）。

規則は `docs/exploratory-analyses.md` E33（設計と予言 S1〜S3 は計算の前に書いた）。テスト，候補（テスト以外の発話），中心は
E31（`compare_b1.Ctx`）と同じ。感情 k ごとに，候補のうち真の感情が k の発話から K 本を選び，P4 の下書き（中心合わせ後の全ソースの
合併の分類器）からの仮説転移（C′ = 0.01，C_b = 8，交差検証なし）で解く。選び方は random（E31 と同じ K 本），pad_low・pad_mid・
pad_high（下書きの点数の空間での塊とソースの感情 k の PAD），easy・hard（下書きが正しい感情に出す確率），committee（ソースごとの
分類器が正しい感情に出す確率のばらつき）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/select_kshot.py --kind audio_wavlm [--target …] [--reps …]
        [--ks …] [--out-dir …]

出力：`results/select_<kind>/<ターゲット>.csv`（method=HT_SEL, source=選び方, cen, cp, cb, rep, K, uar），`.npz`（テストの点数），
`<ターゲット>_sel.csv`（選んだ発話の記録：下書きの当たり，正しい感情の確率，PAD），`provenance.json`，`provenance_log.jsonl`。
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import compare_b1 as CB  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from interventions import map_rows  # noqa: E402
from pair_table import map_dir  # noqa: E402

STRATS = ("random", "pad_low", "pad_mid", "pad_high", "easy", "hard", "committee")
LIN: dict = {}           # (ソース, C_s) → 中心合わせの座標の 1 次関数（E32 の段 2 と同じ．ターゲットによらない）


def softmax(L: np.ndarray) -> np.ndarray:
    Z = np.exp(L - L.max(1, keepdims=True))
    return Z / Z.sum(1, keepdims=True)


def pad(S: np.ndarray, T: np.ndarray, seed: int) -> float:
    """PAD = 2(1 − 2ε)。ε は S と T を見分ける線形の分類器の交差検証の誤り率（多い方を少ない方の本数まで抜く）。"""
    rng = np.random.default_rng(seed)
    n = min(len(S), len(T))
    if n < 2:
        return np.nan
    S = S[rng.choice(len(S), n, replace=False)]
    T = T[rng.choice(len(T), n, replace=False)]
    X, y = np.r_[S, T], np.r_[np.zeros(n), np.ones(n)]
    acc = cross_val_score(LogisticRegression(max_iter=1000), X, y,
                          cv=StratifiedKFold(n_splits=min(5, n), shuffle=True, random_state=seed)).mean()
    return float(2 * (1 - 2 * (1 - acc)))


def pad_pick(Z: np.ndarray, Sk: np.ndarray, K: int, seed: int) -> dict:
    """候補 Z（感情 k，下書きの点数の空間）を塊に分け，PAD が最小・中央・最大の塊の中心に近い順に K 本（位置の番号）。"""
    m = min(10, max(3, len(Z) // (3 * K)))
    m = min(m, len(Z))
    km = KMeans(n_clusters=m, n_init=10, random_state=seed).fit(Z)
    lab = km.labels_
    pads = np.array([pad(Sk, Z[lab == c], seed + 1000 * c) for c in range(m)])
    order = np.argsort(np.where(np.isnan(pads), np.inf, pads))
    order = order[np.isfinite(pads[order])]
    ranks = {"pad_low": list(order), "pad_high": list(order[::-1]),
             "pad_mid": sorted(order, key=lambda c: abs(list(order).index(c) - len(order) // 2))}
    out = {}
    for name, seq in ranks.items():
        picked = []
        for c in seq:
            mem = np.flatnonzero(lab == c)
            dist = np.linalg.norm(Z[mem] - km.cluster_centers_[c], axis=1)
            picked += list(mem[np.argsort(dist)][:K - len(picked)])
            if len(picked) >= K:
                break
        out[name] = (np.array(picked), float(pads[seq[0]]), int((lab == seq[0]).sum()), m)
    return out


def run_target(kind, tgt, data, reps, ks, out):
    ctx = CB.Ctx(tgt, reps, ks, center="ind")
    ref = map_rows(kind, tgt.name)
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
    d = tgt.X.shape[1]
    zero = torch.zeros(d, dtype=logreg.DTYPE, device=CB.DEV)
    one = torch.ones(d, dtype=logreg.DTYPE, device=CB.DEV)
    lin = []
    for s in pool_src:
        C_s = ref[("src_only", s, "", reps[0])][0]
        if any(ref[("src_only", s, "", r)][0] != C_s for r in reps):
            raise ValueError(f"ソースのみの C が反復で違う: {s}")
        if (s, C_s) not in LIN:
            Xs = data[s].X
            LIN[(s, C_s)] = logreg.fit(Xs - Xs.mean(0), M.labels(data[s].y), (C_s,)).moved_to(zero, one, C_s)
        lin.append(LIN[(s, C_s)])
    # ソースの感情 k の発話（ソースごとに中心を引き，下書きの点数から行ごとの平均を引く）
    Ssrc = []
    for s in pool_src:
        Ls = src_pc.logits(data[s].X - data[s].X.mean(0), C_p).cpu().numpy()
        Ssrc.append((Ls - Ls.mean(1, keepdims=True), data[s].y))
    S_by_k = {k: np.concatenate([L[y == k] for L, y in Ssrc]) for k in range(4)}
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    idx = {st: {} for st in STRATS}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = tgt.y[pool]
        Xc = M.rows(tgt.X, pool) - ctx.ct[r]
        Ld = src_pc.logits(Xc, C_p).cpu().numpy()
        P = softmax(Ld)
        p_true = P[np.arange(len(pool)), yq]
        right = Ld.argmax(1) == yq
        Z = Ld - Ld.mean(1, keepdims=True)
        p_src = np.stack([softmax((Xc @ A.T + a0).cpu().numpy())[np.arange(len(pool)), yq] for A, a0 in lin])
        disagree = p_src.std(0)
        pos_of = {int(i): j for j, i in enumerate(pool)}
        for K in ks:
            kk = int(K)
            picks = {st: [] for st in STRATS}
            picks["random"] = [pos_of[int(i)] for i in ctx.idx[(r, K)]]
            for k in range(4):
                cand = np.flatnonzero(yq == k)
                o = np.argsort(p_true[cand])
                picks["hard"] += list(cand[o[:kk]])
                picks["easy"] += list(cand[o[::-1][:kk]])
                picks["committee"] += list(cand[np.argsort(disagree[cand])[::-1][:kk]])
                for name, (pp, pv, size, m) in pad_pick(Z[cand], S_by_k[k], kk, r).items():
                    picks[name] += list(cand[pp])
                    sel_rows.append({"target": tgt.name, "rep": r, "K": K, "emotion": k, "strategy": name, "pad": pv,
                                     "cluster_size": size, "n_clusters": m})
            for st, pp in picks.items():
                pp = np.array(pp)
                if len(pp) != 4 * kk or len(np.unique(pp)) != len(pp):
                    raise ValueError(f"選んだ本数がおかしい: {st} r{r} K{K} {len(pp)}")
                idx[st][(r, K)] = pool[pp]
                sel_rows.append({"target": tgt.name, "rep": r, "K": K, "emotion": -1, "strategy": st,
                                 "draft_right": float(right[pp].mean()), "p_true": float(p_true[pp].mean()),
                                 "disagree": float(disagree[pp].mean())})
    t_sel = time.perf_counter() - t1

    t2 = time.perf_counter()
    for st in STRATS:
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
    pd.DataFrame(sel_rows).to_csv(out / f"{tgt.name}_sel.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"select_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.npz").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/select_kshot.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
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
