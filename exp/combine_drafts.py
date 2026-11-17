"""【探索的】E32 の 5 節 段 2（GPU）：全ソースの下書きを「ソースごとの分類器を重み付きで足したもの」に取り替えた仮説転移。

規則は `docs/exploratory-analyses.md` E32 の 5 節「段 2 の作り方と予言」（予言 D1 は計算の前に書いた）。テストと K 本は
E31（`compare_b1.Ctx`）と同じ，仮説転移は P4 の既定（帰納の中心合わせ，C′ = 0.01，C_b = 8，交差検証なし）。
下書き：P4 と同じソースについて，ソースごとに中心を引いたソースのみの分類器（C は地図のソースのみの C）を中心合わせの座標の
1 次関数に直し，ターゲットのテスト以外の発話（ラベルなし）で尺度をそろえて重み付きで足す。全体の尺度は同じ発話で P4 の下書きに
合わせ，標準化は P4 の下書きの統計を使う。束ね方は equal（主），count，value（段 1 の `results/explore_relation/e32d_weights.csv`．
テストのラベルで測った e による仕組みの確かめ）。`--check` は P4 の下書きそのものでも解き（pooled），E31 の P4 と一致するかを見る。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/combine_drafts.py --kind audio_wavlm [--target …] [--reps …]
        [--ks …] [--drafts …] [--check] [--out-dir …]

出力：`results/combine_<kind>/<ターゲット>.csv`（method=HT_AVG, source=束ね方, cen, cp, cb, rep, K, uar），`.npz`（テストの点数），
`provenance.json`（最後の起動），`provenance_log.jsonl`（起動ごと）。終わったターゲットは飛ばす。
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

DRAFTS = ("equal", "count", "value")
PAIRS = [(k, j) for k in range(4) for j in range(k + 1, 4)]
WEIGHTS = ROOT / "results" / "explore_relation" / "e32d_weights.csv"
LIN: dict = {}           # (ソース, C_s) → 中心合わせの座標の 1 次関数．ソースのデータは全部使うのでターゲットによらず，起動の中で使い回す


def pair_scale(L: torch.Tensor) -> float:
    """ラベルを使わない尺度：感情の対の差の分散（母分散）の 6 対平均の平方根。"""
    Lc = L - L.mean(0)
    return float(torch.stack([(Lc[:, k] - Lc[:, l]).var(unbiased=False) for k, l in PAIRS]).mean().sqrt())


def draft_weights(draft: str, pool: list[str], n_rows: dict, wtab: pd.DataFrame | None) -> dict:
    if draft == "equal":
        w = {s: 1.0 for s in pool}
    elif draft == "count":
        w = {s: float(n_rows[s]) for s in pool}
    else:
        v = wtab.set_index("source").weight
        if set(v.index) != set(pool):
            raise ValueError(f"段 1 の価値の重みのソースが合わない: {sorted(v.index)} 対 {sorted(pool)}")
        w = {s: float(v[s]) for s in pool}
    tot = sum(w.values())
    return {s: x / tot for s, x in w.items()}


def run_target(kind, tgt, data, reps, ks, out, drafts, check, wtab):
    ctx = CB.Ctx(tgt, reps, ks, center="ind")
    ref = map_rows(kind, tgt.name)
    pool = [s for s in data if s != tgt.name and GROUP_SPEC[s][0] != GROUP_SPEC[tgt.name][0]]
    rows, saved = [], {}
    for r in reps:
        saved[f"test_idx_r{r}"] = np.flatnonzero(ctx.te[r])
        saved[f"y_test_r{r}"] = tgt.y[ctx.te[r]]
    t0 = time.perf_counter()
    # P4 の下書き（尺度と標準化の基準．compare_b1 と同じ作り方）
    Xp = torch.cat([data[s].X for s in pool])
    yp = M.labels(np.concatenate([data[s].y for s in pool]))
    C_p = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{tgt.name}.csv", dtype={"K": str}).C.iloc[0])
    src_pc = logreg.fit(Xp - Xp.mean(0), yp, (C_p,))
    del Xp
    # ソースごとの中心を引いたソースのみの分類器 → 中心合わせの座標の 1 次関数 (A, a0)
    d = tgt.X.shape[1]
    zero = torch.zeros(d, dtype=logreg.DTYPE, device=CB.DEV)
    one = torch.ones(d, dtype=logreg.DTYPE, device=CB.DEV)
    lin, cs_of = LIN, {}
    for s in pool:
        for r in reps:
            C_s = ref[("src_only", s, "", r)][0]
            cs_of[(s, r)] = C_s
            if (s, C_s) not in lin:
                Xs = data[s].X
                f = logreg.fit(Xs - Xs.mean(0), M.labels(data[s].y), (C_s,))
                lin[(s, C_s)] = f.moved_to(zero, one, C_s)
    n_rows = {s: len(data[s].y) for s in pool}
    t_fit = time.perf_counter() - t0

    def adder(draft):
        def add(r, K, c, u, cv, L, cen=0, cb=""):
            rows.append({"target": tgt.name, "source": draft, "method": "HT_AVG", "cen": cen, "cp": c, "cb": cb,
                         "rep": r, "K": K, "uar": u})
            saved[f"{draft}_K{K}_r{r}"] = L.astype(np.float32)
        return add

    t1 = time.perf_counter()
    for r in reps:
        Xu = M.rows(tgt.X, np.flatnonzero(~ctx.te[r])) - ctx.ct[r]     # テスト以外の発話（ラベルは使わない）
        ref_scale = pair_scale(src_pc.logits(Xu, C_p))
        comps = {}
        for s in pool:
            A, a0 = lin[(s, cs_of[(s, r)])]
            sc = pair_scale(Xu @ A.T + a0)
            comps[s] = (A / sc, a0 / sc)
        for draft in drafts:
            w = draft_weights(draft, pool, n_rows, wtab[wtab.target == tgt.name] if wtab is not None else None)
            A = sum(w[s] * comps[s][0] for s in pool)
            a0 = sum(w[s] * comps[s][1] for s in pool)
            g = ref_scale / pair_scale(Xu @ A.T + a0)
            A, a0 = g * A, g * a0
            mu, sd = src_pc.mu, src_pc.sd
            anchor = logreg.Fitted(mu, sd, {C_p: A * sd}, {C_p: a0 + mu @ A.T}, {C_p: 0}, {C_p: 0.0})
            CB.ht_block(ctx, [r], anchor, C_p, 1, adder(draft), cps=(CB.DEF_CP,), cbs=(CB.DEF_CB,), with_cv=False)
        if check:
            CB.ht_block(ctx, [r], src_pc, C_p, 1, adder("pooled"), cps=(CB.DEF_CP,), cbs=(CB.DEF_CB,), with_cv=False)
    print(f"  {kind} {tgt.name}：下書き {t_fit:.1f} 秒，仮説転移 {time.perf_counter() - t1:.1f} 秒（ソース {len(pool)}）", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    pd.DataFrame(rows).to_csv(out / f"{tgt.name}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--drafts", nargs="+", default=list(DRAFTS), choices=DRAFTS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"combine_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.npz").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return
    wtab = pd.read_csv(WEIGHTS).query("kind == @args.kind and draft == 'value'") if "value" in args.drafts else None
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/combine_drafts.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "drafts": args.drafts, "check": args.check,
            "cp": CB.DEF_CP, "cb": CB.DEF_CB, "weights": str(WEIGHTS.relative_to(ROOT))}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, data[t], data, args.reps, args.ks, out, args.drafts, args.check, wtab)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
