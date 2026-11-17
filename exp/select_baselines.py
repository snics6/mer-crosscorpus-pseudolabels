"""【探索的】E46（GPU）：ラベル無しで選ぶ既存の方法（能動学習の比べる相手）を，設定 1 の同じ手順・同じ学習器で解く。

規則は `docs/exploratory-analyses.md` E46（設計と予言は計算の前に書いた）。評価・候補・中心・下書き・学習器は E41・E45 と同じ（P4 の既定）。
選び方はどれもラベルを使わず，下書きの状態から 1 回で 4K 本を選ぶ（K で入れ子にできるものは入れ子）。座標は teach と同じ x̃（下書きの統計で標準化した中心合わせ後の特徴）：
  margin    ：下書きの確率の 1 位と 2 位の差が小さい順（不確かさで選ぶ．Scheffer ほか 2001）
  coreset   ：k-center の貪欲法（Sener & Savarese 2018）．候補の平均に最も近い発話から始め，選んだ発話から最も遠い発話を足していく
  typiclust ：候補を 4K 個に k-means で分け，各群で最も典型的な（近い 20 本とのコサイン距離の平均が小さい）発話を選ぶ（Hacohen ほか 2022．K ごとに解き直す）
  badge     ：勾配の埋め込み (e_ŷ − p) ⊗ x̃（ŷ は下書きの予測）に k-means++ の種まきを行う（Ash ほか 2020．最初はノルム最大）
E34 の herding_pred・herding_all・random_pred と，E41 の random_pool・random_true，E45 の C・B・hardp は既存の値を使う。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/select_baselines.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/select_baselines.py --analyze

出力：`results/select_baselines_<kind>/<ターゲット>.csv`（policy, rep, K, uar），`<ターゲット>_sel.csv`，`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e46_targets.csv`・`e46_tests.csv`
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
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from belief_check import knn  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import ht_block_present  # noqa: E402

POLICIES = ["margin", "coreset", "typiclust", "badge"]
K_TYP = 20
KM_ITERS = 50
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def order_margin(P: torch.Tensor) -> list:
    top = P.topk(2, dim=1).values
    return torch.argsort(top[:, 0] - top[:, 1]).tolist()


def order_coreset(X: torch.Tensor, n: int) -> list:
    first = int(torch.argmin(((X - X.mean(0)) ** 2).sum(1)))
    order = [first]
    dmin = ((X - X[first]) ** 2).sum(1)
    for _ in range(n - 1):
        i = int(torch.argmax(dmin))
        order.append(i)
        dmin = torch.minimum(dmin, ((X - X[i]) ** 2).sum(1))
    return order


def kmeans(X: torch.Tensor, k: int, g: torch.Generator) -> torch.Tensor:
    """k-means++ の種まきと Lloyd 法。各発話の群の番号を返す。"""
    n = len(X)
    cen = [int(torch.randint(n, (1,), generator=g, device="cpu"))]
    d2 = ((X - X[cen[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        pr = (d2 / d2.sum()).cpu()
        i = int(torch.multinomial(pr, 1, generator=g))
        cen.append(i)
        d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
    Cm = X[cen].clone()
    for _ in range(KM_ITERS):
        lab = torch.cdist(X, Cm).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                Cm[j] = X[m].mean(0)
    return torch.cdist(X, Cm).argmin(1)


def pick_typiclust(X: torch.Tensor, typ: torch.Tensor, B: int, g: torch.Generator) -> list:
    lab = kmeans(X, B, g)
    out = []
    for j in range(B):
        m = torch.nonzero(lab == j).flatten()
        if len(m):
            out.append(int(m[torch.argmax(typ[m])]))
    rest = [i for i in torch.argsort(typ, descending=True).tolist() if i not in set(out)]      # 空の群の分は典型的な順に補う
    return (out + rest)[:B]


def order_badge(X: torch.Tensor, P: torch.Tensor, n: int, g: torch.Generator) -> list:
    R = torch.nn.functional.one_hot(P.argmax(1), 4).to(P.dtype) - P
    G = (R[:, :, None] * X[:, None, :]).reshape(len(X), -1)
    first = int(torch.argmax((G ** 2).sum(1)))
    order = [first]
    d2 = ((G - G[first]) ** 2).sum(1)
    for _ in range(n - 1):
        pr = d2.clamp(min=0).cpu()
        pr[order] = 0
        i = int(torch.multinomial(pr / pr.sum(), 1, generator=g))
        order.append(i)
        d2 = torch.minimum(d2, ((G - G[i]) ** 2).sum(1))
    return order


def run_target(kind, T, data, reps, ks, out):
    corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731
    S = [s for s in data if corp(s) != corp(T)]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    Xp = torch.cat([data[s].X for s in S])
    yp = M.labels(np.concatenate([data[s].y for s in S]))
    src = logreg.fit(Xp - Xp.mean(0), yp, (C_T,))
    del Xp
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {p: {} for p in POLICIES}
    srows = []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        P = torch.softmax(src.logits(Xc, C_T), 1).float()
        X = ((Xc - src.mu) / src.sd).float()
        gen = lambda name: torch.Generator().manual_seed(zlib.crc32(f"e46|{name}|{T}|{r}".encode()))  # noqa: E731
        orders = {"margin": order_margin(P), "coreset": order_coreset(X, 4 * kmax), "badge": order_badge(X, P, 4 * kmax, gen("badge"))}
        Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nb = knn(X, K_TYP)
        typ = 1.0 / (1.0 - (Xn[:, None, :] * Xn[nb]).sum(2)).mean(1).clamp(min=1e-6)    # 近い 20 本とのコサイン距離の平均の逆数
        for K in ks:
            n = 4 * int(K)
            sel = {p: orders[p][:n] for p in orders}
            sel["typiclust"] = pick_typiclust(X, typ, n, gen(f"typiclust{K}"))
            for p, pp in sel.items():
                pp = np.array(pp)
                if len(np.unique(pp)) != n:
                    raise ValueError(f"選んだ本数がおかしい: {p} r{r} K{K}")
                idx[p][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": p, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max())})
    rows = []
    for p in idx:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[p]

        def add(r, K, c, u, cv, L, cen=0, cb="", p=p):
            rows.append({"target": T, "policy": p, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C_T, add)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    e41 = pd.read_csv(OUT_REL / "e41_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e45f = OUT_REL / "e45_targets.csv"
    e45 = pd.read_csv(e45f, dtype={"K": str}).set_index(["kind", "target", "K"]) if e45f.exists() else None
    tabs = []
    for kind in kinds:
        d = ROOT / "results" / f"select_baselines_{kind}"
        fs = [x for x in sorted(d.glob("*.csv")) if not x.stem.endswith("_sel")]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(x, dtype={"K": str}) for x in fs])
        a = a[a.rep.isin(EC.REPS)]
        reps = sorted(a.rep.unique())
        u = a.groupby(["target", "policy", "K"]).uar.mean() * 100
        for T in sorted(a.target.unique()):
            lf = pd.read_csv(ROOT / "results" / f"select_lf_{kind}" / f"{T}.csv", dtype={"K": str})
            lf = lf[lf.rep.isin(reps)].groupby(["source", "K"]).uar.mean() * 100
            for K in ("1", "5", "10", "25"):
                b = e41.loc[(kind, T, K)]
                row = {"kind": kind, "target": T, "K": K}
                for x in ("random_true", "random_pool", "LLL", "PPP"):
                    row[x] = float(b[x])
                for x in ("herding_pred", "herding_all", "random_pred"):
                    row[x] = float(lf.loc[(x, K)])
                for p in POLICIES:
                    row[p] = float(u.loc[(T, p, K)])
                if e45 is not None and (kind, T, K) in e45.index:
                    for v in ("C", "B", "hardp"):
                        row[f"e45_{v}"] = float(e45.loc[(kind, T, K), v])
                tabs.append(row)
    D = pd.DataFrame(tabs)
    base = ["random_pool", "random_pred", "herding_pred", "herding_all"] + POLICIES
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g), "random_true": float(g.random_true.mean()), "LLL": float(g.LLL.mean())}
        for x in base + [c for c in ("e45_C", "e45_B", "e45_hardp") if c in g]:
            row[x] = float(g[x].mean())
            row[f"{x}-pool"] = float((g[x] - g.random_pool).mean())
        if "e45_C" in g:
            best = g[base].mean().idxmax()
            row["best_baseline"] = best
            row["C-best_baseline"] = float((g.e45_C - g[best]).mean())
            row["C>best_baseline"] = int((g.e45_C > g[best]).sum())
        tests.append(row)
    R = pd.DataFrame(tests)
    D.to_csv(OUT_REL / "e46_targets.csv", index=False)
    R.to_csv(OUT_REL / "e46_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(2).T.to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=MAIN)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"select_baselines_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/select_baselines.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "policies": POLICIES, "k_typ": K_TYP,
            "km_iters": KM_ITERS, "cp": CB.DEF_CP, "cb": CB.DEF_CB, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, args.ks, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
