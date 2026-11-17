"""【探索的】E42（GPU）：候補の感情の見込みを，ターゲットの発話どうしの近さでならすと，分類器の予測そのものより正確になるか。

規則は `docs/exploratory-analyses.md` E42（設計と予言 B1〜B3 は計算の前に書いた）。評価・候補・中心・下書きは E37・E41 と同じ（P4 の既定）。
近所は候補の x̃（下書きの統計で標準化した中心合わせ後の特徴）のコサイン類似度の k 近傍（どちらかの k 本に入れば辺），
ならし方はラベル拡散 Q ← αSQ + (1 − α)Y₀（S = D^{−1/2} W D^{−1/2}，60 回）。
  段 1：Y₀ = 下書きの確率 p。候補全部で測る
  段 2：E41 の random_pool と同じ 4K 本にラベルが付く。p_K はその 4K 本で解いた仮説転移（E41 の学習器）の確率，
        Y₀ はラベルの付いた行を 1-hot，残りを p_K とする。ラベルの付いていない候補で測る
テストの発話は使わない（段 2 の p_K のテストの UAR だけを，E41 の random_pool との照合のために記録する）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/belief_check.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/belief_check.py --analyze

出力：`results/belief_check_<kind>/<ターゲット>.csv`（stage, K, rep, method, k, alpha と量），`<ターゲット>_nbr.csv`（近所の記述），
`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e42_targets.csv`・`e42_tests.csv`
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
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import present_weights  # noqa: E402

GRID = [(k, a) for k in (10, 30) for a in (0.2, 0.5, 0.8)]
MAIN_SET = (10, 0.5)
TEMPS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_ITER = 60
KMAX_NBR = 30
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def knn(X: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """コサイン類似度の近い順の k 本（自分を除く）。(n, k)"""
    Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
    out = []
    for a in range(0, len(X), chunk):
        s = Xn[a:a + chunk] @ Xn.T
        s[torch.arange(len(s)), torch.arange(a, a + len(s))] = -float("inf")
        out.append(s.topk(k, dim=1).indices)
    return torch.cat(out)


def norm_adj(nbr: torch.Tensor, k: int) -> torch.Tensor:
    """どちらかの k 本に入れば重み 1 の辺を張り，S = D^{−1/2} W D^{−1/2} を疎行列で返す。"""
    n = len(nbr)
    r = torch.arange(n, device=nbr.device).repeat_interleave(k)
    c = nbr[:, :k].reshape(-1)
    ij = torch.cat([torch.stack([r, c]), torch.stack([c, r])], 1)
    W = torch.sparse_coo_tensor(ij, torch.ones(ij.shape[1], device=nbr.device), (n, n), check_invariants=False).coalesce()
    ij = W.indices()
    deg = torch.zeros(n, device=nbr.device).index_add_(0, ij[0], torch.ones(ij.shape[1], device=nbr.device))
    dinv = deg.rsqrt()
    return torch.sparse_coo_tensor(ij, dinv[ij[0]] * dinv[ij[1]], (n, n), check_invariants=False).coalesce()


def spread(S: torch.Tensor, Y0: torch.Tensor, alpha: float) -> torch.Tensor:
    Q = Y0.clone()
    for _ in range(N_ITER):
        Q = alpha * torch.sparse.mm(S, Q) + (1 - alpha) * Y0
    Q = Q.clamp(min=0)
    return Q / Q.sum(1, keepdim=True).clamp(min=1e-12)


def scores(P: torch.Tensor, y: np.ndarray, base: torch.Tensor | None = None) -> dict:
    yt = torch.as_tensor(y, device=P.device)
    pred = P.argmax(1)
    row = {"n": len(y), "uar": CB.uar(y, pred.cpu().numpy()), "acc": float((pred == yt).float().mean())}
    oh = torch.nn.functional.one_hot(yt, 4).to(P.dtype)
    row["logloss"] = float(-P.gather(1, yt[:, None]).clamp(min=1e-12).log().mean())
    row["brier"] = float(((P - oh) ** 2).sum(1).mean())
    lls = []
    for T in TEMPS:
        PT = P.clamp(min=1e-12) ** (1.0 / T)
        PT = PT / PT.sum(1, keepdim=True)
        lls.append(float(-PT.gather(1, yt[:, None]).clamp(min=1e-12).log().mean()))
    row["logloss_T"], row["T_best"] = min(lls), TEMPS[int(np.argmin(lls))]
    if base is not None:
        bp = base.argmax(1)
        flip = pred != bp
        row["n_flip"] = int(flip.sum())
        row["n_q_only"] = int((flip & (pred == yt)).sum())
        row["n_p_only"] = int((flip & (bp == yt)).sum())
    return row


def run_target(kind, T, data, reps, ks, out):
    corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731
    ev = data[T]
    pool_T = [s for s in data if corp(s) != corp(T)]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    Xp = torch.cat([data[s].X for s in pool_T])
    yp = M.labels(np.concatenate([data[s].y for s in pool_T]))
    src = logreg.fit(Xp - Xp.mean(0), yp, (C_T,))
    del Xp
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    rows, nrows = [], []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        P0 = torch.softmax(src.logits(Xc, C_T), 1).float()
        nbr = knn(((Xc - src.mu) / src.sd).float(), KMAX_NBR)
        Ss = {k: norm_adj(nbr, k) for k in sorted({k for k, _ in GRID})}
        # 近所の記述（k = 10．正解のラベルを使う）
        nb = nbr[:, :10].cpu().numpy()
        _, pid = np.unique(ev.person[pool], return_inverse=True)
        vote = np.zeros((len(pool), 4))
        for j in range(10):
            np.add.at(vote, (np.arange(len(pool)), yq[nb[:, j]]), 1.0 + 1e-3 * (10 - j))      # 同数なら近い方
        pi = np.bincount(yq, minlength=4) / len(yq)
        nrows.append({"target": T, "rep": r, "n": len(pool), "same_emotion": float((yq[nb] == yq[:, None]).mean()),
                      "chance": float((pi ** 2).sum()), "same_speaker": float((pid[nb] == pid[:, None]).mean()),
                      "knn_true_uar": CB.uar(yq, vote.argmax(1)), "draft_uar": CB.uar(yq, P0.argmax(1).cpu().numpy())})
        # 段 1
        rows.append({"target": T, "rep": r, "stage": 1, "K": "0", "method": "p", "k": 0, "alpha": 0.0, **scores(P0, yq)})
        for k, a in GRID:
            rows.append({"target": T, "rep": r, "stage": 1, "K": "0", "method": "q", "k": k, "alpha": a,
                         **scores(spread(Ss[k], P0, a), yq, P0)})
        # 段 2：E41 の random_pool と同じ 4K 本．学習器は `select_labelfree.ht_block_present` と同じ問
        perm = np.random.default_rng(zlib.crc32(f"e41pool|{ev.name}|{r}".encode())).permutation(len(pool))
        used = np.unique(pool[perm[:4 * kmax]])
        X = M.rows(ev.X, used) - ctx.ct[r]
        yX = M.labels(ev.y[used])
        masks, weights = [], []
        for K in ks:
            m = np.isin(used, pool[perm[:4 * int(K)]])
            w = torch.zeros(len(yX), dtype=logreg.DTYPE, device=CB.DEV)
            mt = torch.as_tensor(m, device=CB.DEV)
            w[mt] = present_weights(yX[mt])
            masks.append(m)
            weights.append(w)
        fits = logreg.fit_batch(X, yX, masks, (CB.DEF_CP,), anchor=src, anchor_c=C_T, stats=(src.mu, src.sd),
                                Cb=[CB.DEF_CB] * len(masks), weights=weights)
        Xte = M.rows(ev.X, np.flatnonzero(ctx.te[r])) - ctx.ct[r]
        yte = ev.y[ctx.te[r]]
        for K, f in zip(ks, fits):
            lab = perm[:4 * int(K)]
            un = np.ones(len(pool), bool)
            un[lab] = False
            PK = torch.softmax(f.logits(Xc, CB.DEF_CP), 1).float()
            Y0 = PK.clone()
            Y0[torch.as_tensor(lab, device=PK.device)] = torch.nn.functional.one_hot(
                torch.as_tensor(yq[lab], device=PK.device), 4).float()
            ut = torch.as_tensor(un, device=PK.device)
            test_uar = CB.uar(yte, f.logits(Xte, CB.DEF_CP).argmax(1).cpu().numpy())
            rows.append({"target": T, "rep": r, "stage": 2, "K": K, "method": "p", "k": 0, "alpha": 0.0,
                         "test_uar": test_uar, **scores(PK[ut], yq[un])})
            for k, a in GRID:
                rows.append({"target": T, "rep": r, "stage": 2, "K": K, "method": "q", "k": k, "alpha": a,
                             **scores(spread(Ss[k], Y0, a)[ut], yq[un], PK[ut])})
    pd.DataFrame(nrows).to_csv(out / f"{T}_nbr.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    tabs, nbrs, flips, chks = [], [], [], []
    for kind in kinds:
        d = ROOT / "results" / f"belief_check_{kind}"
        fs = [f for f in sorted(d.glob("*.csv")) if not f.stem.endswith("_nbr")]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
        a = a[a.rep.isin(EC.REPS)]
        a["set"] = np.where(a.method == "p", "p", "q_k" + a.k.astype(str) + "_a" + a.alpha.astype(str))
        g = a.groupby(["target", "stage", "K", "set"])[["uar", "acc", "logloss", "brier", "logloss_T"]].mean()
        for (T, st, K), sub in g.groupby(level=[0, 1, 2]):
            sub = sub.droplevel([0, 1, 2])
            row = {"kind": kind, "target": T, "stage": st, "K": K}
            for s, v in sub.iterrows():
                for m in ("uar", "acc", "logloss", "brier", "logloss_T"):
                    row[f"{s}:{m}"] = float(v[m] * (100 if m in ("uar", "acc") else 1))
            tabs.append(row)
        tu = a[(a.stage == 2) & (a.method == "p")].groupby(["target", "K"]).test_uar.mean() * 100
        for T in sorted(a.target.unique()):                                          # 照合：p_K のテストの UAR は E41 の random_pool と同じ問
            f41 = ROOT / "results" / f"teach_ablate_{kind}" / f"{T}.csv"
            if f41.exists():
                e = pd.read_csv(f41, dtype={"K": str})
                e = e[(e.policy == "random_pool") & e.rep.isin(a.rep.unique())].groupby("K").uar.mean() * 100
                for K in e.index:
                    chks.append({"kind": kind, "target": T, "K": K, "diff": float(tu.loc[(T, K)] - e.loc[K])})
        fl = a[a.method == "q"].groupby(["stage", "K", "set"])[["n", "n_flip", "n_q_only", "n_p_only"]].sum().reset_index()
        fl.insert(0, "kind", kind)
        flips.append(fl)
        nb = pd.concat([pd.read_csv(f) for f in sorted(d.glob("*_nbr.csv"))])
        nb = nb[nb.rep.isin(EC.REPS)].groupby("target").mean(numeric_only=True).reset_index()
        nb.insert(0, "kind", kind)
        nbrs.append(nb)
    D = pd.DataFrame(tabs)
    F = pd.concat(flips)
    main = f"q_k{MAIN_SET[0]}_a{MAIN_SET[1]}"
    tests = []
    for (kind, st, K), g in D.groupby(["kind", "stage", "K"], sort=False):
        row = {"kind": kind, "stage": st, "K": K, "targets": len(g), "p_uar": float(g["p:uar"].mean())}
        for s in [f"q_k{k}_a{a}" for k, a in GRID]:
            row[f"{s}-p:uar"] = float((g[f"{s}:uar"] - g["p:uar"]).mean())
        row["main>p"] = int((g[f"{main}:uar"] > g["p:uar"]).sum())
        for m in ("acc", "logloss", "brier", "logloss_T"):
            row[f"main-p:{m}"] = float((g[f"{main}:{m}"] - g[f"p:{m}"]).mean())
        f = F[(F.kind == kind) & (F.stage == st) & (F.K == K) & (F.set == main)].iloc[0]
        row["flip_rate"] = f.n_flip / f.n
        row["q_only|flip"] = f.n_q_only / max(f.n_flip, 1)
        row["p_only|flip"] = f.n_p_only / max(f.n_flip, 1)
        tests.append(row)
    R = pd.DataFrame(tests)
    if chks:
        C = pd.DataFrame(chks).groupby(["kind", "K"])["diff"].apply(lambda s: s.abs().max()).rename("chk_e41_max").reset_index()
        C["stage"] = 2
        R = R.merge(C, on=["kind", "stage", "K"], how="left")
    for kind in ("audio_wavlm", "audio_hubert"):
        s1 = R[(R.kind == kind) & (R.stage == 1)]
        s2 = R[(R.kind == kind) & (R.stage == 2) & (R.K == "5")]
        if len(s1):
            R.loc[R.kind == kind, "B1"] = bool(s1[f"{main}-p:uar"].iloc[0] > 0)
            R.loc[R.kind == kind, "B2"] = bool(s1["q_only|flip"].iloc[0] > s1["p_only|flip"].iloc[0])
        if len(s2):
            R.loc[R.kind == kind, "B3"] = bool(s2[f"{main}-p:uar"].iloc[0] > 0)
    N = pd.concat(nbrs)
    D.to_csv(OUT_REL / "e42_targets.csv", index=False)
    R.to_csv(OUT_REL / "e42_tests.csv", index=False)
    N.to_csv(OUT_REL / "e42_nbr.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(N.groupby("kind")[["same_emotion", "chance", "same_speaker", "knn_true_uar", "draft_uar"]].mean().round(3).to_string())
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
    out = args.out_dir or ROOT / "results" / f"belief_check_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/belief_check.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "grid": GRID, "main": MAIN_SET,
            "temps": TEMPS, "n_iter": N_ITER, "cp": CB.DEF_CP, "cb": CB.DEF_CB,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
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
