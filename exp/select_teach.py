"""【探索的】E37（GPU）：ラベルのある候補から，線形にした仮説転移の学習器を教える K 本を選ぶ（機械教示）。学習器の歩幅 c はメタ学習。

規則は `docs/exploratory-analyses.md` E37（設計と予言 T1・T2 は計算の前に書いた）。評価・候補・中心・下書き・学習器は E33 と同じ（P4 の既定）。
学習器の模型：候補 i を教えると候補全体の点数が ΔL_n = c (x̃_nᵀ x̃_i) r_iᵀ だけ動く（自分自身の点数は動かさない）。感情 0〜3 の順に 1 本ずつ，
その感情の未選択の候補のうち，加えたときの「未選択の候補全体での感情の対 6 つの d の平均」が最大の発話を選び，模型の点数と残差を更新する。
ターゲット T では 4 つの c を反復 0〜9 で解き（role=target），T のソースコーパス c′ ごとに T と c′ のコーパスの群を除くソースの合併
（C = 0.01）を下書きにして c′ で反復 0〜4 を解く（role=pseudo．c′ の候補のラベルを使う．T のテストのラベルは使わない）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/select_teach.py --kind audio_wavlm [--target …]
    ./.venv/bin/python exp/select_teach.py --analyze        # 判定（CPU．select_herd_・select_herdfeat_ が要る）

出力：`results/select_teach_<kind>/<ターゲット>.csv`（role, eval, policy, rep, K, uar），`<ターゲット>_model.csv`（模型の d），
`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e37_targets.csv`・`e37_tests.csv`
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
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_herding import herd  # noqa: E402
from select_labelfree import ht_block_present  # noqa: E402

CS = (0.0003, 0.001, 0.003, 0.01)
PAIRS = [(k, l) for k in range(4) for l in range(k + 1, 4)]
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_PSEUDO = 0.01
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def pname(c: float) -> str:
    return f"teach_c{c:g}"


def teach_order(Xt: torch.Tensor, y: np.ndarray, L0: torch.Tensor, c: float, n_steps: int) -> tuple[list[int], list[float]]:
    """模型の学習器を教える順（候補の中の位置）と，各段の後の模型の d（6 対の平均．未選択の候補の上）。"""
    X = Xt.float()
    Km = X @ X.T
    Km.fill_diagonal_(0.0)
    K2 = Km * Km
    yt = torch.as_tensor(np.array(y), device=X.device)
    onehot = torch.nn.functional.one_hot(yt, 4).float()
    L = L0.float().clone()
    avail = torch.ones(len(y), dtype=torch.bool, device=X.device)
    order, trace = [], []
    for step in range(n_steps):
        cls = step % 4
        R = onehot - torch.softmax(L, 1)
        Z = torch.stack([L[:, k] - L[:, l] for k, l in PAIRS], 1)                    # (n, 6)
        Ic = torch.stack([((yt == k) & avail).float() for k in range(4)], 1)          # (n, 4)
        nc = Ic.sum(0).clamp(min=1.0)
        W = Ic / nc
        Bcols = torch.stack([W[:, cc] * Z[:, j] for j, (k, l) in enumerate(PAIRS) for cc in (k, l)], 1)   # (n, 12)
        G = Km @ torch.cat([W, Bcols], 1)                                            # (n, 16)：A_c(i)，B^j_c(i)
        Q = K2 @ W                                                                   # (n, 4)：Q_c(i)
        m = (W.T @ Z)                                                                # (4, 6)：m^j_c
        s = (W.T @ (Z * Z))                                                          # (4, 6)：s^j_c
        cand = (yt == cls) & avail
        obj = torch.zeros(len(y), device=X.device)
        for j, (k, l) in enumerate(PAIRS):
            dl = R[:, k] - R[:, l]
            mk = m[k, j] + c * dl * G[:, k]
            ml = m[l, j] + c * dl * G[:, l]
            sk = s[k, j] + 2 * c * dl * G[:, 4 + 2 * j] + (c * dl) ** 2 * Q[:, k]
            sl = s[l, j] + 2 * c * dl * G[:, 4 + 2 * j + 1] + (c * dl) ** 2 * Q[:, l]
            obj = obj + (mk - ml) / torch.sqrt(((sk - mk ** 2) + (sl - ml ** 2)).clamp(min=1e-12) / 2)
        obj = obj / len(PAIRS)
        if not bool(cand.any()):
            raise ValueError(f"感情 {cls} の候補が尽きた（{step} 本目）")
        obj[~cand] = -float("inf")
        i = int(torch.argmax(obj))
        trace.append(float(obj[i]))
        order.append(i)
        avail[i] = False
        L = L + c * Km[:, i:i + 1] * R[i:i + 1, :]
    return order, trace


def evaluate(role, meta_for, ev, src, C, reps, ks, rows, mrows):
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {pname(c): {} for c in CS}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        L0 = src.logits(Xc, C)
        Xt = (Xc - src.mu) / src.sd
        for c in CS:
            order, trace = teach_order(Xt, yq, L0, c, 4 * kmax)
            for K in ks:
                pp = np.array(order[: 4 * int(K)])
                if len(np.unique(pp)) != len(pp) or (np.bincount(yq[pp], minlength=4) != int(K)).any():
                    raise ValueError(f"選んだ本数がおかしい: c={c} r{r} K{K}")
                idx[pname(c)][(r, K)] = pool[pp]
                mrows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "policy": pname(c), "rep": r, "K": K,
                              "model_d": trace[4 * int(K) - 1]})
    if role == "target":                                    # 比べる相手を同じ GPU で計算し直す（E36 の GPU によるずれの対策）
        for pn in ("herding", "herding_feat", "random"):
            idx[pn] = {}
        for r in reps:
            pool = np.flatnonzero(~ctx.te[r])
            yq = ev.y[pool]
            Xc = M.rows(ev.X, pool) - ctx.ct[r]
            Ld = src.logits(Xc, C)
            Xt = (Xc - src.mu) / src.sd
            Rt = torch.nn.functional.one_hot(torch.as_tensor(np.array(yq), device=CB.DEV), 4).to(Xt.dtype) - torch.softmax(Ld, 1)
            ordh, ordf = {}, {}
            for k in range(4):                              # E33 の追加と同じ φ = (a r, r ⊗ x̃)，E33 の追加 2 と同じ x̃
                cand = np.flatnonzero(yq == k)
                ci = torch.as_tensor(cand, device=CB.DEV)
                R, X = Rt[ci], Xt[ci]
                RX = (R[:, :, None] * X[:, None, :]).reshape(len(cand), -1)
                vr, vx = float(R.var(0, unbiased=False).sum()), float(RX.var(0, unbiased=False).sum())
                a = np.sqrt(vx / vr) if vr > 0 else 1.0
                ordh[k] = cand[herd(torch.cat([a * R, RX], 1), kmax)]
                ordf[k] = cand[herd(X, kmax)]
            pos_of = {int(i): j for j, i in enumerate(pool)}
            for K in ks:
                kk = int(K)
                idx["herding"][(r, K)] = pool[np.concatenate([ordh[k][:kk] for k in range(4)])]
                idx["herding_feat"][(r, K)] = pool[np.concatenate([ordf[k][:kk] for k in range(4)])]
                idx["random"][(r, K)] = pool[[pos_of[int(i)] for i in ctx.idx[(r, K)]]]
    for pn in idx:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[pn]

        def add(r, K, c, u, cv, L, cen=0, cb="", pn=pn):
            rows.append({"role": role, "meta_for": meta_for, "eval": ev.name, "policy": pn, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C, add)


def run_target(kind, T, data, reps, ks, out, cache, pseudo=True):
    t0 = time.perf_counter()
    rows, mrows = [], []
    corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731
    pool_T = [s for s in data if corp(s) != corp(T)]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])

    def draft(pool, C):
        key = (tuple(sorted(pool)), C)
        if key not in cache:
            Xp = torch.cat([data[s].X for s in pool])
            yp = M.labels(np.concatenate([data[s].y for s in pool]))
            cache[key] = logreg.fit(Xp - Xp.mean(0), yp, (C,))
            del Xp
        return cache[key]

    evaluate("target", T, data[T], draft(pool_T, C_T), C_T, reps, ks, rows, mrows)
    t1 = time.perf_counter()
    for c in (pool_T if pseudo else []):                   # --no-pseudo：反復を足すときは仮のターゲット（反復 0〜4．前回と同じ）を解き直さない
        pool_c = [s for s in pool_T if corp(s) != corp(c)]
        evaluate("pseudo", T, data[c], draft(pool_c, C_PSEUDO), C_PSEUDO, PSEUDO_REPS, ks, rows, mrows)
    print(f"  {kind} {T}：ターゲット {t1 - t0:.0f} 秒，仮のターゲット {len(pool_T)} 個 {time.perf_counter() - t1:.0f} 秒", flush=True)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)
    pd.DataFrame(mrows).to_csv(out / f"{T}_model.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN, suffix: str = "") -> int:
    tabs, tests = [], []
    for kind in kinds:
        fs = [f for f in sorted((ROOT / "results" / f"select_teach_{kind}").glob("*.csv")) if not f.stem.endswith("_model")]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
        pseudo = a[a.role == "pseudo"].groupby(["meta_for", "policy"]).uar.mean()
        tgt = a[(a.role == "target") & a.rep.isin(EC.REPS)].groupby(["eval", "policy", "K"]).uar.mean() * 100
        for T in sorted(a.meta_for.unique()):
            star = pseudo.loc[T][[pname(c) for c in CS]].idxmax()
            fh, ff = (ROOT / "results" / f"select_herd_{kind}" / f"{T}.csv", ROOT / "results" / f"select_herdfeat_{kind}" / f"{T}.csv")
            hd = pd.read_csv(fh, dtype={"K": str}) if fh.exists() else None    # 言葉系には E33 の元の出力が無い（照合の列は空）
            hf = pd.read_csv(ff, dtype={"K": str}) if ff.exists() else None
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K, "policy_star": star, "meta": float(tgt.loc[(T, star, K)]),
                       "oracle": float(tgt.loc[T].xs(K, level="K")[[pname(c) for c in CS]].max()),
                       "policy_oracle": str(tgt.loc[T].xs(K, level="K")[[pname(c) for c in CS]].idxmax())}
                for st, f in (("herding", hd), ("random", hd), ("herding_feat", hf)):
                    row[f"{st}_e33"] = np.nan if f is None else float(f[(f.source == st) & (f.K == K) & f.rep.isin(EC.REPS)].uar.mean() * 100)
                    row[st] = float(tgt.loc[(T, st, K)])            # 主：同じ GPU で計算し直した値
                for c in CS:
                    row[pname(c)] = float(tgt.loc[(T, pname(c), K)])
                tabs.append(row)
    D = pd.DataFrame(tabs)
    for (kind, K), g in D.groupby(["kind", "K"]):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for b in ("herding", "random", "herding_feat"):
            diff = g.meta - g[b]
            row[f"meta-{b}"] = float(diff.mean())
            row[f"meta>{b}"] = int((diff > 0).sum())
        row["oracle-herding"] = float((g.oracle - g.herding).mean())
        for st in ("herding", "random", "herding_feat"):
            row[f"check_{st}_vs_e33"] = float((g[st] - g[f"{st}_e33"]).abs().max())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "audio_hubert"):
        m1 = (R.kind == kind) & R.K.isin(["5", "10"])
        m2 = (R.kind == kind) & R.K.isin(["1", "5", "10"])
        R.loc[R.kind == kind, "T1"] = bool((R[m1]["meta-herding"] > 0).all()) if m1.sum() == 2 else np.nan
        R.loc[R.kind == kind, "T2"] = bool((R[m2]["meta-random"] > 0).all()) if m2.sum() == 3 else np.nan
    D.to_csv(OUT_REL / f"e37_targets{suffix}.csv", index=False)
    R.to_csv(OUT_REL / f"e37_tests{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(D.groupby("kind").policy_star.value_counts().to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=MAIN, help="--analyze の種類（言葉系は --suffix _text で別のファイルに書く）")
    ap.add_argument("--suffix", default="", help="--analyze の出力名の接尾辞")
    ap.add_argument("--no-pseudo", action="store_true", help="仮のターゲットを解かない（ターゲットの反復を足すとき）")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds, args.suffix)
    out = args.out_dir or ROOT / "results" / f"select_teach_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/select_teach.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "pseudo_reps": [] if args.no_pseudo else PSEUDO_REPS,
            "steps": list(CS), "C_pseudo": C_PSEUDO, "cp": CB.DEF_CP, "cb": CB.DEF_CB,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    cache: dict = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, args.ks, out, cache, pseudo=not args.no_pseudo)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
