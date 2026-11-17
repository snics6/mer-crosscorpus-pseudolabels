"""【探索的】E53（GPU）：条件 2 の主な比べを言葉（T．RoBERTa の特徴）で行う（生徒は E39 の学習器）。

規則は `docs/exploratory-analyses.md` E53（設計と予言は計算の前に書いた）。E49・E50・E52 を言葉の 4 ターゲットでまとめて行う。
生徒は E40 と同じ作りの E39 の学習器で，設定（γ, ρ_s, C_b, C′, ρ_M）は E39 が言葉でターゲットごとに選んだ値（`e39_text_targets.csv`．random の K 本で選んだもの）。
教える選び方の歩幅は E37 が言葉でメタ学習した値（`e37_targets_text.csv` の policy_star），模型の座標は上の ρ_M の計量で直す。
見込み q は E44 の C と同じ作り（λ は仮のターゲットを 1 つずつ外して選ぶ．言葉では E44 を流していないのでここで選ぶ）。
④ で混ぜる重みの τ は，言葉では E47 を流していないので 40 に固定する（E47 で選ばれたのは 40 と 160 が 13 ターゲットずつ）。
別のエンコーダ（DeBERTa）は C を固定した地図も E39 の値も無いので使わない（E52 の pqo は行わない）。
  random_true・random_pool・herding_pred・herding_pred_p4 ：E49 と同じ
  margin・badge・coreset・typiclust                        ：E50 と同じ（margin・badge は E39 の下書きの確率）
  LLL                                                       ：条件 1 の教える選び方
  q:fixed ：q を直さずに入れた教える選び方（E45・E49 の C:0000 にあたる）
  q:cycle ：q を ④ で直す（E52 の q:cycle）
  pq:cycle・pq:count ：仲間 {出発点 p, q} の見込みを ④ で直す（E52 の pq）．番は 0→3 の順か，付いた本当のラベルが最も少ない感情
比べのために P4 の生徒（仮説転移 4 の既定）で random_pool と random_true も解く（P4:random_pool，P4:random_true）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/text_cond2.py [--target …] [--reps …]
    ./.venv/bin/python exp/text_cond2.py --analyze

出力：`results/text_cond2_text_roberta/<ターゲット>.csv`（policy, rho_M, cb, cp, rep, K, uar），`<ターゲット>_q.csv`，`<ターゲット>_sel.csv`，
`<ターゲット>_meta.csv`，`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e53_targets.csv`・`e53_tests.csv`・`e53_belief.csv`
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
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from belief_check import knn  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_baselines import K_TYP, order_badge, order_coreset, order_margin, pick_typiclust  # noqa: E402
from select_herding import herd  # noqa: E402
from select_labelfree import fill, ht_block_present  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402
from teach_stack import Live, c_pool_on, draft_logits, fit_stack, logp, score, softmax_t  # noqa: E402
from teach_student39 import Student, solve_students  # noqa: E402

KIND = "text_roberta"
TAU = 40.0
RANDY = ["random_true", "random_pool", "herding_pred", "herding_pred_p4"]
AL = ["margin", "badge", "coreset", "typiclust"]
TEACHY = ["LLL", "q:fixed", "q:cycle", "pq:cycle", "pq:count"]
P4REF = ["P4:random_pool", "P4:random_true"]
OUT_REL = ROOT / "results" / "explore_relation"


def choose_lam(S, P, Y):
    """E44 の C の λ を，仮のターゲット（S の群）を 1 つずつ外した UAR の平均で選ぶ（committee.run_target と同じ手順）。"""
    sc = {lam: [] for lam in CM.LAMS}
    for t in S:
        Gt = [g for g in S if g != t]
        Ftr, Atr, ytr, wtr = CM.pool_data(P, Y, S, Gt)
        Fte, Ate, _, _ = CM.pool_data(P, Y, S, [t])
        for lam in CM.LAMS:
            sc[lam].append(CM.uar(Y[t], CM.Pool(S).fit(Ftr, Atr, ytr, wtr, lam).predict(Fte, Ate)))
    return max(sc, key=lambda k: float(np.mean(sc[k]))), sc


def run_target(T, data, f, Cs, P, Y, reps, ks, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    s_ = pd.read_csv(OUT_REL / "e39_text_targets.csv", dtype={"K": str}).query("kind == @KIND").drop_duplicates("target").set_index("target").loc[T]
    r_set = (float(s_.rho_M), float(s_.cb), float(s_.cp))
    c_t = float(pd.read_csv(OUT_REL / "e37_targets_text.csv", dtype={"K": str}).query("kind == @KIND").drop_duplicates("target")
                .set_index("target").loc[T, "policy_star"].split("_c")[1])
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{KIND}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    lam, sc = choose_lam(S, P, Y)
    mrows = [{"target": T, "belief": "q", "pseudo": t, "lam": l, "uar": u} for l, us in sc.items() for t, u in zip(S, us)]
    pool_q = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam)
    mem = {g: {"p": softmax_t(draft_logits(data, S, g, float(s_.gamma), float(s_.rho_s), cache)), "q": c_pool_on(g, S, P, Y, lam)} for g in S}
    pool_pq, lam_pq = fit_stack(T, "pq", S, mem, Y, mrows)
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {p: {} for p in RANDY + AL + TEACHY}
    rows, srows, qrows, students = [], [], [], {}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, float(s_.gamma), float(s_.rho_s), [r_set[0]])
        students[r] = st
        Ps = [softmax_t(f[s].logits(Xc, Cs[s])) for s in S]
        tens = lambda cc: (torch.as_tensor(np.stack(cc, 1), dtype=logreg.DTYPE, device=logreg.DEVICE),  # noqa: E731
                           torch.ones(len(yq), len(cc), dtype=logreg.DTYPE, device=logreg.DEVICE))
        p = softmax_t(st.L0)
        q0 = {"q": pool_q.predict(*tens([logp(x) for x in Ps]))}
        q0["pq"] = pool_pq.predict(*tens([logp(p), logp(q0["q"])]))
        qrows.append({"target": T, "policy": "p", "rep": r, "m": 0, **score(p, np.array([], int), yq)})
        X = st.Xstd.float()
        Xt = (st.Xstd @ st.mats[r_set[0]][0]).float()
        Km = Xt @ Xt.T
        Km.fill_diagonal_(0.0)
        pm_fn = lambda R, st=st, yq=yq: st.fit_pm(R, yq, *r_set)  # noqa: E731
        orders = {"LLL": ablate_order(Km, yq, st.L0, c_t, 4 * kmax, "LLL")}
        ohq = torch.nn.functional.one_hot(torch.as_tensor(q0["q"].argmax(1), device=Xt.device), 4).float()
        orders["q:fixed"] = ablate_order(Km, yq, st.L0, c_t, 4 * kmax, "PPP", belief=lambda L, P_: ohq)
        qrows.append({"target": T, "policy": "q:fixed", "rep": r, "m": 0, **score(q0["q"], np.array([], int), yq)})
        for v in ("q:cycle", "pq:cycle", "pq:count"):
            b, turn = v.split(":")
            live = Live(q0[b], yq, TAU, pm_fn, Xt.device)
            orders[v] = ablate_order(Km, yq, st.L0, c_t, 4 * kmax, "PPP", belief=live, belief_order=True, turn=turn)
            qrows += [{"target": T, "policy": v, "rep": r, **x} for x in live.log]
        del Km
        P39 = torch.softmax(st.L0, 1).float()
        gen = lambda name: torch.Generator().manual_seed(zlib.crc32(f"e46|{name}|{T}|{r}".encode()))  # noqa: E731
        al = {"margin": order_margin(P39), "badge": order_badge(X, P39, 4 * kmax, gen("badge")), "coreset": order_coreset(X, 4 * kmax)}
        Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nb = knn(X, K_TYP)
        typ = 1.0 / (1.0 - (Xn[:, None, :] * Xn[nb]).sum(2)).mean(1).clamp(min=1e-6)
        pos_of = {int(i): j for j, i in enumerate(pool)}
        perm = np.random.default_rng(zlib.crc32(f"e41pool|{T}|{r}".encode())).permutation(len(pool))
        preds = {"herding_pred": st.L0.argmax(1).cpu().numpy(), "herding_pred_p4": src_pc.logits(Xc, C_T).argmax(1).cpu().numpy()}
        for K in ks:
            kk = int(K)
            picks = {p_: np.array(orders[p_][0][:4 * kk]) for p_ in TEACHY}
            picks.update({p_: np.array(al[p_][:4 * kk]) for p_ in al})
            picks["typiclust"] = np.array(pick_typiclust(X, typ, 4 * kk, gen(f"typiclust{K}")))
            picks["random_true"] = np.array([pos_of[int(i)] for i in ctx.idx[(r, K)]])
            picks["random_pool"] = perm[:4 * kk]
            for hname, pred in preds.items():
                groups = [np.flatnonzero(pred == g) for g in range(4)]
                hord = [list(G[herd(st.Xstd[torch.as_tensor(G, device=CB.DEV)], min(len(G), 4 * kk))]) if len(G) else [] for G in groups]
                picks[hname] = np.array(fill(groups, [o[:kk] for o in hord], kk, lambda j, ps, hord=hord: hord[j][len(ps)]))
            for p_, pp in picks.items():
                if len(pp) != 4 * kk or len(np.unique(pp)) != len(pp):
                    raise ValueError(f"選んだ本数がおかしい: {p_} r{r} K{K}")
                idx[p_][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": p_, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max())})
    solve_students(T, ev, ctx, reps, ks, students, idx, [r_set[0]], C_T, rows)
    for ref in P4REF:                                                           # P4 の生徒（仮説転移 4 の既定．E41 と同じ）
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[ref.split(":")[1]]

        def add(r, K, c, u, cv, L, cen=0, cb="", ref=ref):
            rows.append({"target": T, "policy": ref, "rho_M": np.nan, "cb": "", "cp": np.nan, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src_pc, C_T, add)
    meta = pd.DataFrame(mrows)
    meta["chosen"] = [(b == "q" and l == lam) or (b == "pq" and l == lam_pq) for b, l in zip(meta.belief, meta.lam)]
    meta.to_csv(out / f"{T}_meta.csv", index=False)
    pd.DataFrame(qrows).to_csv(out / f"{T}_q.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze() -> int:
    s39 = pd.read_csv(OUT_REL / "e39_text_targets.csv", dtype={"K": str, "cb": str}).query("kind == @KIND").set_index(["target", "K"])
    d = ROOT / "results" / f"text_cond2_{KIND}"
    tabs, btabs = [], []
    for x in sorted(f for f in d.glob("*.csv") if not f.stem.endswith(("_sel", "_q", "_meta"))):
        T = x.stem
        a = pd.read_csv(x, dtype={"K": str, "cb": str})
        a = a[a.rep.isin(EC.REPS)]
        s = s39.loc[(T, "5")]
        rk = (float(s.rho_M), f"{float(s.cb):g}", float(s.cp))
        u = a[a.policy.isin(RANDY + AL + TEACHY)].groupby(["policy", "rho_M", "cb", "cp", "K"]).uar.mean() * 100
        u4 = a[a.policy.isin(P4REF)].groupby(["policy", "K"]).uar.mean() * 100
        q = pd.read_csv(d / f"{T}_q.csv")
        q = q[q.rep.isin(EC.REPS)]
        b0 = q[q.m == 0].groupby("policy").uar.mean() * 100
        btabs.append({"target": T, "p": float(b0["p"]), "q": float(b0["q:fixed"]), "pq": float(b0["pq:cycle"])})
        for K in ("1", "5", "10", "25"):
            row = {"target": T, "K": K}
            for p_ in RANDY + AL + TEACHY:
                row[p_] = float(u.loc[(p_,) + rk + (K,)])
            for p_ in P4REF:
                row[p_] = float(u4.loc[(p_, K)])
            row["chk_e39"] = abs(row["random_true"] - float(s39.loc[(T, K), "meta"]))
            tabs.append(row)
    D, Bt = pd.DataFrame(tabs), pd.DataFrame(btabs)
    lf = ["herding_pred", "herding_pred_p4"] + AL + ["q:fixed", "q:cycle", "pq:cycle", "pq:count"]
    tests = []
    for K, g in D.groupby("K", sort=False):
        row = {"K": K, "targets": len(g)}
        for x in RANDY + AL + TEACHY + P4REF:
            row[x] = float(g[x].mean())
        for x in lf + ["random_true", "LLL"]:
            diff = g[x] - g.random_pool
            row[f"{x}-random_pool"] = float(diff.mean())
            row[f"{x}>random_pool"] = int((diff > 0).sum())
        row["random_pool-P4:random_pool"] = float((g.random_pool - g["P4:random_pool"]).mean())
        row["random_true-P4:random_true"] = float((g.random_true - g["P4:random_true"]).mean())
        row["chk_e39"] = float(g.chk_e39.max())
        tests.append(row)
    R = pd.DataFrame(tests)
    r5, r1 = R[R.K == "5"].iloc[0], R[R.K == "1"].iloc[0]
    R["TX1"] = bool(max(r5[f"{x}-random_pool"] for x in lf) < 1.0)
    R["TX2"] = bool(r5["random_pool-P4:random_pool"] >= 1.0)
    R["TX3"] = bool(r1["pq:cycle-random_pool"] > 0)
    D.to_csv(OUT_REL / "e53_targets.csv", index=False)
    R.to_csv(OUT_REL / "e53_tests.csv", index=False)
    Bt.to_csv(OUT_REL / "e53_belief.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=["TX1", "TX2", "TX3"]).set_index("K").round(2).T.to_string())
    print("\n== 見込みの当たり方（候補の上の UAR．直す前）")
    print(Bt.round(2).to_string(index=False))
    print("\n== K=5 のターゲットごと（random_pool との差）")
    k5 = D[D.K == "5"].set_index("target")
    print(k5[lf + ["random_true", "LLL"]].sub(k5.random_pool, axis=0).round(2).to_string())
    print(R[["TX1", "TX2", "TX3"]].iloc[:1].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        return analyze()
    out = args.out_dir or ROOT / "results" / f"text_cond2_{KIND}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(KIND)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{KIND}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, KIND) for g in groups}
    f, Cs, P, Y = CM.prepare(KIND, data)
    prov = {"script": "exp/text_cond2.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(KIND)), "K": args.ks, "reps": args.reps, "tau": TAU, "lams": CM.LAMS, "src_C": Cs,
            "policies": RANDY + AL + TEACHY + P4REF, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(t, data, f, Cs, P, Y, args.reps, args.ks, out, cache)
        print(f"{KIND} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
