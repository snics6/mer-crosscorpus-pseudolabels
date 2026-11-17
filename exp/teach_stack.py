"""【探索的】E52（GPU）：生徒の出発点を仲間に入れた見込みで教える選び方を動かし，番を付いた本当のラベルの本数で回す（条件 2．生徒は E39 の学習器）。

規則は `docs/exploratory-analyses.md` E52（設計と予言は計算の前に書いた）。E49 で，見込み q が出発点 p より当たるターゲットでは教える選び方が勝ち，
当たらないターゲットでは負けた．そこで，出発点そのものを仲間に入れた見込みを，E44 の C と同じ形の log-linear pool（仲間の log 確率を 4×4 の表で写して平均し，
表は単位行列へ引く）で作る．仲間は少数の強いものに限り，単位行列の表のときに等しい重みの掛け合わせになるようにする：
  q   ：E44 の C（同じエンコーダのソースごとの分類器の票をまとめたもの．これまでの見込み）
  pq  ：仲間 {p, q}．p は E39 の出発点の候補の上の確率
  pqo ：仲間 {p, q, p_o, q_o}．p_o・q_o は別のエンコーダ（A なら HuBERT）の E39 の出発点と E44 の C
表と切片は，T のソースを 1 つずつ仮のターゲット g にして学ぶ．g の上の仲間は，g のコーパスを除くソースで作る（p・p_o は E39 の作り方（合併の C は 0.01．E40 と同じ），
q・q_o は E44 の C の作り方で λ は E44 がそのエンコーダの T 用に選んだ値）．表の λ は仮のターゲットを 1 つずつ外した UAR で選ぶ（E44 と同じ）．
番の回し方：cycle（0→3 の順．これまで）と count（付いた本当のラベルが最も少ない感情．同数なら番の順）．
見込みは E48 の ④（4 本付くたびに，付いたラベルで学び直した生徒の確率と混ぜる．τ は E47 の値）で直す．
生徒と設定は E49 と同じ E40 の生徒で，教える選び方の模型の座標・混ぜる生徒・判定はすべて E40 が random 用に選んだ組（ρ_M，C_b，C′）．歩幅は E40 の teach の組の c。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/teach_stack.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/teach_stack.py --analyze

出力：`results/teach_stack_<kind>/<ターゲット>.csv`（policy, rho_M, cb, cp, rep, K, uar），`<ターゲット>_q.csv`（直すたびの，まだラベルの無い候補での q の UAR．
m = 0 は見込みそのものの当たり方），`<ターゲット>_sel.csv`（感情の本数，番の感情と本当の感情の一致＝当てた割合，補った回数），`<ターゲット>_meta.csv`（λ と仮のターゲットの UAR），
`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e52_targets.csv`・`e52_tests.csv`・`e52_belief.csv`・`e52_sel.csv`
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from belief_correct import Task  # noqa: E402
from check_lll_cues import PAIR  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402
from teach_student39 import Student, solve_students  # noqa: E402

BELIEFS = ["q", "pq", "pqo"]
TURNS = ["cycle", "count"]
VARIANTS = [f"{b}:{t}" for b in BELIEFS for t in TURNS]
EVERY = 4
C_PSEUDO = 0.01
MAIN = ["audio_wavlm", "audio_hubert"]
OUT_REL = ROOT / "results" / "explore_relation"


def logp(p):
    return np.log(np.clip(p, 1e-12, None))


def softmax_t(L):
    return torch.softmax(L, 1).cpu().numpy()


def draft_logits(data, S, g, gamma, rho_s, cache):
    """仮のターゲット g の全件の上の E39 の出発点の点数（g のコーパスを除くソースで作る．合併の C は 0.01）。"""
    pool_g = [s for s in S if CM.corp(s) != CM.corp(g)]
    src_pc, lin, n_of = M5.pool_info(data, pool_g, C_PSEUDO, cache)
    Xu = data[g].X - data[g].X.mean(0)
    return Student(data[g], Xu, src_pc, lin, n_of, pool_g, C_PSEUDO, gamma, rho_s, [1.0]).L0


def c_pool_on(t, S, P, Y, lam):
    """仮のターゲット t の全件の上の E44 の C（t のコーパスを除くソースの票を，t 以外の群で学んだまとめ方でまとめる）。"""
    St = [s for s in S if CM.corp(s) != CM.corp(t)]
    Gt = [g for g in S if g != t]
    pool = CM.Pool(St).fit(*CM.pool_data(P, Y, St, Gt), lam)
    Ft, At, _, _ = CM.pool_data(P, Y, St, [t])
    return pool.predict(Ft, At)


NAMES = {"pq": ["p", "q"], "pqo": ["p", "q", "p_o", "q_o"]}


def stack_data(names, groups, mem, Y):
    """仲間 names の log 確率（n, M, 4）を groups の全件について並べる（重みは CM.pool_data と同じ．仲間はいつも使える）。"""
    F = [np.stack([logp(mem[g][nm]) for nm in names], 1) for g in groups]
    A = [np.ones((len(Y[g]), len(names))) for g in groups]
    ys = [Y[g] for g in groups]
    ws = [1.0 / (len(groups) * 4 * np.bincount(Y[g], minlength=4)[Y[g]]) for g in groups]
    tt = lambda a, dt=logreg.DTYPE: torch.as_tensor(np.concatenate(a), dtype=dt, device=logreg.DEVICE)  # noqa: E731
    return tt(F), tt(A), tt(ys, torch.long), tt(ws)


def fit_stack(T, belief, S, mem, Y, mrows):
    """仲間の表を仮のターゲットで学ぶ．λ は仮のターゲットを 1 つずつ外した UAR の平均で選ぶ（E44 の C と同じ手順）。"""
    names = NAMES[belief]
    scores = {}
    for lam in CM.LAMS:
        for t in S:
            Gt = [g for g in S if g != t]
            pool_t = CM.Pool(names).fit(*stack_data(names, Gt, mem, Y), lam)
            Ft, At, yt, _ = stack_data(names, [t], mem, Y)
            u = CM.uar(yt.cpu().numpy(), pool_t.predict(Ft, At))
            scores.setdefault(lam, []).append(u)
            mrows.append({"target": T, "belief": belief, "pseudo": t, "lam": lam, "uar": u})
    lam = max(scores, key=lambda k: float(np.mean(scores[k])))
    return CM.Pool(names).fit(*stack_data(names, S, mem, Y), lam), lam


def score(q, R, y):
    un = np.ones(len(y), bool)
    un[R] = False
    a = q[un].argmax(1)
    return {"uar": CB.uar(y[un], a), "acc": float((a == y[un]).mean()), "n_un": int(un.sum())}


class Live:
    """ラベルが EVERY 本付くたびに，付いたラベルで学び直した生徒の確率と見込みを混ぜ（E48 の ④），その最大の感情の 1-hot を返す。"""

    def __init__(self, q0, y, tau, pm_fn, dev):
        self.q0, self.y, self.tau, self.pm_fn, self.dev = q0, np.asarray(y), tau, pm_fn, dev
        self.m, self.log = -1, []
        self._set(q0, np.array([], int))

    def _set(self, q, R):
        self.oh = torch.nn.functional.one_hot(torch.as_tensor(q.argmax(1), device=self.dev), 4).float()
        self.m = len(R)
        self.log.append({"m": self.m, **score(q, R, self.y)})

    def __call__(self, L, P, order):
        m = len(order)
        if m > self.m and m % EVERY == 0:
            R = np.array(order)
            self._set(Task.mix(self.q0, self.pm_fn(R), m, self.tau), R)
        return self.oh


def run_target(kind, T, data, data_o, f, Cs, P, Y, f_o, Cs_o, P_o, reps, ks, out, cache, cache_o):
    other = PAIR[kind]
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"])
    s_, s_o = sel.loc[(kind, T)], sel.loc[(other, T)]
    r_set = (float(s_.r_rho_M), float(s_.r_cb), float(s_.r_cp))
    c_t = float(s_.t_c)
    rhos_m = sorted({float(s_.t_rho_M), float(s_.r_rho_M)})
    tau = float(pd.read_csv(ROOT / "results" / f"belief_correct_{kind}" / f"{T}.csv")["knob_C:mix"].iloc[0])
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    C_To = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{other}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    lam44 = {k: float(pd.read_csv(ROOT / "results" / f"committee_{k}" / f"{T}.csv").query("variant == 'C'").lam.iloc[0]) for k in (kind, other)}
    pool_q = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam44[kind])
    pool_qo = CM.Pool(S).fit(*CM.pool_data(P_o, Y, S, S), lam44[other])
    mem = {g: {"p": softmax_t(draft_logits(data, S, g, float(s_.gamma), float(s_.rho_s), cache)),
               "p_o": softmax_t(draft_logits(data_o, S, g, float(s_o.gamma), float(s_o.rho_s), cache_o)),
               "q": c_pool_on(g, S, P, Y, lam44[kind]), "q_o": c_pool_on(g, S, P_o, Y, lam44[other])} for g in S}
    mrows, pools, lams = [], {}, {}
    for b in ("pq", "pqo"):
        pools[b], lams[b] = fit_stack(T, b, S, mem, Y, mrows)
    for g in S:                                                                    # 記録：仮のターゲットの上の仲間の当たり方
        for nm, v in mem[g].items():
            mrows.append({"target": T, "belief": nm, "pseudo": g, "lam": np.nan, "uar": CM.uar(Y[g], v)})
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    src_o, lin_o, n_of_o = M5.pool_info(data_o, S, C_To, cache_o)
    ev, ev_o = data[T], data_o[T]
    ctx, ctx_o = CB.Ctx(ev, reps, ks, center="ind"), CB.Ctx(ev_o, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {v: {} for v in VARIANTS}
    rows, srows, qrows, students = [], [], [], {}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        if not (np.flatnonzero(~ctx_o.te[r]) == pool).all():
            raise ValueError(f"候補が 2 つのエンコーダで違う: {T} r{r}")
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        Xc_o = M.rows(ev_o.X, pool) - ctx_o.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, float(s_.gamma), float(s_.rho_s), rhos_m)
        st_o = Student(ev_o, Xc_o, src_o, lin_o, n_of_o, S, C_To, float(s_o.gamma), float(s_o.rho_s), [1.0])
        students[r] = st
        Ps = [softmax_t(f[s].logits(Xc, Cs[s])) for s in S]
        Ps_o = [softmax_t(f_o[s].logits(Xc_o, Cs_o[s])) for s in S]
        tens = lambda cc: (torch.as_tensor(np.stack(cc, 1), dtype=logreg.DTYPE, device=logreg.DEVICE),  # noqa: E731
                           torch.ones(len(yq), len(cc), dtype=logreg.DTYPE, device=logreg.DEVICE))
        m1 = {"p": softmax_t(st.L0), "p_o": softmax_t(st_o.L0), "q": pool_q.predict(*tens([logp(x) for x in Ps])),
              "q_o": pool_qo.predict(*tens([logp(x) for x in Ps_o]))}
        q0 = {"q": m1["q"], **{b: pools[b].predict(*tens([logp(m1[nm]) for nm in NAMES[b]])) for b in ("pq", "pqo")}}
        qrows += [{"target": T, "policy": nm, "rep": r, "m": 0, **score(v, np.array([], int), yq)} for nm, v in m1.items() if nm != "q"]
        Xt = (st.Xstd @ st.mats[r_set[0]][0]).float()
        Km = Xt @ Xt.T
        Km.fill_diagonal_(0.0)
        pm_fn = lambda R, st=st, yq=yq: st.fit_pm(R, yq, *r_set)  # noqa: E731
        for v in VARIANTS:
            b, turn = v.split(":")
            live = Live(q0[b], yq, tau, pm_fn, Xt.device)
            order, turns, fills = ablate_order(Km, yq, st.L0, c_t, 4 * kmax, "PPP", belief=live, belief_order=True, turn=turn)
            qrows += [{"target": T, "policy": v, "rep": r, **x} for x in live.log]
            for K in ks:
                n = 4 * int(K)
                pp = np.array(order[:n])
                if len(np.unique(pp)) != n:
                    raise ValueError(f"選んだ本数がおかしい: {v} r{r} K{K}")
                idx[v][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": v, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max()),
                              "turn_match": float((yq[pp] == np.array(turns[:n])).mean()), "filled": int(sum(fills[:n]))})
        del Km
    solve_students(T, ev, ctx, reps, ks, students, idx, rhos_m, C_T, rows)
    meta = pd.DataFrame(mrows)
    meta["chosen"] = [lams.get(b) == l for b, l in zip(meta.belief, meta.lam)]
    meta.to_csv(out / f"{T}_meta.csv", index=False)
    pd.DataFrame(qrows).to_csv(out / f"{T}_q.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"])
    e49 = pd.read_csv(OUT_REL / "e49_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    f50 = OUT_REL / "e50_targets.csv"
    e50 = pd.read_csv(f50, dtype={"K": str}).set_index(["kind", "target", "K"]) if f50.exists() else None
    refs = {"random_pool": "random_pool", "herding_pred": "herding_pred", "herding_pred_p4": "herding_pred_p4", "random_true": "random_true",
            "LLL": "LLL", "C:0000@r": "C:0000@r", "C:0001@r": "C:0001@r", "C:0001": "C:0001"}
    al = ["margin", "badge", "coreset", "typiclust"]
    tabs, btabs, stabs = [], [], []
    for kind in kinds:
        d = ROOT / "results" / f"teach_stack_{kind}"
        fs = [x for x in sorted(d.glob("*.csv")) if not x.stem.endswith(("_sel", "_q", "_meta"))]
        if not fs:
            continue
        rsel = pd.concat([pd.read_csv(x) for x in sorted((ROOT / "results" / f"teach_student39_{kind}").glob("*_sel.csv"))])
        rsel = rsel[(rsel.policy == "random_pool") & rsel.rep.isin(EC.REPS)].groupby(["target", "K"])[["missing", "min_count"]].mean()
        for x in fs:
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str, "cb": str})
            a = a[a.rep.isin(EC.REPS)]
            u = a.groupby(["policy", "rho_M", "cb", "cp", "K"]).uar.mean() * 100
            s = sel.loc[(kind, T)]
            rk = (float(s.r_rho_M), f"{float(s.r_cb):g}", float(s.r_cp))
            q = pd.read_csv(d / f"{T}_q.csv")
            q = q[q.rep.isin(EC.REPS)]
            b0 = q[q.m == 0].groupby("policy").uar.mean() * 100
            bel = {"p": float(b0["p"]), "p_o": float(b0["p_o"]), "q_o": float(b0["q_o"]), **{b: float(b0[f"{b}:cycle"]) for b in BELIEFS}}
            btabs.append({"kind": kind, "target": T, **bel, **{f"m{m}:{v}": float(q[(q.policy == v) & (q.m == m)].uar.mean() * 100)
                                                                 for v in VARIANTS for m in (4, 20, 40, 96)}})
            sl = pd.read_csv(d / f"{T}_sel.csv")
            sl = sl[sl.rep.isin(EC.REPS)]
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K}
                for v in VARIANTS:
                    row[v] = float(u.loc[(v,) + rk + (K,)])
                for k_, v_ in refs.items():
                    row[k_] = float(e49.loc[(kind, T, K), v_])
                if e50 is not None:
                    for p_ in al:
                        row[p_] = float(e50.loc[(kind, T, K), p_])
                row.update({f"bel_{b}": bel[b] for b in bel})
                tabs.append(row)
                g = sl[sl.K == int(K)].groupby("policy")[["missing", "min_count", "turn_match", "filled"]].mean()
                srow = {"kind": kind, "target": T, "K": K, "random_pool_missing": float(rsel.loc[(T, int(K)), "missing"]),
                        "random_pool_min_count": float(rsel.loc[(T, int(K)), "min_count"])}
                for v in VARIANTS:
                    for c in g.columns:
                        srow[f"{c}:{v}"] = float(g.loc[v, c])
                stabs.append(srow)
    D, Bt, St = pd.DataFrame(tabs), pd.DataFrame(btabs), pd.DataFrame(stabs)
    main_v = "pqo:count"
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for x in VARIANTS + list(refs) + ([p_ for p_ in al if p_ in g] if e50 is not None else []):
            row[x] = float(g[x].mean())
        for v in VARIANTS:
            for ref in ("random_pool", "herding_pred", "C:0001@r"):
                diff = g[v] - g[ref]
                row[f"{v}-{ref}"] = float(diff.mean())
                row[f"{v}>{ref}"] = int((diff > 0).sum())
        for b in ("pq", "pqo"):
            row[f"{b}:count-{b}:cycle"] = float((g[f"{b}:count"] - g[f"{b}:cycle"]).mean())
        row["q:count-q:cycle"] = float((g["q:count"] - g["q:cycle"]).mean())
        sg = St[(St.kind == kind) & (St.K == K)]
        row["missing:pqo:count"] = float(sg[f"missing:{main_v}"].mean())
        row["missing:random_pool"] = float(sg.random_pool_missing.mean())
        row["hit:pqo:count"] = float(sg[f"turn_match:{main_v}"].mean())
        row["hit:pqo:cycle"] = float(sg["turn_match:pqo:cycle"].mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    k5 = D[D.K == "5"]
    rho = float(spearmanr(k5.bel_pqo - k5.bel_p, k5[main_v] - k5.random_pool).statistic) if len(k5) > 2 else np.nan
    R["spearman_X3"] = rho
    for kind in MAIN:
        r5 = R[(R.kind == kind) & (R.K == "5")]
        if len(r5):
            r5 = r5.iloc[0]
            R.loc[R.kind == kind, "X1"] = bool(r5[f"{main_v}-random_pool"] > 0)
            R.loc[R.kind == kind, "X2"] = bool(r5[f"{main_v}-herding_pred"] > 0)
            R.loc[R.kind == kind, "X4"] = bool(r5["missing:pqo:count"] <= r5["missing:random_pool"])
    R["X3"] = bool(rho >= 0.5) if not np.isnan(rho) else np.nan
    D.to_csv(OUT_REL / "e52_targets.csv", index=False)
    R.to_csv(OUT_REL / "e52_tests.csv", index=False)
    Bt.to_csv(OUT_REL / "e52_belief.csv", index=False)
    St.to_csv(OUT_REL / "e52_sel.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in R if c in ("X1", "X2", "X3", "X4")]).round(2).T.to_string())
    print("\n== 見込みの当たり方（候補の上の UAR．m = 0．ターゲットの平均）")
    print(Bt.groupby("kind")[["p", "p_o", "q_o"] + BELIEFS].mean().round(2).to_string())
    print("\n== 見込みの当たり方（ターゲットごと）")
    print(Bt[["kind", "target", "p", "q", "pq", "p_o", "q_o", "pqo"]].round(1).to_string(index=False))
    print("\n== K=5 のターゲットごと（主な版 − random_pool と，見込み pqo − 出発点 p）")
    print(k5.assign(gain=k5[main_v] - k5.random_pool, adv=k5.bel_pqo - k5.bel_p)[["kind", "target", "gain", "adv"]].round(2).to_string(index=False))
    vc = [c for c in ("X1", "X2", "X3", "X4") if c in R]
    print(R.dropna(subset=["X1"]).drop_duplicates("kind")[["kind"] + vc].to_string(index=False))
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
    out = args.out_dir or ROOT / "results" / f"teach_stack_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    other = PAIR[args.kind]
    data = {g: M.load(g, args.kind) for g in groups}
    data_o = {g: M.load(g, other) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    f_o, Cs_o, P_o, _ = CM.prepare(other, data_o)
    prov = {"script": "exp/teach_stack.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "variants": VARIANTS, "every": EVERY, "other": other,
            "c_pseudo": C_PSEUDO, "lams": CM.LAMS, "src_C": Cs, "src_C_other": Cs_o,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache, cache_o = {}, {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, data_o, f, Cs, P, Y, f_o, Cs_o, P_o, args.reps, args.ks, out, cache, cache_o)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
