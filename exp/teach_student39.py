"""【探索的】E49（GPU）：条件 2 の主な比べを，生徒を E39 の学習器（5 つの値を選ぶ）に替えて測り直す。

規則は `docs/exploratory-analyses.md` E49（設計と予言は計算の前に書いた）。生徒は E40 と同じ（E39 の下書き（γ, ρ_s はターゲットごとに E39 が選んだ値），
罰則の計量 M_ρM で座標を直した仮説転移）で，感情の重みだけ present_weights にする（4 つそろえば E40 と同じ）。設定（教える選び方の歩幅 c，ρ_M，C_b，C′）は
E40 がターゲットごとに仮のターゲットで選んだ値を使う：教える選び方の版は teach の組（t_*），それ以外は random の組（r_*）。
教える選び方の中の模型は，E40 と同じく計量で直した座標 x̃ M_ρM^{−1/2} と E39 の下書きの点数から始める。選び方：
  random_true    ：ラベルを知り，感情ごとにランダム（E31 と同じ K 本．E40 の random と一致することを確かめる）
  LLL            ：条件 1 の教える選び方（E40 の teach と一致することを確かめる）
  random_pool    ：ラベル無しで候補から 4K 本（E41 と同じ順）
  herding_pred   ：E39 の下書きの予測の群の中で herding（E34 の作りで，群だけ替える）
  herding_pred_p4：P4 の出発点の予測の群の中で herding（E34 と同じ K 本）
  C:0000         ：E44 の C を直さずに入れた教える選び方（E45 の作り）
  C:0001         ：ラベルが 4 本付くたびに，付いたラベルで学び直した生徒の確率 p_m と q を混ぜて入れ直す（E48 の作り．τ は E47 の値）
各版は ρ_M ∈ {t_ρM, r_ρM} の生徒の C′ × C_b を束ねて解いて全部残し，判定では上の決まりの組を使う。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/teach_student39.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/teach_student39.py --analyze

出力：`results/teach_student39_<kind>/<ターゲット>.csv`（policy, rho_M, cb, cp, rep, K, uar），`<ターゲット>_q.csv`，`<ターゲット>_sel.csv`，
`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e49_targets.csv`・`e49_tests.csv`・`e49_checks.csv`
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
import combine_drafts as CD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
import select_teach as ST  # noqa: E402
from belief_correct import AGGS, CORR, Task, knob_key  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_herding import herd  # noqa: E402
from select_labelfree import fill, present_weights  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402

TEACHY = ["LLL", "C:0000", "C:0001"]
RANDY = ["random_true", "random_pool", "herding_pred", "herding_pred_p4"]
EVERY = 4
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


class Student:
    """E40 の生徒（E39 の下書きからの仮説転移．座標は計量で直す）。1 つの反復の候補の上で作る。"""

    def __init__(self, ev, Xu, src_pc, lin, n_of, pool, C, gamma, rho_s, rhos_m):
        dev, dt = src_pc.mu.device, src_pc.mu.dtype
        self.zero = torch.zeros(ev.X.shape[1], dtype=dt, device=dev)
        self.one = torch.ones(ev.X.shape[1], dtype=dt, device=dev)
        self.C, self.src = C, src_pc
        ref_scale = CD.pair_scale(src_pc.logits(Xu, C))
        comps = {}
        for s in pool:                                                         # E40（teach_joint.evaluate）と同じ計算
            A, a0 = lin[(s, rho_s)]
            sc = CD.pair_scale(Xu @ A.T + a0)
            comps[s] = (A / sc, a0 / sc)
        w = {s: float(n_of[s]) ** gamma for s in pool}
        tot = sum(w.values())
        A = sum(w[s] / tot * comps[s][0] for s in pool)
        a0 = sum(w[s] / tot * comps[s][1] for s in pool)
        g = ref_scale / CD.pair_scale(Xu @ A.T + a0)
        A, a0 = g * A, g * a0
        self.W_std, self.b_std = A * src_pc.sd, a0 + src_pc.mu @ A.T
        self.Xstd = (Xu - src_pc.mu) / src_pc.sd
        Zp = self.Xstd - self.Xstd.mean(0)
        lam, V = torch.linalg.eigh(Zp.T @ Zp / len(Xu))
        lam = lam.clamp(min=0.0)
        tau = lam.mean()
        del Zp
        self.mats = {}
        for rho_m in rhos_m:
            e_ = (1 - rho_m) * lam / tau + rho_m
            self.mats[rho_m] = ((V * e_.rsqrt()) @ V.T, (V * e_.sqrt()) @ V.T)
        self.L0 = Xu @ A.T + a0

    def anchor(self, rho_m):
        Mh = self.mats[rho_m][1]
        return logreg.Fitted(self.zero, self.one, {self.C: self.W_std @ Mh}, {self.C: self.b_std}, {self.C: 0}, {self.C: 0.0})

    def fit_pm(self, R, y, rho_m, cb, cp) -> np.ndarray:
        """候補の位置 R のラベルで学び直した生徒の，候補の上の確率。"""
        Am = self.mats[rho_m][0]
        X = self.Xstd[torch.as_tensor(R, device=self.Xstd.device)] @ Am
        yX = M.labels(y[R])
        w = torch.zeros(len(yX), dtype=logreg.DTYPE, device=CB.DEV)
        w[:] = present_weights(yX)
        fit = logreg.fit_batch(X, yX, [np.ones(len(R), bool)], (cp,), anchor=self.anchor(rho_m), anchor_c=self.C,
                               stats=(self.zero, self.one), Cb=[cb], weights=[w])[0]
        return torch.softmax(fit.logits(self.Xstd @ Am, cp), 1).cpu().numpy()


class Live:
    """ラベルが EVERY 本付くたびに q を直し，その最大の感情の 1-hot を返す見込み（E48 の Live の p_m を生徒に替えたもの）。"""

    def __init__(self, task, combo, knobs, pm_fn, dev):
        self.task, self.combo, self.knobs, self.pm_fn, self.dev = task, combo, knobs, pm_fn, dev
        self.m, self.log = -1, []
        self._set(task.q0["C"], np.array([], int))

    def _set(self, q, R):
        self.oh = torch.nn.functional.one_hot(torch.as_tensor(q.argmax(1), device=self.dev), 4).float()
        self.m = len(R)
        self.log.append({"m": self.m, **self.task.score(q, R)})

    def __call__(self, L, P, order):
        m = len(order)
        if m > self.m and m % EVERY == 0:
            R = np.array(order)
            pm = self.pm_fn(R) if self.combo[3] == "1" else None
            self._set(self.task.corrected("C", self.combo, R, m, pm, self.knobs), R)
        return self.oh


def run_target(kind, T, data, f, Cs, P, Y, reps, ks, out, cache):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    H, Hs = CM.habits(P, Y, S, S)
    e44 = pd.read_csv(ROOT / "results" / f"committee_{kind}" / f"{T}.csv")
    pool_C = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), float(e44[e44.variant == "C"].lam.iloc[0]))
    e47 = pd.read_csv(ROOT / "results" / f"belief_correct_{kind}" / f"{T}.csv")
    knobs = {knob_key(c, "C"): float(e47[f"knob_C:{c}"].iloc[0]) for c in CORR}
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str, "t_cb": str, "r_cb": str}).drop_duplicates(["kind", "target"])
    sel = sel.set_index(["kind", "target"]).loc[(kind, T)]
    t_set = (float(sel.t_rho_M), float(sel.t_cb), float(sel.t_cp))
    c_t = float(sel.t_c)
    rhos_m = sorted({float(sel.t_rho_M), float(sel.r_rho_M)})
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {p: {} for p in TEACHY + RANDY}
    rows, srows, qrows = [], [], []
    students = {}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, float(sel.gamma), float(sel.rho_s), rhos_m)
        students[r] = st
        X = st.Xstd.float()                                                     # x̃（E48 と同じ座標．近所に使う．herding は E34 と同じく st.Xstd）
        Ps = [torch.softmax(f[s].logits(Xc, Cs[s]), 1).cpu().numpy() for s in S]
        task = Task(Ps, S, H, Hs, pool_C, yq, X, Xc, src_pc, C_T)
        Xt = (st.Xstd @ st.mats[t_set[0]][0]).float()                          # 教える選び方の模型の座標（E40 と同じ）
        Km = Xt @ Xt.T
        Km.fill_diagonal_(0.0)
        orders = {"LLL": ablate_order(Km, yq, st.L0, c_t, 4 * kmax, "LLL")}
        oh0 = torch.nn.functional.one_hot(torch.as_tensor(task.q0["C"].argmax(1), device=X.device), 4).float()
        orders["C:0000"] = ablate_order(Km, yq, st.L0, c_t, 4 * kmax, "PPP", belief=lambda L, P_: oh0)
        live = Live(task, "0001", knobs, lambda R: st.fit_pm(R, yq, *t_set), X.device)
        orders["C:0001"] = ablate_order(Km, yq, st.L0, c_t, 4 * kmax, "PPP", belief=live, belief_order=True)
        qrows += [{"target": T, "policy": "C:0001", "rep": r, **x} for x in live.log]
        del Km
        pos_of = {int(i): j for j, i in enumerate(pool)}
        perm = np.random.default_rng(zlib.crc32(f"e41pool|{T}|{r}".encode())).permutation(len(pool))
        preds = {"herding_pred": st.L0.argmax(1).cpu().numpy(), "herding_pred_p4": src_pc.logits(Xc, C_T).argmax(1).cpu().numpy()}
        for K in ks:
            kk = int(K)
            picks = {p: np.array(orders[p][0][:4 * kk]) for p in TEACHY}
            picks["random_true"] = np.array([pos_of[int(i)] for i in ctx.idx[(r, K)]])
            picks["random_pool"] = perm[:4 * kk]
            for hname, pred in preds.items():                                  # E34（select_labelfree.run_target）と同じ作り
                groups = [np.flatnonzero(pred == g) for g in range(4)]
                hord = [list(G[herd(st.Xstd[torch.as_tensor(G, device=CB.DEV)], min(len(G), 4 * kk))]) if len(G) else [] for G in groups]
                picks[hname] = np.array(fill(groups, [o[:kk] for o in hord], kk, lambda j, ps, hord=hord: hord[j][len(ps)]))
            for p, pp in picks.items():
                if len(pp) != 4 * kk or len(np.unique(pp)) != len(pp):
                    raise ValueError(f"選んだ本数がおかしい: {p} r{r} K{K}")
                idx[p][(r, K)] = pool[pp]
                cnt = np.bincount(yq[pp], minlength=4)
                srows.append({"target": T, "policy": p, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                              "min_count": int(cnt.min()), "max_count": int(cnt.max())})
    solve_students(T, ev, ctx, reps, ks, students, idx, rhos_m, C_T, rows)
    pd.DataFrame(qrows).to_csv(out / f"{T}_q.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


def solve_students(T, ev, ctx, reps, ks, students, idx, rhos_m, C_T, rows):
    """選び方ごとの K 本（idx[選び方][(r, K)]）で，各反復の生徒を ρ_M ∈ rhos_m × C′ × C_b の全部で解き，テストの UAR を rows に足す。"""
    for p in idx:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[p]
        for r in reps:
            st = students[r]
            used = np.unique(np.concatenate([sub.idx[(r, K)] for K in ks]))
            jobs, masks = CB.jobs_for(sub, [r], {int(i): k for k, i in enumerate(used)}, len(used), False)
            yX = M.labels(ev.y[used])
            weights = []
            for m in masks:
                w = torch.zeros(len(yX), dtype=logreg.DTYPE, device=CB.DEV)
                mt = torch.as_tensor(m, device=CB.DEV)
                w[mt] = present_weights(yX[mt])
                weights.append(w)
            for rho_m in rhos_m:
                Am = st.mats[rho_m][0]
                Xs_used = ((M.rows(ev.X, used) - ctx.ct[r] - st.src.mu) / st.src.sd) @ Am
                fits = logreg.fit_batch(Xs_used, yX, masks * len(M5_CBS), M5_CPS, anchor=st.anchor(rho_m), anchor_c=C_T,
                                        stats=(st.zero, st.one), Cb=[cb for cb in M5_CBS for _ in masks], weights=weights * len(M5_CBS))
                X_of = lambda r_, ii, Am=Am, st=st: ((M.rows(ev.X, ii) - ctx.ct[r] - st.src.mu) / st.src.sd) @ Am  # noqa: E731
                for k, cb in enumerate(M5_CBS):
                    CB.collect(sub, jobs, fits[k * len(masks):(k + 1) * len(masks)], M5_CPS, X_of, None,
                               lambda rr, K, cp, u, cv, L, cb=cb, rho_m=rho_m, p=p: rows.append(
                                   {"target": T, "policy": p, "rho_M": rho_m, "cb": f"{cb:g}", "cp": cp, "rep": rr, "K": K, "uar": u}), {})


M5_CPS = (0.003, 0.01, 0.03, 0.1, 0.3)                                         # E40 の C′ と C_b の候補
M5_CBS = (0.5, 2.0, 8.0)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str, "t_cb": str, "r_cb": str}).drop_duplicates(["kind", "target"])
    sel = sel.set_index(["kind", "target"])
    e40 = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    e46 = pd.read_csv(OUT_REL / "e46_targets.csv", dtype={"K": str}).set_index(["kind", "target", "K"])
    tabs, qtabs = [], []
    for kind in kinds:
        d = ROOT / "results" / f"teach_student39_{kind}"
        fs = [x for x in sorted(d.glob("*.csv")) if not x.stem.endswith(("_sel", "_q"))]
        if not fs:
            continue
        for x in fs:
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str, "cb": str})
            a = a[a.rep.isin(EC.REPS)]
            u = a.groupby(["policy", "rho_M", "cb", "cp", "K"]).uar.mean() * 100
            s = sel.loc[(kind, T)]
            tk = (float(s.t_rho_M), f"{float(s.t_cb):g}", float(s.t_cp))
            rk = (float(s.r_rho_M), f"{float(s.r_cb):g}", float(s.r_cp))
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K}
                for p in TEACHY + RANDY:
                    row[p] = float(u.loc[(p,) + (tk if p in TEACHY else rk) + (K,)])
                    row[f"{p}@r"] = float(u.loc[(p,) + rk + (K,)])               # 記述：すべて random の組の生徒で
                row["chk_e40_random"] = abs(row["random_true"] - float(e40.loc[(kind, T, K), "random"]))
                row["chk_e40_teach"] = abs(row["LLL"] - float(e40.loc[(kind, T, K), "teach"]))
                for x_ in ("random_pool", "herding_pred", "random_true", "LLL", "e45_C"):
                    row[f"P4:{x_}"] = float(e46.loc[(kind, T, K), x_])
                tabs.append(row)
        q = pd.concat([pd.read_csv(x) for x in sorted(d.glob("*_q.csv"))])
        q = q[q.rep.isin(EC.REPS) & q.m.isin([0, 4, 20, 40, 100])]
        qtabs.append(q.groupby(["target", "m"]).uar.mean().mul(100).rename("q_uar").reset_index().assign(kind=kind))
    D = pd.DataFrame(tabs)
    Qd = pd.concat(qtabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        row = {"kind": kind, "K": K, "targets": len(g)}
        for x in TEACHY + RANDY + ["P4:random_pool", "P4:herding_pred", "P4:e45_C", "P4:LLL"]:
            row[x] = float(g[x].mean())
        for a_, b_ in [("C:0001", "herding_pred"), ("C:0000", "herding_pred"), ("C:0001", "herding_pred_p4"), ("C:0001", "C:0000"),
                       ("C:0000", "random_pool"), ("C:0001", "random_pool"), ("herding_pred", "random_pool"), ("LLL", "C:0001"),
                       ("random_pool", "P4:random_pool"), ("C:0001", "P4:e45_C"), ("C:0001@r", "herding_pred@r"), ("C:0000@r", "random_pool@r")]:
            diff = g[a_] - g[b_]
            row[f"{a_}-{b_}"] = float(diff.mean())
            row[f"{a_}>{b_}"] = int((diff > 0).sum())
        row["chk_e40_random"] = float(g.chk_e40_random.max())
        row["chk_e40_teach"] = float(g.chk_e40_teach.max())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "audio_hubert"):
        k5 = R[(R.kind == kind) & (R.K == "5")]
        if len(k5):
            k5 = k5.iloc[0]
            R.loc[R.kind == kind, "Y1"] = bool(k5["C:0001-herding_pred"] > 0)
            R.loc[R.kind == kind, "Y2"] = bool(k5["C:0000-random_pool"] > 0)
            R.loc[R.kind == kind, "Y3"] = bool(k5["C:0001-C:0000"] > 0)
    D.to_csv(OUT_REL / "e49_targets.csv", index=False)
    R.to_csv(OUT_REL / "e49_tests.csv", index=False)
    Qd.to_csv(OUT_REL / "e49_q.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in R if c in ("Y1", "Y2", "Y3")]).round(2).T.to_string())
    print("\n== C:0001 の q の UAR（まだラベルの無い候補．m 本付いた時点．ターゲットの平均）")
    print(Qd.groupby(["kind", "m"]).q_uar.mean().unstack("m").round(2).to_string())
    vc = [c for c in ("Y1", "Y2", "Y3") if c in R]
    if vc:
        print(R.dropna(subset=vc).drop_duplicates("kind")[["kind"] + vc].to_string(index=False))
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
    out = args.out_dir or ROOT / "results" / f"teach_student39_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/teach_student39.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "teachy": TEACHY, "randy": RANDY, "every": EVERY,
            "cps": M5_CPS, "cbs": M5_CBS, "src_C": Cs, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, f, Cs, P, Y, args.reps, args.ks, out, cache)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
