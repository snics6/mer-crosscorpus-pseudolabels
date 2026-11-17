"""【探索的】E47（GPU）：ラベルが付くたびに見込み q を直すと，残りの候補での当たる割合はどれだけ上がるか（条件 2 の段 2）。

規則は `docs/exploratory-analyses.md` E47（設計と予言は計算の前に書いた）。見込みは E44 の C と B（ソースの票を癖でまとめたもの）。
ラベルを付ける順は 2 通り：random（E41 の random_pool と同じ順）と teach（E45 の C の版が選んだ順．歩幅は E37 の値）。
ラベルが m = 4・20・40・100 本（K = 1・5・10・25）付いた時点で，q を直す方法 4 つをオン・オフした 16 通りで直し，
まだラベルの無い候補での UAR を測る。直し方（2 つ以上オンのときはこの順にかける）：
  hab（①癖の表を直す）：B は確率のままの癖の表の各行に，付いたラベルの発話での各ソースの確率を足し込む（仮のターゲットの表は N0 件ぶん）。
                         C はまとめ方の数（W, b）を，付いたラベルに合うように学び直す（元の値へ引く強さ μ）
  rel（②q の言い方を読み替える）：付いたラベルで「q が k と言った発話の本当の感情」の割合 M[k, y] を数え（単位行列を a 件ぶん足す），q を q·M にする
  spr（③近所へ広げる）：E42 のラベル拡散（近い 10 本）で，付いたラベルの行を 1-hot に固定して広げる（強さ α）
  mix（④学び直した分類器と混ぜる）：付いたラベルで学び直した仮説転移 4 の確率 p_m と q を，重み w = m／(m + τ) で掛け合わせて混ぜる
つまみ（N0・μ・a・α・τ）は，直し方を 1 つだけオンにした状態で，仮のターゲット（T のソースを 1 つずつ）の UAR で選び，組み合わせではその値に固定する。
仮のターゲットでは，その群の全件を候補とし，ラベルはランダムな順で付ける（2 通り）。出発点は残りのソースの合併（C = 0.01）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/belief_correct.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/belief_correct.py --analyze

出力：`results/belief_correct_<kind>/<ターゲット>.csv`（rep, agg, order, K, combo, uar, …．combo は hab・rel・spr・mix の 0／1），
`<ターゲット>_pseudo.csv`（仮のターゲットでのつまみごとの UAR と，選んだつまみでの 16 通りの UAR），`provenance.json`，`provenance_log.jsonl`。
判定は `results/explore_relation/e47_targets.csv`・`e47_tests.csv`・`e47_pseudo.csv`
"""

from __future__ import annotations

import argparse
import itertools
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
from belief_check import knn, norm_adj, spread  # noqa: E402
from data import groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import present_weights  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402

AGGS = ["C", "B"]
CORR = ["hab", "rel", "spr", "mix"]
KNOBS = {("hab", "B"): (2.0, 10.0, 50.0, 250.0), ("hab", "C"): (0.01, 0.1, 1.0, 10.0), ("rel", None): (1.0, 5.0, 20.0, 100.0),
         ("spr", None): (0.05, 0.2, 0.5, 0.8), ("mix", None): (10.0, 40.0, 160.0, 640.0)}
CKPT = (4, 20, 40, 100)
COMBOS = ["".join(map(str, c)) for c in itertools.product((0, 1), repeat=4)]
K_NBR = 10
PSEUDO_REPS = (0, 1)
PSEUDO_C = 0.01
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def knob_key(corr, agg):
    return (corr, agg) if corr == "hab" else (corr, None)


class Task:
    """1 つの候補の集まり（ターゲットの反復，または仮のターゲット）での見込みと直し方。"""

    def __init__(self, Ps, members, H, Hs, pool, y, Xs, Xc, src, C_anchor):
        self.Ps, self.members, self.H, self.Hs, self.pool, self.y = Ps, members, H, Hs, pool, np.asarray(y)
        self.n = len(self.y)
        self.Xc, self.src, self.C_anchor = Xc, src, C_anchor
        self.F = torch.as_tensor(np.stack([np.log(np.clip(p, 1e-12, None)) for p in Ps], 1), dtype=logreg.DTYPE, device=logreg.DEVICE)
        self.A = torch.ones(self.n, len(Ps), dtype=logreg.DTYPE, device=logreg.DEVICE)
        self.S = norm_adj(knn(Xs, K_NBR), K_NBR)
        self.q0 = {"C": pool.predict(self.F, self.A), "B": CM.agg_B(Ps, [Hs[s] for s in members])}

    def fit_pm(self, orders: dict) -> dict:
        """付いたラベルで学び直した仮説転移 4（C′ = 0.01，C_b = 8，感情の重みは付いた中にある感情だけ）の候補の上の確率。"""
        keys = [(o, m) for o in orders for m in CKPT]
        used = np.unique(np.concatenate([orders[o][:max(CKPT)] for o in orders]))
        X = self.Xc[torch.as_tensor(used, device=self.Xc.device)]
        yX = M.labels(self.y[used])
        masks, weights = [], []
        for o, m in keys:
            mk = np.isin(used, orders[o][:m])
            w = torch.zeros(len(yX), dtype=logreg.DTYPE, device=CB.DEV)
            mt = torch.as_tensor(mk, device=CB.DEV)
            w[mt] = present_weights(yX[mt])
            masks.append(mk)
            weights.append(w)
        fits = logreg.fit_batch(X, yX, masks, (CB.DEF_CP,), anchor=self.src, anchor_c=self.C_anchor, stats=(self.src.mu, self.src.sd),
                                Cb=[CB.DEF_CB] * len(masks), weights=weights)
        return {k: torch.softmax(f.logits(self.Xc, CB.DEF_CP), 1).cpu().numpy() for k, f in zip(keys, fits)}

    def hab(self, agg, R, knob):
        yR = self.y[R]
        if agg == "B":
            Hs = []
            for s, p in zip(self.members, self.Ps):
                h = self.Hs[s].copy()
                for y in range(4):
                    m = yR == y
                    h[y] = (knob * h[y] + p[R][m].sum(0)) / (knob + m.sum())
                Hs.append(h)
            return CM.agg_B(self.Ps, Hs)
        Rt = torch.as_tensor(R, device=self.F.device)
        yt = torch.as_tensor(yR, device=self.F.device)
        w = present_weights(yt)
        return finetune(self.pool, self.F[Rt], self.A[Rt], yt, w / w.sum(), knob).predict(self.F, self.A)

    def rel(self, q, R, knob):
        k = q[R].argmax(1)
        Mx = knob * np.eye(4)
        np.add.at(Mx, (k, self.y[R]), 1.0)
        return q @ (Mx / Mx.sum(1, keepdims=True))

    def spr(self, q, R, knob):
        Y0 = torch.as_tensor(q, dtype=torch.float32, device=CB.DEV).clone()
        Rt = torch.as_tensor(R, device=CB.DEV)
        Y0[Rt] = torch.nn.functional.one_hot(torch.as_tensor(self.y[R], device=CB.DEV), 4).float()
        return spread(self.S, Y0, knob).cpu().numpy()

    @staticmethod
    def mix(q, pm, m, knob):
        w = m / (m + knob)
        L = (1 - w) * np.log(np.clip(q, 1e-12, None)) + w * np.log(np.clip(pm, 1e-12, None))
        return CM.softmax_np(L)

    def corrected(self, agg, combo, R, m, pm, knobs, hab_cache=None):
        h, r, s, x = (c == "1" for c in combo)
        q = (hab_cache if hab_cache is not None else self.hab(agg, R, knobs[knob_key("hab", agg)])) if h else self.q0[agg]
        if r:
            q = self.rel(q, R, knobs[("rel", None)])
        if s:
            q = self.spr(q, R, knobs[("spr", None)])
        if x:
            q = self.mix(q, pm, m, knobs[("mix", None)])
        return q

    def score(self, q, R, pm=None):
        un = np.ones(self.n, bool)
        un[R] = False
        a = q[un].argmax(1)
        yu = self.y[un]
        row = {"uar": CB.uar(yu, a), "acc": float((a == yu).mean()), "n_un": int(un.sum())}
        if pm is not None:
            b = pm[un].argmax(1)
            fl = a != b
            row.update({"dis": float(fl.mean()), "q_only|dis": float((fl & (a == yu)).sum() / max(fl.sum(), 1)),
                        "p_only|dis": float((fl & (b == yu)).sum() / max(fl.sum(), 1))})
        return row


def finetune(pool, F, A, y, w, mu):
    """まとめ方の数を，付いたラベルに合うように学び直す（元の値へ引く強さ mu）。"""
    W0, b0 = pool.W, pool.b
    W = W0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        ll = torch.nn.functional.cross_entropy(pool._logits(F, A, W, b), y, reduction="none")
        loss = (w * ll).sum() + mu * (((W - W0) ** 2).sum() + ((b - b0) ** 2).sum())
        loss.backward()
        return loss
    opt.step(closure)
    new = CM.Pool(pool.S)
    new.W, new.b = W.detach(), b.detach()
    return new


# ---------------------------------------------------------------- 仮のターゲットでつまみを選ぶ

def choose_knobs(T, S, P, Y, data, lam, out_rows):
    """直し方を 1 つだけオンにした状態で，つまみを仮のターゲットの UAR（t・順・m の平均）で選ぶ。選んだつまみで 16 通りも測る。"""
    tasks = []
    for t in S:
        St = [s for s in S if s != t and CM.corp(s) != CM.corp(t)]
        Gt = [g for g in S if g != t]
        Ht, Hst = CM.habits(P, Y, St, Gt)
        pool_t = CM.Pool(St).fit(*CM.pool_data(P, Y, St, Gt), lam)
        Xp = torch.cat([data[s].X for s in St])
        yp = M.labels(np.concatenate([data[s].y for s in St]))
        src_t = logreg.fit(Xp - Xp.mean(0), yp, (PSEUDO_C,))
        del Xp
        Xc = data[t].X - data[t].X.mean(0)
        task = Task([P[(s, t)] for s in St], St, Ht, Hst, pool_t, Y[t], ((Xc - src_t.mu) / src_t.sd).float(), Xc, src_t, PSEUDO_C)
        orders = {f"r{r}": np.random.default_rng(zlib.crc32(f"e47pseudo|{T}|{t}|{r}".encode())).permutation(task.n) for r in PSEUDO_REPS}
        tasks.append((t, task, orders, task.fit_pm(orders)))
    rows = []
    for t, task, orders, pms in tasks:
        for o in orders:
            for m in CKPT:
                R = orders[o][:m]
                for agg in AGGS:
                    rows.append({"pseudo": t, "order": o, "m": m, "agg": agg, "corr": "none", "knob": np.nan, **task.score(task.q0[agg], R)})
                    for corr in CORR:
                        for kv in KNOBS[knob_key(corr, agg)]:
                            kn = {knob_key(c, agg): KNOBS[knob_key(c, agg)][0] for c in CORR}
                            kn[knob_key(corr, agg)] = kv
                            combo = "".join("1" if c == corr else "0" for c in CORR)
                            q = task.corrected(agg, combo, R, m, pms[(o, m)], kn)
                            rows.append({"pseudo": t, "order": o, "m": m, "agg": agg, "corr": corr, "knob": kv, **task.score(q, R)})
    kt = pd.DataFrame(rows)
    knobs = {}
    for agg in AGGS:
        knobs[agg] = {}
        for corr in CORR:
            g = kt[(kt["agg"] == agg) & (kt["corr"] == corr)].groupby("knob").uar.mean()
            knobs[agg][knob_key(corr, agg)] = float(g.idxmax())
    for t, task, orders, pms in tasks:                                                      # 選んだつまみでの 16 通り（段 3 に進めるものを選ぶため）
        for o in orders:
            for m in CKPT:
                R = orders[o][:m]
                for agg in AGGS:
                    hc = task.hab(agg, R, knobs[agg][knob_key("hab", agg)])
                    for combo in COMBOS:
                        q = task.corrected(agg, combo, R, m, pms[(o, m)], knobs[agg], hc)
                        rows.append({"pseudo": t, "order": o, "m": m, "agg": agg, "corr": f"combo{combo}", "knob": np.nan, **task.score(q, R)})
    out_rows += [{"target": T, **r} for r in rows]
    return knobs


# ---------------------------------------------------------------- ターゲットごと

def run_target(kind, T, data, f, Cs, P, Y, reps, out):
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    H, Hs = CM.habits(P, Y, S, S)
    e44 = pd.read_csv(ROOT / "results" / f"committee_{kind}" / f"{T}.csv")
    lam = float(e44[e44.variant == "C"].lam.iloc[0])
    prow = []
    knobs = choose_knobs(T, S, P, Y, data, lam, prow)
    pd.DataFrame(prow).to_csv(out / f"{T}_pseudo.csv", index=False)
    pool_C = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    Xp = torch.cat([data[s].X for s in S])
    yp = M.labels(np.concatenate([data[s].y for s in S]))
    src = logreg.fit(Xp - Xp.mean(0), yp, (C_T,))
    del Xp
    star = pd.read_csv(OUT_REL / "e37_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"]).policy_star
    c_main = float(star.loc[(kind, T)].split("_c")[1])
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ["5"], center="ind")
    rows = []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        X = ((Xc - src.mu) / src.sd).float()
        Ps = [torch.softmax(f[s].logits(Xc, Cs[s]), 1).cpu().numpy() for s in S]
        task = Task(Ps, S, H, Hs, pool_C, yq, X, Xc, src, C_T)
        L0 = src.logits(Xc, C_T)
        Km = X @ X.T
        Km.fill_diagonal_(0.0)
        ohC = torch.nn.functional.one_hot(torch.as_tensor(task.q0["C"].argmax(1), device=X.device), 4).float()
        teach = np.array(ablate_order(Km, yq, L0, c_main, max(CKPT), "PPP", belief=lambda L, P_: ohC)[0])
        del Km
        orders = {"random": np.random.default_rng(zlib.crc32(f"e41pool|{T}|{r}".encode())).permutation(len(pool)), "teach": teach}
        pms = task.fit_pm(orders)
        for o in orders:
            for m in CKPT:
                R = orders[o][:m]
                pm = pms[(o, m)]
                rows.append({"target": T, "rep": r, "agg": "pm", "order": o, "K": m // 4, "combo": "-", **task.score(pm, R)})
                for agg in AGGS:
                    hc = task.hab(agg, R, knobs[agg][knob_key("hab", agg)])
                    for combo in COMBOS:
                        q = task.corrected(agg, combo, R, m, pm, knobs[agg], hc)
                        rows.append({"target": T, "rep": r, "agg": agg, "order": o, "K": m // 4, "combo": combo, **task.score(q, R, pm)})
    kn = {f"{agg}:{c}": knobs[agg][knob_key(c, agg)] for agg in AGGS for c in CORR}
    df = pd.DataFrame(rows)
    for k, v in kn.items():
        df[f"knob_{k}"] = v
    df.to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    tabs, pse = [], []
    for kind in kinds:
        d = ROOT / "results" / f"belief_correct_{kind}"
        fs = [x for x in sorted(d.glob("*.csv")) if not x.stem.endswith("_pseudo")]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(x) for x in fs])
        a = a[a.rep.isin(EC.REPS)]
        g = a.groupby(["target", "agg", "order", "K", "combo"])[["uar", "dis", "q_only|dis", "p_only|dis"]].mean().reset_index()
        g.insert(0, "kind", kind)
        tabs.append(g)
        ps = pd.concat([pd.read_csv(x) for x in sorted(d.glob("*_pseudo.csv"))])
        ps = ps[ps["corr"].str.startswith("combo")].assign(combo=lambda z: z["corr"].str[5:])
        ps = ps.groupby(["target", "agg", "combo"]).uar.mean().reset_index()
        ps.insert(0, "kind", kind)
        pse.append(ps)
    D = pd.concat(tabs)
    D["uar"] *= 100
    P_ = pd.concat(pse)
    P_["uar"] *= 100
    base = D[D.combo == "0000"].set_index(["kind", "target", "agg", "order", "K"]).uar
    pmv = D[D["agg"] == "pm"].set_index(["kind", "target", "order", "K"]).uar
    Q = D[D["agg"] != "pm"].copy()
    Q["-none"] = Q.uar.values - base.reindex(pd.MultiIndex.from_frame(Q[["kind", "target", "agg", "order", "K"]])).values
    Q["-pm"] = Q.uar.values - pmv.reindex(pd.MultiIndex.from_frame(Q[["kind", "target", "order", "K"]])).values
    R = Q.groupby(["kind", "agg", "order", "K", "combo"])[["uar", "-none", "-pm", "dis", "q_only|dis", "p_only|dis"]].mean().reset_index()
    R["n>none"] = Q.assign(w=Q["-none"] > 0).groupby(["kind", "agg", "order", "K", "combo"]).w.sum().values
    pv = P_.groupby(["kind", "agg", "combo"]).uar.mean().rename("pseudo_uar").reset_index()
    R = R.merge(pv, on=["kind", "agg", "combo"], how="left")
    pmm = D[D["agg"] == "pm"].groupby(["kind", "order", "K"]).uar.mean().rename("pm_uar").reset_index()
    R = R.merge(pmm, on=["kind", "order", "K"], how="left")
    D.to_csv(OUT_REL / "e47_targets.csv", index=False)
    R.to_csv(OUT_REL / "e47_tests.csv", index=False)
    P_.to_csv(OUT_REL / "e47_pseudo.csv", index=False)
    pd.set_option("display.width", 250)
    single = {"0000": "none", "1000": "hab", "0100": "rel", "0010": "spr", "0001": "mix"}
    for kind in kinds:
        for agg in AGGS:
            sub = R[(R.kind == kind) & (R["agg"] == agg)]
            if not len(sub):
                continue
            print(f"== {kind} {agg}（単独の直し方の −none．K = 1・5・10・25）")
            for o in ("random", "teach"):
                t = sub[(sub.order == o) & sub.combo.isin(single)].pivot(index="combo", columns="K", values="-none").rename(index=single)
                t["pseudo"] = sub[(sub.order == o) & sub.combo.isin(single)].set_index("combo").pseudo_uar.groupby(level=0).first().rename(index=single)
                print(o, "\n", t.round(2).to_string())
                base_ = sub[(sub.order == o) & (sub.combo == "0000")].set_index("K")[["uar", "pm_uar"]]
                print("  none と pm の UAR\n", base_.round(2).T.to_string())
            top = sub[sub.order == "random"].drop_duplicates("combo").nlargest(3, "pseudo_uar")[["combo", "pseudo_uar"]]
            print("  仮のターゲットで上位の 3 通り：", top.round(2).values.tolist())
    for kind in ("audio_wavlm", "audio_hubert"):                                              # 判定
        sub = R[(R.kind == kind)]
        if not len(sub):
            continue
        out = {}
        for agg in AGGS:
            s = sub[sub["agg"] == agg].set_index(["order", "K", "combo"])["-none"]
            out[f"W1_{agg}"] = bool(s.loc[("random", 25, "0001")] >= 2)
            out[f"W2_{agg}"] = bool(max(s.loc[("random", 5, c)] for c in ("1000", "0100", "0010", "0001")) < 2)
            out[f"W3_{agg}"] = bool(s.loc[("teach", 5, "0100")] < s.loc[("random", 5, "0100")])
        print(kind, out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=MAIN)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"belief_correct_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data = {g: M.load(g, args.kind) for g in groups}
    f, Cs, P, Y = CM.prepare(args.kind, data)
    prov = {"script": "exp/belief_correct.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "reps": args.reps, "aggs": AGGS, "corr": CORR,
            "knobs": {f"{k[0]}:{k[1]}": v for k, v in KNOBS.items()}, "ckpt": CKPT, "k_nbr": K_NBR, "pseudo_reps": PSEUDO_REPS,
            "pseudo_C": PSEUDO_C, "src_C": Cs, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, f, Cs, P, Y, args.reps, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
