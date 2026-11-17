"""【探索的】E61（GPU）：E59・E60 の比べる相手——標準の半教師あり学習・蒸留・共訓練・出力の補正を，同じ生徒・同じ K・同じ仮のターゲットの選び方で測る。

規則は `docs/exploratory-analyses.md` E61（設計と予言は計算の前に書いた）。生徒・K ∈ {0, 1, 5, 10, 25}・反復 0〜9・候補・中心は，
--kind（同じモダリティ．A・V・T）では E59 の追加，--world（融合の 4 世界）では E60 と同じ．相手のつまみは K ごとに，T のソースを 1 つずつ
仮のターゲットにして選ぶ（反復 0〜4．「none（何もしない）」も候補）．固定版（文献の既定値）とオラクル（テストで最良）は判定で出す．
  none    ：E39（K 本だけで学び直す．K=0 は出発点）
  self    ：自己訓練．生徒の予測を確信度 τ 以上の発話だけ硬いラベルにして学び直す（3 巡）．τ ∈ {0.5, 0.7, 0.9}
  entmin  ：エントロピー最小化．候補の予測のエントロピーを罰則にして学び直す（none の解から始める）
  kd      ：温度付きの蒸留．先生の確率を温度 T でなだらかにして，釣り合いを取らずに弱いラベルにする．T ∈ {1, 2, 4}．
            先生は --kind では q・pq，--world ではモダリティの生徒に融合の出発点 pF，融合の生徒にモダリティの出発点の積 poe0
  gd      ：一般化蒸留（--world のモダリティの生徒だけ）．先生は K 本で学び直した融合の生徒 pFK（K=0 では pF）
  hard    ：硬い疑似ラベル（--world のモダリティの生徒だけ）．先生 pF の予測を τ 以上の発話だけ硬いラベルにする
  cotrain ：共訓練（--world）．モダリティごとの生徒が，自分が最も確信する発話を感情ごとに c 本ずつ硬いラベル（重み 1）にして共通のラベル集合に足す（10 巡）．
            c ∈ {2, 5, 20}．運用時 1 つなら各生徒，複数なら生徒の確率の積（cotrain_prod）と，最後のラベル集合で学び直した融合の生徒（cotrain_fused）
  coreg   ：共正則化（--world）．モダリティごとの生徒を，候補の上で中心化した出力の差の 2 乗を罰則にして同時に解く．複数なら出力の和（coreg_sum）
  saerens ：学び直さず，候補の予測から感情の割合を EM で推定して出力に掛ける（Saerens ほか 2002）
  massbal ：学び直さず，候補の上の予測の感情ごとの総量が等しくなるよう出力を補正する（対照 p の学び直さない版）
  oursx   ：私たちの形の拡張（E60 の追加．--world）．先生 ∈ {pFK（K 本で学び直した融合の生徒），ref（pFK を釣り合いを取って m_in=500 で蒸留した
            モダリティごとの生徒の確率の積．2 段目）}，重みの付け方 ∈ {bal（感情の釣り合いを取る），raw（先生の確率そのまま），hardτ（τ 以上の硬いラベル）}，m_eff．
            モダリティの生徒には両方の先生，融合の生徒には ref．K ごとに仮のターゲットで選ぶ（none も候補）
弱いラベル・罰則の総重み m_eff ∈ {20, 100, 500, n}（--kind），{100, 500, 2000, n}（--world）．n は候補 1 本あたり重み 1（文献どおり）
--fused-only（--world）：融合の生徒とその相手だけを解く．モダリティの生徒それぞれの相手（gd・hard・kd・self・entmin・oursx）は省き，
            2 段目の先生 ref を作るための学び直しと，共訓練・共正則化（比べるのは融合の値）は残す．AV・VT・AT に使う（研究者の指示 2026-10-23）

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/baselines_distill.py --kind audio_wavlm [--target …] [--reps …]
    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/baselines_distill.py --world fuse-audio-video [--target …] [--reps …]
    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/baselines_distill.py --world fuse-audio-video --fused-only
    ./.venv/bin/python exp/baselines_distill.py --analyze [--names …] [--results-dir …（試走用．読む場所と判定の書き場所）]

出力：`results/baselines_<kind か world>/<ターゲット>.csv`（student, method, param, meff, rep, K, uar），`<ターゲット>_pseudo.csv`，
`<ターゲット>_meta.csv`（生徒・方法・K ごとの候補の値と選んだもの），`provenance.json`，`provenance_log.jsonl`．
判定は `results/explore_relation/e61_targets.csv`・`e61_tests.csv`
"""

from __future__ import annotations

import argparse
import json
import os
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
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import uar_of, weak_rows  # noqa: E402
from geom_e0 import settings  # noqa: E402
from multimodal_distill import DIMS, WORLDS, WORLDS2, beliefs1, blocks, make_students, sub_group  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import present_weights  # noqa: E402
from teach_stack import c_pool_on, draft_logits, fit_stack, logp, softmax_t  # noqa: E402
from teach_student39 import Student  # noqa: E402
from text_cond2 import choose_lam  # noqa: E402

KINDS = ["audio_wavlm", "video", "text_roberta"]
KS = ("0", "1", "5", "10", "25")
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_FIT = 0.01
TAUS = ("0.5", "0.7", "0.9")
TEMPS = (1, 2, 4)
ROUNDS_SELF = 3
ROUNDS_CO = 10
CO_C = (2, 5, 20)
MEFF_A = ("20", "100", "500", "n")
MEFF_B = ("100", "500", "2000", "n")
FIXED = {"self": ("0.9", "n"), "entmin": ("-", "n"), "gd": ("pFKT2", "n"), "hard": ("0.9", "n"), "cotrain": ("c2", "n"), "cotrain_prod": ("c2", "n"),
         "cotrain_fused": ("c2", "n"), "coreg": ("-", "n"), "coreg_sum": ("-", "n"), "saerens": ("-", "0"), "massbal": ("-", "0")}
KD_FIXED_TEACHER = {"kind": "q", "mod": "pF", "fused": "poe0"}                 # kd の固定版の先生（T=2・重み 1）
OUT_REL = ROOT / "results" / "explore_relation"
DT, DEV = logreg.DTYPE, logreg.DEVICE
STR_COLS = {"K": str, "meff": str, "param": str}


def mass(meff, n):
    return float(n) if meff == "n" else float(meff)


def tens(x):
    return x.to(DT) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=DT, device=DEV)


def temp_rows(P, T, meff):
    """先生の確率 P（n×4）を温度 T でなだらかにし，釣り合いを取らない弱いラベルの行の重み（長さ 4n．行 k·n + i が発話 i の感情 k）。"""
    PT = torch.softmax(torch.log(tens(P).clamp(min=1e-12)) / T, 1)
    r = PT.T.reshape(-1)
    return r if meff == "n" else r / r.sum() * float(meff)


def hard_rows(P, tau, meff):
    """確信度が τ 以上の発話だけ，予測した感情の硬いラベル（重み 1）．m_eff を指定すれば総重みをそれにそろえる。"""
    Pt = tens(P)
    conf, lab = Pt.max(1)
    m = conf >= float(tau)
    r = torch.zeros(4, len(Pt), dtype=DT, device=DEV)
    r[lab[m], torch.nonzero(m).squeeze(1)] = 1.0
    r = r.reshape(-1)
    if meff != "n" and float(r.sum()) > 0:
        r = r / r.sum() * float(meff)
    return r


def saerens(Pp, Pt, iters=200):
    """Saerens ほか 2002 の EM：候補の予測 Pp（学習時の感情の割合は等しい）から運用時の割合 π を推定し，テストの予測 Pt に π を掛けた感情。"""
    pi = np.full(4, 0.25)
    for _ in range(iters):
        A = Pp * pi
        A /= A.sum(1, keepdims=True)
        pi_new = A.mean(0)
        done = np.abs(pi_new - pi).max() < 1e-9
        pi = pi_new
        if done:
            break
    return (Pt * pi).argmax(1)


def massbal(Pp, Pt, iters=500):
    """候補の上の予測の感情ごとの総量が等しくなる補正 a（出力に掛ける）を求め，テストの予測 Pt·a の感情。"""
    a = np.ones(4)
    for _ in range(iters):
        A = Pp * a
        A /= A.sum(1, keepdims=True)
        a_new = a * 0.25 / np.clip(A.mean(0), 1e-12, None)
        a_new /= a_new.sum()
        done = np.abs(a_new - a).max() < 1e-12
        a = a_new
        if done:
            break
    return (Pt * a).argmax(1)


class Prob:
    """1 つの生徒の候補の上の学習問題（E39 の学習器．罰則の中心は出発点）．X = Z×4（行 k·n + i が発話 i の感情 k．弱いラベルの重みはそのまま，
    本当のラベルの重みは行 y_i·n + i に足す．distill_belief.solve_all の [Z×4; Z] と同じ問を 4n 行で解く）。"""

    def __init__(self, s, yq, label_sets):
        self.s = s
        Z = s["Z"]
        self.n, self.d = Z.shape
        self.X = Z.repeat(4, 1)
        yt = M.labels(yq)
        self.y = torch.arange(4, device=DEV).repeat_interleave(self.n)
        self.wt = {}
        for K, idx in label_sets.items():
            w = torch.zeros(4 * self.n, dtype=DT, device=DEV)
            if len(idx):
                it = torch.as_tensor(np.asarray(idx), device=DEV)
                w[yt[it] * self.n + it] = present_weights(yt[it])
            self.wt[K] = w
        self.yt = yt
        self.zero = torch.zeros(self.d, dtype=DT, device=DEV)
        self.one = torch.ones_like(self.zero)
        self.W0, self.b0 = s["st"].W_std @ s["st"].mats[s["rho_m"]][1], s["st"].b_std
        s["st"].Xstd = None                                                      # 以後は Z だけ使う（GPU メモリを空ける）

    def solve(self, items, u=None, inits=None):
        """items = [(key, 弱いラベルの行の重み (4n,) か None, K)] を束ねて解く → {key: (W, b)}．
        u[j] はエントロピーの罰則の 1 本あたりの重み（entmin），inits[j] は出発点の (W, b)（None なら罰則の中心から）．重みが全部 0 の問は出発点そのもの。"""
        s, cp = self.s, self.s["cp"]
        keys, weights, us, ini, out = [], [], [], [], {}
        for j, (key, w4, K) in enumerate(items):
            w = self.wt[K] if w4 is None else self.wt[K] + w4
            uj = 0.0 if u is None else float(u[j])
            if float(w.sum()) == 0.0 and uj == 0.0:
                out[key] = (self.W0, self.b0)
                continue
            keys.append(key)
            weights.append(w)
            us.append(uj)
            wb = None if inits is None else inits[j]
            ini.append(None if wb is None else logreg.Fitted(self.zero, self.one, {cp: wb[0]}, {cp: wb[1]}, {cp: 0}, {cp: 0.0}))
        if not keys:
            return out
        extra = None
        if any(us):
            uv = torch.tensor(us, dtype=DT, device=DEV)
            n1 = self.n

            def fn(L, ix):                                                       # 候補のエントロピーの和 × 重み（出力は感情の行に依らないので最初の n 行）
                lp = torch.log_softmax(L[:n1], 2)
                return -(lp.exp() * lp).sum(2).sum(0) * (uv if ix is None else uv[ix])
            extra = (fn, [x * self.n for x in us])
        fits = logreg.fit_batch(self.X, self.y, [np.ones(len(self.y), bool)] * len(keys), (cp,), anchor=s["st"].anchor(s["rho_m"]),
                                anchor_c=s["st"].C, stats=(self.zero, self.one), Cb=[s["cb"]] * len(keys), weights=weights, extra=extra,
                                inits=ini if any(i is not None for i in ini) else None)
        for key, f in zip(keys, fits):
            out[key] = (f.W[cp], f.b[cp])
        return out

    def probs(self, Wb, Zx):
        return torch.softmax(Zx @ Wb[0].T + Wb[1], 1)


def oursx_rows(pr, yte, meffs, teachers, rows, tag, student):
    """私たちの形の拡張：先生（K ごとの候補の上の確率）× 重みの付け方 {bal, raw, hard τ} × m_eff を解き，テストの UAR を rows に足す。"""
    n = pr.n

    def rows_of(P, wgt, meff):
        if wgt == "bal":
            return weak_rows(P, mass(meff, n))
        if wgt == "raw":
            return temp_rows(P, 1, meff)
        return hard_rows(P, wgt[4:], meff)                                       # "hard0.9" → τ = 0.9
    items = [((name, wgt, meff, K), rows_of(P[K], wgt, meff), K)
             for name, P in teachers.items() for wgt in ("bal", "raw", *(f"hard{t}" for t in TAUS)) for meff in meffs for K in KS]
    for (name, wgt, meff, K), Wb in pr.solve(items).items():
        rows.append({**tag, "student": student, "method": "oursx", "param": f"{name}:{wgt}", "meff": meff, "K": K,
                     "uar": float(uar_of(Wb[0], Wb[1], pr.s["Zte"], yte))})


def run_student(pr, yte, meffs, teachers, rows, tag, student, gd=None, hard=None, oursx=None):
    """1 つの生徒で none・saerens・massbal・self・entmin・kd（・gd・hard・oursx）を解き，テストの UAR を rows に足す．戻りは none の解と候補の上の確率（K ごと）。"""
    s, n = pr.s, pr.n

    def add(method, param, meff, K, u):
        rows.append({**tag, "student": student, "method": method, "param": param, "meff": meff, "K": K, "uar": float(u)})
    uar = lambda Wb: uar_of(Wb[0], Wb[1], s["Zte"], yte)  # noqa: E731
    none = pr.solve([(K, None, K) for K in KS])
    P0 = {K: pr.probs(none[K], s["Z"]) for K in KS}
    for K in KS:
        add("none", "-", "0", K, uar(none[K]))
        Pp, Pt = P0[K].cpu().numpy(), pr.probs(none[K], s["Zte"]).cpu().numpy()
        add("saerens", "-", "0", K, CB.uar(yte, saerens(Pp, Pt)))
        add("massbal", "-", "0", K, CB.uar(yte, massbal(Pp, Pt)))
    keys = [(tau, meff, K) for tau in TAUS for meff in meffs for K in KS]
    cur = {key: P0[key[2]] for key in keys}
    for _ in range(ROUNDS_SELF):
        sol = pr.solve([(key, hard_rows(cur[key], key[0], key[1]), key[2]) for key in keys])
        cur = {key: pr.probs(sol[key], s["Z"]) for key in keys}
    for (tau, meff, K), Wb in sol.items():
        add("self", tau, meff, K, uar(Wb))
    keys = [(meff, K) for meff in meffs for K in KS]
    sol = pr.solve([(key, None, key[1]) for key in keys], u=[mass(meff, n) / n for meff, K in keys], inits=[none[K] for meff, K in keys])
    for (meff, K), Wb in sol.items():
        add("entmin", "-", meff, K, uar(Wb))
    items = [((name, T, meff, K), temp_rows(P, T, meff), K) for name, P in teachers.items() for T in TEMPS for meff in meffs for K in KS]
    for (name, T, meff, K), Wb in pr.solve(items).items():
        add("kd", f"{name}T{T}", meff, K, uar(Wb))
    if gd is not None:
        items = [((T, meff, K), temp_rows(gd[K], T, meff), K) for T in TEMPS for meff in meffs for K in KS]
        for (T, meff, K), Wb in pr.solve(items).items():
            add("gd", f"pFKT{T}", meff, K, uar(Wb))
    if hard is not None:
        items = [((tau, meff, K), hard_rows(hard, tau, meff), K) for tau in TAUS for meff in meffs for K in KS]
        for (tau, meff, K), Wb in pr.solve(items).items():
            add("hard", tau, meff, K, uar(Wb))
    if oursx is not None:
        oursx_rows(pr, yte, meffs, oursx, rows, tag, student)
    return none, P0


def cotrain(prs, fused, yte, rows, tag, world):
    """共訓練（Blum & Mitchell 1998 の原型）：モダリティごとの生徒が，自分が最も確信する発話を感情ごとに c 本ずつ硬いラベル（重み 1）にして
    共通のラベル集合に足す（ROUNDS_CO 巡．先に選んだ生徒が優先）．最後に各生徒・生徒の確率の積・そのラベル集合で学び直した融合の生徒を測る。"""
    views = list(prs)
    n = fused.n
    keys = [(c, K) for c in CO_C for K in KS]
    pl = {key: torch.zeros(4, n, dtype=DT, device=DEV) for key in keys}
    taken = {key: set((torch.nonzero(prs[views[0]].wt[key[1]]).squeeze(1) % n).tolist()) for key in keys}

    def add(student, method, c, K, u):
        rows.append({**tag, "student": student, "method": method, "param": f"c{c}", "meff": "n", "K": K, "uar": float(u)})
    for t in range(ROUNDS_CO + 1):
        sols = {v: prs[v].solve([(key, pl[key].reshape(-1), key[1]) for key in keys]) for v in views}
        if t == ROUNDS_CO:
            break
        picks = {key: [] for key in keys}
        for v in views:
            for key in keys:
                conf, lab = prs[v].probs(sols[v][key], prs[v].s["Z"]).max(1)
                used = taken[key] | {i for i, _ in picks[key]}
                conf = conf.clone()
                if used:
                    conf[torch.as_tensor(sorted(used), device=DEV)] = -1.0
                for k in range(4):
                    ck = torch.where(lab == k, conf, torch.full_like(conf, -1.0))
                    top = torch.topk(ck, min(key[0], n))
                    picks[key] += [(int(i), k) for i, c_ in zip(top.indices.tolist(), top.values.tolist()) if c_ >= 0]
        for key in keys:
            for i, k in picks[key]:
                if i not in taken[key]:
                    pl[key][k, i] = 1.0
                    taken[key].add(i)
    prod = {key: torch.ones(len(yte), 4, dtype=DT, device=DEV) for key in keys}
    for v in views:
        for key, Wb in sols[v].items():
            add(v, "cotrain", key[0], key[1], uar_of(Wb[0], Wb[1], prs[v].s["Zte"], yte))
            prod[key] = prod[key] * prs[v].probs(Wb, prs[v].s["Zte"])
    for key in keys:
        add(world, "cotrain_prod", key[0], key[1], CB.uar(yte, prod[key].argmax(1).cpu().numpy()))
    for key, Wb in fused.solve([(key, pl[key].reshape(-1), key[1]) for key in keys]).items():
        add(world, "cotrain_fused", key[0], key[1], uar_of(Wb[0], Wb[1], fused.s["Zte"], yte))


def coreg(prs, yte, meffs, rows, tag, world):
    """共正則化（Sindhwani & Niyogi 2005 の形）：モダリティごとの生徒を，候補の上で中心化した出力の差の 2 乗（重み m_eff/n）を罰則にして
    同時に解く（各生徒の罰則・座標は E39 のまま．S = ラベルの重み + m_eff．m_eff の 4 つの問を束ねて L-BFGS）．複数モダリティでは出力の和。"""
    views = list(prs)
    n = prs[views[0]].n
    B = len(meffs)
    mv = torch.tensor([mass(m, n) for m in meffs], dtype=DT, device=DEV)
    yt = prs[views[0]].yt

    def add(student, meth, meff, K, u):
        rows.append({**tag, "student": student, "method": meth, "param": "-", "meff": meff, "K": K, "uar": float(u)})
    for K in KS:
        wt = prs[views[0]].wt[K].view(4, n).sum(0)                                  # 発話ごとのラベルの重み（行 y_i·n + i にだけ入っている）
        S = wt.sum() + mv
        params = {v: (prs[v].W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True),
                      prs[v].b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True)) for v in views}
        opt = torch.optim.LBFGS([p for v in views for p in params[v]], lr=1.0, max_iter=3000, max_eval=6000, tolerance_grad=1e-6,
                                tolerance_change=1e-14, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total = torch.zeros(B, dtype=DT, device=DEV)
            Lc = []
            for v in views:
                pr = prs[v]
                W, b = params[v]
                L = torch.einsum("nd,bkd->nbk", pr.s["Z"], W) + b[None]
                ce = -torch.log_softmax(L, 2).gather(2, yt.view(n, 1, 1).expand(n, B, 1)).squeeze(2)
                total = total + (wt[:, None] * ce).sum(0) + ((W - pr.W0) ** 2).sum((1, 2)) / (2 * pr.s["cp"]) + ((b - pr.b0) ** 2).sum(1) / (2 * pr.s["cb"])
                Lc.append(L - L.mean(2, keepdim=True))
            dis = sum(((Lc[a] - Lc[c]) ** 2).sum((0, 2)) for a in range(len(views)) for c in range(a + 1, len(views)))
            f = ((total + dis * (mv / n) / 2) / S).sum()
            f.backward()
            return f
        opt.step(closure)
        with torch.no_grad():
            Lsum = 0
            for v in views:
                W, b = params[v]
                Lte = torch.einsum("nd,bkd->nbk", prs[v].s["Zte"], W) + b[None]
                for j, meff in enumerate(meffs):
                    add(v, "coreg", meff, K, CB.uar(yte, Lte[:, j].argmax(1).cpu().numpy()))
                Lsum = Lsum + Lte
            for j, meff in enumerate(meffs):
                add(world, "coreg_sum", meff, K, CB.uar(yte, Lsum[:, j].argmax(1).cpu().numpy()))


def run_world_unit(world, stu, yq, yte, label_sets, rows, tag, fused_only=False):
    """1 つの反復の候補の上で，世界の全部の生徒の全部の相手を解く（fused_only ならモダリティの生徒それぞれの相手は省く）。"""
    mods = WORLDS[world]
    prs = {k: Prob(stu[k], yq, label_sets) for k in mods + [world]}
    b1 = beliefs1(stu, world)
    _, P0_f = run_student(prs[world], yte, MEFF_B, {"poe0": tens(b1["poe"])}, rows, tag, world)
    pF = tens(stu[world]["P"])
    lg = {K: 0 for K in KS}                                                      # 2 段目の先生 ref：pFK で鍛えたモダリティの生徒（釣り合い，m_in=500）の積
    for m in mods:
        sol = prs[m].solve([(K, weak_rows(P0_f[K], 500), K) for K in KS])
        for K in KS:
            lg[K] = lg[K] + prs[m].probs(sol[K], prs[m].s["Z"]).clamp(min=1e-12).log()
    refK = {K: torch.softmax(lg[K], 1) for K in KS}
    if not fused_only:
        for m in mods:
            run_student(prs[m], yte, MEFF_B, {"pF": pF}, rows, tag, m, gd=P0_f, hard=pF, oursx={"pFK": P0_f, "ref": refK})
    oursx_rows(prs[world], yte, MEFF_B, {"ref": refK}, rows, tag, world)
    cotrain({m: prs[m] for m in mods}, prs[world], yte, rows, tag, world)
    coreg({m: prs[m] for m in mods}, yte, MEFF_B, rows, tag, world)


def choose(pdf, student):
    """仮のターゲットの値から，方法・K ごとに (候補の方法, param, m_eff) を選ぶ（候補に none を含める）．戻りは meta の行。"""
    sub0 = pdf[pdf.student == student]
    mrows = []
    for method in sorted(set(sub0.method) - {"none"}):
        for K in KS:
            sub = sub0[(sub0.K == K) & (sub0.method.isin([method, "none"]))]
            sc = sub.groupby(["method", "param", "meff"]).uar.mean()
            best = sc.idxmax()
            mrows += [{"student": student, "method": method, "K": K, "cand": m_, "param": p_, "meff": e_, "uar_pseudo": float(u_),
                       "chosen": (m_, p_, e_) == best} for (m_, p_, e_), u_ in sc.items()]
    return mrows


def label_sets_of(ctx, r, pool):
    pos_of = {int(i): j for j, i in enumerate(pool)}
    return {"0": np.array([], int), **{K: np.array([pos_of[int(i)] for i in ctx.idx[(r, K)]]) for K in ("1", "5", "10", "25")}}


def run_world(world, T, data_f, data_m, reps, out, cache, pseudo_targets=None, fused_only=False):
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in WORLDS[world] + [world]}
    prows = []
    for c in S:                                                                  # 仮のターゲット
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c or (pseudo_targets and c not in pseudo_targets):
            continue
        info_c = {k: M5.pool_info(data_m[k] if k != world else data_f, pool_c, C_FIT, cache[k]) for k in sets}
        evc = data_f[c]
        ctxc = CB.Ctx(evc, PSEUDO_REPS, ["1", "5", "10", "25"], center="ind")
        for r in PSEUDO_REPS:
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            stu = make_students(world, c, data_f, data_m, pool_c, pool, te, sets, info_c)
            run_world_unit(world, stu, evc.y[pool], evc.y[te], label_sets_of(ctxc, r, pool), prows, {"pseudo": c, "rep": r}, fused_only)
    pdf = pd.DataFrame(prows)
    pdf.to_csv(out / f"{T}_pseudo.csv", index=False)
    mrows = []
    for student in WORLDS[world] + [world]:
        if ((pdf.student == student) & (pdf.method == "none")).any():            # fused_only ではモダリティの生徒に E39 の行が無い
            mrows += choose(pdf, student)
    pd.DataFrame([{"target": T, **x} for x in mrows]).to_csv(out / f"{T}_meta.csv", index=False)
    info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, C_FIT, cache[k]) for k in sets}
    ev = data_f[T]
    ctx = CB.Ctx(ev, reps, ["1", "5", "10", "25"], center="ind")
    rows = []
    for r in reps:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
        run_world_unit(world, stu, ev.y[pool], ev.y[te], label_sets_of(ctx, r, pool), rows, {"target": T, "rep": r}, fused_only)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


def run_kind(kind, T, data, f, Cs, P, Y, reps, out, cache, pseudo_targets=None):
    """同じモダリティ（E59 の追加と同じ生徒・見込み q・pq）。"""
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    gamma, rho_s, rho_m, cb, cp, _ = settings(kind, T)
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    f44 = ROOT / "results" / f"committee_{kind}" / f"{T}.csv"
    if f44.exists():
        lam44 = float(pd.read_csv(f44).query("variant == 'C'").lam.iloc[0])
    else:                                                                        # E44 を流していない種類（T）は仮のターゲットで選ぶ
        lam44, _ = choose_lam(S, P, Y)
    pool_q = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), lam44)
    mem = {g: {"p": softmax_t(draft_logits(data, S, g, gamma, rho_s, cache)), "q": c_pool_on(g, S, P, Y, lam44)} for g in S}
    pool_pq, _ = fit_stack(T, "pq", S, mem, Y, [])

    def student_of(ev, Xc, src, lin, n_of, pool_s, C, te, ct):
        st = Student(ev, Xc, src, lin, n_of, pool_s, C, gamma, rho_s, [rho_m])
        Am = st.mats[rho_m][0]
        return {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am, "Zte": ((M.rows(ev.X, te) - ct - st.src.mu) / st.src.sd) @ Am}
    prows = []
    for c in S:                                                                  # 仮のターゲット
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c or (pseudo_targets and c not in pseudo_targets):
            continue
        Gc = [g for g in S if g != c]
        pool_pq_c, _ = fit_stack(T, "pq", Gc, mem, Y, []) if len(Gc) >= 2 else (None, None)
        src_c, lin_c, n_c = M5.pool_info(data, pool_c, C_FIT, cache)
        evc = data[c]
        ctxc = CB.Ctx(evc, PSEUDO_REPS, ["1", "5", "10", "25"], center="ind")
        for r in PSEUDO_REPS:
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            s = student_of(evc, M.rows(evc.X, pool) - ctxc.ct[r], src_c, lin_c, n_c, pool_c, C_FIT, te, ctxc.ct[r])
            bel = {"q": mem[c]["q"][pool]}
            if pool_pq_c is not None:
                F = torch.as_tensor(np.stack([logp(mem[c]["p"][pool]), logp(bel["q"])], 1), dtype=DT, device=DEV)
                bel["pq"] = pool_pq_c.predict(F, torch.ones(len(pool), 2, dtype=DT, device=DEV))
            run_student(Prob(s, evc.y[pool], label_sets_of(ctxc, r, pool)), evc.y[te], MEFF_A, bel, prows, {"pseudo": c, "rep": r}, kind)
    pdf = pd.DataFrame(prows)
    pdf.to_csv(out / f"{T}_pseudo.csv", index=False)
    pd.DataFrame([{"target": T, **x} for x in choose(pdf, kind)]).to_csv(out / f"{T}_meta.csv", index=False)
    src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
    ev = data[T]
    ctx = CB.Ctx(ev, reps, ["1", "5", "10", "25"], center="ind")
    rows = []
    for r in reps:
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        s = student_of(ev, Xc, src_pc, lin, n_of, S, C_T, te, ctx.ct[r])
        Ps = [softmax_t(f[s_].logits(Xc, Cs[s_])) for s_ in S]
        tens2 = lambda cc: (torch.as_tensor(np.stack(cc, 1), dtype=DT, device=DEV), torch.ones(len(pool), len(cc), dtype=DT, device=DEV))  # noqa: E731
        p = softmax_t(s["st"].L0)
        q = pool_q.predict(*tens2([logp(x) for x in Ps]))
        bel = {"q": q, "pq": pool_pq.predict(*tens2([logp(p), logp(q)]))}
        run_student(Prob(s, ev.y[pool], label_sets_of(ctx, r, pool)), ev.y[te], MEFF_A, bel, rows, {"target": T, "rep": r}, kind)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def read(path):
    return pd.read_csv(path, dtype=STR_COLS, keep_default_na=False)


def analyze(names=None, results=None, out_rel=None, suffix="") -> int:
    """既定は第 1 エンコーダの種類と世界（e61_*.csv）．第 2 エンコーダの世界は --names fuse2-… --suffix _2nd で e61_2nd_*.csv に書く（ours は e60_2nd_targets.csv）。"""
    results, out_rel = results or ROOT / "results", out_rel or OUT_REL
    names = names or [d.name[len("baselines_"):] for d in sorted(results.glob("baselines_*")) if d.is_dir() and d.name[len("baselines_"):] not in WORLDS2]
    e59 = OUT_REL / "e59k_targets.csv"
    e60 = OUT_REL / f"e60{suffix}_targets.csv"
    ours_k = pd.read_csv(e59, dtype={"K": str}).set_index(["kind", "target", "K"]) if e59.exists() else None
    ours_w = pd.read_csv(e60, dtype={"K": str}).set_index(["world", "target", "student", "K"]) if e60.exists() else None
    tabs = []
    for name in names:
        d = results / f"baselines_{name}"
        world = name in WORLDS
        for x in sorted(f_ for f_ in d.glob("*.csv") if not f_.stem.endswith(("_pseudo", "_meta"))):
            T = x.stem
            a = read(x)
            a = a[a.rep.isin(EC.REPS)]
            meta = read(d / f"{T}_meta.csv")
            u = a.groupby(["student", "method", "param", "meff", "K"]).uar.mean() * 100
            for student in sorted(set(a.student)):
                if (student, "none", "-", "0", "0") not in u.index:                     # --fused-only の世界ではモダリティの生徒に E39 の行が無い
                    continue
                fused = world and student == name
                for K in KS:
                    row = {"setting": name, "target": T, "student": student, "K": K, "e39": float(u.loc[(student, "none", "-", "0", K)])}
                    try:
                        o = ours_w.loc[(name, T, student, K)] if world else ours_k.loc[(name, T, K)]
                        row["ours"], row["p_ctrl"] = float(o.meta), float(o.p_ctrl)
                    except (KeyError, AttributeError, TypeError):
                        row["ours"], row["p_ctrl"] = np.nan, np.nan
                    mk = meta[(meta.student == student) & (meta.K == K)]
                    for method in sorted(set(mk.method)):
                        ch = mk[(mk.method == method) & (mk.chosen == True)].iloc[0]  # noqa: E712
                        row[f"{method}:chosen"] = f"{ch.cand}|{ch.param}|{ch.meff}"
                        row[f"{method}:tuned"] = float(u.loc[(student, ch.cand, ch.param, ch.meff, K)])
                        fp, fm = FIXED.get(method, (None, None)) if method != "kd" else (KD_FIXED_TEACHER["fused" if fused else ("mod" if world else "kind")] + "T2", "n")
                        row[f"{method}:fixed"] = float(u.loc[(student, method, fp, fm, K)]) if (student, method, fp, fm, K) in u.index else np.nan
                        row[f"{method}:oracle"] = float(max(v for (s_, m_, p_, e_, K_), v in u.items() if s_ == student and m_ == method and K_ == K))
                    tabs.append(row)
    D = pd.DataFrame(tabs)
    methods = sorted({c.split(":")[0] for c in D.columns if c.endswith(":tuned")})
    tests = []
    for (name, student, K), g in D.groupby(["setting", "student", "K"], sort=False):
        row = {"setting": name, "student": student, "K": K, "targets": len(g), "e39": float(g.e39.mean()), "ours": float(g.ours.mean()),
               "p_ctrl": float(g.p_ctrl.mean()), "ours-e39": float((g.ours - g.e39).mean())}
        for m_ in methods:
            if f"{m_}:tuned" not in g or g[f"{m_}:tuned"].isna().all():
                continue
            for v in ("tuned", "fixed", "oracle"):
                row[f"{m_}:{v}"] = float(g[f"{m_}:{v}"].mean())
                row[f"{m_}:{v}-ours"] = float((g[f"{m_}:{v}"] - g.ours).mean())
                row[f"{m_}:{v}-e39"] = float((g[f"{m_}:{v}"] - g.e39).mean())
            row[f"{m_}:tuned>ours"] = int((g[f"{m_}:tuned"] > g.ours).sum())
        tests.append(row)
    R = pd.DataFrame(tests)
    r = R.set_index(["setting", "student", "K"])
    for c in ("B1", "B2", "B3", "B4", "B5", "B7"):
        R[c] = pd.Series(np.nan, index=R.index, dtype=object)                    # 真偽と NaN を混ぜるので object にしておく
    for name in names:                                                            # 予言 B1〜B7（docs/exploratory-analyses.md E61）
        if name in WORLDS:
            for student in WORLDS[name] + [name]:
                if (name, student, "5") not in r.index:
                    continue
                m = R[(R.setting == name) & (R.student == student)]
                if student in DIMS:
                    R.loc[m.index, "B4"] = bool(all(abs(r.loc[(name, student, K), "gd:tuned-ours"]) <= 0.5 and r.loc[(name, student, K), "gd:fixed-ours"] <= 0.1
                                                    for K in ("0", "5")))
                    R.loc[m.index, "B7"] = bool(all(r.loc[(name, student, K), "oursx:tuned"] - r.loc[(name, student, K), "gd:tuned"] >= -0.1
                                                    for K in ("0", "5")))
                else:
                    R.loc[m.index, "B5"] = bool(all(r.loc[(name, student, K), f"{m_}:tuned-ours"] <= 0.1 for K in ("0", "5")
                                                    for m_ in ("cotrain_prod", "coreg_sum")))
                    R.loc[m.index, "B7"] = bool(all(r.loc[(name, student, K), "oursx:tuned"] - r.loc[(name, student, K), f"{m_}:tuned"] >= -0.1
                                                    for K in ("0", "5") for m_ in ("cotrain_prod", "coreg_sum")))
        elif (name, name, "5") in r.index:
            m = R[(R.setting == name) & (R.student == name)]
            cols = [c for c in R.columns if c.endswith(":tuned-ours")]
            R.loc[m.index, "B1"] = bool(all(r.loc[(name, name, K), cols].max() < 0.3 for K in ("0", "1", "5")))
            R.loc[m.index, "B2"] = bool(abs(r.loc[(name, name, "0"), "massbal:fixed"] - r.loc[(name, name, "0"), "p_ctrl"]) <= 0.5)   # 方法そのもの（none との選択を挟まない）
            R.loc[m.index, "B3"] = bool(r.loc[(name, name, "0"), "saerens:fixed-e39"] < 0)
    D.to_csv(out_rel / f"e61{suffix}_targets.csv", index=False)
    R.to_csv(out_rel / f"e61{suffix}_tests.csv", index=False)
    pd.set_option("display.width", 250)
    for (name, student), g in R.groupby(["setting", "student"], sort=False):
        print(f"\n== {name} の生徒 {student}（ターゲット {int(g.targets.iloc[0])}．値は E39 との差．ours は E59 の追加／E60 の選んだ形）")
        cols = ["ours-e39"] + [c for c in g.columns if c.endswith(("-e39",)) and c != "ours-e39"]
        print(g.set_index("K")[cols].round(2).T.to_string())
    vc = [c for c in ("B1", "B2", "B3", "B4", "B5", "B7") if c in R]
    print()
    print(R.dropna(subset=vc, how="all").drop_duplicates(["setting", "student"])[["setting", "student"] + vc].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--world")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--pseudo-reps", nargs="+", type=int, help="試走用．既定は 0〜4")
    ap.add_argument("--pseudo-targets", nargs="+", help="試走用．仮のターゲットをこの群だけにする")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--names", nargs="+", help="判定する種類と世界（既定は results/baselines_* の全部）")
    ap.add_argument("--fused-only", action="store_true", help="--world で融合の生徒とその相手だけを解く（AV・VT・AT）")
    ap.add_argument("--results-dir", type=Path, help="試走用．--analyze が baselines_* を読む場所と判定を書く場所")
    ap.add_argument("--suffix", default="", help="--analyze の出力名の接尾辞（第 2 エンコーダは _2nd）")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.names, args.results_dir, args.results_dir, args.suffix)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MERDA_GPU_FRACTION", "1.0")))
    global PSEUDO_REPS
    if args.pseudo_reps is not None:
        PSEUDO_REPS = args.pseudo_reps
    name = args.kind or args.world
    out = args.out_dir or ROOT / "results" / f"baselines_{name}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(name)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{name}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/baselines_distill.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(name)), "reps": args.reps, "ks": KS, "pseudo_reps": PSEUDO_REPS, "pseudo_targets": args.pseudo_targets,
            "c_fit": C_FIT, "taus": TAUS, "temps": TEMPS, "rounds_self": ROUNDS_SELF, "rounds_co": ROUNDS_CO, "co_c": CO_C,
            "meffs": MEFF_B if args.world else MEFF_A, "fused_only": args.fused_only, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.world:
        data_f = {g: M.load(g, name) for g in groups}
        data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(name)}
        cache = {k: {} for k in WORLDS[name] + [name]}
    else:
        data = {g: M.load(g, name) for g in groups}
        f, Cs, P, Y = CM.prepare(name, data)
        cache = {}
    for t in todo:
        t0 = time.perf_counter()
        if args.world:
            run_world(name, t, data_f, data_m, args.reps, out, cache, args.pseudo_targets, args.fused_only)
        else:
            run_kind(name, t, data, f, Cs, P, Y, args.reps, out, cache, args.pseudo_targets)
        mem = f"　GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"{name} {t}：{time.perf_counter() - t0:.0f} 秒{mem}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
