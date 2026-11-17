"""DAI 実験 1：コーパスごとの識別器と，中核の手法（設計は `docs/設計/実験1_設計.md`．計画は `docs/計画/計画_2026-11-04.tex`）。

段（--stage）：
  source  ：コーパスごと × 種類（A,V,T,AV,AT,VT,AVT）の識別器を作って保存する（C は塊の 5 分割の交差検証）。
  pseudo  ：評価のターゲット T ごとに，ソース 3 つの 1 つずつを仮のターゲット c にし，残り 2 つの識別器で回す。
            段 1：比較の基準（K 本だけ）を引き戻しの強さの格子で解き，(T, 種類, K) ごとに引き戻しの強さを選ぶ。
            段 2：予測を付ける側を作り，積のモデルのモダリティの指数を (T, K) ごとに選ぶ（候補のラベルでの UAR）。
            段 3：生徒（A,V,T,AVT）の全手法を，格子の全部の点で解く。
  select  ：仮のターゲットの成績で，(T, 生徒, K, 手法) ごとに設定を選び，固定のファイルを書く。
  eval    ：評価のターゲット T を，ソース 3 つで回す（段 1〜3 と同じ．引き戻しの強さとモダリティの指数は select の値）。

保存（保存の契約．計画 8.7 節）：単位（T, c, 分割, 種）ごとに
  `units/<T>_<c>_f<f>_r<r>.npz`：K 本・候補・テストの sample_id，標準化の平均と標準偏差，
    段 1 の全部の問の W・b，段 2 の予測を付ける側の確率（候補とテスト），学び直した各モダリティの識別器の W・b，
    段 3 の全部の問の W・b
  `rows.csv`：全部の問の UAR（テスト）．予測を付ける側の UAR（候補のラベルとテストのラベル）
評価の回の c は "ALL"。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
C_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)   # ソースの識別器の C（第 2 版の試走で 28 個中 19 個が 1e-3 の端を選んだので下に広げた）
TETHER = [(cw, cb) for cw in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0) for cb in (1.0, 10.0)]   # 出発点への引き戻し (C′, C_b)．端に寄らないよう両側に 1 点ずつ広げた
RHO_GRID = (0.0625, 0.25, 1.0, 4.0)                                  # 疑似ラベルの重みの合計 ÷ 候補の本数（定義の検査 D1．絶対の本数だとコーパスで 1 本あたりの重みが変わる）
A_GRID = (1, 4, 16)                                                  # K 本の倍率
KD_T = (1, 2, 4)                                                     # 温度つき蒸留の温度
AG_TAU = (0.5, 0.7, 0.9)                                             # 一致した発話だけの疑似ラベルの確信度のしきい値
AG_F = (0.25, 1, 4)                                                  # その疑似ラベル 1 本あたりの重み
MIX_ALPHA = (0.2, 0.5, 0.8)                                          # 混ぜる手（Zhang ほか 2021 の式 (8)・(9)）の生徒自身の予測の重み α
MIX_TAU = 0.9                                                        # 同．混ぜた確率の最大がこれを超える発話だけに硬いラベル（原論文 4.2 節の値で固定）
MIX_F = (0.25, 1, 4)                                                 # 同．その硬いラベル 1 本あたりの重み
CT_ROUNDS = 3                                                        # 柔らかい共訓練の巡の数
COREG_RHO = (0.0, 0.0625, 0.25, 1.0, 4.0)                            # 共正則化の一致の重み．0 は「使わない」（前の実験 E61 の調整版が選んだ．gate-code の 3 回目 M9）
TGT_C = C_GRID                                                       # K 本だけでゼロから学ぶ版の C
W_EXP = (0.0, 0.5, 1.0, 2.0)                                         # 積のモデルのモダリティの指数
W_GRID = [w for w in itertools.product(W_EXP, repeat=3) if any(w)]
DEFAULTS = {"joint": {"rho": 1.0, "a": 1}, "rolesep": {"rho": 1.0, "a": 1}, "kd": {"temp": 2, "rho": 1.0},
            "agree": {"tau": 0.9, "f": 1}, "mix": {"alpha": 0.2, "f": 1}, "cotrain": {"rho": 1.0}, "coreg": {"rho": 1.0}, "tgtonly": {"C": 1.0},
            "w": (1.0, 1.0, 1.0)}                                            # 選択なしの値（設計の文書 7 節の表．順 3 の前に固定）
KS_ALL = (0,) + DD.KS


def mval(rho, n):
    return float(rho) * float(n)


def git_commit() -> tuple[str, bool]:
    c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "exp", "splits", "measure", "features",
                                 "data/metadata"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())     # 未追跡のファイルも数える（I9）
    return c, dirty


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ENC = {"enc": DD.ENC1}                                                  # main が --encoder で替える（R7）


class Corpus:
    """1 コーパスの表と特徴（GPU に載せる）。"""
    cache: dict = {}

    def __init__(self, name: str):
        df, feats = DD.load_corpus(name, ENC["enc"])
        self.name, self.df = name, df
        self.y = df["label_4"].to_numpy().astype(int)
        self.X = {k: S.t(DD.kind_matrix(feats, k)) for k in DD.KINDS}

    @classmethod
    def get(cls, name):
        if name not in cls.cache:
            cls.cache[name] = Corpus(name)
        return cls.cache[name]


# ------------------------------------------------------------------ 段 source

def stage_source(out: Path, corpora=DD.CORPORA) -> None:
    (out / "source").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in corpora:
        cp = Corpus.get(c)
        for kind in DD.KINDS:
            t0 = time.time()
            X = cp.X[kind]
            cv = np.zeros((DD.N_FOLDS, len(C_GRID)))
            for f in range(DD.N_FOLDS):
                tr = np.flatnonzero(DD.pool_mask(cp.df, f)); te = np.flatnonzero(cp.df["fold"].to_numpy() == f)
                mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
                Xs = (X - mu) / sd
                G = S.balanced_onehot(cp.y[tr], np.arange(len(tr)), len(tr))[None].expand(len(C_GRID), -1, -1)
                r = S.fit(Xs[tr], G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.array(C_GRID), Cb=math.inf)
                cv[f] = S.uar_batch(cp.y[te], S.logits(Xs[te], r["W"], r["b"]))
            mean = cv.mean(0)
            best = int(np.flatnonzero(mean == mean.max())[0])               # 同点なら小さい C（強い正則化）
            mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
            Xs = (X - mu) / sd
            G = S.balanced_onehot(cp.y, np.arange(len(cp.y)), len(cp.y))[None].expand(len(C_GRID), -1, -1)
            ra = S.fit(Xs, G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.array(C_GRID), Cb=math.inf)   # 8 つの C の最終の識別器（R4-src）
            r = {"W": ra["W"][best:best + 1], "b": ra["b"][best:best + 1], "grad_max": float(ra["grad_p"][best])}
            np.savez_compressed(out / "source" / f"{c}_{kind}.npz", W=r["W"][0].cpu().numpy(), b=r["b"][0].cpu().numpy(),
                                W_allC=f32(ra["W"]), b_allC=f32(ra["b"]), grad_allC=ra["grad_p"],
                                mu=mu.cpu().numpy(), sd=sd.cpu().numpy(), C=C_GRID[best], cv=cv, C_grid=np.array(C_GRID),
                                sample_id=cp.df["sample_id"].to_numpy().astype(str), grad_max=r["grad_max"])
            rows.append({"corpus": c, "kind": kind, "C": C_GRID[best], "cv_uar": float(mean[best]),
                         **{f"cv_{C}": float(v) for C, v in zip(C_GRID, mean)}, "grad_max": r["grad_max"],
                         "W_norm": float(r["W"][0].norm()), "b_range": float(r["b"][0].max() - r["b"][0].min()),
                         "commit": RUN["commit"], "sec": time.time() - t0})
            print(f"source {c} {kind}: C={C_GRID[best]} 交差検証の UAR {mean[best]:.2f} ({time.time()-t0:.0f} 秒)", flush=True)
    sm = out / "source" / "summary.csv"
    old = pd.read_csv(sm) if sm.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old["corpus"].isin(new["corpus"])]
    pd.concat([old, new]).to_csv(sm, index=False)


def load_source(out: Path, corpus: str, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(out / "source" / f"{corpus}_{kind}.npz")
    return S.t(z["W"]), S.t(z["b"])


def load_source_stats(out: Path, corpus: str, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(out / "source" / f"{corpus}_{kind}.npz")
    return S.t(z["mu"]), S.t(z["sd"])


# ------------------------------------------------------------------ 単位（T, c, 分割, 種）

class Unit:
    """1 つの単位の分割，K 本，標準化した特徴，出発点。"""

    def __init__(self, out: Path, T: str, c: str, srcs: list[str], f: int, r: int):
        self.T, self.c, self.srcs, self.f, self.r = T, c, srcs, f, r
        cp = Corpus.get(c)
        self.cp = cp
        self.pool = np.flatnonzero(DD.pool_mask(cp.df, f))
        self.test = np.flatnonzero(cp.df["fold"].to_numpy() == f)
        self.ypool, self.ytest = cp.y[self.pool], cp.y[self.test]
        korder = DD.k_orders(cp.y, self.pool, c, f, r)
        self.kpos = {0: np.array([], int), **{K: np.searchsorted(self.pool, korder[K]) for K in DD.KS}}
        assert all((self.pool[self.kpos[K]] == korder[K]).all() for K in DD.KS)
        self.Xp, self.Xt, self.stats, self.start, self.src, self.src_stats = {}, {}, {}, {}, {}, {}
        for kind in DD.KINDS:
            X = cp.X[kind]
            mu, sd = X[self.pool].mean(0), X[self.pool].std(0).clamp(min=1e-6)   # 候補（K 本を含み，テストを含まない）の統計
            self.Xp[kind], self.Xt[kind] = (X[self.pool] - mu) / sd, (X[self.test] - mu) / sd
            self.stats[kind] = (mu.cpu().numpy(), sd.cpu().numpy())
            Ws, bs = zip(*[load_source(out, s, kind) for s in srcs])
            self.src_stats[kind] = [load_source_stats(out, s, kind) for s in srcs]
            self.start[kind] = (torch.stack(Ws).mean(0), torch.stack(bs).mean(0))   # 点数の等しい重みの平均（8.4 節）
            self.src[kind] = list(zip(srcs, Ws, bs))
        self.n = len(self.pool)

    def name(self):
        return f"{self.T}_{self.c}_f{self.f}_r{self.r}"

    def gK(self, K, scale=1.0):
        return S.balanced_onehot(self.ypool, self.kpos[K], self.n, scale)


def probs(X, W, b):
    return torch.softmax(S.logits(X, W, b), -1)


def f32(x: torch.Tensor) -> np.ndarray:
    return x.float().cpu().numpy()


def pred8(Z: torch.Tensor) -> np.ndarray:
    """(P, n, 4) の点数 → テストの発話ごとの判定（int8）．区間と感情ごとの再現率は保存したこの判定から作る（統計の検査 S3，m39）。"""
    return Z.argmax(-1).to(torch.int8).cpu().numpy()


# ------------------------------------------------------------------ 段 1：比較の基準と全ラベルの参照

def pass1(u: Unit, store: dict, rows: list) -> None:
    """種類ごとに，引き戻しの格子の全部の点で，K 本だけ（K≥1）と候補の全ラベル（上限の参照 W*）を解く（評価の回も同じ．定義の検査 D17）．
    あわせて，出発点の W のまま K 本で切片だけを直す版（引き戻しの C_b は格子の 2 値．R6）と，K 本だけでゼロから学ぶ版（C の格子）を解く。"""
    for kind in DD.KINDS:
        W0, b0 = u.start[kind]
        Z0 = S.logits(u.Xt[kind], W0, b0)
        rows.append(dict(stage=1, method="start", kind=kind, K=0, uar=S.uar(u.ytest, Z0.argmax(1).cpu().numpy()) * 100))
        store[f"p1_{kind}_start_pred"] = pred8(Z0)
        # ターゲットの統計で標準化しない出発点（各ソースの識別器を，そのソースの平均と標準偏差のまま当てて点数を平均．R4，a3）
        Xraw = u.cp.X[kind][u.test]
        Zs = sum(S.logits((Xraw - mu) / sd, Wc, bc) for (_, Wc, bc), (mu, sd) in zip(u.src[kind], u.src_stats[kind])) / len(u.src[kind])
        rows.append(dict(stage=1, method="start_srcstd", kind=kind, K=0, uar=S.uar(u.ytest, Zs.argmax(1).cpu().numpy()) * 100))
        store[f"p1_{kind}_start_srcstd_pred"] = pred8(Zs)
        items = [(K, cw, cb) for K in DD.KS + ("all",) for cw, cb in TETHER]
        G = torch.stack([S.balanced_onehot(u.ypool, np.arange(u.n), u.n) if K == "all" else u.gK(K) for K, _, _ in items])
        res = S.fit(u.Xp[kind], G, W0, b0, Cw=[x[1] for x in items], Cb=[x[2] for x in items])
        Z = S.logits(u.Xt[kind], res["W"], res["b"])
        ua = S.uar_batch(u.ytest, Z)
        for (K, cw, cb), v, gp, sv, st, fn in zip(items, ua, res["grad_p"], G.sum((1, 2)).cpu().numpy(), res["stalled_p"], res["finite_p"]):
            rows.append(dict(stage=1, method="oracle" if K == "all" else "none", kind=kind, K=-1 if K == "all" else K,
                             Cw=cw, Cb=cb, uar=v, S=float(sv), grad=float(gp), stalled=bool(st), finite=bool(fn)))
        store[f"p1_{kind}_W"], store[f"p1_{kind}_b"] = f32(res["W"]), f32(res["b"])
        store[f"p1_{kind}_pred"] = pred8(Z)
        store[f"p1_{kind}_items"] = np.array([f"{K}|{cw}|{cb}" for K, cw, cb in items])
        rows.append(dict(stage=1, method="_conv", kind=kind, grad_max=res["grad_max"], n_iter=res["n_iter"], n_stalled=res["stalled"]))
        if kind not in DD.STUDENTS:
            continue
        # 出発点の W のまま K 本で切片だけ（C_b は 1 と 10）
        itb = [(K, cb) for K in DD.KS for cb in (1.0, 10.0)]
        Gb = torch.stack([u.gK(K) for K, _ in itb])
        rb = S.fit(u.Xp[kind], Gb, W0, b0, Cw=math.inf, Cb=[x[1] for x in itb], fix_W=True)
        Zb = S.logits(u.Xt[kind], rb["W"], rb["b"])
        ub = S.uar_batch(u.ytest, Zb)
        for (K, cb), v, gp, st, fn in zip(itb, ub, rb["grad_p"], rb["stalled_p"], rb["finite_p"]):
            rows.append(dict(stage=1, method="kbias", kind=kind, K=K, Cb=cb, uar=v, S=4.0 * K, grad=float(gp), stalled=bool(st), finite=bool(fn)))
        store[f"p1_{kind}_kbias_b"] = f32(rb["b"])
        store[f"p1_{kind}_kbias_pred"] = pred8(Zb)
        store[f"p1_{kind}_kbias_items"] = np.array([f"{K}|{cb}" for K, cb in itb])
        # K 本だけでゼロから（W0 = 0，切片に罰則なし，C の格子）
        itt = [(K, C) for K in DD.KS for C in TGT_C]
        Gt = torch.stack([u.gK(K) for K, _ in itt])
        d = u.Xp[kind].shape[1]
        rt = S.fit(u.Xp[kind], Gt, torch.zeros(4, d), torch.zeros(4), Cw=[x[1] for x in itt], Cb=math.inf)
        Zt = S.logits(u.Xt[kind], rt["W"], rt["b"])
        ut = S.uar_batch(u.ytest, Zt)
        for (K, C), v, gp, st, fn in zip(itt, ut, rt["grad_p"], rt["stalled_p"], rt["finite_p"]):
            rows.append(dict(stage=1, method="tgtonly", kind=kind, K=K, C=C, uar=v, S=4.0 * K, grad=float(gp), stalled=bool(st), finite=bool(fn)))
        store[f"p1_{kind}_tgt_W"], store[f"p1_{kind}_tgt_b"] = f32(rt["W"]), f32(rt["b"])
        store[f"p1_{kind}_tgt_pred"] = pred8(Zt)
        store[f"p1_{kind}_tgt_items"] = np.array([f"{K}|{C}" for K, C in itt])


# ------------------------------------------------------------------ 段 2：予測を付ける側

def none_wb(u: Unit, p1, kind: str, K: int, tether: dict):
    """比較の基準 none[K] の (W, b)．K=0 は出発点．段 1 の保存から，選んだ引き戻しの強さの解を読む。"""
    if K == 0:
        return u.start[kind]
    cw, cb = tether[(kind, K)]
    items = [str(x) for x in p1[f"p1_{kind}_items"]]
    j = items.index(f"{K}|{cw}|{cb}")
    return S.t(p1[f"p1_{kind}_W"][j]), S.t(p1[f"p1_{kind}_b"][j])


def none_probs(u: Unit, p1, K: int, tether: dict) -> dict:
    tk = {}
    for kind in DD.KINDS:
        W, b = none_wb(u, p1, kind, K, tether)
        tk[f"none_{kind}"] = (probs(u.Xp[kind], W, b), probs(u.Xt[kind], W, b))
    return tk


def pass2(u: Unit, p1, store: dict, rows: list, tether: dict) -> None:
    """K ごとに，積のモデルの各モダリティを学び直す（融合 AVT の none[K] の予測を疑似ラベルにし，重みの合計は候補の本数．
    K 本も入れる）．全部の指数の積と各種類の予測の当たり方を，候補のラベル（選ぶのに使う）とテストのラベル（記録だけ）で測る。"""
    for K in KS_ALL:
        tk = none_probs(u, p1, K, tether)
        F = tk["none_AVT"][0]
        for m in "AVT":
            W0, b0 = u.start[m]
            cw, cb = tether[(m, max(K, 1))]
            G = (u.gK(K) if K else torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)) + S.soft_rows(F, float(u.n))
            res = S.fit(u.Xp[m], G[None], W0, b0, Cw=cw, Cb=cb)
            store[f"p2_K{K}_ret_{m}_W"], store[f"p2_K{K}_ret_{m}_b"] = f32(res["W"][0]), f32(res["b"][0])
            tk[f"ret_{m}"] = (probs(u.Xp[m], res["W"][0], res["b"][0]), probs(u.Xt[m], res["W"][0], res["b"][0]))
            rows.append(dict(stage=2, method="_conv", kind=f"ret_{m}", K=K, grad_max=res["grad_max"], n_iter=res["n_iter"],
                             n_stalled=res["stalled"], finite=bool(res["finite_p"].all())))
        Lp = torch.stack([tk[f"ret_{m}"][0].clamp(min=1e-12).log() for m in "AVT"])
        Lt = torch.stack([tk[f"ret_{m}"][1].clamp(min=1e-12).log() for m in "AVT"])
        nk = np.setdiff1d(np.arange(u.n), u.kpos[K])                      # K 本を除いた候補（m35）
        for w in W_GRID:
            wt = S.t(np.array(w))[:, None, None]
            up = S.uar(u.ypool[nk], (wt * Lp).sum(0).argmax(1).cpu().numpy()[nk]) * 100
            ut = S.uar(u.ytest, (wt * Lt).sum(0).argmax(1).cpu().numpy()) * 100
            rows.append(dict(stage=2, method="product", teacher=str(w), K=K, uar_pool=up, uar=ut))
        tk["poe_none"] = (poe(tk, "AVT", 0), poe(tk, "AVT", 1))
        for name in [f"none_{k}" for k in DD.KINDS] + [f"ret_{m}" for m in "AVT"] + ["poe_none"]:
            P, Q = tk[name]
            rows.append(dict(stage=2, method="teacher", teacher=name, K=K,
                             uar_pool=S.uar(u.ypool[nk], P.argmax(1).cpu().numpy()[nk]) * 100, uar=S.uar(u.ytest, Q.argmax(1).cpu().numpy()) * 100))
            store[f"p2_K{K}_{name}_pool"], store[f"p2_K{K}_{name}_test"] = P.half().cpu().numpy(), Q.half().cpu().numpy()


def build_tk(u: Unit, p1, p2, tether: dict) -> dict:
    """段 1・段 2 の保存（float32 の W・b）から，K ごとの予測を作り直す。"""
    TK = {}
    for K in KS_ALL:
        tk = none_probs(u, p1, K, tether)
        for m in "AVT":
            W, b = S.t(p2[f"p2_K{K}_ret_{m}_W"]), S.t(p2[f"p2_K{K}_ret_{m}_b"])
            tk[f"ret_{m}"] = (probs(u.Xp[m], W, b), probs(u.Xt[m], W, b))
        TK[K] = tk
    return TK


def product(tk: dict, w, which: int, src: str = "ret"):
    """モダリティの確率の重みつきの積（対数の重みつき和の softmax）．src="ret" は学び直した識別器，"none" は学び直さない none[K]。"""
    L = sum(float(wi) * tk[f"{src}_{m}"][which].clamp(min=1e-12).log() for wi, m in zip(w, "AVT"))
    return torch.softmax(L, -1)


def poe(tk: dict, mods: str, which: int, src="none"):
    L = sum(tk[f"{src}_{m}"][which].clamp(min=1e-12).log() for m in mods)
    return torch.softmax(L, -1)


# ------------------------------------------------------------------ 段 3：生徒

def cotrain(u: Unit, tk: dict, K: int, tether: dict, store: dict, rows: list) -> dict:
    """柔らかい共訓練（Blum & Mitchell 1998 の多視点の考えを，柔らかいラベルで線形の識別器に当てた版）．
    音声・顔・言葉の識別器を none[K] から始め，CT_ROUNDS 巡のあいだ順に，各モダリティを「ほかの 2 つの現在の確率の積」を疑似ラベル
    （感情ごとの総量をそろえ，重みの合計 ρ × 候補の本数）と K 本で，出発点から学び直す．ρ の格子ごとに別々に回す．
    戻り：{ρ: {m: (候補の確率, テストの確率)}}。"""
    gk = u.gK(K) if K else torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
    cur = {rho: {m: tk[f"none_{m}"] for m in "AVT"} for rho in RHO_GRID}
    for rnd in range(CT_ROUNDS):
        for m in "AVT":
            W0, b0 = u.start[m]
            cw, cb = tether[(m, max(K, 1))]
            G = []
            for rho in RHO_GRID:
                L = sum(cur[rho][o][0].clamp(min=1e-12).log() for o in "AVT" if o != m)
                G.append(gk + S.soft_rows(torch.softmax(L, -1), mval(rho, u.n)))
            res = S.fit(u.Xp[m], torch.stack(G), W0, b0, Cw=cw, Cb=cb)
            if rnd == CT_ROUNDS - 1:                                    # 最後の巡の疑似ラベル（重みの集まりを後で数えるため．D11）
                store[f"p3c_K{K}_{m}_pseudo"] = torch.stack([x - gk for x in G]).half().cpu().numpy()
            for jr, rho in enumerate(RHO_GRID):
                cur[rho][m] = (probs(u.Xp[m], res["W"][jr], res["b"][jr]), probs(u.Xt[m], res["W"][jr], res["b"][jr]))
            if rnd == CT_ROUNDS - 1:
                store[f"p3c_K{K}_{m}_W"], store[f"p3c_K{K}_{m}_b"] = f32(res["W"]), f32(res["b"])
                store[f"p3c_K{K}_{m}_pred"] = pred8(S.logits(u.Xt[m], res["W"], res["b"]))   # 並びは RHO_GRID
            rows.append(dict(stage=3, method="_conv_cotrain", kind=m, K=K, round=rnd, grad_max=res["grad_max"],
                             n_stalled=res["stalled"], finite=bool(res["finite_p"].all())))
    return cur


def coreg(u: Unit, K: int, tether: dict, store: dict, rows: list) -> None:
    """共正則化（Sindhwani & Rosenberg 2008 の多視点の考え．前の実験 E61 の coreg と同じ問）．ρ の格子の問をまとめて
    `dai_solver.fit_coreg`（共通の学習器と同じ問ごとの L-BFGS．tol 1e-6）で解く：
      Σ_m [K 本の交差エントロピー ＋ ‖W_m − W0_m‖²/(2C′_m) ＋ ‖b_m − b0_m‖²/(2C_b,m)] ＋ (ρ/2) Σ_i Σ_{m<m'} ‖c_m,i − c_m',i‖²
    （c は感情の平均を引いた点数）を S = 3ΣG_K ＋ ρn で割ったもの．凸．単一の生徒はその識別器，AVT の生徒は 3 つの点数の和
    （つないだ特徴の上の線形の識別器と同じ形）．判定と W・b を保存し，行を足す（第 3 版の torch の L-BFGS は 2 万反復でも
    許容の勾配に届かず遅かったので替えた）。"""
    gk = u.gK(K) if K else torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
    mods = "AVT"
    tw = [tether[(m, max(K, 1))] for m in mods]
    res = S.fit_coreg([u.Xp[m] for m in mods], gk, [u.start[m][0] for m in mods], [u.start[m][1] for m in mods],
                      [x[0] for x in tw], [x[1] for x in tw], list(COREG_RHO))
    Zsum = 0
    for i, m in enumerate(mods):
        W, b = res["Ws"][i], res["bs"][i]
        Zt = S.logits(u.Xt[m], W, b)                                     # (P, n_test, 4)
        Zsum = Zsum + Zt
        pr = pred8(Zt)
        store[f"p3g_K{K}_{m}_pred"] = pr                                 # 並びは COREG_RHO
        store[f"p3g_K{K}_{m}_W"], store[f"p3g_K{K}_{m}_b"] = f32(W), f32(b)
        for j, rho in enumerate(COREG_RHO):
            rows.append(dict(stage=3, student=m, K=K, method="coreg", teacher="cr", rho=rho, grad=float(res["grad_p"][j]),
                             stalled=bool(res["stalled_p"][j]), finite=bool(res["finite_p"][j]), S=float(res["S"][j]),
                             uar=100 * S.uar(u.ytest, pr[j].astype(int))))
    prs = pred8(Zsum)
    store[f"p3g_K{K}_AVT_pred"] = prs
    for j, rho in enumerate(COREG_RHO):
        rows.append(dict(stage=3, student="AVT", K=K, method="coreg", teacher="cr", rho=rho, grad=float(res["grad_p"][j]),
                         stalled=bool(res["stalled_p"][j]), finite=bool(res["finite_p"][j]), S=float(res["S"][j]),
                         uar=100 * S.uar(u.ytest, prs[j].astype(int))))
    rows.append(dict(stage=3, method="_conv_coreg", K=K, grad_max=float(res["grad_p"].max()), n_iter=int(res["n_iter_p"].max()),
                     n_stalled=int(res["stalled_p"].sum()), finite=bool(res["finite_p"].all())))


def make_G(u: Unit, method: str, tv_Q, own, gk, it: dict):
    """手法の名前とつまみから (G, G2, T2) を作る（格子の回と引き戻しの感度の回で同じ式を使う）。"""
    zero = torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
    Q = tv_Q
    if method in ("joint", "tt", "joint_tether"):
        return float(it.get("a", 1)) * gk + S.soft_rows(Q, mval(it.get("rho", 1.0), u.n)), zero, 1.0
    if method == "kd":
        Tt = float(it["temp"])
        Qt = torch.softmax(Q.clamp(min=1e-12).log() / Tt, -1)
        return gk, S.soft_rows(Qt, mval(it["rho"], u.n), balance=False), Tt
    if method == "agree":
        keep = (Q.argmax(1) == own.argmax(1)) & (Q.max(1).values >= float(it["tau"]))
        H = torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
        H[keep, Q.argmax(1)[keep]] = 1.0
        it["n_kept"] = int(keep.sum())
        return gk + float(it["f"]) * H, zero, 1.0
    if method == "mix":                                               # Zhang ほか 2021 の式 (8)・(9)：混ぜた確率の argmax を，最大が τ を超える発話だけに
        al = float(it["alpha"])
        M = al * own + (1 - al) * Q
        keep = M.max(1).values > MIX_TAU
        H = torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
        H[keep, M.argmax(1)[keep]] = 1.0
        it["n_kept"] = int(keep.sum())
        return gk + float(it["f"]) * H, zero, 1.0
    raise ValueError(method)


def solve_items(u: Unit, s: str, K: int, specs: list, Cw, Cb, W0, b0, rows: list, store: dict, key: str, extra=None) -> None:
    """specs：[(行の dict, G, G2, T2)] をまとめて解き，行と判定と W・b を残す。"""
    if not specs:
        return
    G = torch.stack([x[1] for x in specs]); G2 = torch.stack([x[2] for x in specs])
    res = S.fit(u.Xp[s], G, W0, b0, Cw=Cw, Cb=Cb, G2=G2, T2=np.array([x[3] for x in specs]))
    Z = S.logits(u.Xt[s], res["W"], res["b"])
    ua = S.uar_batch(u.ytest, Z)
    Ssum = (G.sum((1, 2)) + G2.sum((1, 2))).cpu().numpy()
    for (it, *_), v, sv, gp, st, fn in zip(specs, ua, Ssum, res["grad_p"], res["stalled_p"], res["finite_p"]):
        rows.append(dict(stage=3, student=s, K=K, uar=v, S=float(sv), grad=float(gp), stalled=bool(st), finite=bool(fn),
                         **(extra or {}), **it))
    store[f"{key}_pred"] = pred8(Z)
    store[f"{key}_W"], store[f"{key}_b"] = f32(res["W"]), f32(res["b"])
    store[f"{key}_items"] = np.array([json.dumps(it, sort_keys=True) for it, *_ in specs])
    rows.append(dict(stage=3, method="_conv", student=s, K=K, key=key, grad_max=res["grad_max"], n_iter=res["n_iter"],
                     n_stalled=res["stalled"]))


def pass3(u: Unit, store: dict, rows: list, tether: dict, TK: dict, wsel: dict, fsel: dict | None = None) -> None:
    """fsel：評価の回だけ，固定した (生徒, K) ごとの提案の主の版の点と族 3 の相手の点（引き戻しの感度に使う）。"""
    for K in KS_ALL:
        tk = TK[K]
        gk = u.gK(K) if K else torch.zeros(u.n, 4, device=S.DEV, dtype=S.DT)
        teach = {"sel": product(tk, wsel[K], 0), "one": product(tk, (1, 1, 1), 0)}
        nk = np.setdiff1d(np.arange(u.n), u.kpos[K])
        for tv, Q in teach.items():
            # 予測の鋭さと，感情ごとの総量をそろえたときの重みの集まり方（定義の検査 D2・D11）
            Wt = S.soft_rows(Q, float(u.n))
            colsum = Q.sum(0).cpu().numpy()
            neff = ((Wt.sum(0) ** 2) / (Wt ** 2).sum(0).clamp(min=1e-30)).cpu().numpy()
            ent = float(-(Q * Q.clamp(min=1e-12).log()).sum(1).mean())
            rows.append(dict(stage=3, method="_teacher_stats", teacher=tv, K=K, maxp=float(Q.max(1).values.mean()), entropy=ent,
                             uar_pool=S.uar(u.ypool[nk], Q.argmax(1).cpu().numpy()[nk]) * 100,
                             **{f"colsum_{k}": float(colsum[k]) for k in range(4)}, **{f"neff_{k}": float(neff[k]) for k in range(4)}))
        ct = cotrain(u, tk, K, tether, store, rows)
        coreg(u, K, tether, store, rows)
        for s in DD.STUDENTS:
            W0, b0 = u.start[s]
            cw, cb = tether[(s, max(K, 1))]
            own = tk[f"none_{s}"][0]
            specs = []
            for tv, Q in teach.items():
                grid = ([dict(method="joint", teacher=tv, rho=r, a=a) for r in RHO_GRID for a in A_GRID] +
                        [dict(method="kd", teacher=tv, temp=T_, rho=r) for T_ in KD_T for r in RHO_GRID] +
                        [dict(method="agree", teacher=tv, tau=ta, f=fw) for ta in AG_TAU for fw in AG_F] +
                        [dict(method="mix", teacher=tv, alpha=al, f=fw) for al in MIX_ALPHA for fw in MIX_F])
                for it in grid:
                    specs.append((it, *make_G(u, it["method"], Q, own, gk, it)))
            # 予測を付ける側の種類（記述の比べ．ρ = 1，a = 1）．生徒自身のモダリティの none[K] は入れない（計画 4.3 節）
            tts = {f"single_{m}": tk[f"none_{m}"][0] for m in "AVT" if m != s}
            tts.update({f"pair_{k}": tk[f"none_{k}"][0] for k in ("AV", "AT", "VT")})
            tts["fused_AVT"] = tk["none_AVT"][0]
            tts["poe_none"] = poe(tk, "AVT", 0)
            tts["noret_sel"] = product(tk, wsel[K], 0, src="none")          # 学び直さない積（選んだ指数．m48）
            tts.update({f"ret_{m}": tk[f"ret_{m}"][0] for m in "AVT"})
            tts["start_AVT"] = probs(u.Xp["AVT"], *u.start["AVT"])
            for kind in ["AVT"] + [m for m in "AVT" if m != s]:            # ソースのコーパスごとの識別器（I1）
                tts.update({f"src_{c}_{kind}": probs(u.Xp[kind], Wc, bc) for c, Wc, bc in u.src[kind]})
            for name, Qx in tts.items():
                it = dict(method="tt", teacher=name, rho=1.0, a=1)
                specs.append((it, *make_G(u, "tt", Qx, own, gk, it)))
                Wt = S.soft_rows(Qx, float(u.n))                            # 重みの集まり（D11．予測を付ける側の種類にも）
                neff = ((Wt.sum(0) ** 2) / (Wt ** 2).sum(0).clamp(min=1e-30)).cpu().numpy()
                csum = Qx.sum(0).cpu().numpy()
                rows.append(dict(stage=3, method="_teacher_stats", student=s, teacher=f"tt:{name}", K=K, maxp=float(Qx.max(1).values.mean()),
                                 entropy=float(-(Qx * Qx.clamp(min=1e-12).log()).sum(1).mean()),
                                 uar_pool=S.uar(u.ypool[nk], Qx.argmax(1).cpu().numpy()[nk]) * 100,
                                 **{f"colsum_{k}": float(csum[k]) for k in range(4)}, **{f"neff_{k}": float(neff[k]) for k in range(4)}))
            # 共訓練：AVT の生徒は，共訓練した 3 つの確率の積を疑似ラベルにして学ぶ（単一の生徒は共訓練の識別器そのもの）
            if s == "AVT":
                for rho in RHO_GRID:
                    L = sum(ct[rho][m][0].clamp(min=1e-12).log() for m in "AVT")
                    it = dict(method="cotrain", teacher="ct", rho=rho)
                    specs.append((it, *make_G(u, "joint", torch.softmax(L, -1), own, gk, dict(rho=rho, a=1))))
            solve_items(u, s, K, specs, cw, cb, W0, b0, rows, store, f"p3_{s}_K{K}")
            if s != "AVT":
                for rho in RHO_GRID:
                    v = 100 * S.uar(u.ytest, ct[rho][s][1].argmax(1).cpu().numpy())
                    rows.append(dict(stage=3, student=s, K=K, method="cotrain", teacher="ct", rho=rho, uar=v))
            # 引き戻しの曲線（R4．評価の回だけ）：固定した提案の点と族 3 の相手の点を引き戻しの格子の全点で解く．選び直さない（設計 9.3 節）
            spec_t, cws, cbs = [], [], []
            pts = []                                                       # 評価の回だけ（設計 9.3 節．相手と同じ扱い．定義の検査 D1・D4）
            if fsel is not None:
                pts += [dict(fsel[(s, K)]["prop"], method="joint_tether", tag="prop_sel")]
                pp = fsel[(s, K)]["partner"]
                if pp is not None:
                    pts += [dict(pp, tag="partner_sel")]
            for pt in pts:
                m_ = "joint" if pt["method"] == "joint_tether" else pt["method"]
                for tcw, tcb in TETHER:
                    it = dict(pt, Cw=tcw, Cb=tcb)
                    if m_ != "joint":
                        it["method"] = pt["method"] + "_tether"
                    spec_t.append((it, *make_G(u, m_, teach[pt["teacher"]], own, gk, it)))
                    cws.append(tcw); cbs.append(tcb)
            solve_items(u, s, K, spec_t, cws, cbs, W0, b0, rows, store, f"p3t_{s}_K{K}")
            # 出発点を使わない提案（R6．W0 = 0，切片に罰則なし，C の格子．選択なしの点 ρ=1，a=1，両方の版）
            spec_z = []
            for tv, Q in teach.items():
                for C in TGT_C:
                    it = dict(method="joint_scratch", teacher=tv, rho=1.0, a=1, C=C)
                    spec_z.append((it, *make_G(u, "joint", Q, own, gk, it)))
            d = u.Xp[s].shape[1]
            solve_items(u, s, K, spec_z, [x[0]["C"] for x in spec_z], math.inf, torch.zeros(4, d), torch.zeros(4), rows, store,
                        f"p3z_{s}_K{K}")
            # 役割を分けた版：段 a は予測だけ（K 本なし），段 b は W を固定して K 本で切片だけ
            X, Xt = u.Xp[s], u.Xt[s]
            itA, GA = [], []
            for tv, Qx in teach.items():
                for rho in RHO_GRID:
                    itA.append(dict(teacher=tv, rho=rho)); GA.append(S.soft_rows(Qx, mval(rho, u.n)))
            ra = S.fit(X, torch.stack(GA), W0, b0, Cw=cw, Cb=cb)
            store[f"p3r_{s}_K{K}_Wa"], store[f"p3r_{s}_K{K}_ba"] = f32(ra["W"]), f32(ra["b"])
            store[f"p3r_{s}_K{K}_pred_a"] = pred8(S.logits(Xt, ra["W"], ra["b"]))   # 並びは (予測の版, ρ)
            if K == 0:
                ua = S.uar_batch(u.ytest, S.logits(Xt, ra["W"], ra["b"]))
                for it, v, gp, fn in zip(itA, ua, ra["grad_p"], ra["finite_p"]):
                    for a in A_GRID:
                        rows.append(dict(stage=3, student=s, K=K, method="rolesep", a=a, uar=v, grad=float(gp), finite=bool(fn),
                                         S=float(mval(it["rho"], u.n)), **it))
                continue
            itB, WB, bB, GB = [], [], [], []
            for jj, it in enumerate(itA):
                for a in A_GRID:
                    itB.append(dict(**it, a=a)); WB.append(ra["W"][jj]); bB.append(ra["b"][jj]); GB.append(a * gk)
            rb = S.fit(X, torch.stack(GB), torch.stack(WB), torch.stack(bB), Cw=math.inf, Cb=cb, fix_W=True)
            ZB = S.logits(Xt, rb["W"], rb["b"])
            ua = S.uar_batch(u.ytest, ZB)
            store[f"p3r_{s}_K{K}_pred_b"] = pred8(ZB)                         # 並びは (予測の版, ρ, a)
            for it, v, gp, fn, ga in zip(itB, ua, rb["grad_p"], rb["finite_p"], np.repeat(ra["grad_p"], len(A_GRID))):
                rows.append(dict(stage=3, student=s, K=K, method="rolesep", uar=v, grad=float(max(gp, ga)), finite=bool(fn),
                                 S=float(4 * K * it["a"]), **it))
            store[f"p3r_{s}_K{K}_bb"] = f32(rb["b"])
            rows.append(dict(stage=3, method="_conv_rolesep", student=s, K=K, grad_max=max(ra["grad_max"], rb["grad_max"]),
                             n_stalled=ra["stalled"] + rb["stalled"], finite=bool(ra["finite_p"].all() and rb["finite_p"].all())))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ 進め方

def units_for(T: str, mode: str, folds, seeds):
    srcs = DD.others(T)
    if mode == "pseudo":
        for c in srcs:
            for f in folds:
                for r in seeds:
                    yield c, [s for s in srcs if s != c], f, r
    else:
        for f in folds:
            for r in seeds:
                yield T, srcs, f, r


RUN = {"commit": "", "dirty": False, "frozen_sha256": "", "gpu": "", "gate_sha256": ""}   # main が埋める


def committed_clean(path: Path) -> bool:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=ROOT).returncode == 0
    return tracked and clean


EVAL_REQUIRED = [                                                   # 評価の回の前にコミットしてあるべきファイル（台本の側に固定．m51）
    "docs/設計/実験1_設計.md", "docs/設計/実験2_設計.md", "docs/設計/実験3_設計.md", "docs/計画/計画_2026-11-04.tex",
    "frozen/dai2_frozen.json", "frozen/dai3_frozen.json", "frozen/subsets.csv",
    "exp/dai1.py", "exp/dai_solver.py", "exp/dai_data.py", "exp/dai1_select.py", "exp/dai1_stats.py",
    "exp/dai_subsets.py", "exp/dai2.py", "exp/dai2_grad.py", "exp/dai2_select.py", "exp/dai3.py",
    "docs/設計/前半/proposal_v8.md", "exp/compare_dai.py", "exp/compare_dai2.py",
]


APPROVAL_ROLES = ("review_ml", "review_affect", "stats", "leak", "compare", "define", "code")


def latest_report(folder: Path, role: str) -> str:
    """`<役>.md`（第 1 版），`<役>_v2.md`，`<役>_v3.md` … のうち最も新しい版の名前。"""
    best, name = 0, ""
    for f in folder.glob(f"{role}*.md"):
        m = re.fullmatch(re.escape(role) + r"(?:_v(\d+)([a-z]?))?\.md", f.name)
        if m:
            v = (int(m.group(1) or 1), m.group(2) or "")                # _v7b は _v7 より新しい（m92）
            if best == 0 or v > best:
                best, name = v, f.name
    return name


def check_eval_gate(frozen_rel: str) -> str:
    """評価の回を流す条件（ML の査読者の M15，漏れの検査 L8）：`frozen/eval_gate.json` がコミット済みで，
    そこに並べたファイル（実験 1 の固定のファイル，実験 2・3 の設計，種の数と検出できる差の表，査読の OK の報告など）が
    すべてコミット済みで sha256 が一致すること．戻り：eval_gate.json の sha256。"""
    gp = ROOT / "frozen" / "eval_gate.json"
    if not gp.exists() or not committed_clean(gp):
        sys.exit("frozen/eval_gate.json が無いかコミットされていない（実験 2・3 の固定の前に評価の回は流さない）")
    gate = json.load(open(gp))
    req = gate["required"]
    missing = [r for r in EVAL_REQUIRED + [frozen_rel] if r not in req]
    if missing:
        sys.exit(f"eval_gate.json に必須のファイルが無い：{missing}（必須の一覧は台本の EVAL_REQUIRED．m51，a13，比較の検査 C20）")
    for rel, h in req.items():
        fp = ROOT / rel
        if not fp.exists() or sha(fp) != h or not committed_clean(fp):
            sys.exit(f"eval_gate.json の {rel} が無いか，sha256 が違うか，コミットされていない")
    # 実験 1〜3 のそれぞれで，査読者 2 人と関門 5 体の「最新の版の」報告が OK・通すであること（CLAUDE.md の規則．m64）
    folders = {"DAI-1": ("checks/DAI-1/",), "DAI-2": ("checks/DAI-2_3/", "checks/DAI-2/"),
               "DAI-3": ("checks/DAI-2_3/", "checks/DAI-3/")}             # 実験ごとの報告の置き場を台本に固定（比較の検査の 4 回目）
    for exp_name, files in gate["approvals"].items():
        for role in APPROVAL_ROLES:
            if not files[role].startswith(folders[exp_name]):
                sys.exit(f"{exp_name} の {role} の報告 {files[role]} が実験の置き場 {folders[exp_name]} に無い")
            fp = ROOT / files[role]
            if fp.as_posix() not in {(ROOT / r).as_posix() for r in req}:
                sys.exit(f"{exp_name} の {role} の報告が required に無い")
            latest = latest_report(fp.parent, role)
            if latest != fp.name:
                sys.exit(f"{exp_name} の {role} は新しい版 {latest} がある（{fp.name} を並べている）")
            lines = open(fp, encoding="utf-8").read().splitlines()
            head = "\n".join(lines[:12])
            word = "OK" if role.startswith("review_") else "通す"
            # 共有の報告（DAI-2_3）は実験ごとの行「判定（DAI-2）：通す」で確かめる．実験の名前の無い「判定：通す」は報告の全部に当たる．
            # 関門の書式「## 判定」の次の行が「通す」で始まる形も認める（m92）
            shared = "DAI-2_3" in files[role]
            if shared:                                                  # 共有の報告は実験ごとの行「判定（DAI-2）：…」だけを読む（m102）
                ok = rf"^判定（{re.escape(exp_name)}）[：:]\s*\**{word}(?:\**)?(?:$|[．。（])"
                sec = None
            else:
                ok = rf"^判定(?:（{re.escape(exp_name)}）)?[：:]\s*\**{word}(?:\**)?(?:$|[．。（])"
                sec = re.search(rf"^## 判定\s*\n+\s*\**{word}(?:$|[．。（])", head, re.M)   # 「OK でない」を通さない（M8）
            if not re.search(ok, head, re.M) and not sec:
                sys.exit(f"{exp_name} の {role} の報告（{files[role]}）が「判定：OK」（関門は「判定：通す」）でない")
    if set(gate["approvals"]) != {"DAI-1", "DAI-2", "DAI-3"}:
        sys.exit("eval_gate.json の approvals は DAI-1・DAI-2・DAI-3 の 3 つが要る")
    return sha(gp)


def run_pass(out, tag, T, c, srcs, f, r, which, tether=None, wsel=None, fsel=None) -> pd.DataFrame:
    """1 単位の 1 つの段を回して保存する．保存があれば読むだけ（途中から再開できる）．
    再開のときは，保存したコミット（と評価の回では固定のファイルの sha256）が今の実行と一致するかを確かめる（比較の検査 C6，漏れの検査 L8）。"""
    cname = "ALL" if tag == "eval" else c
    base = out / tag / "units" / f"{T}_{cname}_f{f}_r{r}"
    path, rpath = Path(f"{base}_p{which}.npz"), Path(f"{base}_p{which}.csv")
    tag_t = None if tether is None else np.array([f"{k[0]}|{k[1]}|{v[0]}|{v[1]}" for k, v in sorted(tether.items())])
    tag_w = None if wsel is None else np.array([f"{K}|{w}" for K, w in sorted(wsel.items())])
    if path.exists() and rpath.exists():
        old = pd.read_csv(rpath, low_memory=False)
        z = np.load(path)
        bad = str(old["commit"].iloc[0]) != RUN["commit"] or (tag == "eval" and str(old["frozen_sha256"].iloc[0]) != RUN["frozen_sha256"])
        bad |= tag_t is not None and ("tether" not in z or list(z["tether"]) != list(tag_t))      # I12
        bad |= tag_w is not None and ("wsel" not in z or list(z["wsel"]) != list(tag_w))
        if "finite" in old and (old["finite"].astype(str) == "False").any():
            bad = True                                                 # 再開の経路でも非有限の単位を通さない（N3）
        if bad:
            sys.exit(f"保存済みの単位 {rpath.name} のコミット・固定のファイル・引き戻し・指数・有限のどれかが今の実行と違う（消してから流し直す）")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    for prev in range(1, which):                                         # 前の段の保存のコミットを照合（N2）
        z = np.load(f"{base}_p{prev}.npz")
        if str(z["commit"]) != RUN["commit"]:
            sys.exit(f"{base.name} の段 {prev} の保存のコミット {z['commit']} が今の実行と違う")
        if prev == 2 and tag_t is not None and ("tether" not in z or list(z["tether"]) != list(tag_t)):
            sys.exit(f"{base.name} の段 2 の保存の引き戻しが今の引き戻しと違う（gate-code の 3 回目 M3）")
    u = Unit(out, T, c, srcs, f, r)
    store, rows = {}, []
    t0 = time.time()
    if which == 1:
        pass1(u, store, rows)
        store.update(sample_pool=u.cp.df["sample_id"].to_numpy()[u.pool].astype(str),
                     sample_test=u.cp.df["sample_id"].to_numpy()[u.test].astype(str),
                     **{f"sample_K{K}": u.cp.df["sample_id"].to_numpy()[u.pool[u.kpos[K]]].astype(str) for K in DD.KS},
                     **{f"stat_{k}_mu": v[0] for k, v in u.stats.items()}, **{f"stat_{k}_sd": v[1] for k, v in u.stats.items()})
    elif which == 2:
        p1 = np.load(f"{base}_p1.npz")
        pass2(u, p1, store, rows, tether)
    else:
        p1, p2 = np.load(f"{base}_p1.npz"), np.load(f"{base}_p2.npz")
        pass3(u, store, rows, tether, build_tk(u, p1, p2, tether), wsel, fsel)
    if tag_t is not None:                                                # 渡された引き戻しと指数は段に関わらず残す（再開の照合．m54）
        store["tether"] = tag_t
    if tag_w is not None:
        store["wsel"] = tag_w
    store.update(commit=np.array(RUN["commit"]), dirty=np.array(RUN["dirty"]), frozen_sha256=np.array(RUN["frozen_sha256"]),
                 gate_sha256=np.array(RUN["gate_sha256"]))
    if any(("finite" in r and r["finite"] is False) for r in rows):              # 保存の前に確かめる（N3）
        sys.exit(f"{rpath.name}: 勾配か解に NaN・Inf の問がある（計算の失敗として止める．保存しない．I11，N3）")
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in store.items()})
    df = pd.DataFrame(rows).assign(T=T, c=cname, fold=f, seed=r, sec=time.time() - t0, commit=RUN["commit"],
                                   dirty=RUN["dirty"], frozen_sha256=RUN["frozen_sha256"], gpu=RUN["gpu"],
                                   gate_sha256=RUN["gate_sha256"])
    df.to_csv(rpath, index=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{tag} {base.name} 段 {which}: {time.time()-t0:.0f} 秒", flush=True)
    return df


def frozen_points(fz: dict) -> dict:
    """固定のファイルから，(生徒, K) ごとの提案の主の版の点と族 3 の相手の点（共訓練なら None）を作る（引き戻しの感度に使う）。"""
    out = {}
    for s in DD.STUDENTS:
        for K in KS_ALL:
            v = fz["main"][str(K)]
            j = fz["select"][f"{s}|{K}|joint|{v}"]
            part = fz["partner"][f"{s}|{K}"]
            pm, pv = part["method"], part["version"]
            pp = None
            if pm not in ("cotrain", "coreg"):                              # 共訓練と共正則化は引き戻しの曲線を解かない
                pp = {k: x for k, x in fz["select"][f"{s}|{K}|{pm}|{pv}"].items() if k != "pseudo_uar"} | dict(method=pm, teacher=pv)
            out[(s, K)] = {"prop": dict(teacher=v, rho=j["rho"], a=j["a"]), "partner": pp}
    return out


def select_tether(rows: pd.DataFrame) -> dict:
    """(種類, K) ごとに，比較の基準の仮のターゲットの UAR の平均が最大の (C′, C_b)．同点は小さい C′・小さい C_b。"""
    out = {}
    d = rows[(rows.stage == 1) & (rows.method == "none")]
    for (kind, K), g in d.groupby(["kind", "K"]):
        m = g.groupby(["Cw", "Cb"])["uar"].mean().reset_index().sort_values(["uar", "Cw", "Cb"], ascending=[False, True, True])
        out[(kind, int(K))] = (float(m.iloc[0].Cw), float(m.iloc[0].Cb))
    return out


def select_w(rows: pd.DataFrame) -> dict:
    """K ごとに，積のモデルの候補のラベルでの UAR の平均が最大の指数．同点は (1,1,1) に近い方。"""
    d = rows[(rows.stage == 2) & (rows.method == "product")]
    out = {}
    for K, g in d.groupby("K"):
        m = g.groupby("teacher")["uar_pool"].mean()
        best = m.max()
        cands = [eval(w) for w in m.index[m >= best - 1e-9]]
        # 同点は (1,1,1) からの差の絶対値の和が最小，それも同じなら最大の指数が 1 に近いもの，それも同じなら辞書順（m34）
        out[int(K)] = min(cands, key=lambda w: (sum(abs(x - 1) for x in w), abs(max(w) - 1), w))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("source", "pseudo", "select", "eval", "selftest"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai1"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(DD.N_FOLDS)))
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="既定は pseudo で 0 1，eval で固定のファイルの n_seeds_eval")
    ap.add_argument("--pass", dest="passes", default="all", choices=("1", "2", "3", "23", "all"),
                    help="pseudo の段を分けて回す：1（--folds の単位），2（全単位の段 1 から引き戻しを選び，全単位の段 2 と指数を選んで tether_w.json），"
                         "3（tether_w.json を読み，--folds の単位の段 3）．2 は全部の分割と種の段 1 がそろっているときだけ動く")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--encoder", type=int, default=1, choices=(1, 2), help="1：WavLM・ResNet-50・RoBERTa，2：HuBERT・CLIP・DeBERTa")
    args = ap.parse_args()
    ENC["enc"] = DD.ENC1 if args.encoder == 1 else DD.ENC2
    if args.encoder == 2 and args.out_dir == str(ROOT / "results" / "dai1"):
        args.out_dir = str(ROOT / "results" / "dai1_enc2")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_commit()
    if dirty and args.stage in ("select", "eval"):
        sys.exit("選ぶ段と評価の段は未コミットの変更があると流さない（--allow-dirty も受け付けない．m51）")
    if dirty and not args.allow_dirty and args.stage not in ("selftest",):
        sys.exit("exp・measure・features に未コミットの変更がある（試しは --allow-dirty）")
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                script_sha256=sha(Path(__file__)), solver_sha256=sha(HERE / "dai_solver.py"), data_sha256=sha(HERE / "dai_data.py"),
                time=time.strftime("%Y-%m-%d %H:%M:%S"),
                numpy=np.__version__, pandas=pd.__version__,
                features_sha256={p.name: sha(p) for p in sorted((ROOT / "data" / "cache").glob("*.npz"))
                                 if p.stem.split("_")[-1] in DD.CORPORA and any(p.stem.startswith(k + "_") for k in ENC["enc"].values())},
                encoder=ENC["enc"],
                afew_transcript_sha256=sha(ROOT / "data" / "metadata" / "afew_transcript_whisper.csv"),
                metadata_sha256={p.name: sha(p) for p in sorted((ROOT / "data" / "metadata").glob("*.csv"))},   # m55
                splits_sha256={p.name: sha(p) for p in sorted((ROOT / "splits").glob("*.csv"))})
    RUN["commit"], RUN["dirty"] = commit, dirty
    RUN["gpu"] = prov["gpu"]
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")

    if args.stage == "selftest":
        S.selftest(); return
    if args.stage == "source":
        stage_source(out, args.targets); return

    tag = "pseudo" if args.stage == "pseudo" else ("eval" if args.stage == "eval" else "select")
    seeds = args.seeds if args.seeds is not None else ([0, 1] if args.stage == "pseudo" else list(range(10)))
    if args.stage == "pseudo":
        for T in args.targets:
            us = list(units_for(T, "pseudo", args.folds, seeds))
            if args.passes in ("1", "all"):
                for c, s, f, r in us:
                    run_pass(out, "pseudo", T, c, s, f, r, 1)
            if args.passes in ("2", "23", "all"):
                if args.passes == "2" and sorted(args.folds) != list(range(DD.N_FOLDS)):
                    sys.exit("--pass 2 は全部の分割で流す（引き戻しと指数は全単位で選ぶ）")
                r1 = pd.concat([pd.read_csv(out / "pseudo" / "units" / f"{T}_{c}_f{f}_r{r}_p1.csv") for c, _, f, r in us])
                if set(r1.commit.astype(str)) != {RUN["commit"]}:
                    sys.exit(f"{T}: 段 1 の行のコミットが今の実行と違う")
                teth = select_tether(r1)
                r2 = pd.concat([run_pass(out, "pseudo", T, c, s, f, r, 2, tether=teth) for c, s, f, r in us])
                w = select_w(r2)
                json.dump({"tether": {f"{k[0]}|{k[1]}": v for k, v in teth.items()}, "w": {str(k): v for k, v in w.items()},
                           "units": [f"{c}_f{f}_r{r}" for c, _, f, r in us], "commit": RUN["commit"]},
                          open(out / "pseudo" / f"{T}_tether_w.json", "w"), indent=1)
            if args.passes in ("3", "23", "all"):
                tw = json.load(open(out / "pseudo" / f"{T}_tether_w.json"))
                if tw["commit"] != RUN["commit"]:
                    sys.exit(f"{T}: tether_w.json のコミットが今の実行と違う")
                full = [f"{c}_f{f}_r{r}" for c, _, f, r in units_for(T, "pseudo", range(DD.N_FOLDS), seeds)]
                if sorted(tw["units"]) != sorted(full):
                    sys.exit(f"{T}: tether_w.json の単位が全部の分割と種でない")
                teth = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in tw["tether"].items()}
                w = {int(k): tuple(v) for k, v in tw["w"].items()}
                for c, s, f, r in us:
                    run_pass(out, "pseudo", T, c, s, f, r, 3, tether=teth, wsel=w)
        return
    if args.stage == "select":
        import dai1_select
        dai1_select.main(out, args.targets); return
    if args.stage == "eval":
        fz = out / "select" / "frozen.json"
        cp = ROOT / "frozen" / f"{out.name}_frozen.json"
        if not fz.exists() or not cp.exists() or sha(fz) != sha(cp):
            sys.exit("固定のファイルが無いか，frozen/dai1_frozen.json と一致しない")
        if not committed_clean(cp):
            sys.exit("固定のファイルの写しがコミットされていない（漏れの検査 L8）")
        gate_sha = check_eval_gate(f"frozen/{out.name}_frozen.json")
        RUN["gate_sha256"] = gate_sha
        RUN["frozen_sha256"] = sha(fz)
        with open(out / "provenance_log.jsonl", "a") as fh:
            fh.write(json.dumps({"eval_gate_sha256": gate_sha, "frozen_sha256": RUN["frozen_sha256"]}) + "\n")
        sel = json.load(open(fz))
        for T in args.targets:
            teth = {tuple([k.split("|")[0], int(k.split("|")[1])]): tuple(v) for k, v in sel[T]["tether"].items()}
            w = {int(k): tuple(v) for k, v in sel[T]["w"].items()}
            seeds_T = args.seeds if args.seeds is not None else list(range(int(sel[T]["n_seeds_eval"])))
            fsel = frozen_points(sel[T])
            for c, s, f, r in units_for(T, "eval", args.folds, seeds_T):
                for which in (1, 2, 3):
                    run_pass(out, "eval", T, c, s, f, r, which, tether=teth, wsel=w, fsel=fsel)

if __name__ == "__main__":
    main()
