"""実験 18 の下調べ（DAI-18）：6 コーパス（4 コーパス＋CREMA-D＋RAVDESS）で，実験 16・17 の主な版を K = 5・10・25 で流し直す．
設計：docs/設計/実験18_下調べ_設計.md．仮のターゲットの回だけを作る（評価の回は作らない）．各コーパスにあるモダリティだけを使う．

段：
  selftest   人工の数での検算
  source     CREMA-D と RAVDESS のソースの分類器（4 コーパスは実験 1 の保存を使う）
  base       単位ごとに，K 本だけの分類器を引き戻しの強さの格子の全部の点で解き，テストの判定と UAR を残す
  tether     評価のターゲット T ごとに引き戻しの強さを選ぶ（tether.json）
  units      単位ごとに，none・ret・ソースの分類器・近いソースの発話の答えを作り，版ごとの判定を残す
  students   格子の点を選んで固定し（frozen_sel.json），疑似ラベルで生徒を学び直す
  summary    c × K ごとの UAR，差と区間，数え

使い方：.venv/bin/python exp/dai18_probe.py --stage {selftest,source,base,tether,units,students,summary}
試走：--out-dir <一時の置き場> --allow-dirty --allow-partial --targets AFEW CREMA-D --folds 0 --seeds 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai16_probe as P16  # noqa: E402
import dai17_probe as P17  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT_PROD = ROOT / "results" / "dai18_probe"
DESIGN = ROOT / "docs" / "設計" / "実験18_下調べ_設計.md"
CORP6 = ("AFEW", "IEMOCAP", "MOSEI", "MELD", "CREMA-D", "RAVDESS")
NEW = ("CREMA-D", "RAVDESS")
MODS_OF = {**{c: "AVT" for c in DD.CORPORA}, "CREMA-D": "AV", "RAVDESS": "A"}
GROUP6 = {**DD.GROUP, "CREMA-D": "CREMA-D", "RAVDESS": "RAVDESS"}
CID6 = {**DD.CORPUS_ID, "CREMA-D": 4, "RAVDESS": 5}
KS = (5, 10, 25)
K_MAIN = 5
SEEDS = (0, 1)
NF = dai15.NF
TETH = list(dai1.TETHER)
SGRID = [(r, a) for r in dai1.RHO_GRID for a in dai1.A_GRID]
KMAX = 100
KNN_K = (20, 100)
RELS = ("none", "away")
LAMS = (0.0625, 0.25, 1.0)
LAM_OF = {f"l{j}": v for j, v in enumerate(LAMS)}
GRID16 = P16.GRID
NONE_AGREE = 0.995
SEED = 20261114 + 18
FLOOR = 1.0
SCRIPTS = ("dai18_probe.py", "dai17_probe.py", "dai16_probe.py", "dai15.py", "dai12_probe.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
RUN = {"commit": "", "dirty": True}


def fus(c: str) -> str | None:
    return MODS_OF[c] if len(MODS_OF[c]) >= 2 else None


def kinds_of(c: str) -> list:
    return list(MODS_OF[c]) + ([fus(c)] if fus(c) else [])


def cfgs_of(c: str) -> list:
    """近いソースの発話の答えの格子（引く範囲 × 本数 × 信頼度）．モダリティが 1 つなら 3 つを合わせる形は無い．"""
    scopes = ("mod", "joint") if len(MODS_OF[c]) >= 2 else ("mod",)
    return [(sc, k, rl) for sc in scopes for k in KNN_K for rl in RELS]


# ------------------------------------------------------------------ データ

class C6:
    """6 コーパスの 1 つ：そのコーパスのモダリティがそろう発話の表と，種類ごとの特徴（GPU）．"""
    cache: dict = {}

    def __init__(self, name: str):
        if name in DD.CORPORA:
            df, feats = DD.load_corpus(name, DD.ENC1)
        else:
            parts = {m: DD.D.load_group(GROUP6[name], DD.ENC1[m]) for m in MODS_OF[name]}
            ids = sorted(set.intersection(*[set(d["sample_id"]) for d, _ in parts.values()]))
            feats, base = {}, None
            for m, (d, X) in parts.items():
                pos = pd.Series(np.arange(len(d)), index=d["sample_id"].to_numpy())
                ix = pos.loc[ids].to_numpy()
                feats[m] = X[ix].astype(np.float32)
                if base is None:
                    base = d.iloc[ix].reset_index(drop=True)
                else:
                    assert (d.iloc[ix]["label_4"].to_numpy() == base["label_4"].to_numpy()).all()
            df = base[["sample_id", "speaker_id", "speaker_group", "label_4"]].copy()
            df["cluster"] = df["speaker_group"].astype(str).to_numpy()
            df["dialogue"] = ""
            df["fold"] = DD.speaker_folds(df["cluster"], k=DD.N_FOLDS, seed=0).to_numpy()
            assert (df["fold"] >= 0).all()
        self.name, self.df = name, df
        self.y = df["label_4"].to_numpy().astype(int)
        self.X = {k: S.t(DD.kind_matrix(feats, k)) for k in kinds_of(name)}

    @classmethod
    def get(cls, name):
        if name not in cls.cache:
            cls.cache[name] = C6(name)
        return cls.cache[name]


def g_eff_of(c: str) -> float:
    n = C6.get(c).df["cluster"].value_counts().to_numpy()
    return float(n.sum() ** 2 / (n ** 2).sum())


def k_orders6(y: np.ndarray, pool_idx: np.ndarray, corpus: str, fold: int, seed: int) -> dict:
    """dai_data.k_orders と同じ抜き方（コーパスの番号だけ 6 コーパスに広げた）．4 コーパスでは実験 1 と同じ K 本になる．"""
    rng = DD.common.rng_for("dai-K", CID6[corpus], fold, seed)
    perm = {k: rng.permutation(pool_idx[y[pool_idx] == k]) for k in range(4)}
    out = {}
    for K in DD.KS:
        assert all(len(perm[k]) >= K for k in range(4)), (corpus, fold, K)
        out[K] = np.sort(np.concatenate([perm[k][:K] for k in range(4)]))
    return out


# ------------------------------------------------------------------ ソースの分類器

def src_path(out: Path, s: str, kind: str) -> Path:
    return (OUT1 / "source" / f"{s}_{kind}.npz") if s in DD.CORPORA else (out / "source" / f"{s}_{kind}.npz")


OUTDIR = {"out": OUT_PROD}


def load_src(s: str, kind: str):
    z = np.load(src_path(OUTDIR["out"], s, kind))
    return S.t(z["W"]), S.t(z["b"]), float(z["C"])


def stage_source(out: Path) -> None:
    (out / "source").mkdir(parents=True, exist_ok=True)
    pd.concat([C6.get(c).df[["sample_id", "speaker_id", "cluster", "fold", "label_4"]].assign(corpus=c) for c in NEW]).to_csv(out / "splits_new.csv", index=False)
    rows = []
    for c in NEW:
        cp = C6.get(c)
        for kind in kinds_of(c):
            t0 = time.time()
            X = cp.X[kind]
            cv = np.zeros((DD.N_FOLDS, len(dai1.C_GRID)))
            for f in range(DD.N_FOLDS):
                tr = np.flatnonzero(DD.pool_mask(cp.df, f)); te = np.flatnonzero(cp.df["fold"].to_numpy() == f)
                mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
                Xs = (X - mu) / sd
                G = S.balanced_onehot(cp.y[tr], np.arange(len(tr)), len(tr))[None].expand(len(dai1.C_GRID), -1, -1)
                r = S.fit(Xs[tr], G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.array(dai1.C_GRID), Cb=math.inf)
                cv[f] = S.uar_batch(cp.y[te], S.logits(Xs[te], r["W"], r["b"]))
            mean = cv.mean(0)
            best = int(np.flatnonzero(mean == mean.max())[0])
            mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
            Xs = (X - mu) / sd
            G = S.balanced_onehot(cp.y, np.arange(len(cp.y)), len(cp.y))[None]
            ra = S.fit(Xs, G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.array([dai1.C_GRID[best]]), Cb=math.inf)
            if float(ra["grad_max"]) > P16.GRAD_TOL:
                raise SystemExit(f"source：{c} {kind} の勾配の最大 {float(ra['grad_max']):.1e}")
            np.savez_compressed(out / "source" / f"{c}_{kind}.npz", W=ra["W"][0].cpu().numpy(), b=ra["b"][0].cpu().numpy(), C=dai1.C_GRID[best], cv=cv,
                                C_grid=np.array(dai1.C_GRID), mu=mu.cpu().numpy(), sd=sd.cpu().numpy(), sample_id=cp.df["sample_id"].to_numpy().astype(str))
            rows.append(dict(corpus=c, kind=kind, C=dai1.C_GRID[best], cv_uar=float(mean[best]), commit=RUN["commit"], sec=time.time() - t0))
            print(f"source {c} {kind}：C={dai1.C_GRID[best]}（{time.time() - t0:.0f} 秒）", flush=True)
    pd.DataFrame(rows).to_csv(out / "source" / "summary.csv", index=False)


CVP: dict = {}
SRCP: dict = {}


def cv_probs(c: str, m: str) -> np.ndarray:
    """コーパス c の中の交差検証の確率（その発話を学習に使っていない分類器．罰則の強さはソースの分類器の C）．"""
    if (c, m) not in CVP:
        cp = C6.get(c)
        X, y = cp.X[m], cp.y
        C = load_src(c, m)[2]
        P = np.full((len(y), 4), np.nan)
        for f in range(DD.N_FOLDS):
            tr = np.flatnonzero(DD.pool_mask(cp.df, f)); te = np.flatnonzero(cp.df["fold"].to_numpy() == f)
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
            Xs = (X - mu) / sd
            G = S.balanced_onehot(y[tr], np.arange(len(tr)), len(tr))[None]
            r = S.fit(Xs[tr], G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.array([C]), Cb=math.inf)
            if float(r["grad_max"]) > P16.GRAD_TOL:
                raise SystemExit(f"cv_probs：{c} {m} 分割 {f}")
            P[te] = torch.softmax(S.logits(Xs[te], r["W"][0], r["b"][0]), -1).cpu().numpy()
        assert np.isfinite(P).all()
        CVP[(c, m)] = P
    return CVP[(c, m)]


def src_probs_on(s: str, t: str, m: str) -> np.ndarray:
    """ソース s の分類器を，コーパス t の全発話（t の統計で標準化）に当てた確率．"""
    if (s, t, m) not in SRCP:
        X = C6.get(t).X[m]
        Xs = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
        W, b, _ = load_src(s, m)
        SRCP[(s, t, m)] = dai1.probs(Xs, W, b).cpu().numpy().astype(np.float64)
    return SRCP[(s, t, m)]


# ------------------------------------------------------------------ 単位

class Unit6:
    def __init__(self, T: str, c: str, f: int, r: int):
        self.T, self.c, self.f, self.r = T, c, f, r
        cp = C6.get(c)
        self.cp = cp
        self.pool = np.flatnonzero(DD.pool_mask(cp.df, f))
        self.test = np.flatnonzero(cp.df["fold"].to_numpy() == f)
        self.ypool, self.ytest = cp.y[self.pool], cp.y[self.test]
        korder = k_orders6(cp.y, self.pool, c, f, r)
        self.kpos = {K: np.searchsorted(self.pool, korder[K]) for K in DD.KS}
        assert all((self.pool[self.kpos[K]] == korder[K]).all() for K in DD.KS)
        self.mods = MODS_OF[c]
        self.fus = fus(c)
        self.kinds = kinds_of(c)
        self.srcs = {k: [s for s in CORP6 if s not in (T, c) and set(k) <= set(MODS_OF[s])] for k in self.kinds}
        assert all(len(v) >= 1 for v in self.srcs.values()), (T, c, self.srcs)
        self.Xp, self.Xt, self.start, self.src = {}, {}, {}, {}
        for kind in self.kinds:
            X = cp.X[kind]
            mu, sd = X[self.pool].mean(0), X[self.pool].std(0).clamp(min=1e-6)
            self.Xp[kind], self.Xt[kind] = (X[self.pool] - mu) / sd, (X[self.test] - mu) / sd
            Ws, bs = zip(*[load_src(s, kind)[:2] for s in self.srcs[kind]])
            self.start[kind] = (torch.stack(Ws).mean(0), torch.stack(bs).mean(0))
            self.src[kind] = list(zip(self.srcs[kind], Ws, bs))
        self.n = len(self.pool)
        df = cp.df
        self.cl, self.cl_pool = df["cluster"].to_numpy().astype(str)[self.test], df["cluster"].to_numpy().astype(str)[self.pool]
        self.sid = df["sample_id"].to_numpy().astype(str)[self.test]

    def gK(self, K, idx=None):
        return S.balanced_onehot(self.ypool, self.kpos[K] if idx is None else idx, self.n)

    def name(self):
        return f"{self.T}_{self.c}_f{self.f}_r{self.r}"


def all_units(targets=None, folds=None, seeds=None) -> list:
    out = []
    for T in (targets or CORP6):
        for c in CORP6:
            if c == T:
                continue
            for f in (folds if folds is not None else range(DD.N_FOLDS)):
                for r in (seeds if seeds is not None else SEEDS):
                    out.append((T, c, f, r))
    return out


# ------------------------------------------------------------------ base：K 本だけの分類器の格子

def run_base(u: Unit6) -> tuple[list, dict]:
    rows, arr = [], {}
    arr["y_test"], arr["cl_test"] = u.ytest.astype(np.int8), u.cl
    for kind in u.kinds:
        W0, b0 = u.start[kind]
        items = [(K, cw, cb) for K in KS for cw, cb in TETH]
        G = torch.stack([u.gK(K) for K, _, _ in items])
        res = S.fit(u.Xp[kind], G, W0, b0, Cw=[x[1] for x in items], Cb=[x[2] for x in items])
        if float(res["grad_max"]) > P16.GRAD_TOL or not res["finite_p"].all():
            raise SystemExit(f"base：{u.name()} {kind} が解けていない（{float(res['grad_max']):.1e}）")
        Z = S.logits(u.Xt[kind], res["W"], res["b"])
        ua = S.uar_batch(u.ytest, Z)
        arr[f"pred_{kind}"] = Z.argmax(-1).to(torch.int8).cpu().numpy()
        for gi, ((K, cw, cb), v) in enumerate(zip(items, ua)):
            rows.append(dict(kind=kind, K=K, Cw=cw, Cb=cb, grid=gi % len(TETH), uar=float(v)))
        Z0 = S.logits(u.Xt[kind], W0, b0)
        rows.append(dict(kind=kind, K=0, Cw=np.nan, Cb=np.nan, grid=-1, uar=100 * S.uar(u.ytest, Z0.argmax(1).cpu().numpy())))
        rows.append(dict(kind=kind, K=-2, grid=-2, grad=float(res["grad_max"]), stalled=int(res["stalled"])))
    return rows, arr


def stage_tether(out: Path, allow_partial: bool) -> None:
    """評価のターゲット T ごとに，T の仮のターゲットの単位の none の UAR の平均が最大の引き戻しの強さを種類 × K ごとに選ぶ（同点は小さい C′・小さい C_b）．"""
    n_exp = len(all_units())
    files, d = load_csvs(out, "base", n_exp, allow_partial)             # base の出力が 300 個そろい，コミットが 1 つであることを確かめる（gate-code 指摘 2）
    if not len(d):
        raise SystemExit("base の出力が無い")
    d = d[d.K.isin(KS)]
    out_j = {}
    for T in CORP6:
        g = d[d["T"] == T].groupby(["kind", "K", "Cw", "Cb"]).uar.mean().reset_index()
        sel = {}
        for (kind, K), q in g.groupby(["kind", "K"]):
            q = q.sort_values(["uar", "Cw", "Cb"], ascending=[False, True, True])
            sel[f"{kind}|{int(K)}"] = [float(q.iloc[0].Cw), float(q.iloc[0].Cb)]
        for K in KS:                                                    # 融合 AV（CREMA-D だけが持つ）の強さは，同じ T の AVT の強さを使う（CREMA-D のテストで選ばない）
            if f"AVT|{K}" in sel:
                sel[f"AV|{K}|own"] = sel.get(f"AV|{K}")
                sel[f"AV|{K}"] = sel[f"AVT|{K}"]
        out_j[T] = sel
    json.dump(dict(tether=out_j, n_rows=len(d), n_units=len(files), n_units_expected=n_exp, base_commits=sorted(d.commit.astype(str).unique().tolist()),
                   time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=RUN["commit"]), open(out / "tether.json", "w"), indent=1)
    print(f"tether：{sum(len(v) for v in out_j.values())} 組", flush=True)


# ------------------------------------------------------------------ units

def fit_none(u: Unit6, kind: str, K: int, teth: dict, idx=None):
    cw, cb = teth[f"{kind}|{K}"]
    W0, b0 = u.start[kind]
    sets = [u.kpos[K]] if idx is None else idx
    G = torch.stack([u.gK(K, s) for s in sets])
    r = S.fit(u.Xp[kind], G, W0, b0, Cw=cw, Cb=cb)
    if float(r["grad_max"]) > P16.GRAD_TOL or not r["finite_p"].all():
        raise SystemExit(f"none：{u.name()} {kind} K={K}")
    return r


def ret_models(u: Unit6, K: int, teth: dict, sets: list) -> dict:
    """K 本の部分集合ごとに，微調整した分類器の候補の上の確率 {m: (部分集合, n, 4)} とテストの上の確率 {m: (部分集合, nt, 4)}．
    融合があれば融合の K 本だけの分類器の予測を柔らかいラベルにして各モダリティを学び直す（実験 1 の段 2）．無ければ K 本だけの分類器．"""
    outp, outt = {}, {}
    if u.fus:
        rF = fit_none(u, u.fus, K, teth, sets)
        F = torch.softmax(S.logits(u.Xp[u.fus], rF["W"], rF["b"]), -1)
        G0 = torch.stack([u.gK(K, s) for s in sets])
        for m in u.mods:
            G = G0 + torch.stack([S.soft_rows(F[j], float(u.n)) for j in range(len(sets))])
            cw, cb = teth[f"{m}|{K}"]
            W0, b0 = u.start[m]
            rm = S.fit(u.Xp[m], G, W0, b0, Cw=cw, Cb=cb)
            if float(rm["grad_max"]) > P16.GRAD_TOL or not rm["finite_p"].all():
                raise SystemExit(f"ret：{u.name()} {m} K={K}")
            outp[m] = torch.softmax(S.logits(u.Xp[m], rm["W"], rm["b"]), -1).cpu().numpy().astype(np.float64)
            outt[m] = torch.softmax(S.logits(u.Xt[m], rm["W"], rm["b"]), -1).cpu().numpy().astype(np.float64)
    else:
        m = u.mods
        r = fit_none(u, m, K, teth, sets)
        outp[m] = torch.softmax(S.logits(u.Xp[m], r["W"], r["b"]), -1).cpu().numpy().astype(np.float64)
        outt[m] = torch.softmax(S.logits(u.Xt[m], r["W"], r["b"]), -1).cpu().numpy().astype(np.float64)
    return outp, outt


SRCSTAT: dict = {}


def home_rel6(c: str, m: str) -> np.ndarray:
    """実験 16 の home_rel と同じ定義：コーパスの中の交差検証の分割ごとに d′ を出して平均する．"""
    P, y, fo = cv_probs(c, m), C6.get(c).y, C6.get(c).df["fold"].to_numpy()
    return np.mean([P16.dprime(P[fo == f], y[fo == f])[0] for f in range(DD.N_FOLDS)], 0)


def ext_rel(srcs_m: dict, mods: str) -> dict:
    """ソースで測った信頼度（実験 16 の ext）：モダリティ m ごとに，ソースの中の交差検証の d′（home）の平均 × 残り方（away ÷ home をソースの対で足したもの）．"""
    key = tuple((m, tuple(srcs_m[m])) for m in mods)
    if key not in SRCSTAT:
        out = {}
        for m in mods:
            ss = srcs_m[m]
            home = {s: home_rel6(s, m) for s in ss}
            pairs = [(s, t) for s in ss for t in ss if s != t]
            away = {(s, t): P16.dprime(src_probs_on(s, t, m), C6.get(t).y)[0] for s, t in pairs}
            ret = float(np.clip(sum(away[p].sum() for p in pairs) / max(sum(home[s].sum() for s, t in pairs), 1e-12), 0.0, 1.5)) if pairs else 1.0
            out[m] = (np.mean([home[s] for s in ss], 0) * ret, ret, np.mean([home[s] for s in ss], 0),
                      np.mean([away[p_] for p_ in pairs], 0) if pairs else np.full(4, np.nan))
        SRCSTAT[key] = out
    return SRCSTAT[key]


SRCDB6: dict = {}


def source_db6(rows: list, m: str, clf: list) -> dict:
    """近い発話を引く表（モダリティ m）：行は rows のコーパスの全発話，点は clf のソースの分類器（行のコーパスの分は交差検証）．"""
    key = (tuple(rows), m, tuple(clf))
    if key not in SRCDB6:
        feat, logit, ys, corp, bal, ra_raw, ra = [], [], [], [], [], [], []
        for ci, s in enumerate(rows):
            cp = C6.get(s)
            X, y = cp.X[m], cp.y
            feat.append(torch.nn.functional.normalize(((X - X.mean(0)) / X.std(0).clamp(min=1e-6)).float(), dim=1))
            E = np.concatenate([P16.centered(cv_probs(s, m) if q == s else src_probs_on(q, s, m)) for q in clf], 1)
            logit.append(P17.unit_norm(P17.zs(E)[0]))
            cnt = np.bincount(y, minlength=4).astype(float)
            bal.append(len(y) / (4.0 * np.maximum(cnt[y], 1.0)))
            ii = np.arange(len(y))
            others = [q for q in clf if q != s]
            r_ = np.mean([src_probs_on(q, s, m)[ii, y] for q in others], 0) if others else np.ones(len(y))
            ra_raw.append(r_); ra.append(P17.within_label(r_, y))
            ys.append(y); corp.append(np.full(len(y), ci))
        SRCDB6[key] = dict(feat=torch.cat(feat), logit=torch.cat(logit), y=np.concatenate(ys), corp=np.concatenate(corp), bal=np.concatenate(bal),
                           r_none=np.ones(sum(map(len, ys))), r_away=np.concatenate(ra), r_away_raw=np.concatenate(ra_raw), sizes=[len(a) for a in ys])
    return SRCDB6[key]


def joint_db(u: Unit6) -> dict:
    """3 つ（2 つ）のモダリティを合わせて引く表：行は融合のソース（すべてのモダリティを持つソース）．信頼度はモダリティの平均をコーパス × ラベルの中で割る．"""
    rows = u.srcs[u.fus]
    dbs = [source_db6(rows, m, u.srcs[m]) for m in u.mods]
    d0 = dbs[0]
    off = np.concatenate([[0], np.cumsum(d0["sizes"])])
    raw = np.mean([d["r_away_raw"] for d in dbs], 0)
    nrm = np.concatenate([P17.within_label(raw[a:b], d0["y"][a:b]) for a, b in zip(off[:-1], off[1:])])
    return dict(d0, r_away=nrm, r_away_raw=raw, dbs=dbs)


def target_pos(u: Unit6, src_p: dict, src_t: dict, m: str):
    """ターゲットの候補とテストの位置：特徴と，そのモダリティのソースの分類器の点（候補の統計で標準化）．"""
    Ep = np.concatenate([P16.centered(src_p[(s, m)]) for s in u.srcs[m]], 1)
    Et = np.concatenate([P16.centered(src_t[(s, m)]) for s in u.srcs[m]], 1)
    Ep, mu, sd = P17.zs(Ep)
    return ((torch.nn.functional.normalize(u.Xp[m].float(), dim=1), P17.unit_norm(Ep)),
            (torch.nn.functional.normalize(u.Xt[m].float(), dim=1), P17.unit_norm(P17.zs(Et, mu, sd)[0])))


def mn_weights(form: str, lam: float, DK_M, DK_N, DO_M, DO_N, K: int, nm: int) -> np.ndarray:
    """実験 17 の mn_weights を，微調整した分類器の数 nm に広げたもの．近いソースの発話の答えの行に lam × (nm ÷ 答えの数) を掛ける．"""
    nE = DK_N.shape[0]
    DKx, DOx = np.concatenate([DK_M, DK_N]), np.concatenate([DO_M, DO_N])
    Wk = {"eq": np.ones((nm + nE, 4)), "F2K": np.repeat(P16.mix_rel(DKx, DKx, 4 * K, P17.N0_LEVEL, 0.0, "eq").mean(1, keepdims=True), 4, 1),
          "F6K": DKx, "F6orc": DOx}[form]
    ref = float(Wk[:nm].mean())
    W = (np.ones((nm + nE, 4)) if ref <= 0 else Wk / ref).copy()
    W[nm:] *= lam * nm / nE
    return W


def src_avg(src: dict, u: Unit6, which: int) -> np.ndarray:
    """モダリティごとにソースの分類器の点を平均して足したもの．which：0 は候補，1 はテスト（src は {(s, m): (候補, テスト)}）．"""
    return sum(np.mean([P16.centered(src[(s, m)][which]) for s in u.srcs[m]], 0) for m in u.mods)


def run_unit(u: Unit6, teth: dict) -> tuple[list, dict]:
    rows, arr = [], {}
    yt, yp = u.ytest, u.ypool
    nm = len(u.mods)
    arr["y_test"], arr["cl_test"], arr["sid_test"], arr["y_pool"], arr["cl_pool"] = yt.astype(np.int8), u.cl, u.sid, yp.astype(np.int8), u.cl_pool
    for kind in u.kinds:
        arr[f"srcs_{kind}"] = np.array(u.srcs[kind])                       # 単位ごとのソースの一覧
    src = {(s, m): (dai1.probs(u.Xp[m], W, b).cpu().numpy().astype(np.float64), dai1.probs(u.Xt[m], W, b).cpu().numpy().astype(np.float64))
           for m in u.mods for s, W, b in u.src[m]}
    for (s, m), (P, Q) in src.items():
        arr[f"p_pool_src_{s}_{m}"], arr[f"p_test_src_{s}_{m}"] = P.astype(np.float32), Q.astype(np.float32)
    src_p = {k: v[0] for k, v in src.items()}
    src_t = {k: v[1] for k, v in src.items()}
    # 近いソースの発話（K に依らない）
    knn_p, knn_t = {}, {}
    pos = {m: target_pos(u, src_p, src_t, m) for m in u.mods}
    lists = {}
    for m in u.mods:
        db = source_db6(u.srcs[m], m, u.srcs[m])
        ip = P17.topk_z([(pos[m][0][0], db["feat"]), (pos[m][0][1], db["logit"])])[0]
        it = P17.topk_z([(pos[m][1][0], db["feat"]), (pos[m][1][1], db["logit"])])[0]
        lists[m] = (db, ip, it)
        arr[f"nn_test_{m}"] = it.astype(np.int32)
    if nm >= 2:
        jd = joint_db(u)
        partsP = [(pos[m][0][j], jd["dbs"][i]["feat" if j == 0 else "logit"]) for i, m in enumerate(u.mods) for j in (0, 1)]
        partsT = [(pos[m][1][j], jd["dbs"][i]["feat" if j == 0 else "logit"]) for i, m in enumerate(u.mods) for j in (0, 1)]
        lists["J"] = (jd, P17.topk_z(partsP)[0], P17.topk_z(partsT)[0])
        arr["nn_test_J"] = lists["J"][2].astype(np.int32)
    CF = cfgs_of(u.c)
    for ci, (sc, k, rl) in enumerate(CF):
        slots = list(u.mods) if sc == "mod" else ["J"]
        knn_p[ci] = np.stack([P17.votes(lists[s_][1], lists[s_][0], k, rl) for s_ in slots])
        knn_t[ci] = np.stack([P17.votes(lists[s_][2], lists[s_][0], k, rl) for s_ in slots])
        arr[f"p_pool_knn_c{ci}"], arr[f"p_test_knn_c{ci}"] = knn_p[ci].astype(np.float32), knn_t[ci].astype(np.float32)
        for s_ in slots:                                                   # 記述：近い 100 本の感情ごとの当たる割合の平均（D3 の 1 つ目の量）
            if rl == "none" and k == KMAX:
                Y = lists[s_][0]["y"][lists[s_][2]]
                hit = Y == yt[:, None]
                rows.append(dict(K=-1, version="_nn", modality=s_, rank_bal=float(np.mean([hit[yt == e].mean() for e in range(4) if (yt == e).any()])),
                                 d1=float(np.mean([hit[yt == e][:, :5].mean() for e in range(4) if (yt == e).any()]) -
                                          np.mean([hit[yt == e][:, 20:].mean() for e in range(4) if (yt == e).any()]))))
    ext = ext_rel(u.srcs, u.mods) if nm >= 2 else None
    if ext:
        arr["D_M_ext"] = np.stack([ext[m][0] for m in u.mods])
        arr["D_M_home"], arr["D_M_away"] = np.stack([ext[m][2] for m in u.mods]), np.stack([ext[m][3] for m in u.mods])
        arr["ret_M"] = np.array([ext[m][1] for m in u.mods])
    for K in KS:
        kpos = u.kpos[K]
        yk = yp[kpos]
        nk = np.setdiff1d(np.arange(u.n), kpos)
        ynk = yp[nk]
        fold = dai15.kfold_of(yk)
        sets = [kpos] + [kpos[fold != j] for j in range(NF)]
        Rp, Rt = ret_models(u, K, teth, sets)
        retp = {m: Rp[m][0] for m in u.mods}
        rett = {m: Rt[m][0] for m in u.mods}
        held = {m: np.stack([Rp[m][1 + fold[a]][kpos[a]] for a in range(len(kpos))]) for m in u.mods}
        for m in u.mods:
            arr[f"p_pool_ret_{m}_K{K}"], arr[f"p_test_ret_{m}_K{K}"] = retp[m].astype(np.float32), rett[m].astype(np.float32)
        # none（記述）
        for kind in u.kinds:
            r = fit_none(u, kind, K, teth)
            Zt = S.logits(u.Xt[kind], r["W"][0], r["b"][0])
            arr[f"pred_raw_K{K}_none_{kind}"] = Zt.argmax(1).to(torch.int8).cpu().numpy()
            rows.append(dict(K=K, version=f"none_{kind}", uar_raw=100 * S.uar(yt, Zt.argmax(1).cpu().numpy())))
        LMp = np.stack([P16.centered(retp[m]) for m in u.mods])
        LMt = np.stack([P16.centered(rett[m]) for m in u.mods])
        LMh = np.stack([P16.centered(held[m]) for m in u.mods])
        DK_M = np.stack([P16.dprime(held[m], yk)[0] for m in u.mods])
        DO_M = np.stack([P16.dprime(retp[m][nk], ynk)[0] for m in u.mods])
        arr[f"D_M_K_K{K}"], arr[f"D_M_orc_K{K}"] = DK_M, DO_M
        target = np.exp(P16.norm_log(LMp.sum(0))).mean(0)
        specs = {"prod": (LMp.sum(0), LMt.sum(0))}
        SAp = src_avg(src, u, 0); SAt = src_avg(src, u, 1)
        for nmj, lam in LAM_OF.items():
            specs[f"MS_eq_{nmj}"] = (LMp.sum(0) + lam * SAp, LMt.sum(0) + lam * SAt)
        if nm >= 2:
            Dext = arr["D_M_ext"]
            Dm = {"orc": DO_M, "K": DK_M, "ext": Dext, **{f"mix{i}": P16.mix_rel(DK_M, Dext, 4 * K, n0, b_, pr) for i, (n0, b_, pr) in enumerate(GRID16)}}
            for r_, D in Dm.items():
                W = P16.to_w(D)
                specs[f"M_F6_{r_}"] = (np.einsum("ky,kny->ny", W, LMp), np.einsum("ky,kny->ny", W, LMt))
                if r_ == "orc" or r_.startswith("mix"):
                    Wn = P16.noint(W)
                    specs[f"M_F6n_{r_}"] = (np.einsum("ky,kny->ny", Wn, LMp), np.einsum("ky,kny->ny", Wn, LMt))
                if r_ == "orc":
                    w2 = W.mean(1)
                    specs["M_F2_orc"] = (np.tensordot(w2, LMp, axes=1), np.tensordot(w2, LMt, axes=1))
        for ci in range(len(CF)):
            LNp = np.stack([P16.centered(p_) for p_ in knn_p[ci]])
            LNt = np.stack([P16.centered(p_) for p_ in knn_t[ci]])
            nE = LNp.shape[0]
            DK_N = np.stack([P16.dprime(p_[kpos], yk)[0] for p_ in knn_p[ci]])
            DO_N = np.stack([P16.dprime(p_[nk], ynk)[0] for p_ in knn_p[ci]])
            arr[f"D_N_K_K{K}_c{ci}"], arr[f"D_N_orc_K{K}_c{ci}"] = DK_N, DO_N
            specs[f"N_eq_c{ci}"] = (LNp.sum(0) * nm / nE, LNt.sum(0) * nm / nE)
            Lp, Lt = np.concatenate([LMp, LNp]), np.concatenate([LMt, LNt])
            for form in ("eq", "F2K", "F6K", "F6orc"):
                for j, lam in enumerate(LAMS):
                    W = mn_weights(form, lam, DK_M, DK_N, DO_M, DO_N, K, nm)
                    specs[f"MN_{form}_c{ci}_l{j}"] = (np.einsum("ky,kny->ny", W, Lp), np.einsum("ky,kny->ny", W, Lt))
        for v, (sp, st) in specs.items():
            Sp, St = P16.norm_log(sp), P16.norm_log(st)
            if not (np.isfinite(Sp).all() and np.isfinite(St).all()):
                raise SystemExit(f"{u.name()}：まとめた点が有限でない（{v}）")
            dl, rs = P16.match_shift(Sp, target)
            pr, pa, pp = St.argmax(1), (St + dl).argmax(1), (Sp + dl)[nk].argmax(1)
            arr[f"pred_raw_K{K}_{v}"], arr[f"pred_adj_K{K}_{v}"], arr[f"pred_pool_K{K}_{v}"] = pr.astype(np.int8), pa.astype(np.int8), pp.astype(np.int8)
            rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, pr), uar_adj=100 * S.uar(yt, pa), uar_pool=100 * S.uar(ynk, pp), adj_resid=rs))
        arr[f"kpos_K{K}"] = kpos.astype(np.int32)
        rows.append(dict(K=K, version="_check", gpu_max_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0)))
    return rows, arr


# ------------------------------------------------------------------ 選び方

def seldef_of(c: str) -> dict:
    CF = cfgs_of(c)
    d = {"MS_eq_sel": [f"MS_eq_l{j}" for j in range(len(LAMS))], "N_eq_sel": [f"N_eq_c{i}" for i in range(len(CF))],
         **{f"MN_{f}_sel": [f"MN_{f}_c{i}_l{j}" for i in range(len(CF)) for j in range(len(LAMS))] for f in ("eq", "F2K", "F6K")}}
    if len(MODS_OF[c]) >= 2:
        d["M_F6_mixsel"] = [f"M_F6_mix{i}" for i in range(len(GRID16))]
    return d


def tied(name: str, v: str) -> str | None:
    if name == "MN_F6K_sel":
        return v.replace("MN_F6K_", "MN_F6orc_")
    if name == "MN_eq_sel":
        return f"MS_eq_{v.rsplit('_', 1)[1]}"
    if name == "M_F6_mixsel":
        return v.replace("M_F6_", "M_F6n_")
    return None


TIED_NAME = {"MN_F6K_sel": "MN_F6orc_sel", "MN_eq_sel": "MS_eq_tied", "M_F6_mixsel": "M_F6n_mixsel"}


def select_points(csvs: pd.DataFrame) -> tuple[dict, list]:
    """コーパス c を評価するときは，c を評価のターゲット T にした単位（c を含まない）のテストの UAR（adj）の平均が最大の点．同点は並びで先．"""
    d = csvs[csvs.version.astype(str).str.match(r"^(M_F6_mix|MN_|MS_eq_l|N_eq_c)")]
    piv = d.pivot_table(index=["T", "c", "fold", "seed", "K"], columns="version", values="uar_adj")
    choice, rows = {}, []
    for c in CORP6:
        for K in KS:
            for name, cands in seldef_of(c).items():
                if not all(v in piv.columns for v in cands):
                    continue
                q = piv.loc[(piv.index.get_level_values("T") == c) & (piv.index.get_level_values("K") == K), cands].dropna()   # 候補がすべてそろう単位だけ
                if name == "MS_eq_sel":                                     # 比べる MN_eq_sel と同じ単位で選ぶ（3 つを合わせて引く形が無い RAVDESS の単位を除く．probe_v2 指摘 1）
                    q = q[[len(MODS_OF[cc]) >= 2 for cc in q.index.get_level_values("c")]]
                if len(q) == 0:
                    continue
                vals = q.mean().to_numpy()
                i = int(np.argmax(vals))
                choice[(c, K, name)] = cands[i]
                lam = LAM_OF.get(cands[i].rsplit("_", 1)[1], np.nan)
                rows.append(dict(c=c, K=K, version=name, chosen=cands[i], n_tied=int(np.sum(vals >= vals.max() - 1e-9)), n_cands=len(cands), n_units=len(q),
                                 pseudo_targets=",".join(sorted(set(q.index.get_level_values("c")))), lam=lam, at_min_lam=bool(lam == LAMS[0]),
                                 mean_uar_adj=float(vals.max())))
                tv = tied(name, cands[i])
                if tv:
                    choice[(c, K, TIED_NAME[name])] = tv
                    rows.append(dict(c=c, K=K, version=TIED_NAME[name], chosen=tv, tied_to=name))
    return choice, rows


# ------------------------------------------------------------------ 生徒

TEACH = ("prod", "M_F6_mixsel", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel", "MS_eq_sel", "MS_eq_tied")


def teacher_pool(z, u: Unit6, K: int, version: str) -> np.ndarray:
    """単位の保存から，版の候補の上の確率（adj）を作り直す（単位の段と同じ式）．"""
    LM = np.stack([P16.centered(z[f"p_pool_ret_{m}_K{K}"].astype(np.float64)) for m in u.mods])
    nm = len(u.mods)
    target = np.exp(P16.norm_log(LM.sum(0))).mean(0)
    if version == "prod":
        return np.exp(P16.norm_log(LM.sum(0)))
    if version.startswith("MS_eq_"):
        SA = sum(np.mean([P16.centered(z[f"p_pool_src_{s}_{m}"].astype(np.float64)) for s in u.srcs[m]], 0) for m in u.mods)
        sc = LM.sum(0) + LAM_OF[version.rsplit("_", 1)[1]] * SA
    elif version.startswith("M_F6_mix"):
        i = int(version[len("M_F6_mix"):])
        n0, b_, pr = GRID16[i]
        W = P16.to_w(P16.mix_rel(z[f"D_M_K_K{K}"], z["D_M_ext"], 4 * K, n0, b_, pr))
        sc = np.einsum("ky,kny->ny", W, LM)
    else:
        _, form, cpart, lpart = version.split("_")
        ci = int(cpart[1:])
        LN = np.stack([P16.centered(p_.astype(np.float64)) for p_ in z[f"p_pool_knn_c{ci}"]])
        W = mn_weights(form, LAM_OF[lpart], z[f"D_M_K_K{K}"], z[f"D_N_K_K{K}_c{ci}"], z[f"D_M_orc_K{K}"], z[f"D_N_orc_K{K}_c{ci}"], K, nm)
        sc = np.einsum("ky,kny->ny", W, np.concatenate([LM, LN]))
    Sp = P16.norm_log(sc)
    dl, _ = P16.match_shift(Sp, target)
    return np.exp(P16.norm_log(Sp + dl))


def run_students(u: Unit6, z, choice: dict, teth: dict) -> tuple[list, dict]:
    rows, arr = [], {}
    yt, yp = u.ytest, u.ypool
    arr["y_test"], arr["cl_test"] = yt.astype(np.int8), u.cl
    nG = len(SGRID)
    for K in KS:
        kpos = u.kpos[K]
        assert (z[f"kpos_K{K}"] == kpos).all()
        nk = np.setdiff1d(np.arange(u.n), kpos)
        teach = {}
        for t_ in TEACH:
            v = "prod" if t_ == "prod" else choice.get((u.c, K, t_))
            if v is None:
                continue
            Q = teacher_pool(z, u, K, v)
            ag = float((Q[nk].argmax(1) == z[f"pred_pool_K{K}_{v}"].astype(int)).mean())
            rows.append(dict(K=K, student="_teacher", teacher=t_, chosen=v, uar_pool=100 * S.uar(yp[nk], Q[nk].argmax(1)), maxp=float(Q.max(1).mean()), agree_saved=ag))
            if ag < NONE_AGREE:
                raise SystemExit(f"{u.name()}：疑似ラベルの作り直しが保存と違う（{v} K={K} 一致 {ag:.4f}）")
            teach[t_] = Q
        names = list(teach)
        gk = u.gK(K)
        for kind in u.kinds:
            W0, b0 = u.start[kind]
            cw, cb = teth[f"{kind}|{K}"]
            G = torch.stack([float(a_) * gk + S.soft_rows(S.t(teach[t_]), dai1.mval(r_, u.n)) for t_ in names for r_, a_ in SGRID])
            res = S.fit(u.Xp[kind], G, W0, b0, Cw=cw, Cb=cb)
            if float(res["grad_max"]) > P16.GRAD_TOL or not res["finite_p"].all():
                raise SystemExit(f"{u.name()}：生徒 {kind} K={K} が解けていない")
            Z = S.logits(u.Xt[kind], res["W"], res["b"])
            Zp = S.logits(u.Xp[kind], res["W"], res["b"])
            rn = fit_none(u, kind, K, teth)
            nt = S.logits(u.Xt[kind], rn["W"][0], rn["b"][0]).argmax(1).cpu().numpy()
            npl = S.logits(u.Xp[kind], rn["W"][0], rn["b"][0]).argmax(1).cpu().numpy()
            arr[f"pred_K{K}_{kind}_none"] = nt.astype(np.int8)
            arr[f"pred_pool_K{K}_{kind}_none"] = npl.astype(np.int8)                # 候補の上の K 本だけの分類器の判定（条件 2 の量．gate-code 指摘 1）
            rows.append(dict(K=K, student=kind, teacher="_check", grad=float(res["grad_max"]), stalled=int(res["stalled"]), uar_none=100 * S.uar(yt, nt),
                             uar_pool_none=100 * S.uar(yp[nk], npl[nk])))
            for j, t_ in enumerate(names):
                for gi in range(nG):
                    q = j * nG + gi
                    pr = Z[q].argmax(1).cpu().numpy()
                    pp = Zp[q].argmax(1).cpu().numpy()
                    arr[f"pred_K{K}_{kind}_{t_}_g{gi}"] = pr.astype(np.int8)
                    rows.append(dict(K=K, student=kind, teacher=t_, grid=gi, uar=100 * S.uar(yt, pr), uar_pool=100 * S.uar(yp[nk], pp[nk]),
                                     mimic=float((pp[nk] == teach[t_][nk].argmax(1)).mean())))
    return rows, arr


# ------------------------------------------------------------------ 段の回し方

def loop_units(out: Path, sub: str, fn, targets, folds, seeds, extra=None) -> None:
    d = out / sub
    d.mkdir(parents=True, exist_ok=True)
    for T, c, f, r in all_units(targets, folds, seeds):
        name = f"{T}_{c}_f{f}_r{r}"
        if (d / f"{name}.csv").exists():
            if (pd.read_csv(d / f"{name}.csv").commit == RUN["commit"]).all():
                print(f"{sub} {name}：済み（飛ばす）", flush=True)
                continue
            raise SystemExit(f"{sub} {name} が別のコミットで既にある")
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        u = Unit6(T, c, f, r)
        rows, arr = fn(u, extra)
        np.savez_compressed(d / f"{name}.npz", **arr)                       # npz を先に書き，csv を済みの印にする（途中で止まっても再開できる）
        pd.DataFrame(rows).assign(T=T, c=c, fold=f, seed=r, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0).to_csv(d / f"{name}.csv", index=False)
        gm = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0
        print(f"{sub} {name}：{time.time() - t0:.0f} 秒．行 {len(rows)}．GPU の最大 {gm:.0f} MB", flush=True)
        del u, rows, arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_csvs(out: Path, sub: str, n_exp: int, allow_partial: bool) -> tuple[list, pd.DataFrame]:
    files = sorted(p.with_suffix(".npz") for p in (out / sub).glob("*.csv"))
    if len(files) != n_exp and not allow_partial:
        raise SystemExit(f"{sub} の出力が {len(files)} 個（{n_exp} のはず）")
    csvs = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in files], ignore_index=True) if files else pd.DataFrame()
    if len(csvs) and csvs.commit.nunique() != 1 and not allow_partial:
        raise SystemExit(f"{sub} のコミットが 1 つでない")
    return files, csvs


def row_rng(*key) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32("|".join(map(str, key)).encode())])


A_CONTRASTS = ([("M_F6_orc", "prod"), ("M_F6n_orc", "prod"), ("M_F2_orc", "prod"), ("M_F6_K", "prod"), ("M_F6_ext", "prod"), ("M_F6_mixsel", "prod"),
                ("M_F6_orc", "M_F6n_orc"), ("M_F6_mixsel", "M_F6n_mixsel"), ("MS_eq_sel", "prod"), ("MN_eq_sel", "prod"), ("MN_F2K_sel", "prod"),
                ("MN_F6K_sel", "prod"), ("MN_F6orc_sel", "prod"), ("MN_eq_sel", "MS_eq_sel"), ("MN_eq_sel", "MS_eq_tied"), ("N_eq_sel", "prod"),
                ("MN_F2K_sel", "MN_eq_sel")])
RULES_A = [("R1", "M_F6_orc", "prod"), ("R1", "MN_F6orc_sel", "prod"), ("R2", "M_F6_mixsel", "prod"), ("R2", "M_F6_K", "prod"), ("R2", "M_F6_ext", "prod"),
           ("R2", "MN_eq_sel", "prod"), ("R2", "MN_F2K_sel", "prod"), ("R2", "MN_F6K_sel", "prod"), ("帰し方", "M_F6_orc", "M_F6n_orc"),
           ("帰し方", "M_F6_mixsel", "M_F6n_mixsel"), ("帰し方", "MN_eq_sel", "MS_eq_sel"), ("帰し方", "MN_eq_sel", "MS_eq_tied"), ("対照", "MS_eq_sel", "prod")]


def count_row(d: pd.DataFrame) -> dict:
    up, dn = d[d.verdict == "上回る"].c.tolist(), d[d.verdict == "下回る"].c.tolist()
    return dict(n_corpora=len(d), n_up=len(up), n_down=len(dn), hit=len(up) >= 2, down2=len(dn) >= 2, n_up_orig4=len([x for x in up if x in DD.CORPORA]),
                n_down_orig4=len([x for x in dn if x in DD.CORPORA]), up=",".join(up), down=",".join(dn), new=",".join(f"{r.c}:{r.est:+.2f}" for r in d.itertuples() if r.c in NEW))


def stage_summary(out: Path, allow_partial: bool) -> None:
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)
    geff = {c: g_eff_of(c) for c in CORP6}
    geff["RAVDESS"] = 19.99                                                 # 24 話者を全分割から 1 人ずつ除く jackknife にする（自由度は 18 で控えめ．設計 5 節）

    def level(dd: dict) -> float:
        return float(np.mean([np.mean([ST.uar_full(dd[k]) for k in dd if k[0] == f]) for f in sorted({k[0] for k in dd})]))

    def eff(A, B, c, key, pool=False):
        if set(A) != set(B) and not allow_partial:
            raise SystemExit(f"差の単位がそろわない：{key}（{len(A)} と {len(B)}）")
        es = (P12.effect_stats_pool if pool else ST.effect_stats)(A, B, geff[c], row_rng(*key))
        return dict(est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], df=es.get("df", np.nan), method=es["method"], n_fr=es["n_units"],
                    verdict=P16.verdict(es["lo"], es["hi"], es["effect"]))

    n_units = len(all_units())
    files, csvs = load_csvs(out, "units", n_units, allow_partial)
    choice, chrows = select_points(csvs)
    pd.DataFrame(chrows).to_csv(sdir / "sel.csv", index=False)
    miss = [(c, K, nm_) for c in CORP6 for K in KS for nm_ in seldef_of(c) if (c, K, nm_) not in choice]
    if miss and not allow_partial:
        raise SystemExit(f"版の選択が欠けた組がある：{miss[:5]}（計 {len(miss)}）")
    layers = ("adj", "raw", "pool")
    cells = {w: {} for w in layers}
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        f, r = int(f[1:]), int(r[1:])
        z = np.load(p, allow_pickle=False)
        yt = z["y_test"].astype(int)
        for K in KS:
            nkm = np.ones(len(z["y_pool"]), bool)
            nkm[z[f"kpos_K{K}"]] = False
            ypk, clp = z["y_pool"].astype(int)[nkm], z["cl_pool"][nkm]
            names = {"prod": "prod", "M_F6_orc": "M_F6_orc", "M_F6n_orc": "M_F6n_orc", "M_F2_orc": "M_F2_orc", "M_F6_K": "M_F6_K", "M_F6_ext": "M_F6_ext"}
            names.update({nm_: v for (cc, KK, nm_), v in choice.items() if cc == c and KK == K})
            for v, src in names.items():
                for w in layers:
                    kk = f"pred_{w}_K{K}_{src}"
                    if kk not in z.files:
                        continue
                    cnt = ST.counts(ypk, z[kk].astype(int), clp) if w == "pool" else ST.counts(yt, z[kk].astype(int), z["cl_test"])
                    P12._add(cells[w].setdefault((c, K, v), {}), (f, r), cnt, f"{p.stem} {w} {v}")
            for kind in kinds_of(c):
                kk = f"pred_raw_K{K}_none_{kind}"
                P12._add(cells["raw"].setdefault((c, K, f"none_{kind}"), {}), (f, r), ST.counts(yt, z[kk].astype(int), z["cl_test"]), f"{p.stem} none")
    emo = {}
    for (c, K, v), dd in cells["adj"].items():
        cor = sum(x[1].sum(0) for x in dd.values()); tot = sum(x[2].sum(0) for x in dd.values())
        emo[(c, K, v)] = (cor, tot)
    for w in cells:
        P12._finish(cells[w])
    pd.DataFrame([dict(c=c, K=K, version=v, emotion=i, recall=100 * cor[i] / max(tot[i], 1)) for (c, K, v), (cor, tot) in sorted(emo.items()) for i in range(4)]).to_csv(
        sdir / "per_emotion.csv", index=False)
    pd.DataFrame([dict(c=c, K=K, version=v, **{f"uar_{w}": (level(cells[w][(c, K, v)]) if (c, K, v) in cells[w] else np.nan) for w in layers})
                  for (c, K, v) in sorted(set(cells["raw"]) | set(cells["adj"]))]).to_csv(sdir / "levels.csv", index=False)
    effs = []
    for c in CORP6:
        for K in KS:
            for a, b in A_CONTRASTS:
                for w in layers:
                    A, B = cells[w].get((c, K, a)), cells[w].get((c, K, b))
                    if A is None or B is None:
                        continue
                    effs.append(dict(c=c, K=K, a=a, b=b, where=w, **eff(A, B, c, ("A", c, K, a, b, w), pool=(w == "pool"))))
    effs = pd.DataFrame(effs)
    effs.to_csv(sdir / "effects.csv", index=False)
    crit = []
    for rule, a, b in RULES_A:
        for K in KS:
            d = effs[(effs.a == a) & (effs.b == b) & (effs["where"] == "adj") & (effs.K == K)]
            if len(d):
                crit.append(dict(rule=rule, a=a, b=b, K=K, main=(K == K_MAIN), **count_row(d)))
    pd.DataFrame(crit).to_csv(sdir / "criteria.csv", index=False)
    nn = csvs[csvs.version == "_nn"]
    nn = nn[nn.seed == nn.groupby(["T", "c", "fold"]).seed.transform("min")]
    nn.groupby(["c", "modality"])[["rank_bal", "d1"]].agg(["mean", "min", "max", "count"]).pipe(lambda q: q.set_axis([f"{x}_{y}" for x, y in q.columns], axis=1)).reset_index().to_csv(
        sdir / "neighbors.csv", index=False)
    json.dump(dict(n_units=len(files), adj_resid_max=float(csvs.adj_resid.max()), gpu_max_mb=float(csvs.gpu_max_mb.max()),
                   sec_units=float(csvs.groupby(["T", "c", "fold", "seed"]).sec.first().sum())), open(sdir / "convergence.json", "w"), indent=1)
    print(f"summary（まとめた答え）：単位 {len(files)}，差の行 {len(effs)}", flush=True)
    # 生徒
    sfiles, sc = load_csvs(out, "students", n_units, allow_partial)
    if sfiles:
        sc["role"] = sc.student.astype(str).map(lambda k: "fusion" if len(k) >= 2 else k)
        gs = sc[sc.grid.notna() & (sc.grid >= 0)].groupby(["T", "K", "role", "teacher", "grid"]).uar.mean()   # 融合は AVT と AV をまとめて「融合」として選ぶ
        sch, srows = {}, []
        for c in CORP6:
            for K in KS:
                for kind in kinds_of(c):
                    role = "fusion" if len(kind) >= 2 else kind
                    for t_ in TEACH:
                        if t_ == "M_F6_mixsel" and len(MODS_OF[c]) < 2:            # RAVDESS には無い疑似ラベル（code_v2 記録 2）
                            continue
                        vals = [gs.get((c, K, role, t_, gi), np.nan) for gi in range(len(SGRID))]
                        if not np.isfinite(vals).any():
                            continue
                        gi = int(np.nanargmax(vals))
                        sch[(c, K, kind, t_)] = gi
                        srows.append(dict(c=c, K=K, student=kind, role=role, teacher=t_, grid=gi, rho=SGRID[gi][0], a=SGRID[gi][1], mean_uar=float(np.nanmax(vals))))
        pd.DataFrame(srows).to_csv(sdir / "students_sel.csv", index=False)
        exp_combos = [(c, K, kind, t_) for c in CORP6 for K in KS for kind in kinds_of(c) for t_ in TEACH if t_ != "M_F6_mixsel" or len(MODS_OF[c]) >= 2]
        missing = [x for x in exp_combos if x not in sch]
        if missing and not allow_partial:
            raise SystemExit(f"生徒の格子の点が選べない組がある：{missing[:5]}（計 {len(missing)}）")
        cells = {}
        for p in sfiles:
            T, c, f, r = p.stem.rsplit("_", 3)
            f, r = int(f[1:]), int(r[1:])
            z = np.load(p, allow_pickle=False)
            yt = z["y_test"].astype(int)
            for K in KS:
                for kind in kinds_of(c):
                    P12._add(cells.setdefault((c, K, kind, "none"), {}), (f, r), ST.counts(yt, z[f"pred_K{K}_{kind}_none"].astype(int), z["cl_test"]), p.stem)
                    for t_ in TEACH:
                        if (c, K, kind, t_) in sch and f"pred_K{K}_{kind}_{t_}_g{sch[(c, K, kind, t_)]}" in z.files:
                            P12._add(cells.setdefault((c, K, kind, t_), {}), (f, r),
                                     ST.counts(yt, z[f"pred_K{K}_{kind}_{t_}_g{sch[(c, K, kind, t_)]}"].astype(int), z["cl_test"]), p.stem)
        P12._finish(cells)
        pd.DataFrame([dict(c=c, K=K, student=s_, teacher=t_, uar=level(dd)) for (c, K, s_, t_), dd in sorted(cells.items())]).to_csv(sdir / "students_levels.csv", index=False)
        se_ = []
        pairs = [(t_, "prod") for t_ in TEACH[1:]] + [("MN_eq_sel", "MS_eq_sel"), ("MN_eq_sel", "MS_eq_tied"), ("prod", "none")]
        for c in CORP6:
            for K in KS:
                for kind in kinds_of(c):
                    for a, b in pairs:
                        A, B = cells.get((c, K, kind, a)), cells.get((c, K, kind, b))
                        if A is None or B is None:
                            continue
                        se_.append(dict(c=c, K=K, student=kind, a=a, b=b, **eff(A, B, c, ("C", c, K, kind, a, b))))
        se_ = pd.DataFrame(se_)
        se_.to_csv(sdir / "students_effects.csv", index=False)
        se_["role"] = se_.student.map(lambda k: "fusion" if len(k) >= 2 else k)
        crit = []
        for (K, role, a, b), q in se_.groupby(["K", "role", "a", "b"]):
            crit.append(dict(K=K, student=role, a=a, b=b, main=bool(K == K_MAIN and b == "prod" and a in ("M_F6_mixsel", "MN_eq_sel", "MN_F2K_sel", "MN_F6K_sel")), **count_row(q)))
        pd.DataFrame(crit).to_csv(sdir / "students_criteria.csv", index=False)
        # 3 つの条件の量（記述）：選んだ生徒の格子の点で，疑似ラベルが K 本だけの分類器より当たる幅（候補の K 本以外）と，生徒が疑似ラベルを再現した割合
        tt = sc[sc.student == "_teacher"][["T", "c", "fold", "seed", "K", "teacher", "uar_pool", "maxp"]].rename(columns={"uar_pool": "uar_pool_teacher"})
        st = sc[sc.grid.notna() & (sc.grid >= 0) & ~sc.student.isin(["_teacher"]) & (sc.teacher != "_check")].copy()
        st = st[[sch.get((r_.c, int(r_.K), r_.student, r_.teacher), -1) == int(r_.grid) for r_ in st.itertuples()]]
        st = st[["T", "c", "fold", "seed", "K", "student", "teacher", "uar", "uar_pool", "mimic"]]
        nn_ = sc[sc.teacher == "_check"][["T", "c", "fold", "seed", "K", "student", "uar_none", "uar_pool_none"]]
        cond = st.merge(tt, on=["T", "c", "fold", "seed", "K", "teacher"]).merge(nn_, on=["T", "c", "fold", "seed", "K", "student"])
        if len(cond) != len(st) and not allow_partial:
            raise SystemExit(f"条件の量の行がそろわない（{len(cond)} 行，生徒の行 {len(st)}）")
        cond["edge"] = cond.uar_pool_teacher - cond.uar_pool_none                 # 条件 2 の量：疑似ラベルの UAR − K 本だけの分類器の UAR（候補）
        cond["gain_none"] = cond.uar - cond.uar_none
        cond.groupby(["c", "K", "student", "teacher"])[["edge", "mimic", "gain_none", "uar_pool_teacher", "uar_pool_none", "maxp"]].mean().reset_index().to_csv(
            sdir / "students_conditions.csv", index=False)
        tt = sc[sc.student == "_teacher"]
        json.dump(dict(n_units=len(sfiles), teacher_agree_min=float(tt.agree_saved.min()), grad_max=float(sc[sc.teacher == "_check"].grad.max())),
                  open(sdir / "students_checks.json", "w"), indent=1)
        print(f"summary（生徒）：単位 {len(sfiles)}，差の行 {len(se_)}", flush=True)


# ------------------------------------------------------------------ 検算

def selftest() -> None:
    assert all_units() and len(all_units()) == 300
    assert cfgs_of("RAVDESS") == [("mod", k, rl) for k in KNN_K for rl in RELS] and len(cfgs_of("AFEW")) == 8
    assert kinds_of("CREMA-D") == ["A", "V", "AV"] and kinds_of("RAVDESS") == ["A"] and kinds_of("MELD") == ["A", "V", "T", "AVT"]
    D3, D1 = np.full((2, 4), 0.8), np.full((1, 4), 0.8)
    W = mn_weights("eq", 0.25, D3, D1, D3, D1, 5, 2)
    assert W.shape == (3, 4) and np.allclose(W[:2], 1) and np.allclose(W[2], 0.5)
    for f in ("F2K", "F6K", "F6orc"):
        assert np.allclose(mn_weights(f, 0.25, D3, D1, D3, D1, 5, 2), W)
    W3 = mn_weights("eq", 1.0, np.full((3, 4), .5), np.full((3, 4), .5), np.full((3, 4), .5), np.full((3, 4), .5), 5, 3)
    assert np.allclose(W3, P17.mn_weights("eq", 1.0, np.full((3, 4), .5), np.full((3, 4), .5), np.full((3, 4), .5), np.full((3, 4), .5), 5))
    for c in CORP6:
        sd = seldef_of(c)
        assert ("M_F6_mixsel" in sd) == (len(MODS_OF[c]) >= 2)
    print("selftest：通った")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "source", "base", "tether", "units", "students", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    out = Path(args.out_dir).resolve()
    OUTDIR["out"] = out
    prod = out == OUT_PROD.resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある")
    if prod and not dai1.committed_clean(DESIGN):
        sys.exit("設計の文書がコミットされていない")
    if prod and (args.targets or args.folds is not None or args.seeds is not None or args.allow_partial):
        sys.exit("本番の置き場では対象を絞らない")
    dai1.ENC["enc"] = DD.ENC1
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, python=platform.python_version(), this_script_sha256=dai15.sha(Path(__file__).resolve()),
                scripts={n: dai15.sha(HERE / n) for n in SCRIPTS if (HERE / n).exists()}, design_sha256=(dai15.sha(DESIGN) if DESIGN.exists() else ""),
                inputs_sha256={**{f"dai1_source/{q.name}": dai15.sha(q) for q in sorted((OUT1 / "source").glob("*.npz"))},
                               **{f"splits/{q.name}": dai15.sha(q) for q in sorted((ROOT / "splits").glob("*.csv"))},
                               **{f"out/{q.name}": dai15.sha(q) for q in (out / "tether.json", out / "frozen_sel.json", out / "splits_new.csv") if q.exists()},
                               **{f"out/source/{q.name}": dai15.sha(q) for q in sorted((out / "source").glob("*")) if q.is_file()},
                               **{f"metadata/{q.name}": dai15.sha(q) for q in sorted((ROOT / "data" / "metadata").glob("*.csv"))},
                               **{f"cache/{q.name}": dai15.sha(q) for q in sorted((ROOT / "data" / "cache").glob("*.npz"))}},
                KS=KS, SEEDS=SEEDS, TETH=TETH, SGRID=SGRID, KNN_K=KNN_K, RELS=RELS, LAMS=LAMS, MODS_OF=MODS_OF, CORP6=CORP6, rng_seed=SEED)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "source":
        stage_source(out)
    elif args.stage == "base":
        loop_units(out, "base", lambda u, _: run_base(u), args.targets, args.folds, args.seeds)
    elif args.stage == "tether":
        stage_tether(out, args.allow_partial)
    elif args.stage == "units":
        teth = json.load(open(out / "tether.json"))["tether"]
        loop_units(out, "units", lambda u, _: run_unit(u, teth[u.T]), args.targets, args.folds, args.seeds)
    elif args.stage == "students":
        teth = json.load(open(out / "tether.json"))["tether"]
        files, csvs = load_csvs(out, "units", len(all_units()), args.allow_partial)
        choice, _ = select_points(csvs)
        fp = out / "frozen_sel.json"
        rec = {f"{c}|{K}|{nm_}": v for (c, K, nm_), v in sorted(choice.items())}
        if fp.exists() and json.load(open(fp))["choice"] != rec:
            raise SystemExit("frozen_sel.json が選び直した点と違う")
        json.dump(dict(choice=rec, time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=commit), open(fp, "w"), ensure_ascii=False, indent=1)
        loop_units(out, "students", lambda u, _: run_students(u, np.load(out / "units" / f"{u.name()}.npz", allow_pickle=False), choice, teth[u.T]),
                   args.targets, args.folds, args.seeds)
    else:
        stage_summary(out, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
