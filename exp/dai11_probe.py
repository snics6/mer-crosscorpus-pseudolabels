"""DAI-11 の下調べ：同じモダリティの別のエンコーダの予測を疑似ラベルにすると，凍結した特徴の上の分類器は上がるか．
新しい情報源の 3 つの要件（出所が学び直す分類器より当たるか，誤りの重なり，特徴で表せる分）を測る
（設計 docs/設計/実験11_下調べ_設計.md 第 3 版）。

DAI-1 の仮のターゲットの回の保存（results/dai1/pseudo/units/）と特徴だけを読む．新しく学ぶのは，第 2 エンコーダのソースの分類器と
K 本だけの版，疑似ラベルで学び直した第 1 エンコーダの分類器，表せる分のリッジ回帰である．評価の回の保存は読まない．

第 2 エンコーダのソースの段：python exp/dai11_probe.py --out results/dai11_probe --stage source2
段 A（引き戻しの格子）：      python exp/dai11_probe.py --out results/dai11_probe --stage enc2k [--units AFEW_IEMOCAP_f0_r0]
選ぶ段：                      python exp/dai11_probe.py --out results/dai11_probe --stage select [--allow-partial]
段 B（単位）：                python exp/dai11_probe.py --out results/dai11_probe --stage units [--units AFEW_IEMOCAP_f0_r0]
集計の段：                    python exp/dai11_probe.py --out results/dai11_probe --stage summarize [--allow-partial]
自己検査：                    python exp/dai11_probe.py --stage selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy import stats as SS
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402   subset_masks（MOSEI の channel，AFEW の asr_ok，MELD の face_single，IEMOCAP の marker）
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = HERE.parent
UD = ROOT / "results" / "dai1" / "pseudo" / "units"
SRC1 = ROOT / "results" / "dai1"                      # 第 1 エンコーダのソースの分類器（dai1.load_source が out/source/ を読む）
FROZEN = ROOT / "frozen" / "dai1_frozen.json"
KS_ALL = (0, 1, 5, 10, 25)
KS_RULE = (1, 5, 10, 25)
STUDENTS = ("A", "V", "T", "AVT")
MODS = "AVT"
BOTH = {**DD.ENC1, **{k + "2": v for k, v in DD.ENC2.items()}}     # 6 つの特徴（A, V, T, A2, V2, T2）
TEACHERS = {"A": ("self", "enc2", "fuse1", "other", "enc2fuse"), "V": ("self", "enc2", "fuse1", "other", "enc2fuse"),
            "T": ("self", "enc2", "fuse1", "other", "enc2fuse"), "AVT": ("self", "enc2", "other")}
INFO_TEACHERS = ("enc2", "fuse1", "other", "enc2fuse")           # R3 の順位相関に入れる出所（self は入れない）
OTHER_MOD = {"enc2": 0, "fuse1": 1, "other": 1, "enc2fuse": 1}   # 出所が別のモダリティを含むか（A・V・T の学び直す分類器にとって）


def other_mod(student: str, teacher: str) -> float:
    """「別のモダリティを含む」の印は受け取る分類器ごとに決める．融合 AVT にとってはどの出所も同じ 3 モダリティから作るので 0（gate-probe 第 2 回の 2，C15）。"""
    return 0.0 if student == "AVT" else float(OTHER_MOD[teacher])


W_TOL = 1e-4                                          # fuse1 で学び直した W と DAI-1 の ret の W の最大の差の許し（観測 7.7e-6，K の取り違えの差 5.7e-3 の間．gate-code 第 2 回の 2）
LAM_REL = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)        # リッジの λ ÷ 候補の本数
N_UNITS = 120
SEED = 0
N_PERM = 2000                                         # R3 の偶然の水準の並べ替えの回数
MISMATCH_FRAC = 0.001                                 # none の判定の作り直しと DAI-1 の保存の不一致の許し
PROB_TOL = 0.02                                       # none の候補の確率の作り直しと DAI-1 の保存（float16）の最大の差の許し
FLOOR = 0.5                                           # 4.1 節の下限（UAR のポイント）
MIN_DELTA_SHARE = 0.5                                 # 取り込める割合を出す Δ（セルの平均）の下限
BIG_DELTA = 5.0                                       # R3：Δ ≥ 5 なのに得 < 0.5 のセルを数える
PAIRS = (("enc2", "none"), ("self", "none"), ("other", "none"), ("fuse1", "none"), ("enc2fuse", "none"),
         ("enc2", "fuse1"), ("enc2fuse", "fuse1"), ("fuse1", "self"), ("enc2fuse", "enc2"), ("enc2", "other"), ("enc2fuse", "other"),
         ("enc2", "self"), ("enc2none", "none"), ("none", "noneloco"),
         ("enc2", "noneloco"), ("self", "noneloco"), ("other", "noneloco"), ("fuse1", "noneloco"), ("enc2fuse", "noneloco"))   # 学び直した版 − none_loco（gate-probe 第 2 回の 4）
RULES = (("enc2", "none", "R1"), ("enc2", "noneloco", "R1loco"), ("self", "none", "R1p"), ("enc2", "self", "R1info"), ("other", "none", "R0"),
         ("fuse1", "none", "R1f1"), ("enc2fuse", "none", "R1f"), ("enc2", "fuse1", "R2a"), ("enc2fuse", "fuse1", "R2b"), ("fuse1", "self", "R2c"),
         ("enc2fuse", "enc2", "R2d"), ("enc2", "other", "R2e"), ("enc2fuse", "other", "R2f"), ("none", "noneloco", "SEL"))
METRICS = ("uar", "war", "waf", "mf1")


# ------------------------------------------------------------------ 指標

def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """UAR（現れる感情の再現率の平均），WAR（正解率），WAF（標本数の重みの F1），macro-F1（4 感情），感情ごとの再現率（%）。"""
    y = np.asarray(y, int); pred = np.asarray(pred, int)
    rec = np.full(4, np.nan)
    for e in range(4):
        m = y == e
        if m.any():
            rec[e] = 100.0 * float((pred[m] == e).mean())
    out = dict(uar=float(np.nanmean(rec)), war=100.0 * float((pred == y).mean()),
               waf=100.0 * float(f1_score(y, pred, labels=[0, 1, 2, 3], average="weighted", zero_division=0)),
               mf1=100.0 * float(f1_score(y, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)))
    for e in range(4):
        out[f"rec{e}"] = float(rec[e])
    return out


def phi_coef(a: np.ndarray, b: np.ndarray) -> float:
    """2 つの 0/1 の列の相関係数（どちらかが一定なら NaN）。"""
    a = a.astype(float); b = b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def kappa_coef(a: np.ndarray, b: np.ndarray) -> float:
    """2 つの 0/1 の列の Cohen の κ（偶然の一致を補正した一致）。"""
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe >= 1.0 - 1e-12 else float((po - pe) / (1 - pe))


def overlap_stats(ct: np.ndarray, cs: np.ndarray) -> dict:
    """要件 2 の量（3.4 節）：comp2 = P(出所が正解 | none が外した) − P(出所が正解)，κ，φ，2×2 の本数。"""
    n_tt, n_tf, n_ft, n_ff = int((ct & cs).sum()), int((ct & ~cs).sum()), int((~ct & cs).sum()), int((~ct & ~cs).sum())
    n_swrong = n_tf + n_ff
    comp2 = (n_tf / n_swrong - float(ct.mean())) * 100.0 if n_swrong > 0 else float("nan")
    return dict(comp2=comp2, kappa=kappa_coef(ct, cs), phi=phi_coef(ct, cs), n_tt=n_tt, n_tf=n_tf, n_ft=n_ft, n_ff=n_ff)


# ------------------------------------------------------------------ 読み込み

def parse_tether(arr) -> dict:
    out = {}
    for s_ in arr:
        kind, K, cw, cb = str(s_).split("|")
        out[(kind, int(K))] = (float(cw), float(cb))
    return out


def p1_item(z1, kind: str, tag: str) -> int:
    items = [str(x) for x in z1[f"p1_{kind}_items"]]
    if items.count(tag) != 1:
        raise SystemExit(f"p1_{kind}_items に {tag} が {items.count(tag)} 個ある")
    return items.index(tag)


def tether_tag(K, cw, cb) -> str:
    return f"{K}|{float(cw)}|{float(cb)}"


class Both:
    """1 コーパスの表と，2 つのエンコーダの特徴（GPU に載せる）。"""
    cache: dict = {}

    def __init__(self, name: str):
        df, feats = DD.load_corpus(name, BOTH)
        self.name, self.df = name, df
        self.y = df["label_4"].to_numpy().astype(int)
        self.X = {}
        for m in MODS:
            self.X[m] = S.t(feats[m]); self.X[m + "2"] = S.t(feats[m + "2"])
        self.X["AVT"] = S.t(np.hstack([feats[m] for m in MODS]))
        self.X["AVT2"] = S.t(np.hstack([feats[m + "2"] for m in MODS]))

    @classmethod
    def get(cls, name):
        if name not in cls.cache:
            cls.cache[name] = Both(name)
        return cls.cache[name]


class Unit2:
    """1 つの単位の分割，K 本，2 つのエンコーダの標準化した特徴と出発点（dai1.Unit と同じ手順．2 つ目のエンコーダは kind+"2"）。"""

    def __init__(self, out2: Path, T: str, c: str, srcs: list[str], f: int, r: int):
        self.T, self.c, self.srcs, self.f, self.r = T, c, srcs, f, r
        cp = Both.get(c)
        self.cp = cp
        self.pool = np.flatnonzero(DD.pool_mask(cp.df, f))
        self.test = np.flatnonzero(cp.df["fold"].to_numpy() == f)
        self.ypool, self.ytest = cp.y[self.pool], cp.y[self.test]
        korder = DD.k_orders(cp.y, self.pool, c, f, r)
        self.kpos = {0: np.array([], int), **{K: np.searchsorted(self.pool, korder[K]) for K in DD.KS}}
        assert all((self.pool[self.kpos[K]] == korder[K]).all() for K in DD.KS)
        self.n = len(self.pool)
        self.Xp, self.Xt, self.start = {}, {}, {}
        for kind in STUDENTS:
            for k2, src_dir in ((kind, SRC1), (kind + "2", out2)):
                X = cp.X[k2]
                mu, sd = X[self.pool].mean(0), X[self.pool].std(0).clamp(min=1e-6)      # 候補（K 本を含み，テストを含まない）の統計
                self.Xp[k2], self.Xt[k2] = (X[self.pool] - mu) / sd, (X[self.test] - mu) / sd
                Ws, bs = zip(*[dai1.load_source(src_dir, s_, kind) for s_ in srcs])
                self.start[k2] = (torch.stack(Ws).mean(0), torch.stack(bs).mean(0))     # 等しい重みの平均（DAI-1 と同じ）

    def gK(self, K, scale=1.0):
        return S.balanced_onehot(self.ypool, self.kpos[K], self.n, scale)

    def nk(self, K) -> np.ndarray:
        return np.setdiff1d(np.arange(self.n), self.kpos[K])


def probs(X, W, b):
    return torch.softmax(S.logits(X, W, b), -1)


def argmax8(Z) -> np.ndarray:
    return Z.argmax(-1).to(torch.int8).cpu().numpy()


def f32(x: torch.Tensor) -> np.ndarray:
    return x.float().cpu().numpy()


def none_wb(u: Unit2, z1, kind: str, K: int, tether: dict):
    if K == 0:
        return u.start[kind]
    cw, cb = tether[(kind, K)]
    j = p1_item(z1, kind, tether_tag(K, cw, cb))
    return S.t(z1[f"p1_{kind}_W"][j]), S.t(z1[f"p1_{kind}_b"][j])


def unit_parts(p3: Path):
    name = p3.name[: -len("_p3.npz")]
    T, c, fs, rs = name.split("_")
    return name, T, c, int(fs[1:]), int(rs[1:])


def load_unit_npz(name: str, fz: dict, T: str):
    z1, z2, z3 = (np.load(UD / f"{name}_p{i}.npz") for i in (1, 2, 3))
    for i, z in enumerate((z1, z2, z3), 1):
        if str(z["commit"]) != fz[T]["pseudo_commit"]:
            raise SystemExit(f"{name}_p{i} のコミット {z['commit']} が固定のファイルの pseudo_commit と違う")
    return z1, z2, z3


# ------------------------------------------------------------------ リッジ回帰（表せる分．交差の予測）

def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """切片つきのリッジ回帰（切片は罰しない）．(d+1, q) の係数。"""
    n, d = X.shape
    X1 = torch.cat([X, torch.ones(n, 1, dtype=X.dtype, device=X.device)], 1)
    A = X1.T @ X1
    pen = torch.full((d + 1,), float(lam), dtype=X.dtype, device=X.device); pen[-1] = 0.0
    A = A + torch.diag(pen)
    return torch.linalg.solve(A, X1.T @ Y)


def ridge_predict(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return X @ B[:-1] + B[-1]


def representability(X: torch.Tensor, Y: torch.Tensor, half: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """候補の特徴 X (n, d) から出所の対数確率 Y (n, q) へのリッジ回帰．各半分をもう一方で当てはめた係数で予測する（交差の予測）．
    λ は列の組（4 列 = 1 つの出所 × K）ごとに，交差の予測の二乗誤差が最小の点を選ぶ（ラベルを使わない）．
    戻り：交差の予測 (n, q)（当てはめに使っていない側の予測だけ），選んだ λ/n (q/4,)，交差の R² (q/4,)。"""
    n = X.shape[0]
    q4 = Y.shape[1] // 4
    dev = X.device
    preds = torch.empty(len(LAM_REL), n, Y.shape[1], dtype=Y.dtype, device=dev)
    for h in (0, 1):
        tr = torch.as_tensor(np.flatnonzero(half != h), device=dev); te = torch.as_tensor(np.flatnonzero(half == h), device=dev)
        for i, lr in enumerate(LAM_REL):
            B = ridge_fit(X[tr], Y[tr], lr * n)
            preds[i, te] = ridge_predict(X[te], B)
    err = (preds - Y[None]).pow(2).sum(1).reshape(len(LAM_REL), q4, 4).sum(-1)          # (λ, q4)
    var = (Y - Y.mean(0)).pow(2).sum(0).reshape(q4, 4).sum(-1)                           # (q4,)
    best = err.argmin(0).cpu().numpy()
    r2 = 1.0 - err.cpu().numpy()[best, np.arange(q4)] / np.maximum(var.cpu().numpy(), 1e-12)
    sel = torch.as_tensor(np.repeat(best, 4), device=dev)
    Yhat = preds[sel, :, torch.arange(Y.shape[1], device=dev)].T                         # (n, q)
    return Yhat, np.array([LAM_REL[i] for i in best]), r2


# ------------------------------------------------------------------ 段 A：第 2 エンコーダの K 本だけの版（引き戻しの格子）

def run_unit_A(p3: Path, out: Path, fz: dict, commit: str) -> dict:
    name, T, c, f, r = unit_parts(p3)
    rows_dir = out / "rowsA"
    rows_dir.mkdir(parents=True, exist_ok=True)
    pcsv = rows_dir / f"{name}_enc2k.csv"
    if pcsv.exists():
        raise SystemExit(f"{pcsv} は既にある（重複を作らない）")
    t0 = time.time()
    load_unit_npz(name, fz, T)
    srcs = [x for x in DD.CORPORA if x not in (T, c)]
    u = Unit2(out / "enc2", T, c, srcs, f, r)
    rows = []
    for m in STUDENTS:
        k2 = m + "2"
        W0, b0 = u.start[k2]
        rows.append(dict(unit=name, T=T, c=c, fold=f, seed=r, kind=m, K=0, Cw=np.nan, Cb=np.nan,
                         uar=S.uar(u.ytest, S.logits(u.Xt[k2], W0, b0).argmax(1).cpu().numpy()) * 100, grad=np.nan, stalled=False))
        items = [(K, cw, cb) for K in DD.KS for cw, cb in dai1.TETHER]
        G = torch.stack([u.gK(K) for K, _, _ in items])
        res = S.fit(u.Xp[k2], G, W0, b0, Cw=[x[1] for x in items], Cb=[x[2] for x in items])
        if not bool(res["finite_p"].all()):                                     # I11：有限でない解は保存しない
            raise SystemExit(f"A {name} {k2}：有限でない解がある（{int((~res['finite_p']).sum())} 問）")
        ua = S.uar_batch(u.ytest, S.logits(u.Xt[k2], res["W"], res["b"]))
        for (K, cw, cb), v, gp, st, fn in zip(items, ua, res["grad_p"], res["stalled_p"], res["finite_p"]):
            rows.append(dict(unit=name, T=T, c=c, fold=f, seed=r, kind=m, K=K, Cw=cw, Cb=cb, uar=float(v), grad=float(gp), stalled=bool(st), finite=bool(fn)))
    df = pd.DataFrame(rows); df["commit"] = commit
    df.to_csv(pcsv, index=False)
    sec = time.time() - t0
    print(f"A {name}: 行 {len(df)} ({sec:.0f} 秒)", flush=True)
    return dict(unit=name, sec=sec)


def select_like_dai1(d: pd.DataFrame) -> dict:
    """(kind, K) ごとに，UAR の平均が最大の (Cw, Cb)．同点は小さい Cw・小さい Cb（dai1.select_tether と同じ規則）。"""
    out = {}
    for (kind, K), g in d.groupby(["kind", "K"]):
        m = g.groupby(["Cw", "Cb"])["uar"].mean().reset_index().sort_values(["uar", "Cw", "Cb"], ascending=[False, True, True])
        out[(kind, int(K))] = (float(m.iloc[0].Cw), float(m.iloc[0].Cb))
    return out


def stage_select(out: Path, allow_partial: bool, commit: str) -> None:
    files = sorted((out / "rowsA").glob("*_enc2k.csv"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"段 A の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    A = pd.concat([pd.read_csv(p) for p in files])
    if set(A["commit"].astype(str)) != {commit}:
        raise SystemExit(f"段 A の行のコミットが今の実行と違う：{set(A['commit'].astype(str))}")
    (out / "select").mkdir(exist_ok=True)
    sel, grid_rows = {}, []
    for T, dT in A[A["K"] > 0].groupby("T"):
        teth = select_like_dai1(dT)
        sel[T] = {"tether": {f"{k}|{K}": list(v) for (k, K), v in teth.items()}, "units": sorted(dT["unit"].unique()), "n_units": int(dT["unit"].nunique())}
        for (kind, K), g in dT.groupby(["kind", "K"]):
            gm = g.groupby(["Cw", "Cb"])["uar"].mean().reset_index()
            cw, cb = teth[(kind, int(K))]
            per_unit_best = g.groupby("unit")["uar"].max()
            per_unit_sel = g[(g["Cw"] == cw) & (g["Cb"] == cb)].set_index("unit")["uar"]
            for _, row in gm.iterrows():
                grid_rows.append(dict(T=T, kind=kind, K=int(K), Cw=row.Cw, Cb=row.Cb, mean_uar=row.uar, selected=bool(row.Cw == cw and row.Cb == cb),
                                      grid_oracle_gap_per_TkindK=float((per_unit_best - per_unit_sel.reindex(per_unit_best.index)).mean())))   # (T, 種類, K) ごとの 1 つの値を 14 行に繰り返す（m-h）
    pd.DataFrame(grid_rows).to_csv(out / "select" / "enc2_grid.csv", index=False)
    json.dump({"commit": commit, "rule": "dai1.select_tether と同じ（T の単位のテストの UAR の平均が最大．同点は小さい Cw・Cb）", **sel},
              open(out / "select" / "enc2_tether.json", "w"), indent=1, ensure_ascii=False)
    # none_loco：T の単位のうち仮のターゲット c を除いた単位の p1 の行から選ぶ
    loco = {}
    units_T = {}
    for p in sorted(UD.glob("*_p1.csv")):
        name = p.name[: -len("_p1.csv")]
        T = name.split("_")[0]
        units_T.setdefault(T, []).append(name)
    for T, names in units_T.items():
        if allow_partial and T not in sel:
            continue
        rows = []
        for name in names:
            d = pd.read_csv(UD / f"{name}_p1.csv")
            d = d[(d["stage"] == 1) & (d["method"] == "none")].copy()
            d["unit"] = name; d["c_unit"] = name.split("_")[1]
            rows.append(d)
        R = pd.concat(rows)
        loco[T] = {}
        for c in DD.others(T):
            Rc = R[R["c_unit"] != c]
            teth = select_like_dai1(Rc.rename(columns={"kind": "kind"}))
            loco[T][c] = {"tether": {f"{k}|{K}": list(v) for (k, K), v in teth.items()}, "n_units": int(Rc["unit"].nunique())}
    json.dump({"commit": commit, "rule": "T の単位のうち仮のターゲット c の単位を除いた p1 の none の行で dai1.select_tether と同じ規則", **loco},
              open(out / "select" / "none_loco_tether.json", "w"), indent=1, ensure_ascii=False)
    print(f"選んだ：enc2_tether.json（T {len(sel)}），none_loco_tether.json（T {len(loco)}），enc2_grid.csv（{len(grid_rows)} 行）", flush=True)


# ------------------------------------------------------------------ 段 B：要件の量と学び直し

def run_unit(p3: Path, out: Path, fz: dict, commit: str, sel: dict, loco: dict) -> dict:
    name, T, c, f, r = unit_parts(p3)
    rows_dir = out / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for p in (rows_dir / f"{name}.npz", rows_dir / f"{name}_req.csv"):
        if p.exists():
            raise SystemExit(f"{p} は既にある（重複を作らない．消すか別の --out にする）")
    t0 = time.time()
    srcs = [x for x in DD.CORPORA if x not in (T, c)]
    z1, z2, z3 = load_unit_npz(name, fz, T)
    tether = parse_tether(z3["tether"])
    teth2 = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in sel[T]["tether"].items()}
    tloco = {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in loco[T][c]["tether"].items()}
    u = Unit2(out / "enc2", T, c, srcs, f, r)
    n, nt = u.n, len(u.test)
    store, req_rows, conv, mism = {}, [], {}, {}
    nd = {m: n / u.Xp[m].shape[1] for m in STUDENTS}

    # --- none（第 1 エンコーダの K 本だけの版），none_loco，oracle：保存の W・b を作り直した特徴に当て，判定を照合する
    none_p = {}
    for m in STUDENTS:
        for K in KS_ALL:
            W, b = none_wb(u, z1, m, K, tether)
            none_p[(m, K)] = probs(u.Xp[m], W, b)
            pred = argmax8(S.logits(u.Xt[m], W, b))
            saved = z1[f"p1_{m}_start_pred"] if K == 0 else z1[f"p1_{m}_pred"][p1_item(z1, m, tether_tag(K, *tether[(m, K)]))]
            k_m = int((pred != saved.astype(np.int8)).sum())
            mism[f"{m}_K{K}"] = k_m
            if k_m > max(1, int(MISMATCH_FRAC * nt)):
                raise SystemExit(f"{name} {m} K={K}：none の判定の作り直しが DAI-1 の保存と {k_m} 本違う")
            store[f"K{K}_{m}_none"] = pred
            store[f"pool_none_K{K}_{m}"] = argmax8(none_p[(m, K)])
            store[f"K{K}_{m}_noneloco"] = pred if K == 0 else z1[f"p1_{m}_pred"][p1_item(z1, m, tether_tag(K, *tloco[(m, K)]))].astype(np.int8)
            if K in (0, 5):                                                     # 候補の確率の並びと標準化の照合（float16 の保存）
                saved_p = z2[f"p2_K{K}_none_{m}_pool"].astype(np.float32)
                if saved_p.shape[0] != n:
                    raise SystemExit(f"{name} {m} K={K}：候補の本数が DAI-1 の保存と違う（{n} 対 {saved_p.shape[0]}）")
                d_ = float(np.abs(none_p[(m, K)].float().cpu().numpy() - saved_p).max())
                mism[f"{m}_K{K}_poolprob_maxdiff"] = d_
                if d_ > PROB_TOL:
                    raise SystemExit(f"{name} {m} K={K}：none の候補の確率が DAI-1 の保存と最大 {d_:.3f} 違う")
        store[f"{m}_oracle"] = z1[f"p1_{m}_pred"][p1_item(z1, m, tether_tag("all", *tether[(m, 25)]))].astype(np.int8)

    # --- 出所 other：DAI-1 の ret の W・b から 3 モダリティの確率の積（指数 1,1,1）
    other_p = {}
    for K in KS_ALL:
        L = 0
        for m in MODS:
            W, b = S.t(z2[f"p2_K{K}_ret_{m}_W"]), S.t(z2[f"p2_K{K}_ret_{m}_b"])
            L = L + probs(u.Xp[m], W, b).clamp(min=1e-12).log()
        other_p[K] = torch.softmax(L, -1)

    # --- 出所 enc2：第 2 エンコーダの K 本だけの版（K=0 は出発点，K≥1 は選ぶ段で選んだ引き戻しで出発点から学び直す）
    enc2_p = {}
    for m in STUDENTS:
        k2 = m + "2"
        W0, b0 = u.start[k2]
        enc2_p[(m, 0)] = probs(u.Xp[k2], W0, b0)
        store[f"K0_{m}_enc2none"] = argmax8(S.logits(u.Xt[k2], W0, b0))
        store[f"enc2_W_K0_{m}"], store[f"enc2_b_K0_{m}"] = f32(W0), f32(b0)
        items = [(K, *teth2[(m, K)]) for K in DD.KS]
        G = torch.stack([u.gK(K) for K, _, _ in items])
        res = S.fit(u.Xp[k2], G, W0, b0, Cw=[x[1] for x in items], Cb=[x[2] for x in items])
        if not bool(res["finite_p"].all()):                                     # I11
            raise SystemExit(f"B {name} enc2 {k2}：有限でない解がある（{int((~res['finite_p']).sum())} 問）")
        conv[f"enc2_{m}"] = dict(items=[f"K{K}" for K, _, _ in items], grad_p=[float(x) for x in res["grad_p"]],
                                 stalled_p=[bool(x) for x in res["stalled_p"]], finite_p=[bool(x) for x in res["finite_p"]],
                                 grad_max=float(res["grad_max"]), n_iter=int(res["n_iter"]))
        Zt = S.logits(u.Xt[k2], res["W"], res["b"])
        for j, (K, _, _) in enumerate(items):
            enc2_p[(m, K)] = probs(u.Xp[k2], res["W"][j], res["b"][j])
            store[f"K{K}_{m}_enc2none"] = argmax8(Zt[j])
            store[f"enc2_W_K{K}_{m}"], store[f"enc2_b_K{K}_{m}"] = f32(res["W"][j]), f32(res["b"][j])   # R1・R2（gate-code の 1）

    def teacher(m: str, t_: str, K: int) -> torch.Tensor:
        if t_ == "other":
            return other_p[K]
        if t_ == "enc2":
            return enc2_p[(m, K)]
        if t_ == "enc2fuse":
            return enc2_p[("AVT", K)]
        if t_ == "fuse1":
            return none_p[("AVT", K)]
        if t_ == "self":
            return none_p[(m, K)]
        raise KeyError(t_)

    # --- 段 B の 2：疑似ラベルで学び直す（DAI-1 の ret と同じ形）．先に解いて，D8 の一致率を段 1 の行に入れる
    student_pool = {}
    for m in STUDENTS:
        items = [(t_, K) for t_ in TEACHERS[m] for K in KS_ALL]
        G = torch.stack([(u.gK(K) if K else torch.zeros(n, 4, device=S.DEV, dtype=S.DT)) + S.soft_rows(teacher(m, t_, K), float(n))
                         for t_, K in items])
        W0, b0 = u.start[m]
        res = S.fit(u.Xp[m], G, W0, b0, Cw=[tether[(m, max(K, 1))][0] for _, K in items], Cb=[tether[(m, max(K, 1))][1] for _, K in items])
        if not bool(res["finite_p"].all()):                                     # I11
            raise SystemExit(f"B {name} student {m}：有限でない解がある（{int((~res['finite_p']).sum())} 問）")
        conv[f"student_{m}"] = dict(items=[f"{t_}_K{K}" for t_, K in items], grad_p=[float(x) for x in res["grad_p"]],
                                    stalled_p=[bool(x) for x in res["stalled_p"]], finite_p=[bool(x) for x in res["finite_p"]],
                                    grad_max=float(res["grad_max"]), n_iter=int(res["n_iter"]))
        Zt = S.logits(u.Xt[m], res["W"], res["b"])
        Zp = S.logits(u.Xp[m], res["W"], res["b"])
        for j, (t_, K) in enumerate(items):
            store[f"K{K}_{m}_{t_}"] = argmax8(Zt[j])
            student_pool[(m, t_, K)] = argmax8(Zp[j])
            store[f"pool_student_K{K}_{m}_{t_}"] = student_pool[(m, t_, K)]
            store[f"W_K{K}_{m}_{t_}"], store[f"b_K{K}_{m}_{t_}"] = f32(res["W"][j]), f32(res["b"][j])
            if t_ == "fuse1" and m in MODS:                                       # fuse1 の問は DAI-1 の段 2 の ret_m と同じ問（I7・I14．W で照合）
                dW = float(np.abs(f32(res["W"][j]) - z2[f"p2_K{K}_ret_{m}_W"]).max())
                mism[f"fuse1_ret_Wdiff_{m}_K{K}"] = dW
                if dW > W_TOL:
                    raise SystemExit(f"{name} {m} K={K}：fuse1 で学び直した W が DAI-1 の ret と最大 {dW:.4f} 違う")

    # --- 段 B の 1：要件の量（候補のうち K 本を除いた発話の上）
    cl_pool = u.cp.df["cluster"].to_numpy()[u.pool].astype(str)
    half = (pd.factorize(cl_pool, sort=True)[0] % 2).astype(int)              # 塊を名前の順に分けた 2 つの半分（ラベルを使わない）
    for m in STUDENTS:
        cols = [(t_, K) for t_ in TEACHERS[m] for K in KS_ALL]
        Y = torch.cat([teacher(m, t_, K).clamp(min=1e-12).log() for t_, K in cols], 1)
        Y = Y - Y.reshape(n, len(cols), 4).mean(-1, keepdim=True).expand(-1, -1, 4).reshape(n, -1)   # 行ごとに平均を引く
        Yhat, lam, r2 = representability(u.Xp[m], Y, half)
        if not bool(torch.isfinite(Yhat).all()):                                 # I11
            raise SystemExit(f"B {name} {m}：交差の予測に有限でない値がある")
        for j, (t_, K) in enumerate(cols):
            nk = u.nk(K)
            yk = u.ypool[nk]
            P = teacher(m, t_, K)
            pt_all = P.argmax(1).cpu().numpy()
            pbal_all = (P / P.sum(0, keepdim=True).clamp(min=1e-12)).argmax(1).cpu().numpy()
            Pproj = torch.softmax(Yhat[:, 4 * j: 4 * j + 4], -1)
            pp_all = Pproj.argmax(1).cpu().numpy()
            ppb_all = (Pproj / Pproj.sum(0, keepdim=True).clamp(min=1e-12)).argmax(1).cpu().numpy()   # 射影自身の総量でそろえた射影（gate-probe 第 2 回の 3）
            ppbt_all = (Pproj / P.sum(0, keepdim=True).clamp(min=1e-12)).argmax(1).cpu().numpy()      # 出所の総量でそろえた射影（学び直しの目標の射影に近い．gate-probe 第 3 回）
            store[f"pool_teacher_K{K}_{m}_{t_}"] = pt_all.astype(np.int8)
            store[f"pool_proj_K{K}_{m}_{t_}"] = pp_all.astype(np.int8)
            store[f"pool_projbal_K{K}_{m}_{t_}"] = ppb_all.astype(np.int8)
            store[f"pool_projbal_t_K{K}_{m}_{t_}"] = ppbt_all.astype(np.int8)
            pt, pb, pp, ppb, ppbt = pt_all[nk], pbal_all[nk], pp_all[nk], ppb_all[nk], ppbt_all[nk]
            ps = store[f"pool_none_K{K}_{m}"].astype(int)[nk]
            pstu = student_pool[(m, t_, K)].astype(int)[nk]
            mt, ms, mp, mb = metrics(yk, pt), metrics(yk, ps), metrics(yk, pp), metrics(yk, pb)
            ct, cs = pt == yk, ps == yk
            req_rows.append(dict(unit=name, T=T, c=c, fold=f, seed=r, student=m, teacher=t_, K=K, n_pool=n, n_nk=len(nk), n_over_d=nd[m],
                                 uar_t=mt["uar"], uar_s=ms["uar"], delta_uar=mt["uar"] - ms["uar"], delta_uar_bal=mb["uar"] - ms["uar"],
                                 war_t=mt["war"], war_s=ms["war"], delta_war=mt["war"] - ms["war"],
                                 waf_t=mt["waf"], waf_s=ms["waf"], delta_waf=mt["waf"] - ms["waf"],
                                 mf1_t=mt["mf1"], mf1_s=ms["mf1"], delta_mf1=mt["mf1"] - ms["mf1"],
                                 agree=float((pt == ps).mean()), **overlap_stats(ct, cs),
                                 uar_proj=mp["uar"], proj_gain=mp["uar"] - ms["uar"], lam_rel=float(lam[j]), r2=float(r2[j]),
                                 d8_agree_student_proj=float((pstu == pp).mean()), d8_agree_student_projbal=float((pstu == ppb).mean()),
                                 d8_agree_none_proj=float((ps == pp).mean()), d8_agree_none_projbal=float((ps == ppb).mean()),
                                 d8_agree_student_projbal_t=float((pstu == ppbt).mean()), d8_agree_none_projbal_t=float((ps == ppbt).mean()),
                                 uar_projbal_t=metrics(yk, ppbt)["uar"],
                                 d8_agree_student_teacher=float((pstu == pt).mean()), d8_agree_student_teacherbal=float((pstu == pb).mean()),
                                 uar_student_pool=metrics(yk, pstu)["uar"], uar_projbal=metrics(yk, ppb)["uar"]))

    sid_t = u.cp.df["sample_id"].to_numpy()[u.test].astype(str)
    np.savez_compressed(rows_dir / f"{name}.npz", commit=commit, unit=name, T=T, c=c, fold=f, seed=r, n_pool=n,
                        n_over_d=json.dumps(nd), test_sample_id=sid_t, test_y=u.ytest.astype(np.int8),
                        test_cluster=u.cp.df["cluster"].to_numpy()[u.test].astype(str),
                        pool_y=u.ypool.astype(np.int8), pool_cluster=cl_pool, **{f"kpos_K{K}": u.kpos[K] for K in DD.KS},
                        mismatch=json.dumps(mism), conv=json.dumps(conv), **store)
    pd.DataFrame(req_rows).to_csv(rows_dir / f"{name}_req.csv", index=False)
    sec = time.time() - t0
    print(f"B {name}: 候補 {n} テスト {nt} 保存 {len(store)} 要件の行 {len(req_rows)} "
          f"判定の不一致の最大 {max(v for k, v in mism.items() if 'maxdiff' not in k and 'Wdiff' not in k)} 本 "
          f"確率の差の最大 {max(v for k, v in mism.items() if 'maxdiff' in k):.2e} W の差の最大 {max(v for k, v in mism.items() if 'Wdiff' in k):.2e} ({sec:.0f} 秒)", flush=True)
    return dict(unit=name, sec=sec)


# ------------------------------------------------------------------ 集計

def _add(d, key, cnt, label):
    if key in d:
        n0, c0, t0, m_ = d[key]
        if not (n0 == cnt[0]).all():
            raise SystemExit(f"{label}：塊の並びが同じ (f, r) のほかの T と違う")
        d[key] = (n0, c0 + cnt[1], t0 + cnt[2], m_ + 1)
    else:
        d[key] = (cnt[0], cnt[1].astype(float), cnt[2].astype(float), 1)


def _finish(cells):
    for k in cells:
        for kk, (n0, c0, t0, m_) in cells[k].items():
            cells[k][kk] = (n0, c0 / m_, t0 / m_)


def fold_t(vals_by_fold: dict) -> dict:
    """分割ごとの平均（種と T を平均）の t の区間（自由度 = 分割の数 − 1）。"""
    v = np.array([np.mean(x) for x in vals_by_fold.values()])
    if len(v) < 2:
        return dict(est=float(v.mean()) if len(v) else np.nan, lo=np.nan, hi=np.nan)
    se = v.std(ddof=1) / np.sqrt(len(v))
    q = SS.t.ppf(0.975, len(v) - 1)
    return dict(est=float(v.mean()), lo=float(v.mean() - q * se), hi=float(v.mean() + q * se))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = SS.rankdata(x), SS.rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def block_plan(M: pd.DataFrame, block_cols: tuple) -> list:
    """並べ替えの塊の計画（S17）．塊（block_cols）ごとに，(c, student, K) の組の行の添字を，出所の名前の順にそろえた行列 (組の数, 出所の数) にする．
    同じ塊の中では，すべての組に同じ出所の並べ替えを当てる（同じ出所は K をまたいで同じ単位と同じ確率を共有するため）．
    塊の中で出所の集合が組によって違うときは，その組だけ別の塊にする（AVT は fuse1・enc2fuse を持たない）。"""
    plans = []
    for _, blk in M.groupby(list(block_cols), sort=False):
        groups = {}
        for key, g in blk.groupby(["c", "student", "K"], sort=False):
            if len(g) < 2:
                continue                                                        # 出所が 1 つの組は使わない（gate-code の 3）
            groups.setdefault(tuple(sorted(g["teacher"])), []).append(g.sort_values("teacher").index.to_numpy())
        for teachers, rows in groups.items():
            plans.append(np.stack(rows))                                        # (組の数, 出所の数)．位置の添字
    return plans


def permute_by_plan(v: np.ndarray, plans: list, rng) -> np.ndarray:
    out = v.copy()
    for P in plans:
        perm = rng.permutation(P.shape[1])
        out[P] = v[P[:, perm]]
    return out


def centered(M: pd.DataFrame, col: str) -> np.ndarray:
    g = M.groupby(["c", "student", "K"])[col]
    return (M[col] - g.transform("mean")).to_numpy()


def r3_rows(M: pd.DataFrame, quantities: tuple, controls: tuple, rng, n_perm: int) -> list:
    """R3（4.3 節）：(c, student, K) の中で中心を引いた量と得の順位相関，塊の並べ替えの偶然の水準（|ρ| の 95% 点），
    「別のモダリティを含む」の印と得の部分順位相関（Freedman–Lane：統制の量で回帰した残りを塊で並べ替える）。
    塊は 2 通り：(c, student)（主）と c（副．A・V・T に同じ並べ替え）。"""
    rows = []
    M = M.reset_index(drop=True)
    sizes = M.groupby(["c", "student", "K"])["teacher"].transform("size")
    M = M[sizes >= 2].reset_index(drop=True)
    if len(M) < 4:
        return [dict(kind="r3", quantity=q, rho=np.nan, null95=np.nan, n_cells=len(M)) for q in quantities]
    plans = {"blk_cm": block_plan(M, ("c", "student")), "blk_c": block_plan(M, ("c",))}
    gain = centered(M, "gain")
    for q in quantities:
        x = centered(M, q)
        ok = np.isfinite(x) & np.isfinite(gain)
        rho = _spearman(x[ok], gain[ok]) if ok.sum() >= 4 else np.nan
        for pname, pl in plans.items():
            null = np.array([_spearman(x[ok], permute_by_plan(gain, pl, rng)[ok]) for _ in range(n_perm)])
            rows.append(dict(kind=f"centered_spearman|{pname}", quantity=q, rho=rho,
                             null95=float(np.nanpercentile(np.abs(null), 95)) if np.isfinite(null).any() else np.nan, n_cells=int(ok.sum())))
    # 部分相関：印が塊の中で変わる行だけ（AVT は印が全部 0 なので外れる）．並べ替えの塊は ok の行だけで作り直す（NaN が入らない）
    ind = centered(M, "other_mod")
    keep = np.isfinite(ind) & (M.groupby(["c", "student", "K"])["other_mod"].transform("nunique").to_numpy() >= 2)
    for ctrl in controls:
        ctl = centered(M, ctrl)
        ok = keep & np.isfinite(ctl) & np.isfinite(gain)
        if ok.sum() < 4:
            rows.append(dict(kind=f"partial_other_mod|{ctrl}", quantity="other_mod", rho=np.nan, null95=np.nan, n_cells=int(ok.sum())))
            continue
        Mok = M[ok].reset_index(drop=True)
        plans_ok = {"blk_cm": block_plan(Mok, ("c", "student")), "blk_c": block_plan(Mok, ("c",))}
        rg, ri, rc = SS.rankdata(gain[ok]), SS.rankdata(ind[ok]), SS.rankdata(ctl[ok])
        A = np.column_stack([rc, np.ones(ok.sum())])
        fit_g = A @ np.linalg.lstsq(A, rg, rcond=None)[0]; e_g = rg - fit_g
        e_i = ri - A @ np.linalg.lstsq(A, ri, rcond=None)[0]

        def pcorr(e1, e2):
            return float(np.corrcoef(e1, e2)[0, 1]) if e1.std() > 0 and e2.std() > 0 else np.nan

        rho = pcorr(e_g, e_i)
        for pname, pl in plans_ok.items():
            null = np.empty(n_perm)
            for i in range(n_perm):
                gstar = fit_g + permute_by_plan(e_g, pl, rng)                  # Freedman–Lane：残りを塊で並べ替え，当てはめに足す
                null[i] = pcorr(gstar - A @ np.linalg.lstsq(A, gstar, rcond=None)[0], e_i)
            rows.append(dict(kind=f"partial_other_mod|{ctrl}|{pname}", quantity="other_mod", rho=rho,
                             null95=float(np.nanpercentile(np.abs(null), 95)) if np.isfinite(null).any() else np.nan, n_cells=int(ok.sum())))
    return rows


def summarize(out: Path, allow_partial: bool) -> None:
    files = sorted((out / "rows").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    rng = np.random.default_rng(SEED)
    commits = {str(np.load(p, allow_pickle=False)["commit"]) for p in files}
    if len(commits) != 1:
        raise SystemExit(f"行のコミットが 1 つでない：{commits}")
    missing = [p.name for p in files if not (out / "rows" / f"{p.stem}_req.csv").exists()]
    if missing:
        raise SystemExit(f"npz と _req.csv の対がそろわない：{missing[:5]}")
    for jname in ("enc2_tether.json", "none_loco_tether.json"):
        jc = json.load(open(out / "select" / jname))["commit"]
        if jc not in commits:
            raise SystemExit(f"{jname} のコミット {jc} が行のコミットと違う")
    src_commits = set(pd.read_csv(out / "enc2" / "source" / "summary.csv")["commit"].astype(str))
    if src_commits != commits:
        raise SystemExit(f"第 2 エンコーダのソースの段のコミット {src_commits} が行のコミットと違う")
    rowsA = sorted((out / "rowsA").glob("*_enc2k.csv"))
    if {str(x) for x in pd.concat([pd.read_csv(p, usecols=["commit"]) for p in rowsA])["commit"].unique()} != commits:
        raise SystemExit("段 A の行のコミットが段 B と違う")
    cells, unit_rows, mism_rows, conv_rows = {}, [], [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        T, c, f, r = str(z["T"]), str(z["c"]), int(z["fold"]), int(z["seed"])
        y, cl, sid = z["test_y"].astype(int), z["test_cluster"].astype(str), z["test_sample_id"].astype(str)
        subs = dai3.subset_masks(c, sid, cl)
        for k in z.files:
            if z[k].dtype == np.int8 and z[k].shape == y.shape and (k.startswith("K") or k.endswith("_oracle")):
                prd = z[k].astype(int)
                cnt = ST.counts(y, prd, cl)
                _add(cells.setdefault((c, k), {}), (f, r), cnt, f"{p.name} {k}")
                unit_rows.append(dict(c=c, T=T, fold=f, seed=r, key=k, **metrics(y, prd)))
                for nm, mk in subs.items():
                    mk = np.asarray(mk, bool)
                    if mk.any():
                        _add(cells.setdefault((c, f"{k}|sub_{nm}"), {}), (f, r), ST.counts(y[mk], prd[mk], cl[mk]), f"{p.name} {k} {nm}")
        for k, v in json.loads(str(z["mismatch"])).items():
            mism_rows.append(dict(unit=str(z["unit"]), key=k, value=v))
        for k, v in json.loads(str(z["conv"])).items():
            for it, gp, st, fn in zip(v["items"], v["grad_p"], v["stalled_p"], v.get("finite_p", [True] * len(v["items"]))):
                conv_rows.append(dict(unit=str(z["unit"]), block=k, item=it, grad=gp, stalled=st, finite=fn, grad_max=v["grad_max"], n_iter=v["n_iter"]))
    _finish(cells)
    U = pd.DataFrame(unit_rows)
    U.to_csv(out / "uar_units.csv", index=False)
    pd.DataFrame(mism_rows).to_csv(out / "checks_units.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(out / "conv_units.csv", index=False)
    cellmean = U.groupby(["c", "key"])[list(METRICS) + [f"rec{e}" for e in range(4)]].mean().reset_index()
    cellmean["n_units"] = U.groupby(["c", "key"]).size().to_numpy()
    if not allow_partial and (cellmean["n_units"] != 30).any():
        raise SystemExit(f"単位の数が 30 でないセルがある：{cellmean[cellmean['n_units'] != 30][['c', 'key', 'n_units']].head().to_dict('records')}")
    cellmean.to_csv(out / "cells.csv", index=False)

    # 差と区間（UAR は effect_stats，ほかは分割の t．部分集合は UAR だけ）
    diff_rows = []
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}
    have_c = sorted(U["c"].unique())
    subset_names = sorted({k.split("|sub_")[1] for (c_, k) in cells if "|sub_" in k})
    for c in have_c:
        Uc = U[U["c"] == c]
        for m in STUDENTS:
            for K in KS_ALL:
                for a, b in PAIRS:
                    ka, kb = f"K{K}_{m}_{a}", f"K{K}_{m}_{b}"
                    if (c, ka) not in cells or (c, kb) not in cells:
                        continue
                    A, B = cells[(c, ka)], cells[(c, kb)]
                    if set(A) != set(B):
                        raise SystemExit(f"{c} {ka} と {kb} の単位がそろわない")
                    es = ST.effect_stats(A, B, geff[c], rng)
                    row = dict(c=c, student=m, K=K, a=a, b=b, subset="all", est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"],
                               p=es["p"], method=es["method"], n_units=es["n_units"],
                               **{f"es_{k}": v for k, v in es.items() if k not in ("effect", "lo", "hi", "se", "p", "method", "n_units")})
                    for met in ("war", "waf", "mf1"):
                        da = Uc[Uc["key"] == ka].set_index(["T", "fold", "seed"])[met]
                        db = Uc[Uc["key"] == kb].set_index(["T", "fold", "seed"])[met]
                        d = (da - db).dropna()
                        ft = fold_t({f_: d.xs(f_, level="fold").to_numpy() for f_ in sorted({i[1] for i in d.index})})
                        row.update({f"{met}_est": ft["est"], f"{met}_lo": ft["lo"], f"{met}_hi": ft["hi"]})
                    diff_rows.append(row)
                    for nm in subset_names:
                        if (c, f"{ka}|sub_{nm}") in cells and (c, f"{kb}|sub_{nm}") in cells:
                            es2 = ST.effect_stats(cells[(c, f"{ka}|sub_{nm}")], cells[(c, f"{kb}|sub_{nm}")], geff[c], rng)
                            diff_rows.append(dict(c=c, student=m, K=K, a=a, b=b, subset=nm, est=es2["effect"], lo=es2["lo"], hi=es2["hi"],
                                                  se=es2["se"], p=es2["p"], method=es2["method"], n_units=es2["n_units"]))
    Dd = pd.DataFrame(diff_rows)
    Dd.to_csv(out / "diffs.csv", index=False)

    # 段 B の 1 の量の集計（取り込める割合はセルの平均の比．D8 の一致率）
    R = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_req.csv"))])
    R.to_csv(out / "req_units.csv", index=False)
    num = [c_ for c_ in R.columns if c_ not in ("unit", "T", "c", "fold", "seed", "student", "teacher", "K")]
    Rc = R.groupby(["c", "student", "teacher", "K"])[num].mean().reset_index()
    Rc["n_units"] = R.groupby(["c", "student", "teacher", "K"]).size().to_numpy()
    Rc["share_cell"] = np.where(Rc["delta_uar"] >= MIN_DELTA_SHARE, Rc["proj_gain"] / Rc["delta_uar"].where(Rc["delta_uar"] != 0), np.nan)
    Rc.to_csv(out / "req_cells.csv", index=False)

    # R3：中心を引いた順位相関と偶然の水準，部分相関（4.3 節．塊の並べ替え，Freedman–Lane）
    gains = Dd[(Dd["b"] == "none") & (Dd["subset"] == "all") & (Dd["a"].isin(INFO_TEACHERS))][["c", "student", "K", "a", "est"]]
    gains = gains.rename(columns={"a": "teacher", "est": "gain"})
    M = Rc.merge(gains, on=["c", "student", "teacher", "K"])
    M["other_mod"] = [other_mod(s_, t_) for s_, t_ in zip(M["student"], M["teacher"])]
    M.to_csv(out / "req_vs_gain.csv", index=False)
    rc_rows = []
    Q = ("delta_uar", "delta_uar_bal", "comp2", "kappa", "phi", "proj_gain", "agree", "r2")        # uar_t は中心を引くと Δ と同じ（gate-probe 第 2 回の記録）
    subsets = [("all", M), ("K>=1", M[M["K"] >= 1]), ("excl_AVTxother", M[~((M["student"] == "AVT") & (M["teacher"] == "other"))]),
               ("AVT_students_excl_other", M[(M["student"] != "AVT") & (M["teacher"] != "other")])]             # 切り分けの主の版（A・V・T の分類器で other を除く．m-d）
    subsets += [(f"student={m}", M[M["student"] == m]) for m in STUDENTS] + [(f"c={c}", M[M["c"] == c]) for c in have_c]
    for sub_name, Msub in subsets:
        for row in r3_rows(Msub, Q, ("delta_uar", "proj_gain"), rng, N_PERM):
            rc_rows.append(dict(subset=sub_name, **row))
    big = M[(M["delta_uar"] >= BIG_DELTA)]
    rc_rows.append(dict(subset="delta>=5", kind="count", quantity="n_cells", value=len(big), n_cells=len(M)))
    rc_rows.append(dict(subset="delta>=5 & gain<0.5", kind="count", quantity="n_cells", value=int((big["gain"] < FLOOR).sum()), n_cells=len(big)))
    rc_rows.append(dict(subset="delta>=5 & gain<0.5 & share_cell<0.5", kind="count", quantity="n_cells",
                        value=int(((big["gain"] < FLOOR) & (big["share_cell"] < 0.5)).sum()), n_cells=len(big)))
    info = Rc[Rc["teacher"].isin(INFO_TEACHERS)]
    for suf, label in (("", "射影自身の総量"), ("_t", "出所の総量")):
        i_ = info[f"d8_agree_student_projbal{suf}"] > info[f"d8_agree_none_projbal{suf}"]
        ii_ = info[f"d8_agree_student_projbal{suf}"] > info["d8_agree_student_teacherbal"]
        rc_rows.append(dict(subset=f"D8(i) {label}: agree(student,projbal) > agree(none,projbal)", kind="share", quantity="share_of_cells", value=float(i_.mean()), n_cells=len(info)))
        rc_rows.append(dict(subset=f"D8(ii) {label}: agree(student,projbal) > agree(student,teacherbal)", kind="share", quantity="share_of_cells", value=float(ii_.mean()), n_cells=len(info)))
        rc_rows.append(dict(subset=f"D8(i)&(ii) {label}", kind="share", quantity="share_of_cells", value=float((i_ & ii_).mean()), n_cells=len(info)))
    pd.DataFrame(rc_rows).to_csv(out / "rank_corr.csv", index=False)

    # R1〜R4 の数え上げ（4.2・4.4・4.5 節）
    crit = []
    for c in have_c:
        for m in STUDENTS:
            for a, b, label in RULES:
                d = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["a"] == a) & (Dd["b"] == b) & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                if len(d) == 0:
                    continue
                up = d[(d["lo"] > 0) & (d["est"] >= FLOOR)]; down = d[(d["hi"] < 0) & (d["est"] <= -FLOOR)]
                verdict = "上回った" if len(up) >= 3 else ("下回った" if len(down) >= 3 else "言えない")
                row = dict(c=c, student=m, rule=label, a=a, b=b, n_K=len(d), n_K_up=len(up), n_K_down=len(down), verdict=verdict,
                           K_up=",".join(map(str, up["K"])), K_down=",".join(map(str, down["K"])), est_min=d["est"].min(), est_max=d["est"].max())
                for met in ("war", "waf", "mf1"):                                    # R4
                    row[f"{met}_same_sign_n"] = int((np.sign(d[f"{met}_est"]) == np.sign(d["est"])).sum())
                    row[f"{met}_ci_excl0_n"] = int(((d[f"{met}_lo"] > 0) | (d[f"{met}_hi"] < 0)).sum())
                if c == "MOSEI":                                                       # 4.5 節：上回った・下回った の両方
                    hit = pd.concat([up, down])
                    ds = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["a"] == a) & (Dd["b"] == b) & (Dd["subset"] == "channel") & (Dd["K"].isin(hit["K"]))]
                    ds = ds.merge(hit[["K", "est"]].rename(columns={"est": "est_all"}), on="K")
                    rob = ds[(np.sign(ds["est"]) == np.sign(ds["est_all"])) & ((ds["lo"] > 0) | (ds["hi"] < 0)) & (ds["est"].abs() >= 0.5 * ds["est_all"].abs())]
                    row["chan_n_K_hit"] = len(hit); row["chan_n_K_robust"] = len(rob)
                if a == "enc2" and b == "none":                                        # A1：同じセルの一致率と κ を添える
                    rq = Rc[(Rc["c"] == c) & (Rc["student"] == m) & (Rc["teacher"] == "enc2") & (Rc["K"].isin(KS_RULE))]
                    row["enc2_agree_mean"] = float(rq["agree"].mean()); row["enc2_kappa_mean"] = float(rq["kappa"].mean())
                crit.append(row)
    pd.DataFrame(crit).to_csv(out / "criteria.csv", index=False)
    print(f"書いた：cells.csv，diffs.csv，req_cells.csv，req_vs_gain.csv，rank_corr.csv，criteria.csv，uar_units.csv，req_units.csv，"
          f"checks_units.csv，conv_units.csv（{len(files)} 単位）", flush=True)


# ------------------------------------------------------------------ 自己検査

def selftest() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400); pred = y.copy(); flip = rng.random(400) < 0.3; pred[flip] = (pred[flip] + 1) % 4
    m = metrics(y, pred)
    from sklearn.metrics import recall_score, accuracy_score, cohen_kappa_score
    assert abs(m["uar"] - 100 * recall_score(y, pred, average="macro")) < 1e-9
    assert abs(m["war"] - 100 * accuracy_score(y, pred)) < 1e-9
    # 出所 other は dai1.product（w = 1,1,1）と同じ
    tk = {f"ret_{mm}": (torch.softmax(torch.as_tensor(rng.normal(size=(50, 4))), -1),) for mm in MODS}
    L = sum(tk[f"ret_{mm}"][0].clamp(min=1e-12).log() for mm in MODS)
    assert torch.allclose(torch.softmax(L, -1), dai1.product(tk, (1.0, 1.0, 1.0), 0), atol=1e-6)
    # リッジ（交差の予測）：線形の目標なら交差の R² ≈ 1，λ は最小の点．予測は当てはめに使っていない側のもの
    n, d = 600, 20
    X = torch.as_tensor(rng.normal(size=(n, d))); Bt = torch.as_tensor(rng.normal(size=(d, 8)))
    Y = X @ Bt + 0.01 * torch.as_tensor(rng.normal(size=(n, 8)))
    half = (np.arange(n) % 2)
    Yhat, lam, r2 = representability(X, Y, half)
    assert (r2 > 0.99).all() and (lam == LAM_REL[0]).all(), (r2, lam)
    assert float((Yhat - Y).pow(2).mean()) < 1e-3
    for h in (0, 1):                                                          # 交差であること：半分 h の予測は，他方で当てた係数の予測と一致
        tr = np.flatnonzero(half != h); te = np.flatnonzero(half == h)
        B = ridge_fit(X[tr], Y[tr], LAM_REL[0] * n)
        assert torch.allclose(Yhat[te], ridge_predict(X[te], B), atol=1e-8)
    # 目標が特徴と無関係なら交差の R² は 0 以下
    Yn = torch.as_tensor(rng.normal(size=(n, 4)))
    _, _, r2n = representability(X, Yn, half)
    assert (r2n < 0.05).all(), r2n
    # 要件 2 の量：κ は sklearn と一致．comp2 は正解率の差と恒等でない（ct と cs を独立に作ると comp2 ≈ 0 だが差は大きい）
    a = rng.random(2000) < 0.7; b = rng.random(2000) < 0.4
    o = overlap_stats(a, b)
    assert abs(o["kappa"] - cohen_kappa_score(a.astype(int), b.astype(int))) < 1e-9
    assert abs(o["comp2"]) < 5.0 and abs(100 * (a.mean() - b.mean())) > 20.0, o
    assert o["n_tt"] + o["n_tf"] + o["n_ft"] + o["n_ff"] == 2000
    # 分割の t
    ft = fold_t({0: [1.0, 1.0], 1: [2.0], 2: [3.0]})
    assert abs(ft["est"] - 2.0) < 1e-12 and ft["lo"] < 2.0 < ft["hi"]
    # 塊の並べ替え：組ごとの出所の多重集合は保たれ，同じ塊の中では K をまたいで同じ並べ替えが当たる
    M = pd.DataFrame([dict(c="X", student=st, K=K, teacher=t, q=i, gain=i * 2.0 + K, other_mod=other_mod(st, t))
                      for st in ("A", "AVT") for K in (0, 5) for i, t in enumerate(TEACHERS[st]) if t != "self"])
    pl = block_plan(M, ("c", "student"))
    assert len(pl) == 2 and pl[0].shape == (2, 4) and pl[1].shape == (2, 2), [P.shape for P in pl]
    g = M["gain"].to_numpy()
    for _ in range(20):
        gp = permute_by_plan(g, pl, np.random.default_rng(_))
        for P in pl:
            for row in P:
                assert sorted(gp[row]) == sorted(g[row])
            d0 = gp[P[0]] - g[P[0]]; d1 = gp[P[1]] - g[P[1]]
            assert np.allclose(d0, d1)                                            # K=0 と K=5 に同じ並べ替え（得は i*2 + K なので差が一致）
    rows_r3 = r3_rows(M, ("q",), ("q",), np.random.default_rng(1), 100)
    cs = [r for r in rows_r3 if r["kind"] == "centered_spearman|blk_cm" and r["quantity"] == "q"][0]
    assert abs(cs["rho"] - 1.0) < 1e-9 and cs["null95"] < 1.0, cs
    pc = [r for r in rows_r3 if r["kind"].startswith("partial_other_mod|q|blk_cm")][0]
    assert pc["n_cells"] == 8, pc                                                 # AVT（印が全部 0）は部分相関から外れる
    xx, yy = rng.normal(size=30), rng.normal(size=30)
    assert abs(_spearman(xx, yy) - SS.spearmanr(xx, yy).correlation) < 1e-12
    # 選び方の規則：dai1.select_tether と同じ結果
    rows = pd.DataFrame([dict(stage=1, method="none", kind="A", K=5, Cw=cw, Cb=cb, uar=(10 - abs(cw - 0.01) * 100) - cb * 0.01, unit=f"u{i}")
                         for i in range(3) for cw, cb in dai1.TETHER])
    assert select_like_dai1(rows) == dai1.select_tether(rows)
    print("selftest ok")


# ------------------------------------------------------------------ main

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("source2", "enc2k", "select", "units", "summarize", "selftest"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dai11_probe"))
    ap.add_argument("--units", default="*", help="単位の名前の glob（例：AFEW_IEMOCAP_f0_r0）")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--resume", action="store_true", help="出力が既にある単位を飛ばす（止まった計算の続き．同じコミットのときだけ）")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest(); return
    out = Path(args.out)
    commit, dirty = dai1.git_commit()
    if dirty and not args.allow_dirty:
        sys.exit("exp・measure・features に未コミットの変更がある（試走は --allow-dirty）")
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(), python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                time=time.strftime("%Y-%m-%d %H:%M:%S"),
                script_sha256=sha(Path(__file__)), dai1_sha256=sha(HERE / "dai1.py"), solver_sha256=sha(HERE / "dai_solver.py"),
                data_sha256=sha(HERE / "dai_data.py"), frozen_sha256=sha(FROZEN), stats_sha256=sha(HERE / "dai1_stats.py"), dai3_sha256=sha(HERE / "dai3.py"),
                pseudo_commit={T: json.load(open(FROZEN))[T]["pseudo_commit"] for T in DD.CORPORA},
                features_sha256={p.name: sha(p) for p in sorted((ROOT / "data" / "cache").glob("*.npz"))
                                 if p.stem.split("_")[-1] in DD.CORPORA and any(p.stem.startswith(k + "_") for k in BOTH.values())},
                constants=dict(KS_ALL=KS_ALL, KS_RULE=KS_RULE, LAM_REL=LAM_REL, FLOOR=FLOOR, MISMATCH_FRAC=MISMATCH_FRAC, PROB_TOL=PROB_TOL,
                               TEACHERS=TEACHERS, PAIRS=PAIRS, RULES=RULES, BIG_DELTA=BIG_DELTA, SEED=SEED, N_PERM=N_PERM, TETHER=dai1.TETHER))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "source2":
        out2 = out / "enc2"
        if (out2 / "source").exists():
            sys.exit(f"{out2 / 'source'} は既にある（重複を作らない）")
        dai1.ENC["enc"] = DD.ENC2
        dai1.RUN["commit"], dai1.RUN["dirty"] = commit, dirty
        dai1.stage_source(out2, DD.CORPORA)
        return
    fz = json.load(open(FROZEN))
    if args.stage in ("enc2k", "units") and not (out / "enc2" / "source" / "summary.csv").exists():
        sys.exit("第 2 エンコーダのソースの段（--stage source2）が先")
    if args.stage == "enc2k":
        files = sorted(UD.glob(f"{args.units}_p3.npz"))
        if args.units == "*" and len(files) != N_UNITS:
            sys.exit(f"入力の単位が {len(files)} 個（{N_UNITS} のはず）")
        if args.resume:
            files = [p for p in files if not (out / "rowsA" / f"{p.name[:-len('_p3.npz')]}_enc2k.csv").exists()]
            print(f"続き：残り {len(files)} 単位", flush=True)
        rows = [run_unit_A(p, out, fz, commit) for p in files]
        pd.DataFrame(rows).to_csv(out / f"unitsA_time_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
        return
    if args.stage == "select":
        stage_select(out, args.allow_partial, commit); return
    if args.stage == "units":
        sel = json.load(open(out / "select" / "enc2_tether.json"))
        loco = json.load(open(out / "select" / "none_loco_tether.json"))
        if sel["commit"] != commit or loco["commit"] != commit:
            sys.exit("選ぶ段のコミットが今の実行と違う")
        files = sorted(UD.glob(f"{args.units}_p3.npz"))
        if args.units == "*" and len(files) != N_UNITS:
            sys.exit(f"入力の単位が {len(files)} 個（{N_UNITS} のはず）")
        if args.resume:
            files = [p for p in files if not (out / "rows" / f"{p.name[:-len('_p3.npz')]}.npz").exists()]
            print(f"続き：残り {len(files)} 単位", flush=True)
        rows = [run_unit(p, out, fz, commit, sel, loco) for p in files]
        pd.DataFrame(rows).to_csv(out / f"units_time_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
        return
    if args.stage == "summarize":
        summarize(out, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
