"""実験 16 の下調べ（DAI-16．第 3 版）：ソースで測った「感情ごとの信頼度」と，ターゲットの K 本で測った水準を使って，
声・顔・言葉の分類器の答えを発話ごとにまとめる．まとめた答えが，等しい重みの積（今の提案の主の版の疑似ラベル）より当たるかを，
実験 1 の仮のターゲットの回の保存だけで測る．設計：docs/設計/実験16_下調べ_設計.md．評価の回の保存は開かない．

信頼度：分類器ごと・感情ごとの数．その感情の発話に，ほかの感情の発話より高い点を付ける割合（感情の対ごとの AUC の平均）を d′ に直したもの．
出どころ：home（ソースの中の交差検証），ext（home × 外での残り方），away（別のソースに当てた値），K（ターゲットの K 本），
          mix（水準は K 本，相性は ext．格子），orc（ターゲットの候補の全ラベル．参照）．
使い方：F2（感情で共通の重み），F3（先に疑似ラベル・柔らかい），F4（硬い），F5（繰り返す），F6（感情を順に仮定して比べる），
        F7（答えを表で読み替えて掛ける），F8（まとめ方の重みをソースどうしで学ぶ）．基準は等しい重みの積 prod．
分類器の組：M（微調整した声・顔・言葉の 3 つ），S（ソースごとの分類器），MS（両方．ソースの側に倍率 λ）．

段：
  selftest  人工の数での検算
  units     120 単位（T × 仮のターゲット c × 分割 × 種）．単位ごとに <out>/units/<名前>.npz と .csv
  summary   メタ学習の選択（評価するコーパスを含まない単位で格子の点を選ぶ），c × K ごとの UAR，差と区間，判定の数え，見積もりの当たり方

使い方：
  .venv/bin/python exp/dai16_probe.py --stage selftest
  .venv/bin/python exp/dai16_probe.py --stage units
  .venv/bin/python exp/dai16_probe.py --stage summary
試走：--out-dir <一時の置き場> --allow-dirty --allow-partial --targets AFEW --folds 0 --seeds 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402
import dai12_probe as P12  # noqa: E402
import dai15  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT_PROD = ROOT / "results" / "dai16_probe"
ADDENDUM = ROOT / "docs" / "設計" / "実験16・17_下調べ_K10_追補.md"
DESIGN = ROOT / "docs" / "設計" / "実験16_下調べ_設計.md"
KS = (5, 25)
K_MAIN = 5
NF = dai15.NF                           # K 本を感情ごとに分ける組の数（5）
D_MIN = 0.25                            # 表の対角の平均がこれ以下の分類器は使わない（F7 の表）
EPS = 1e-9                              # F7 の読み替えの前に表の全成分に足す数
AUC_CLIP = (0.02, 0.98)                 # d′ に直す前の AUC の範囲
CHECK_TEACHER = dai15.CHECK_TEACHER     # K 本を全部使って作り直した積と保存の積の差の許し
GRAD_TOL = 1e-5                         # 作り直しの勾配の最大の許し
FLOOR = 1.0                             # 「上回る」「下回る」の点推定の下限（点）
TAU = 1.0                               # 切片を K 本で決めるときの罰則の強さ
TAU_F8 = 0.01                           # F8 の重みを 1 へ引く罰則の強さ
W_MAX = 5.0                             # F8 の重みの上限（下限は 0）
N_ITER = 5                              # F5 の繰り返しの回数
N0S = (0.0, 20.0, 80.0, float("inf"))   # 水準を戻す強さ（ラベル何本ぶん）
BETAS = (0.0, 1.0)                      # 相性を使う強さ
PRIORS = ("eq", "ext")                  # 水準を戻す先（等しい重み，ソースで測った水準）
GRID = [(n0, b, p) for n0 in N0S for b in BETAS for p in PRIORS]    # 16 点
LAMS = (0.25, 1.0)                      # MS の組でソースの分類器に掛ける倍率
ADJ_ITERS = 3000
ADJ_TOL = 1e-6
ASSUME_FLAG = 0.05
N_UNITS = 120
SEED = 20261113
MODS = "AVT"
SCRIPTS = ("dai16_probe.py", "dai15.py", "dai12_probe.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
FORMS = ("F2", "F3", "F4", "F5", "F6")
RSRC = ("home", "ext", "away", "K", "orc")
TABLE_SPECS = {"orc_D": ("orc", "D", 0.0, False), "orc_F": ("orc", "F", 0.0, False), "orc_Dc": ("orc", "D", 0.0, True), "orc_Fc": ("orc", "F", 0.0, True),
               "cf_D_l05": ("cf", "D", 0.5, False), "cf_D_l05c": ("cf", "D", 0.5, True)}
V_M = (tuple(f"M_{f}_{r}" for f in FORMS for r in RSRC) + tuple(f"M_{f}_mix{i}" for f in ("F3", "F6") for i in range(len(GRID)))
       + tuple(f"M_{f}n_{r}" for f in ("F6", "F3") for r in RSRC) + tuple(f"M_{f}n_mix{i}" for f in ("F6", "F3") for i in range(len(GRID))))
# F6n・F3n：重みから「モダリティ × 感情」の成分を消した対照（分類器ごとの水準と，どの分類器にも共通の感情ごとの倍率だけを残す）
V_S = ("S_eq",) + tuple(f"S_{f}_{r}" for f in FORMS for r in RSRC) + tuple(f"S_{f}_mix{i}" for f in ("F3", "F6") for i in range(len(GRID)))
V_MS = tuple(f"MS_{x}_l{j}" for j in range(len(LAMS)) for x in ("eq", "F6_K", "F6_orc") + tuple(f"F6_mix{i}" for i in range(len(GRID))))
V_OTHER = ("M_F8", "S_F8") + tuple(TABLE_SPECS)
BASE = ("prod",) + V_M + V_S + V_MS + V_OTHER                          # 単位の段で作る版（層：raw・adj・kb）
EST_SEL = ("k_bias", "cf_D_l05", "M_F6_K", "M_F3_K", "M_F6_ext")       # 見込みで選ぶ版を単位の段で作る版
EST_NEST = EST_SEL[1:] + tuple(f"M_{f}_mix{i}" for f in ("F6", "F3") for i in range(len(GRID)))   # 入れ子で量を決め直す版
EST = ("k_bias",) + EST_NEST                                           # K 本での見積もりを出す版（格子の全部の点を含む）
SELV = tuple(f"sel_{v}" for v in EST_SEL) + ("sel_best",)              # 見込みで選ぶ版（層は adj だけ）
SEL_MIX = ("M_F6_mixsel", "M_F3_mixsel")                               # summary で，選んだ格子の点の見込みで選ぶ版を作る
MIXSEL = {"M_F6_mixsel": [f"M_F6_mix{i}" for i in range(len(GRID))], "M_F3_mixsel": [f"M_F3_mix{i}" for i in range(len(GRID))],
          "S_F6_mixsel": [f"S_F6_mix{i}" for i in range(len(GRID))], "S_F3_mixsel": [f"S_F3_mix{i}" for i in range(len(GRID))],
          "MS_F6_mixsel": [f"MS_F6_mix{i}_l{j}" for j in range(len(LAMS)) for i in range(len(GRID))]}   # メタ学習で格子の点を選ぶ版（summary で作る）
REFS = ("none_AVT", "joint_AVT", "ret_A", "ret_V", "ret_T", "prod_sel", "prod_ob", "k_bias")
TIED = {"M_F6n_mixsel": ("M_F6_mixsel", lambda v: v.replace("M_F6_", "M_F6n_")), "M_F3n_mixsel": ("M_F3_mixsel", lambda v: v.replace("M_F3_", "M_F3n_")),
        "MS_eq_sel": ("MS_F6_mixsel", lambda v: "MS_eq_l" + v[-1])}       # 選んだ格子の点を借りる版（対照と，同じ倍率 λ の等しい重み）
PER_EMO = ("prod", "prod_ob", "M_F6_orc", "M_F6_K", "M_F6_ext", "M_F3_K", "M_F6_mixsel", "M_F3_mixsel", "cf_D_l05", "none_AVT")
RUN = {"commit": "", "dirty": True}


# ------------------------------------------------------------------ 小さな道具

def lg(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, 1e-300, None))


def centered(P: np.ndarray) -> np.ndarray:
    """確率 (n, 4) → 感情の平均を引いた対数（切片をずらしても，発話どうしの順位は変わらない）．"""
    L = lg(P)
    return L - L.mean(1, keepdims=True)


def norm_log(L: np.ndarray) -> np.ndarray:
    return L - logsumexp(L, axis=1, keepdims=True)


def softmax(L: np.ndarray) -> np.ndarray:
    return np.exp(norm_log(L))


def bal_ll(lq: np.ndarray, y: np.ndarray) -> float:
    v = lq[np.arange(len(y)), y]
    return float(np.mean([v[y == k].mean() for k in range(4) if (y == k).any()]))


def match_shift(Lp: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """ラベルを使わずに，候補の上の softmax(Lp + δ) の平均が target になる切片のずれ δ を不動点の反復で求める（dai15.match_prior と同じ式）．"""
    d = np.zeros(4)
    lt = np.log(np.clip(target, 1e-12, None))
    for _ in range(ADJ_ITERS):
        m = softmax(Lp + d).mean(0)
        if np.abs(m - target).max() < ADJ_TOL * 1e-2:
            break
        d = d + lt - np.log(np.clip(m, 1e-12, None))
        d = d - d.mean()
    return d, float(np.abs(softmax(Lp + d).mean(0) - target).max())


# ------------------------------------------------------------------ 信頼度（感情の対ごとの AUC の平均 → d′）

def auc_pair(pos: np.ndarray, neg: np.ndarray) -> float:
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def dprime(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """分類器の確率 P (n, 4) と本当の感情 y から，感情 c ごとの信頼度を出す．
    c の点（感情の平均を引いた対数）について，本当が c の発話と本当が j の発話の AUC を j ≠ c で平均し，d′ = √2 Φ⁻¹(AUC) に直す（負は 0）．
    感情の割合にも切片にも依らない．発話が無い感情は AUC 0.5（d′ 0）．戻り：(d′ (4,), AUC (4,))．"""
    Lc = centered(P)
    a = np.full(4, 0.5)
    A = np.full((4, 4), np.nan)                                            # A[c, j]：感情 c の点での，本当が c の発話と本当が j の発話の AUC
    for c in range(4):
        pos = Lc[y == c, c]
        if len(pos):
            for j in range(4):
                if j != c and (y == j).any():
                    A[c, j] = auc_pair(pos, Lc[y == j, c])
            if np.isfinite(A[c]).any():
                a[c] = float(np.nanmean(A[c]))
    d = np.maximum(0.0, math.sqrt(2.0) * norm.ppf(np.clip(a, *AUC_CLIP)))
    return d, a, A


def mix_rel(DK: np.ndarray, Dext: np.ndarray, n: int, n0: float, beta: float, prior: str) -> np.ndarray:
    """水準は K 本で測った値を戻し先へ寄せ，相性は ext から取る．DK・Dext：(分類器, 4)．n：ラベルの本数（4K）．
    水準 L = (n·L_K + n0·L_0) / (n + n0)，L_0 は prior が eq なら L_K の分類器の平均，ext なら ext の水準．信頼度 = max(0, L + beta·相性)．"""
    LK = DK.mean(1)
    L0 = np.full_like(LK, LK.mean()) if prior == "eq" else Dext.mean(1)
    w = 0.0 if math.isinf(n0) else n / (n + n0)
    L = w * LK + (1.0 - w) * L0
    prof = Dext - Dext.mean(1, keepdims=True)
    return np.maximum(0.0, L[:, None] + beta * prof)


def tukey_fit(X: np.ndarray) -> np.ndarray:
    """表 (分類器, 4) を「全体の平均 ＋ 行の差 a ＋ 列の差 b ＋ λ a b」で当てはめた値（Tukey 1949 の非加法性の 1 自由度）．
    表が「水準 × 感情の当てやすさ」の掛け算の形（どの感情でも分類器の比が同じ）なら，当てはめは表と一致する．"""
    g = X.mean()
    a, b = X.mean(1) - g, X.mean(0) - g
    den = float((a ** 2).sum() * (b ** 2).sum())
    lam = float((a[:, None] * b[None, :] * X).sum() / den) if den > 1e-12 else 0.0
    return g + a[:, None] + b[None, :] + lam * a[:, None] * b[None, :]


def noint(W: np.ndarray) -> np.ndarray:
    """重みから「分類器 × 感情」の成分を消す：tukey_fit の当てはめ（負は 0）．残るのは，分類器ごとの水準，どの分類器にも共通の感情ごとの倍率，
    その 2 つの掛け算の成分だけである（どの感情でも分類器どうしの比がほぼ同じ重み）．"""
    return np.maximum(0.0, tukey_fit(W))


def dcenter(X: np.ndarray) -> np.ndarray:
    """分類器 × 感情の成分：表から tukey_fit の当てはめを引いた残り．"""
    return X - tukey_fit(X)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else float("nan")


def perm_share(A: np.ndarray, B: np.ndarray) -> float:
    """A の感情の並びを 24 通りに入れ替えたとき，dcenter の相関が元の並びの値以上になる割合（元の並びを含む．偶然の水準の目安）．"""
    obs = corr(dcenter(A), dcenter(B))
    if not np.isfinite(obs):
        return float("nan")
    vals = [corr(dcenter(A[:, list(p)]), dcenter(B)) for p in itertools.permutations(range(4))]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean([v >= obs - 1e-12 for v in vals]))


def to_w(D: np.ndarray, ref: float | None = None) -> np.ndarray:
    """信頼度 → 重み（平均が 1 になるように割る．ref を渡すとその値で割る）．平均が 0 なら全部 1．"""
    m = float(D.mean()) if ref is None else ref
    return np.ones_like(D) if m <= 0 else D / m


# ------------------------------------------------------------------ ソースで測る信頼度（home・away．コーパスごとに 1 回）

HOME: dict = {}
HOMEF: dict = {}
HOMEA: dict = {}
AWAY: dict = {}
AWAYA: dict = {}
SRCP: dict = {}


def home_rel(c: str, m: str) -> np.ndarray:
    """コーパス c の中の交差検証（実験 1 の話者の分割 5 つ．罰則の強さは実験 1 の段 source が選んだ値）で測った，モダリティ m の分類器の信頼度．"""
    if (c, m) not in HOME:
        cp = dai1.Corpus.get(c)
        X, y = cp.X[m], cp.y
        C = float(np.load(OUT1 / "source" / f"{c}_{m}.npz")["C"])
        ds, As = [], []
        for f in range(DD.N_FOLDS):
            tr = np.flatnonzero(DD.pool_mask(cp.df, f))
            te = np.flatnonzero(cp.df["fold"].to_numpy() == f)
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
            Xs = (X - mu) / sd
            G = S.balanced_onehot(y[tr], np.arange(len(tr)), len(tr))[None]
            r = S.fit(Xs[tr], G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=np.array([C]), Cb=math.inf)
            if float(r["grad_max"]) > GRAD_TOL:
                raise SystemExit(f"home_rel：{c} {m} 分割 {f} の勾配の最大が {float(r['grad_max']):.1e}")
            P = torch.softmax(S.logits(Xs[te], r["W"][0], r["b"][0]), -1).cpu().numpy().astype(np.float64)
            rr = dprime(P, y[te])
            ds.append(rr[0]); As.append(rr[2])
        HOME[(c, m)] = np.mean(ds, 0)
        HOMEF[(c, m)] = np.stack(ds)                                          # 分割ごとの値 (5, 4)（記録）
        with np.errstate(all="ignore"):
            HOMEA[(c, m)] = np.nanmean(np.stack(As), 0)                        # 感情の対ごとの AUC の分割の平均 (4, 4)（記録）
    return HOME[(c, m)]


def src_probs_on(s: str, t: str, m: str) -> np.ndarray:
    """ソース s の全件で学んだ分類器（実験 1 の段 source）を，コーパス t の全発話（t の平均と標準偏差で標準化）に当てた確率．"""
    if (s, t, m) not in SRCP:
        cp = dai1.Corpus.get(t)
        X = cp.X[m]
        Xs = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
        W, b = dai1.load_source(OUT1, s, m)
        SRCP[(s, t, m)] = dai1.probs(Xs, W, b).cpu().numpy().astype(np.float64)
    return SRCP[(s, t, m)]


def away_rel(s: str, t: str, m: str) -> np.ndarray:
    if (s, t, m) not in AWAY:
        rr = dprime(src_probs_on(s, t, m), dai1.Corpus.get(t).y)
        AWAY[(s, t, m)], AWAYA[(s, t, m)] = rr[0], rr[2]
    return AWAY[(s, t, m)]


def fit_wb(Ls: np.ndarray, y: np.ndarray, w0: np.ndarray, tau: float, fit_w: bool = True, fit_b: bool = False, sw: np.ndarray | None = None):
    """log q(i | x) ∝ Σ_k w_k Ls[k, x, i] + b_i．重み sw（既定は感情ごとに等しい重み．和は 1）の負の対数尤度 ＋ (tau/2)(‖w − w0‖² ＋ ‖b‖²) を最小にする．
    0 ≤ w ≤ W_MAX．戻り：(w, b)．凸の問．"""
    k, n, _ = Ls.shape
    if sw is None:
        cnt = np.bincount(y, minlength=4).astype(float)
        sw = 1.0 / (cnt[y] * (cnt > 0).sum())
    Y = np.eye(4)[y]
    nw = k if fit_w else 0

    def unpack(th):
        return (th[:nw] if fit_w else w0), (th[nw:] if fit_b else np.zeros(4))

    def fun(th):
        w, b = unpack(th)
        Z = np.tensordot(w, Ls, axes=1) + b
        lp = Z - logsumexp(Z, axis=1, keepdims=True)
        f = -float((sw * lp[np.arange(n), y]).sum()) + 0.5 * tau * (float(((w - w0) ** 2).sum()) * fit_w + float((b ** 2).sum()) * fit_b)
        D = (np.exp(lp) - Y) * sw[:, None]
        g = []
        if fit_w:
            g.append(np.einsum("kni,ni->k", Ls, D) + tau * (w - w0))
        if fit_b:
            g.append(D.sum(0) + tau * b)
        return f, np.concatenate(g)

    th0 = np.concatenate(([w0] if fit_w else []) + ([np.zeros(4)] if fit_b else []))
    bounds = [(0.0, W_MAX)] * nw + [(None, None)] * (4 if fit_b else 0)
    r = minimize(fun, th0, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8))
    g = fun(r.x)[1]
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds]); hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
    free = ~(((r.x <= lo + 1e-9) & (g > 0)) | ((r.x >= hi - 1e-9) & (g < 0)))    # 範囲の端で外向きの勾配は数えない
    if free.any() and float(np.abs(g[free]).max()) > 1e-3:
        raise SystemExit(f"fit_wb：勾配の最大が {float(np.abs(g[free]).max()):.1e}")
    return unpack(r.x)


F8CACHE: dict = {}


def source_stats(srcs: list) -> dict:
    """単位のソースの組だけで決まる量：home・away・残り方・ext（M と S の並び）と，F8 の重み（ソースどうしで学ぶ）．"""
    key = tuple(srcs)
    if key in F8CACHE:
        return F8CACHE[key]
    pairs = [(s, t) for s in srcs for t in srcs if s != t]
    home = {(s, m): home_rel(s, m) for s in srcs for m in MODS}
    away = {(s, m): np.mean([away_rel(s, t, m) for t in srcs if t != s], 0) for s in srcs for m in MODS}
    ret = {m: float(np.clip(sum(away_rel(s, t, m).sum() for s, t in pairs) / max(sum(home[(s, m)].sum() for s, t in pairs), 1e-12), 0.0, 1.5)) for m in MODS}
    out = {"ret": np.array([ret[m] for m in MODS])}
    re_ = np.full((3, 4), np.nan)                                         # 感情ごとの残り方（記録．ext は残り方が感情で共通と置く）
    for j_, m in enumerate(MODS):
        for i in range(4):
            hsum = sum(home[(s, m)][i] for s, t in pairs)
            if hsum > 0.05 * len(pairs):                                      # 大本の d′ の平均が 0.05 以下の感情は出さない（比が不安定）
                re_[j_, i] = float(np.clip(sum(away_rel(s, t, m)[i] for s, t in pairs) / hsum, 0.0, 3.0))
    out["ret_emo"] = re_
    out["S_home"] = np.stack([home[(s, m)] for s in srcs for m in MODS])
    out["S_homefolds"] = np.stack([HOMEF[(s, m)] for s in srcs for m in MODS])
    out["S_homeauc"] = np.stack([HOMEA[(s, m)] for s in srcs for m in MODS])
    with np.errstate(all="ignore"):
        out["S_awayauc"] = np.stack([np.nanmean(np.stack([AWAYA[(s, t, m)] for t in srcs if t != s]), 0) for s in srcs for m in MODS])
    out["S_ext"] = np.stack([home[(s, m)] * ret[m] for s in srcs for m in MODS])
    out["S_away"] = np.stack([away[(s, m)] for s in srcs for m in MODS])
    for nm in ("home", "ext", "away"):
        out[f"M_{nm}"] = np.stack([np.mean([out[f"S_{nm}"][i * 3 + j] for i in range(len(srcs))], 0) for j in range(3)])
    # F8：ソース s の 3 つの分類器をソース t に当て，感情ごとの重み w[m, 感情]（12 個．1 へ引く）と切片を，向き (s → t) ごとの重みの和を等しくして学ぶ
    Ls, ys, sws = [], [], []
    for s, t in pairs:
        y = dai1.Corpus.get(t).y
        L3 = [centered(src_probs_on(s, t, m)) for m in MODS]
        L12 = np.zeros((12, len(y), 4))
        for j in range(3):
            for i in range(4):
                L12[j * 4 + i][:, i] = L3[j][:, i]
        cnt = np.bincount(y, minlength=4).astype(float)
        Ls.append(L12); ys.append(y); sws.append(1.0 / (cnt[y] * (cnt > 0).sum() * len(pairs)))
    w, _ = fit_wb(np.concatenate(Ls, 1), np.concatenate(ys), np.ones(12), TAU_F8, fit_w=True, fit_b=True, sw=np.concatenate(sws))
    out["F8"] = w.reshape(3, 4)
    F8CACHE[key] = out
    return out


# ------------------------------------------------------------------ F7 の表（第 2 版と同じ）

def shape_S(C):
    return dai15.shrink(C, 1.0)


def shape_D(C):
    d = np.diag(C)
    out = np.repeat(((1.0 - d) / 3.0)[:, None], 4, axis=1)
    np.fill_diagonal(out, d)
    return out


def shaped(C, shape, lam=0.0):
    if float(np.mean(np.diag(C))) <= D_MIN:
        return np.full((4, 4), 0.25), True
    base = {"S": shape_S(C), "D": shape_D(C), "F": C}[shape]
    return (1.0 - lam) * base + lam * shape_S(C), False


def posterior_matrix(C):
    """B[j, i] = C[i, j] / Σ_i′ C[i′, j]（「j と答えたとき本当が i である確率」）．"""
    Ce = C + EPS
    Ce = Ce / Ce.sum(1, keepdims=True)
    return (Ce / Ce.sum(0, keepdims=True)).T


def reread(P, C):
    return P @ posterior_matrix(C)


def table_set(C: dict, shp: str, lam: float, common: bool = False):
    if common:
        Cb = np.mean([C[m] for m in MODS], 0)
        C = {m: Cb for m in MODS}
    out, nun = {}, 0
    for m in MODS:
        out[m], un = shaped(C[m], shp, lam)
        nun += int(un)
    return out, nun


# ------------------------------------------------------------------ まとめ方

def combine(form: str, Ls: np.ndarray, W: np.ndarray, q1: np.ndarray | None = None) -> np.ndarray:
    """Ls：分類器の点（感情の平均を引いた対数．(分類器, n, 4)）．W：重み (分類器, 4)．q1：先に当てる疑似ラベルの確率 (n, 4)．
    F2：感情で共通の重み（W の行の平均）．F3：q1 の割合で W の行を混ぜて発話ごとの重みにする．F4：q1 の最大の感情で W を引く．
    F5：F3 を，まとめた答えを次の疑似ラベルにして N_ITER 回繰り返す．F6：感情 i の点は W の i 列で重みづける．戻り：正規化した対数確率．"""
    if form == "F6":
        return norm_log(np.einsum("ky,kny->ny", W, Ls))
    if form == "F2":
        return norm_log(np.tensordot(W.mean(1), Ls, axes=1))
    q = np.eye(4)[q1.argmax(1)] if form == "F4" else q1
    for _ in range(N_ITER if form == "F5" else 1):
        out = norm_log(np.einsum("nk,kny->ny", q @ W.T, Ls))
        q = np.exp(out)
    return out


class View:
    """発話の集まり 1 つの上の分類器の確率．ret：{m: (n, 4)}（微調整した声・顔・言葉），src：{(s, m): (n, 4)}（ソースごとの分類器）．"""

    def __init__(self, ret: dict, src: dict, srcs: list):
        self.ret, self.src, self.srcs = ret, src, list(srcs)
        self.LM = np.stack([centered(ret[m]) for m in MODS])
        self.LS = np.stack([centered(src[(s, m)]) for s in self.srcs for m in MODS])
        self.prod = norm_log(self.LM.sum(0))
        self.q1 = np.exp(self.prod)

    def take(self, idx: np.ndarray) -> "View":
        return View({m: self.ret[m][idx] for m in MODS}, {k: v[idx] for k, v in self.src.items()}, self.srcs)

    def L(self, fam: str) -> np.ndarray:
        return self.LM if fam == "M" else (self.LS if fam == "S" else np.concatenate([self.LM, self.LS]))


def score(view: View, spec: tuple) -> np.ndarray:
    """spec：("w", 組, 形, W) か ("tab", 表の辞書)．戻り：正規化した対数確率 (n, 4)．"""
    if spec[0] == "w":
        out = combine(spec[2], view.L(spec[1]), spec[3], view.q1)
    else:
        out = norm_log(sum(lg(reread(view.ret[m], spec[1][m])) for m in MODS))
    if not np.isfinite(out).all():
        raise SystemExit("score：まとめた点が有限でない")
    return out


def rel_K(view: View, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([dprime(view.ret[m], y)[0] for m in MODS]), np.stack([dprime(view.src[(s, m)], y)[0] for s in view.srcs for m in MODS]))


def build_specs(D: dict, nsrc: int, C_cf: dict, C_orc: dict | None, F8: np.ndarray | None, only: tuple | None = None) -> dict:
    """版の名前 → spec．D：{"M": {出どころ: (3, 4)}, "S": {出どころ: (3·ソースの数, 4)}}．only を渡すとその版だけ作る．"""
    sp = {}
    for fam in ("M", "S"):
        for r in D[fam]:
            W = to_w(D[fam][r])
            for f in (("F3", "F6") if r.startswith("mix") else FORMS):
                sp[f"{fam}_{f}_{r}"] = ("w", fam, f, W)
            if fam == "M":
                sp[f"M_F6n_{r}"] = ("w", "M", "F6", noint(W))
                sp[f"M_F3n_{r}"] = ("w", "M", "F3", noint(W))
    sp["prod"] = ("w", "M", "F2", np.ones((3, 4)))
    sp["S_eq"] = ("w", "S", "F2", np.ones((3 * nsrc, 4)))
    for j, lam in enumerate(LAMS):
        sp[f"MS_eq_l{j}"] = ("w", "MS", "F2", np.concatenate([np.ones((3, 4)), lam * np.ones((3 * nsrc, 4))]))
        for r in D["M"]:
            if r in ("K", "orc") or r.startswith("mix"):
                ref = float(D["M"][r].mean())
                W = np.concatenate([to_w(D["M"][r]), lam * to_w(D["S"][r], ref if ref > 0 else None)])
                sp[f"MS_F6_{r}_l{j}"] = ("w", "MS", "F6", W)
    if F8 is not None:
        sp["M_F8"] = ("w", "M", "F6", F8)
        sp["S_F8"] = ("w", "S", "F6", np.tile(F8, (nsrc, 1)))
    for v, (src, shp, lam, common) in TABLE_SPECS.items():
        C = C_cf if src == "cf" else C_orc
        if C is not None:
            sp[v] = ("tab", table_set(C, shp, lam, common)[0])
    return sp if only is None else {k: v for k, v in sp.items() if k in only}


# ------------------------------------------------------------------ 交差の推定（実験 1 の段 1・2 と同じ手順）

def teachers_mod(u, tether: dict, K: int, kid_sets: list) -> tuple[dict, list]:
    """K 本の部分集合ごとに，融合の K 本だけの分類器 → その確率で声・顔・言葉を学び直す．戻り：({m: 候補の確率 (部分集合, n, 4)}, [勾配の最大, 止まり])．"""
    n = u.n
    G0 = torch.stack([S.balanced_onehot(u.ypool, ks, n) for ks in kid_sets])
    W0, b0 = u.start["AVT"]
    cw, cb = tether[("AVT", K)]
    r = S.fit(u.Xp["AVT"], G0, W0, b0, Cw=cw, Cb=cb)
    if not r["finite_p"].all():
        raise SystemExit("交差の推定の作り直し（融合）が有限でない")
    conv = [float(r["grad_max"]), int(r["stalled"])]
    F = torch.softmax(S.logits(u.Xp["AVT"], r["W"], r["b"]), -1)
    out = {}
    for m in MODS:
        G = G0 + torch.stack([S.soft_rows(F[j], float(n)) for j in range(len(kid_sets))])
        W0, b0 = u.start[m]
        cw, cb = tether[(m, K)]
        rm = S.fit(u.Xp[m], G, W0, b0, Cw=cw, Cb=cb)
        if not rm["finite_p"].all():
            raise SystemExit(f"交差の推定の作り直し（{m}）が有限でない")
        conv = [max(conv[0], float(rm["grad_max"])), conv[1] + int(rm["stalled"])]
        out[m] = torch.softmax(S.logits(u.Xp[m], rm["W"], rm["b"]), -1).cpu().numpy().astype(np.float64)
    if conv[0] > GRAD_TOL:
        raise SystemExit(f"交差の推定の作り直しの勾配の最大が {conv[0]:.1e}（K={K}）")
    return out, conv


# ------------------------------------------------------------------ 1 単位

def run_unit(sv) -> tuple[list, dict]:
    u = sv.u
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    rows, arr = [], {}
    yt, yp = u.ytest, u.ypool
    nsrc = len(u.srcs)
    arr["y_test"] = yt.astype(np.int8)
    arr["cl_test"] = sv.cl
    arr["sid_test"] = sv.sid
    arr["sid_pool"] = sv.sid_pool
    arr["cl_pool"] = sv.cl_pool
    arr["y_pool"] = yp.astype(np.int8)
    arr["srcs"] = np.array(u.srcs)
    arr["grid"] = np.array([f"{n0}|{b}|{p}" for n0, b, p in GRID])
    SR = source_stats(u.srcs)
    for nm in ("ret", "ret_emo", "S_home", "S_homefolds", "S_homeauc", "S_awayauc", "S_ext", "S_away", "M_home", "M_ext", "M_away", "F8"):
        arr[f"SR_{nm}"] = SR[nm]
    src_p = {(s, m): dai1.probs(u.Xp[m], W, b).cpu().numpy().astype(np.float64) for m in MODS for s, W, b in u.src[m]}
    src_t = {(s, m): dai1.probs(u.Xt[m], W, b).cpu().numpy().astype(np.float64) for m in MODS for s, W, b in u.src[m]}
    for (s, m), P in src_p.items():
        arr[f"p_pool_src_{s}_{m}"] = P.astype(np.float32)
        arr[f"p_test_src_{s}_{m}"] = src_t[(s, m)].astype(np.float32)
    for K in KS:
        if sv.fz["main"][str(K)] != "one":
            raise SystemExit(f"{sv.base.name}：実験 1 の主の版が one でない（K={K}）")
        kpos = u.kpos[K]
        yk = yp[kpos]
        nk = np.setdiff1d(np.arange(u.n), kpos)
        ynk = yp[nk]
        pool = View({m: TK[K][f"ret_{m}"][0].cpu().numpy().astype(np.float64) for m in MODS}, src_p, u.srcs)
        test = View({m: TK[K][f"ret_{m}"][1].cpu().numpy().astype(np.float64) for m in MODS}, src_t, u.srcs)
        pnk = pool.take(nk)
        for m in MODS:
            arr[f"p_pool_ret_{m}_K{K}"] = pool.ret[m].astype(np.float32)
            arr[f"p_test_ret_{m}_K{K}"] = test.ret[m].astype(np.float32)
        q0p = dai1.product(TK[K], (1, 1, 1), 0).cpu().numpy().astype(np.float64)
        if float(np.abs(pool.q1 - q0p).max()) > 1e-6:
            raise SystemExit(f"{sv.base.name}：点の和で作った積が保存の積と違う（K={K}）")
        # ---- 交差の推定：K 本の全部，1 組を抜いたもの 5 つ，2 組を抜いたもの 10
        fold = dai15.kfold_of(yk)
        assert all(np.bincount(fold[yk == i], minlength=NF).min() >= 1 for i in range(4)), "組に感情が欠ける"
        pairs = list(itertools.combinations(range(NF), 2))
        sets = [kpos] + [kpos[fold != j] for j in range(NF)] + [kpos[(fold != i) & (fold != j)] for i, j in pairs]
        pidx = {p: 1 + NF + n for n, p in enumerate(pairs)}
        Pm, conv = teachers_mod(u, sv.teth, K, sets)
        dev = float(np.abs(softmax(sum(lg(Pm[m][0]) for m in MODS)) - q0p).max())
        dev_m = {f"teacher_dev_{m}": float(np.abs(Pm[m][0] - pool.ret[m]).max()) for m in MODS}
        if dev > CHECK_TEACHER:
            raise SystemExit(f"{sv.base.name}：K 本を全部使って作り直した積が保存の積と {dev:.2e} 違う（K={K}）")
        held = View({m: np.stack([Pm[m][1 + fold[a]][kpos[a]] for a in range(len(kpos))]) for m in MODS}, {k: v[kpos] for k, v in src_p.items()}, u.srcs)
        # ---- 信頼度の表
        D = {"M": {r: SR[f"M_{r}"] for r in ("home", "ext", "away")}, "S": {r: SR[f"S_{r}"] for r in ("home", "ext", "away")}}
        D["M"]["K"], D["S"]["K"] = rel_K(held, yk)
        D["M"]["orc"], D["S"]["orc"] = rel_K(pnk, ynk)
        for fam in ("M", "S"):
            for i, (n0, beta, prior) in enumerate(GRID):
                D[fam][f"mix{i}"] = mix_rel(D[fam]["K"], D[fam]["ext"], 4 * K, n0, beta, prior)
            for r in ("K", "orc"):
                arr[f"D_{fam}_{r}_K{K}"] = D[fam][r]
        arr[f"AUC_M_K_K{K}"] = np.stack([dprime(held.ret[m], yk)[2] for m in MODS])       # 感情の対ごとの AUC (3, 4, 4)（記録）
        arr[f"AUC_M_orc_K{K}"] = np.stack([dprime(pnk.ret[m], ynk)[2] for m in MODS])
        C_cf = {m: dai15.table_from(held.ret[m], yk) for m in MODS}
        C_orc = {m: dai15.table_from(pnk.ret[m], ynk) for m in MODS}
        specs = build_specs(D, nsrc, C_cf, C_orc, SR["F8"])
        unused = {v: table_set(C_cf if src == "cf" else C_orc, shp, lam, common)[1] for v, (src, shp, lam, common) in TABLE_SPECS.items()}
        assert set(specs) == set(BASE), set(specs) ^ set(BASE)
        # ---- 版ごとの判定：raw（そのまま），adj（候補の上の確率の平均を積に合わせる．ラベルを使わない），kb（adj の上に K 本で決めた切片を足す），
        #      ob（参照：候補の全ラベルで決めた切片を raw に足す．切片を最良にそろえて，向きの良さだけを比べる層）
        target = pool.q1.mean(0)
        lt_adj, resid_max = {}, 0.0
        for v in BASE:
            St, Sp, Sh = score(test, specs[v]), score(pool, specs[v]), score(held, specs[v])
            dl, rs = match_shift(Sp, target)
            resid_max = max(resid_max, rs)
            bk = fit_wb((Sh + dl)[None], yk, np.ones(1), TAU, fit_w=False, fit_b=True)[1]
            bo = fit_wb(Sp[nk][None], ynk, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
            pr, pa, pk, pp, po = St.argmax(1), (St + dl).argmax(1), (St + dl + bk).argmax(1), (Sp + dl)[nk].argmax(1), (St + bo).argmax(1)
            if v == "prod":
                prod_ob_t, prod_ob_p, b_orc = po, (Sp + bo)[nk].argmax(1), bo
            lt_adj[v] = norm_log(St + dl)
            arr[f"pred_raw_K{K}_{v}"] = pr.astype(np.int8)
            arr[f"pred_adj_K{K}_{v}"] = pa.astype(np.int8)
            arr[f"pred_kb_K{K}_{v}"] = pk.astype(np.int8)
            arr[f"pred_pool_K{K}_{v}"] = pp.astype(np.int8)
            arr[f"pred_ob_K{K}_{v}"] = po.astype(np.int8)
            rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, pr), uar_adj=100 * S.uar(yt, pa), uar_kb=100 * S.uar(yt, pk), uar_ob=100 * S.uar(yt, po),
                             uar_pool=100 * S.uar(ynk, pp), ll_adj=bal_ll(lt_adj[v], yt), adj_resid=rs, n_unused=unused.get(v, 0)))
        # ---- K 本での見積もり（入れ子）：組 j の発話は，組 j を抜いて作った確率と，組 j のラベルを使わずに決めた量で当てる
        est_lq = {v: np.empty((len(kpos), 4)) for v in ("prod",) + EST}
        for j in range(NF):
            te, tr = np.flatnonzero(fold == j), np.flatnonzero(fold != j)
            assert not np.isin(kpos[te], sets[1 + j]).any(), "組 j の発話が，組 j を抜いた分類器の学習に入っている"
            for a in tr:                                                        # 量を決める確率を作る分類器は，組 j の発話も，その発話自身も学習に使わない
                sx = sets[pidx[tuple(sorted((int(fold[a]), j)))]]
                assert not np.isin(kpos[te], sx).any() and kpos[a] not in sx, "入れ子の交差に漏れがある"
            trv = View({m: np.stack([Pm[m][pidx[tuple(sorted((int(fold[a]), j)))]][kpos[a]] for a in tr]) for m in MODS},
                       {k: v[kpos[tr]] for k, v in src_p.items()}, u.srcs)
            hv = held.take(te)
            pool_j = View({m: Pm[m][1 + j] for m in MODS}, src_p, u.srcs)   # 組 j を抜いて作った分類器の，候補の上の確率
            tgt_j = pool_j.q1.mean(0)
            DKj = rel_K(trv, yk[tr])
            Dj = {"M": {"K": DKj[0], "ext": D["M"]["ext"]}, "S": {"K": DKj[1], "ext": D["S"]["ext"]}}
            for i, (n0, beta, prior) in enumerate(GRID):                          # 水準を戻す式のラベルの本数は，組 j を除いた本数
                Dj["M"][f"mix{i}"] = mix_rel(Dj["M"]["K"], Dj["M"]["ext"], len(tr), n0, beta, prior)
                Dj["S"][f"mix{i}"] = mix_rel(Dj["S"]["K"], Dj["S"]["ext"], len(tr), n0, beta, prior)
            sj = build_specs(Dj, nsrc, {m: dai15.table_from(trv.ret[m], yk[tr]) for m in MODS}, None, None, only=EST_NEST)
            est_lq["prod"][te] = hv.prod
            est_lq["k_bias"][te] = norm_log(hv.prod + fit_wb(trv.prod[None], yk[tr], np.ones(1), TAU, fit_w=False, fit_b=True)[1])
            for v in EST_NEST:
                dl, _ = match_shift(score(pool_j, sj[v]), tgt_j)
                est_lq[v][te] = norm_log(score(hv, sj[v]) + dl)
        est = {v: (bal_ll(L, yk), 100 * S.uar(yk, L.argmax(1))) for v, L in est_lq.items()}
        gain = {v: est[v][0] - est["prod"][0] for v in EST}
        tpred = {v: arr[f"pred_adj_K{K}_{v}"] for v in EST if v != "k_bias"}
        for v in EST:                                                           # 見積もりは格子の全部の点について残す（選んだ点の見込みを後から引くため）
            rows.append(dict(K=K, version=f"est_{v}", est_ll=est[v][0], est_uar=est[v][1], est_gain=gain[v],
                             uar_base=100 * S.uar(yt, arr[f"pred_kb_K{K}_prod"] if v == "k_bias" else tpred[v])))
        tpred["k_bias"] = arr[f"pred_kb_K{K}_prod"]
        tpred["prod"] = arr[f"pred_raw_K{K}_prod"]
        best = max(("prod",) + EST_SEL, key=lambda v: (est[v][0], v == "prod"))            # 同点は積
        for v in EST_SEL:
            use = v if gain[v] > 0 else "prod"
            arr[f"pred_adj_K{K}_sel_{v}"] = tpred[use]
            rows.append(dict(K=K, version=f"sel_{v}", uar_adj=100 * S.uar(yt, tpred[use]), est_ll=est[v][0], est_uar=est[v][1], est_gain=gain[v],
                             used=str(gain[v] > 0), uar_base=100 * S.uar(yt, tpred[v])))
        arr[f"pred_adj_K{K}_sel_best"] = tpred[best]
        rows.append(dict(K=K, version="sel_best", uar_adj=100 * S.uar(yt, tpred[best]), used=best, est_ll=est[best][0]))
        rows.append(dict(K=K, version="_est_prod", est_ll=est["prod"][0], est_uar=est["prod"][1]))
        # ---- 参照
        Wj, bj = sv.prop("AVT", K)
        refs_t = {"none_AVT": TK[K]["none_AVT"][1].argmax(1).cpu().numpy(), "joint_AVT": S.logits(u.Xt["AVT"], Wj, bj).argmax(1).cpu().numpy(),
                  "prod_sel": sum(w * lg(test.ret[m]) for w, m in zip(sv.wsel[K], MODS)).argmax(1)}
        refs_p = {"none_AVT": TK[K]["none_AVT"][0].argmax(1).cpu().numpy()[nk], "joint_AVT": S.logits(u.Xp["AVT"], Wj, bj).argmax(1).cpu().numpy()[nk],
                  "prod_sel": sum(w * lg(pnk.ret[m]) for w, m in zip(sv.wsel[K], MODS)).argmax(1)}
        for m in MODS:
            refs_t[f"ret_{m}"], refs_p[f"ret_{m}"] = test.ret[m].argmax(1), pnk.ret[m].argmax(1)
        refs_t["prod_ob"], refs_p["prod_ob"] = prod_ob_t, prod_ob_p               # 積に，候補の全ラベルで決めた切片を足したもの（切片だけの対照）
        refs_t["k_bias"], refs_p["k_bias"] = arr[f"pred_kb_K{K}_prod"], arr[f"pred_pool_K{K}_prod"]   # 積 ＋ K 本の切片（候補の上の判定は積のもの）
        for v in REFS:
            arr[f"pred_raw_K{K}_{v}"] = refs_t[v].astype(np.int8)
            arr[f"pred_pool_K{K}_{v}"] = refs_p[v].astype(np.int8)
            rows.append(dict(K=K, version=v, uar_raw=100 * S.uar(yt, refs_t[v]), uar_pool=100 * S.uar(ynk, refs_p[v])))
        # ---- 保存：表，抜いた確率，K 本
        for m in MODS:
            arr[f"C_cf_{m}_K{K}"], arr[f"C_orc_{m}_K{K}"] = C_cf[m], C_orc[m]
            arr[f"pcf_{m}_K{K}"] = held.ret[m].astype(np.float32)
            arr[f"pcf2_{m}_K{K}"] = np.stack([Pm[m][pidx[pq]][kpos] for pq in pairs]).astype(np.float32)   # 2 組を抜いて作った分類器の，K 本の上の確率 (10, 4K, 4)．学習の外なのは抜いた 2 組の行だけ
            arr[f"ppj_{m}_K{K}"] = np.stack([Pm[m][1 + j] for j in range(NF)]).astype(np.float32)          # 組 j を抜いて作った分類器の，候補の上の確率 (5, 候補, 4)
        arr[f"b_orc_K{K}"] = b_orc
        arr[f"kpos_K{K}"] = kpos.astype(np.int32)
        arr[f"kfold_K{K}"] = fold.astype(np.int8)
        # ---- 仮定の表
        Dt_M = np.stack([dprime(test.ret[m], yt)[0] for m in MODS])
        prof = lambda X: X - X.mean(1, keepdims=True)  # noqa: E731
        ass = {}
        for fam in ("M", "S"):
            for r in ("home", "ext", "away", "K"):
                ass[f"intcorr_{fam}_{r}"] = corr(dcenter(D[fam][r]), dcenter(D[fam]["orc"]))          # 分類器 × 感情の成分だけの相関（R0 の主の量）
                ass[f"intperm_{fam}_{r}"] = perm_share(D[fam][r], D[fam]["orc"])
                ass[f"profcorr_{fam}_{r}"] = corr(prof(D[fam][r]), prof(D[fam]["orc"]))                # 共通の感情の難しさを含む相性の相関（記録）
                ass[f"emocorr_{fam}_{r}"] = corr(D[fam][r].mean(0), D[fam]["orc"].mean(0))             # どの分類器にも共通の感情ごとの成分の相関（記録）
                ass[f"levelcorr_{fam}_{r}"] = corr(D[fam][r].mean(1), D[fam]["orc"].mean(1))
                ass[f"topok_{fam}_{r}"] = float(D[fam][r].mean(1).argmax() == D[fam]["orc"].mean(1).argmax())
        wsd = [float(np.sqrt(np.mean([pnk.LM[k_][ynk == c_].var(0).mean() for c_ in range(4) if (ynk == c_).sum() > 1]))) for k_ in range(3)]
        ex = {}
        for a, b in (("A", "V"), ("A", "T"), ("V", "T")):
            ya, yb = pnk.ret[a].argmax(1), pnk.ret[b].argmax(1)
            Ha, Hb = dai15.table_from(np.eye(4)[ya], ynk), dai15.table_from(np.eye(4)[yb], ynk)
            ex[f"excess_{a}{b}"] = float(np.mean([(ya[ynk == i] == yb[ynk == i]).mean() - float(Ha[i] @ Hb[i]) for i in range(4) if (ynk == i).any()]))
        rows.append(dict(K=K, version="_check", teacher_dev=dev, **dev_m, grad=conv[0], stalled=conv[1], adj_resid=resid_max, **ass, **ex,
                         drift_M=float(np.abs(Dt_M - D["M"]["orc"]).mean()),
                         sdratio_M=float(max(wsd) / max(min(wsd), 1e-12)),
                         retsd=(float(np.nanmean(np.nanstd(SR["ret_emo"], 1))) if np.isfinite(SR["ret_emo"]).sum(1).max() > 1 else np.nan),
                         **{f"ret_{m}": float(SR["ret"][i]) for i, m in enumerate(MODS)},
                         **{f"share_pool_{i}": float((ynk == i).mean()) for i in range(4)},
                         **{f"level_M_{r}_{m}": float(D["M"][r][i].mean()) for r in ("home", "ext", "away", "K", "orc") for i, m in enumerate(MODS)}))
    return rows, arr


# ------------------------------------------------------------------ 段

def stage_units(out: Path, targets: list, folds=None, seeds=None) -> None:
    fz1, _ = dai3.load_fz1(OUT1)
    udir = out / "units"
    udir.mkdir(parents=True, exist_ok=True)
    for T in targets:
        fzT = fz1[T]
        todo = dai3.pseudo_units(OUT1, T)
        if folds is not None:
            todo = [x for x in todo if x[1] in folds]
        if seeds is not None:
            todo = [x for x in todo if x[2] in seeds]
        for c, f, r in todo:
            name = f"{T}_{c}_f{f}_r{r}"
            if (udir / f"{name}.csv").exists():
                old = pd.read_csv(udir / f"{name}.csv")
                if (old.commit == RUN["commit"]).all():
                    print(f"{name}：済み（飛ばす）", flush=True)
                    continue
                raise SystemExit(f"{name} が別のコミットで既にある")
            t0 = time.time()
            sv = dai3.Saved(OUT1, "pseudo", T, c, f, r, fzT)
            rows, arr = run_unit(sv)
            df = pd.DataFrame(rows).assign(T=T, c=c, fold=f, seed=r, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0,
                                           dai1_commits=",".join(sorted(sv.commits)))
            np.savez_compressed(udir / f"{name}.npz", **arr)
            df.to_csv(udir / f"{name}.csv", index=False)
            ck = df[df.version == "_check"]
            print(f"{name}：{time.time() - t0:.0f} 秒．行 {len(df)}．作り直しの差 {ck.teacher_dev.max():.1e}．勾配の最大 {ck.grad.max():.1e}．止まり {int(ck.stalled.sum())}．"
                  f"割合をそろえる残り {ck.adj_resid.max():.1e}", flush=True)
            del sv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verdict(lo: float, hi: float, est: float) -> str:
    if lo > 0 and est >= FLOOR:
        return "上回る"
    if hi < 0 and est <= -FLOOR:
        return "下回る"
    return "言えない"


def row_rng(*key) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32("|".join(map(str, key)).encode())])


def contrasts() -> list:
    """(a, b, 層の並び)．層：adj・kb・raw はテスト，pool は候補（K 本を除く．adj の判定）．"""
    full, two = ("adj", "kb", "raw", "pool", "ob"), ("adj", "kb")
    main_M = ("M_F6_mixsel", "M_F3_mixsel", "M_F6_ext", "M_F6_K", "M_F6_orc", "M_F3_orc", "M_F2_orc", "M_F6n_orc", "M_F6n_mixsel", "M_F8", "cf_D_l05", "orc_F", "orc_D")
    con = [(v, "prod", full if v in main_M else two) for v in V_M + ("M_F6_mixsel", "M_F3_mixsel", "M_F6n_mixsel", "M_F3n_mixsel", "M_F8") + tuple(TABLE_SPECS)]
    con += [("prod_ob", "prod", ("adj",))]
    con += [(v, "S_eq", two) for v in V_S[1:] + ("S_F6_mixsel", "S_F3_mixsel", "S_F8")] + [(v, "prod", two) for v in ("S_eq", "S_F6_orc", "S_F6_mixsel", "S_F8")]
    con += [(v, "prod", full if v == "MS_F6_mixsel" else two) for v in V_MS + ("MS_F6_mixsel",)]
    con += [(f"MS_F6_{r}_l{j}", f"MS_eq_l{j}", two) for j in range(len(LAMS)) for r in ("K", "orc")]
    for r in ("home", "ext", "K", "orc"):                                # 使い方の比べ（出どころをそろえる）
        con += [(f"M_F6_{r}", f"M_F3_{r}", two), (f"M_F3_{r}", f"M_F4_{r}", two), (f"M_F5_{r}", f"M_F3_{r}", two), (f"M_F6_{r}", f"M_F2_{r}", two)]
    con += [(f"M_{f}_{r}", f"M_{f}n_{r}", two + ("ob",)) for f in ("F6", "F3") for r in ("home", "ext", "K", "orc", "mixsel")]
    con += [("cf_D_l05", "orc_D", two)]
    con += [("M_F6_mixsel", "M_F3_mixsel", two), ("M_F6_mixsel", "M_F6_K", two), ("M_F6_mixsel", "M_F6_ext", two), ("M_F6_ext", "M_F6_home", two),
            ("M_F6_ext", "M_F6_away", two), ("M_F6_K", "M_F6_orc", two), ("M_F6_mixsel", "M_F6_orc", two), ("M_F3_mixsel", "M_F3_K", two),
            ("orc_F", "orc_Fc", two), ("orc_D", "orc_Dc", two), ("cf_D_l05", "cf_D_l05c", two), ("M_F6_orc", "orc_F", two), ("M_F8", "M_F6_ext", two)]
    con += [(f"sel_{v}", "prod", ("adj",)) for v in EST_SEL + SEL_MIX] + [("sel_best", "prod", ("adj",))]
    con += [(f"sel_{v}", v, ("adj",)) for v in EST_SEL + SEL_MIX] + [("k_bias", "prod", ("adj",))]
    con += [("MS_F6_mixsel", "MS_eq_sel", two), ("MS_eq_sel", "prod", two)]
    con += [(v, ref, ("adj",)) for v in ("prod", "M_F6_mixsel", "M_F3_mixsel", "M_F6_orc", "MS_F6_mixsel") for ref in ("none_AVT", "joint_AVT")]
    con += [("prod_sel", "prod", ("raw",)), ("joint_AVT", "none_AVT", ("raw",))]
    merged: dict = {}
    for a, b, ws in con:                                                # 同じ (a, b) が 2 回出たら層をまとめる（差の行を重ねない）
        merged[(a, b)] = tuple(dict.fromkeys(merged.get((a, b), ()) + tuple(ws)))
    return [(a, b, ws) for (a, b), ws in merged.items()]


CRITERIA = [("R1", "本体の形・全ラベル", "M_F6_orc", "prod"), ("R1", "研究者の形・全ラベル", "M_F3_orc", "prod"), ("R1", "水準だけ・全ラベル", "M_F2_orc", "prod"),
            ("R1", "表の読み替え・全ラベル", "orc_F", "prod"), ("R1", "切片だけ・全ラベル", "prod_ob", "prod"),
            ("R1", "本体の形・全ラベル（切片を全ラベルで決めた層）", "M_F6_orc", "prod", "ob"), ("R1", "研究者の形・全ラベル（切片を全ラベルで決めた層）", "M_F3_orc", "prod", "ob"),
            ("R1", "ソースの分類器・全ラベル", "S_F6_orc", "S_eq"), ("R1", "両方・全ラベル", "MS_F6_orc_l1", "prod"),
            ("R2", "本体の形・水準 K＋相性ソース", "M_F6_mixsel", "prod"), ("R2", "研究者の形・水準 K＋相性ソース", "M_F3_mixsel", "prod"),
            ("R2", "本体の形・ソースだけ", "M_F6_ext", "prod"), ("R2", "本体の形・K 本だけ", "M_F6_K", "prod"), ("R2", "表の読み替え・K 本", "cf_D_l05", "prod"),
            ("R2", "まとめ方を学ぶ", "M_F8", "prod"), ("R2", "ソースの分類器", "S_F6_mixsel", "S_eq"), ("R2", "両方", "MS_F6_mixsel", "prod"),
            ("比べ", "本体 − 研究者（水準 K＋相性ソース）", "M_F6_mixsel", "M_F3_mixsel"), ("比べ", "本体 − 研究者（全ラベル）", "M_F6_orc", "M_F3_orc"),
            ("比べ", "柔らかい − 硬い（K 本）", "M_F3_K", "M_F4_K"), ("比べ", "繰り返す − 柔らかい（K 本）", "M_F5_K", "M_F3_K"),
            ("比べ", "感情ごと − 共通（全ラベル）", "M_F6_orc", "M_F2_orc"),
            ("比べ", "モダリティ × 感情の分（全ラベル）", "M_F6_orc", "M_F6n_orc"), ("比べ", "モダリティ × 感情の分（水準 K＋相性ソース）", "M_F6_mixsel", "M_F6n_mixsel"),
            ("比べ", "モダリティ × 感情の分（ソースだけ）", "M_F6_ext", "M_F6n_ext"),
            ("比べ", "モダリティ × 感情の分・研究者の形（全ラベル）", "M_F3_orc", "M_F3n_orc"),
            ("比べ", "モダリティ × 感情の分・研究者の形（水準 K＋相性ソース）", "M_F3_mixsel", "M_F3n_mixsel"), ("比べ", "混ぜる − K 本だけ", "M_F6_mixsel", "M_F6_K"),
            ("比べ", "混ぜる − ソースだけ", "M_F6_mixsel", "M_F6_ext"), ("比べ", "外で測り直す − 中だけ", "M_F6_ext", "M_F6_home"),
            ("比べ", "本体 − 研究者（K 本）", "M_F6_K", "M_F3_K"), ("比べ", "本体 − 研究者（ソースだけ）", "M_F6_ext", "M_F3_ext"),
            ("比べ", "信頼度の重みの分（両方の組）", "MS_F6_mixsel", "MS_eq_sel"), ("比べ", "ソースの分類器を足した分（等しい重み）", "MS_eq_sel", "prod"),
            ("R3b", "見込みで選ぶ − 元の版：本体の形・水準 K＋相性ソース", "sel_M_F6_mixsel", "M_F6_mixsel"),
            ("R3b", "見込みで選ぶ − 元の版：研究者の形・水準 K＋相性ソース", "sel_M_F3_mixsel", "M_F3_mixsel"),
            ("R3b", "見込みで選ぶ − 元の版：本体の形・K 本", "sel_M_F6_K", "M_F6_K"), ("R3b", "見込みで選ぶ − 元の版：研究者の形・K 本", "sel_M_F3_K", "M_F3_K"),
            ("R3b", "見込みで選ぶ − 元の版：本体の形・ソースだけ", "sel_M_F6_ext", "M_F6_ext"), ("R3b", "見込みで選ぶ − 元の版：表の読み替え", "sel_cf_D_l05", "cf_D_l05"),
            ("R3b", "見込みで選ぶ − 元の版：切片", "sel_k_bias", "k_bias"),
            ("R3", "見込みで選ぶ：本体の形・水準 K＋相性ソース", "sel_M_F6_mixsel", "prod"), ("R3", "見込みで選ぶ：研究者の形・水準 K＋相性ソース", "sel_M_F3_mixsel", "prod"),
            ("R3", "見込みで選ぶ：本体の形・K 本", "sel_M_F6_K", "prod"), ("R3", "見込みで選ぶ：研究者の形・K 本", "sel_M_F3_K", "prod"),
            ("R3", "見込みで選ぶ：本体の形・ソースだけ", "sel_M_F6_ext", "prod"), ("R3", "見込みで選ぶ：表の読み替え", "sel_cf_D_l05", "prod"),
            ("R3", "見込みで選ぶ：切片", "sel_k_bias", "prod"), ("R3", "見込みで選ぶ：最良", "sel_best", "prod")]


def stage_summary(out: Path, allow_partial: bool) -> None:
    files = sorted((out / "units").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    csvs = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in files], ignore_index=True)
    if csvs.commit.nunique() != 1 and not allow_partial:
        raise SystemExit(f"単位のコミットが 1 つでない：{csvs.commit.unique()}")
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)
    # ---- メタ学習の選択：コーパス c を評価するときは，c を評価のターゲットにした単位（c がソースにも仮のターゲットにも入らない）の平均で格子の点を選ぶ
    g = csvs[csvs.version.isin([x for L in MIXSEL.values() for x in L])].groupby(["T", "K", "version"]).uar_adj.mean()
    choice, chrows = {}, []
    for c in DD.CORPORA:
        for K in KS:
            for name, cands in MIXSEL.items():
                vals = [g.get((c, K, v), np.nan) for v in cands]
                if not np.isfinite(vals).any():
                    continue                                              # 試走で c を評価のターゲットにした単位が無いとき
                i = int(np.nanargmax(vals))                               # 同点は並びで先（水準を戻さない側，倍率の小さい側）
                choice[(c, K, name)] = cands[i]
                gi = int(cands[i].split("mix")[1].split("_")[0])
                chrows.append(dict(c=c, K=K, version=name, chosen=cands[i], n0=GRID[gi][0], beta=GRID[gi][1], prior=GRID[gi][2],
                                   lam=(LAMS[int(cands[i][-1])] if name.startswith("MS") else np.nan), mean_uar_adj=float(np.nanmax(vals)),
                                   mean_uar_adj_prod=float(csvs[(csvs["T"] == c) & (csvs.K == K) & (csvs.version == "prod")].uar_adj.mean())))
    pd.DataFrame(chrows).to_csv(sdir / "mixsel.csv", index=False)
    # ---- 塊ごとの数
    layers = {"adj": "pred_adj", "kb": "pred_kb", "raw": "pred_raw", "pool": "pred_pool", "ob": "pred_ob"}
    eg = {(a.T, a.c, a.fold, a.seed, a.K, a.version[4:]): a.est_gain for a in csvs[csvs.version.astype(str).str.startswith("est_")].itertuples()}
    cells = {w: {} for w in layers}
    emo, urows = {}, []
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        f, r = int(f[1:]), int(r[1:])
        z = np.load(p, allow_pickle=False)
        yt = z["y_test"].astype(int)
        for K in KS:
            nkm = np.ones(len(z["y_pool"]), bool)
            nkm[z[f"kpos_K{K}"]] = False
            ypk, clp = z["y_pool"].astype(int)[nkm], z["cl_pool"][nkm]
            names = {v: v for v in BASE + REFS + SELV}
            names.update({name: choice[(c, K, name)] for name in MIXSEL if (c, K, name) in choice})
            names.update({name: fn(choice[(c, K, src)]) for name, (src, fn) in TIED.items() if (c, K, src) in choice})
            for v, src in names.items():
                for w, key in layers.items():
                    kk = f"{key}_K{K}_{src}"
                    if v in REFS and w in ("adj", "kb", "ob"):
                        kk = f"pred_raw_K{K}_{src}"                       # 参照は層を持たない
                    if kk not in z.files:
                        continue
                    pred = z[kk].astype(int)
                    cnt = ST.counts(ypk, pred, clp) if w == "pool" else ST.counts(yt, pred, z["cl_test"])
                    P12._add(cells[w].setdefault((c, K, v), {}), (f, r), cnt, f"{p.stem} {w} {v}")
                    if w == "adj" and v in PER_EMO:
                        e = emo.setdefault((c, K, v), [np.zeros(4), np.zeros(4), np.zeros(4), 0])
                        e[0] += cnt[1].sum(0); e[1] += cnt[2].sum(0); e[2] += np.bincount(pred, minlength=4); e[3] += 1
                    if w == "adj" and v in MIXSEL:
                        urows.append(dict(T=T, c=c, fold=f, seed=r, K=K, version=v, chosen=src, uar_adj=100 * S.uar(yt, pred),
                                          est_gain=eg.get((T, c, f, r, K, src), np.nan)))
                    if w == "adj" and v in SEL_MIX:                               # 選んだ格子の点の見込みが正の単位でその点を，そうでない単位で積を使う
                        use = pred if eg.get((T, c, f, r, K, src), -1.0) > 0 else z[f"pred_raw_K{K}_prod"].astype(int)
                        P12._add(cells["adj"].setdefault((c, K, f"sel_{v}"), {}), (f, r), ST.counts(yt, use, z["cl_test"]), f"{p.stem} sel {v}")
    for w in cells:
        P12._finish(cells[w])
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}

    def level(d: dict) -> float:
        return float(np.mean([np.mean([ST.uar_full(d[k]) for k in d if k[0] == f]) for f in sorted({k[0] for k in d})]))

    lev = [dict(c=c, K=K, version=v, **{f"uar_{w}": (level(cells[w][(c, K, v)]) if (c, K, v) in cells[w] else np.nan) for w in layers}, n_fr=len(d))
           for (c, K, v), d in sorted(cells["adj"].items())]
    pd.DataFrame(lev).to_csv(sdir / "levels.csv", index=False)
    # ---- 差
    eff = []
    for c in DD.CORPORA:
        for K in KS:
            for a, b, ws in contrasts():
                for w in ws:
                    A, B = cells[w].get((c, K, a)), cells[w].get((c, K, b))
                    if A is None or B is None:
                        continue
                    if set(A) != set(B):
                        raise SystemExit(f"{c} K={K} {a} と {b} の単位がそろわない（{w}）")
                    es = (P12.effect_stats_pool if w == "pool" else ST.effect_stats)(A, B, geff[c], row_rng(c, K, a, b, w))
                    eff.append(dict(c=c, K=K, a=a, b=b, where=w, est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], method=es["method"],
                                    n_fr=es["n_units"], n_folds=es["n_folds"], p=es.get("p", np.nan), df=es.get("df", np.nan), v_jk=es.get("v_jk", np.nan),
                                    v_seed=es.get("v_seed", np.nan), folds_dropped=es.get("folds_dropped", ""), verdict=verdict(es["lo"], es["hi"], es["effect"])))
    eff = pd.DataFrame(eff)
    eff.to_csv(sdir / "effects.csv", index=False)
    crit = []
    for rule, axis, a, b, *ml in CRITERIA:
        mainw = ml[0] if ml else "adj"
        for K in KS:
            for w in ((mainw,) if ml else ("adj", "kb", "raw", "pool", "ob")):
                d = eff[(eff.a == a) & (eff.b == b) & (eff["where"] == w) & (eff.K == K)]
                if len(d) == 0:
                    continue
                up, dn = d[d.verdict == "上回る"].c.tolist(), d[d.verdict == "下回る"].c.tolist()
                crit.append(dict(rule=rule, axis=axis, a=a, b=b, K=K, where=w, main=(K == K_MAIN and w == mainw),
                                 proceed=(rule in ("R2", "R3") and b == "prod" and K == K_MAIN and w == "adj"), n_corpora=len(d), n_up=len(up), n_down=len(dn),
                                 n_up_noAFEW=len([x for x in up if x != "AFEW"]), n_down_noAFEW=len([x for x in dn if x != "AFEW"]),
                                 hit=len(up) >= 2, down2=len(dn) >= 2, hit_noAFEW=len([x for x in up if x != "AFEW"]) >= 2, up=",".join(up), down=",".join(dn)))
    pd.DataFrame(crit).to_csv(sdir / "criteria.csv", index=False)
    # ---- 見積もりの当たり方（単位ごとの見込みと，単位ごとのテストの差）
    idx = ["T", "c", "fold", "seed", "K"]
    pr = csvs[csvs.version == "prod"].set_index(idx)[["uar_raw"]].rename(columns={"uar_raw": "uar_prod"})
    ep = csvs[csvs.version == "_est_prod"].set_index(idx)[["est_uar"]].rename(columns={"est_uar": "est_uar_prod"})
    E = csvs[csvs.version.astype(str).str.startswith("est_")].copy()
    E["version"] = E.version.str[4:]
    U = pd.DataFrame(urows)
    extra = []
    if len(U):
        for v in SEL_MIX:                                                       # 選んだ格子の点の行を，選んだ版の名前で写す
            m = U[U.version == v][idx + ["chosen"]].merge(E, left_on=idx + ["chosen"], right_on=idx + ["version"], how="left")
            extra.append(m.assign(version=v))
    E = pd.concat([E] + extra, ignore_index=True)
    er = []
    for v in EST_SEL + SEL_MIX:
        jv = E[E.version == v].set_index(idx).join(pr).join(ep)
        if len(jv) == 0:
            continue
        jv["act"] = jv.uar_base - jv.uar_prod
        for (c, K), gq in jv.groupby(["c", "K"]):
            nz = gq[gq.act != 0]
            er.append(dict(c=c, K=K, version=v, n_units=len(gq), est_gain_ll=float(gq.est_gain.mean()), est_gain_uar=float((gq.est_uar - gq.est_uar_prod).mean()),
                           act_gain_uar=float(gq.act.mean()), share_used=float((gq.est_gain > 0).mean()), n_nonzero=len(nz),
                           sign_agree=(float(((nz.est_gain > 0) == (nz.act > 0)).mean()) if len(nz) else np.nan),
                           spearman=float(gq[["est_gain", "act"]].corr(method="spearman").iloc[0, 1]) if len(gq) > 2 else np.nan))
    pd.DataFrame(er).to_csv(sdir / "estimate.csv", index=False)
    csvs[csvs.version == "sel_best"].groupby(["c", "K"]).used.value_counts().rename("n").reset_index().to_csv(sdir / "sel_best_used.csv", index=False)
    U.to_csv(sdir / "mixsel_units.csv", index=False)
    # ---- 感情ごとの再現率と判定の割合（テスト．adj の層．区間は無い）
    pe = [dict(c=c, K=K, version=v, **{f"recall_{i}": 100 * e[0][i] / e[1][i] if e[1][i] > 0 else np.nan for i in range(4)},
               **{f"pred_share_{i}": float(e[2][i] / e[2].sum()) for i in range(4)}, **{f"n_per_unit_{i}": float(e[1][i] / e[3]) for i in range(4)})
          for (c, K, v), e in sorted(emo.items())]
    pd.DataFrame(pe).to_csv(sdir / "per_emotion.csv", index=False)
    # ---- 信頼度の表の単位の平均
    tb = {}
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        z = np.load(p, allow_pickle=False)
        for nm in ("M_home", "M_ext", "M_away"):
            tb.setdefault((c, 0, nm), []).append(z[f"SR_{nm}"])
        tb.setdefault((c, 0, "F8"), []).append(z["SR_F8"])
        for K in KS:
            for nm in ("K", "orc"):
                tb.setdefault((c, K, f"M_{nm}"), []).append(z[f"D_M_{nm}_K{K}"])
    tr = []
    for (c, K, nm), L in sorted(tb.items()):
        M, sd = np.mean(L, 0), np.std(L, 0)
        for i, m in enumerate(MODS):
            tr.append(dict(c=c, K=K, source=nm, modality=m, n_units=len(L), **{f"d_{j}": float(M[i, j]) for j in range(4)}, **{f"sd_{j}": float(sd[i, j]) for j in range(4)}))
    pd.DataFrame(tr).to_csv(sdir / "reliability.csv", index=False)
    sr = {}                                                               # ソースごとの home と，向き (s → t) ごとの away（仮のターゲットの回はソースが 2 つなので向きは 1 つ）
    for p in files:
        z = np.load(p, allow_pickle=False)
        srcs = [str(x) for x in z["srcs"]]
        for i, s_ in enumerate(srcs):
            for j, m in enumerate(MODS):
                sr[(s_, "home", m)] = z["SR_S_home"][i * 3 + j]
                if len(srcs) == 2:
                    sr[(s_, f"away→{srcs[1 - i]}", m)] = z["SR_S_away"][i * 3 + j]
    pd.DataFrame([dict(source=k[0], where=k[1], modality=k[2], **{f"d_{j}": float(v[j]) for j in range(4)}) for k, v in sorted(sr.items())]
                 ).to_csv(sdir / "reliability_src.csv", index=False)
    chk = csvs[csvs.version == "_check"]
    acols = [x for x in chk.columns if x.startswith(("intcorr_", "intperm_", "profcorr_", "emocorr_", "levelcorr_", "topok_", "excess_", "drift_", "sdratio_", "retsd",
                                                      "ret_", "share_pool_", "level_M_"))]
    am = chk.groupby(["c", "K"])[acols].agg(["mean", "std"])
    am.columns = [f"{a}_{b}" for a, b in am.columns]
    am.reset_index().to_csv(sdir / "assumptions.csv", index=False)
    rx = {}                                                               # R0 と並べる量（保存した信頼度の表から）：成分の大きさと，感情ごとに最も信頼度の高い分類器の一致
    re_rows = {}
    for p in files:
        T, c, f, r = p.stem.rsplit("_", 3)
        z = np.load(p, allow_pickle=False)
        for j_, m in enumerate(MODS):                                         # 感情ごとの残り方（モダリティごと．出せなかった感情は数えない）
            for i in range(4):
                re_rows.setdefault((c, m, i), []).append(float(z["SR_ret_emo"][j_, i]))
        for K in KS:
            for fam in ("M", "S"):
                orc = z[f"D_{fam}_orc_K{K}"]
                top_o = orc.argmax(0)
                for src_ in ("home", "ext", "away", "K"):
                    Ds = z[f"D_{fam}_K_K{K}"] if src_ == "K" else z[f"SR_{fam}_{src_}"]
                    rx.setdefault((c, K, fam, src_), []).append((float(np.abs(dcenter(Ds)).mean()), float(np.abs(dcenter(orc)).mean()),
                                                                 float((Ds.argmax(0) == top_o).mean()), float((Ds.mean(1).argmax() == top_o).mean())))
    pd.DataFrame([dict(c=c, modality=m, emotion=i, n_units=len(v), n_finite=int(np.isfinite(v).sum()),
                       ret_mean=(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan), ret_sd=(float(np.nanstd(v)) if np.isfinite(v).sum() > 1 else np.nan))
                  for (c, m, i), v in sorted(re_rows.items())]).to_csv(sdir / "ret_emo.csv", index=False)
    r0 = []                                                               # R0：分類器 × 感情の成分の相関と，感情の並びを入れ替えたときにそれ以上になる割合
    for (c, K), gq in chk.groupby(["c", "K"]):
        for fam in ("M", "S"):
            for r in ("home", "ext", "away", "K"):
                ic, ip = float(gq[f"intcorr_{fam}_{r}"].mean()), float(gq[f"intperm_{fam}_{r}"].mean())
                xs = np.array(rx[(c, K, fam, r)]).mean(0)                 # 成分の大きさ（出どころ，全ラベル），感情ごとの一致，水準だけの一致
                r0.append(dict(c=c, K=K, fam=fam, source=r, n_units=len(gq), intcorr=ic, intperm=ip, hit=bool(ic >= 0.3 and ip <= 0.2),
                               resid_src=float(xs[0]), resid_orc=float(xs[1]), emo_agree=float(xs[2]), level_agree=float(xs[3]), agree_gain=float(xs[2] - xs[3]),
                               hit_rank=bool(ic >= 0.3 and ip <= 0.2 and xs[2] - xs[3] >= 0.1),
                               profcorr=float(gq[f"profcorr_{fam}_{r}"].mean()), emocorr=float(gq[f"emocorr_{fam}_{r}"].mean()),
                               levelcorr=float(gq[f"levelcorr_{fam}_{r}"].mean()), topok=float(gq[f"topok_{fam}_{r}"].mean())))
    r0 = pd.DataFrame(r0)
    r0.to_csv(sdir / "r0.csv", index=False)
    r0.groupby(["K", "fam", "source"]).apply(lambda q: pd.Series(dict(n_corpora=len(q), n_hit=int(q.hit.sum()), n_hit_noAFEW=int(q[q.c != "AFEW"].hit.sum()),
                                                                       hit3=bool(q.hit.sum() >= 3), hit2_noAFEW=bool(q[q.c != "AFEW"].hit.sum() >= 2),
                                                                       n_hit_rank=int(q.hit_rank.sum()), n_hit_rank_noAFEW=int(q[q.c != "AFEW"].hit_rank.sum()),
                                                                       hit3_rank=bool(q.hit_rank.sum() >= 3), hit2_rank_noAFEW=bool(q[q.c != "AFEW"].hit_rank.sum() >= 2),
                                                                       corpora=",".join(q[q.hit].c), corpora_rank=",".join(q[q.hit_rank].c))),
                                             include_groups=False).reset_index().to_csv(sdir / "r0_count.csv", index=False)
    json.dump(dict(n_units=len(files), teacher_dev_max=float(chk.teacher_dev.max()), grad_max=float(chk.grad.max()), stalled=int(chk.stalled.sum()),
                   adj_resid_max=float(chk.adj_resid.max()), sec_total=float(csvs.groupby(["T", "c", "fold", "seed"]).sec.first().sum())),
              open(sdir / "convergence.json", "w"), ensure_ascii=False, indent=1)
    print(f"summary：単位 {len(files)}，差の行 {len(eff)}，数えの行 {len(crit)}，選んだ格子の行 {len(chrows)}", flush=True)


# ------------------------------------------------------------------ 検算

def selftest() -> None:
    rng = np.random.default_rng(0)
    # 1. AUC と d′：分かれ方が大きいほど大きい．切片をずらしても変わらない．感情の割合を変えてもほぼ変わらない
    n = 4000
    y = rng.integers(0, 4, n)
    Z = rng.normal(size=(n, 4)); Z[np.arange(n), y] += np.array([2.0, 1.0, 0.5, 0.0])[y]
    P = softmax(Z)
    d, a, A44 = dprime(P, y)
    assert d[0] > d[1] > d[2], (d, a)
    assert np.isnan(np.diag(A44)).all() and np.allclose(np.nanmean(A44, 1), a)
    dn, an, _ = dprime(softmax(rng.normal(size=(n, 4))), y)                 # 感情と関係の無い点なら AUC は 0.5 の近く
    assert np.abs(an - 0.5).max() < 0.03, an
    d2 = dprime(softmax(Z + np.array([3.0, -1.0, 0.5, 0.0])), y)[0]
    assert np.abs(d - d2).max() < 1e-9
    keep = (y != 0) | (rng.uniform(size=n) < 0.2)
    d3 = dprime(P[keep], y[keep])[0]
    assert np.abs(d - d3).max() < 0.25
    assert abs(auc_pair(np.array([3.0, 2.0]), np.array([1.0, 0.0])) - 1.0) < 1e-12 and abs(auc_pair(np.array([1.0, 1.0]), np.array([1.0, 1.0])) - 0.5) < 1e-12
    # 2. 重みが全部 1 なら，どの形も等しい重みの積と同じ判定
    Ps = [rng.dirichlet(np.ones(4), size=300) for _ in range(3)]
    Ls = np.stack([centered(p) for p in Ps])
    prod = sum(lg(p) for p in Ps).argmax(1)
    q1 = softmax(Ls.sum(0))
    for f in FORMS:
        assert (combine(f, Ls, np.ones((3, 4)), q1).argmax(1) == prod).all(), f
    assert np.abs(np.exp(combine("F6", Ls, np.ones((3, 4)), q1)) - q1).max() < 1e-12
    # 3. 分類器ごとに得意な感情が違うとき：正確な信頼度の F6 と F3 は積より当たる．F4 は疑似ラベルを言い切る
    n = 6000
    y = rng.integers(0, 4, n)
    Pv = []
    for k in range(3):
        sep = np.full(4, 0.3); sep[k] = 2.5; sep[3] = 0.8
        Zk = rng.normal(size=(n, 4)); Zk[np.arange(n), y] += sep[y]
        Pv.append(softmax(Zk))
    Lv = np.stack([centered(p) for p in Pv])
    Dv = np.stack([dprime(p, y)[0] for p in Pv])
    q1 = softmax(Lv.sum(0))
    u0 = S.uar(y, Lv.sum(0).argmax(1))
    u6 = S.uar(y, combine("F6", Lv, to_w(Dv), q1).argmax(1))
    u3 = S.uar(y, combine("F3", Lv, to_w(Dv), q1).argmax(1))
    assert u6 > u0 + 0.01 and u3 > u0, (u0, u6, u3)
    # 4. 混ぜる版の端：n0 = ∞・相性 1・戻し先 ext なら ext そのもの．n0 = 0・相性 0 なら K の水準だけ．n0 = ∞・相性 0・戻し先 eq なら全部同じ
    DK, DE = rng.uniform(0.2, 2.0, size=(3, 4)), rng.uniform(0.2, 2.0, size=(3, 4))
    assert np.allclose(mix_rel(DK, DE, 20, float("inf"), 1.0, "ext"), DE)
    assert np.allclose(mix_rel(DK, DE, 20, 0.0, 0.0, "eq"), np.repeat(DK.mean(1)[:, None], 4, 1))
    m = mix_rel(DK, DE, 20, float("inf"), 0.0, "eq")
    assert np.allclose(m, m[0, 0])
    assert np.allclose(mix_rel(DK, DE, 20, 20.0, 0.0, "eq").mean(1), 0.5 * DK.mean(1) + 0.5 * DK.mean())
    assert np.allclose(to_w(np.zeros((3, 4))), 1.0) and abs(to_w(DK).mean() - 1.0) < 1e-12
    Wn = noint(to_w(DK))
    assert np.abs(dcenter(Wn)).max() < 1e-9 or (Wn == 0).any()                   # 加法の版は分類器 × 感情の成分が 0
    add = np.array([0.5, 1.0, 1.5])[:, None] + np.array([0.0, 0.2, -0.1, 0.3])[None, :]
    assert np.allclose(noint(add), add) and np.abs(dcenter(add)).max() < 1e-12
    mul = np.array([0.5, 1.0, 1.5])[:, None] * np.array([0.6, 1.2, 0.9, 1.4])[None, :]     # 掛け算だけの表：分類器の比がどの感情でも同じ
    assert np.abs(dcenter(mul)).max() < 1e-12 and np.allclose(noint(mul), mul)
    A0 = rng.normal(size=(3, 4))
    assert np.abs(dcenter(tukey_fit(A0))).max() < 1e-9 and perm_share(A0, A0) <= 1 / 24 + 1e-9
    Lm = np.stack([centered(p) for p in Ps])                                              # 掛け算だけの重みの F6 は，F2（感情で共通の重み）と同じ順位の点に列ごとの倍率を掛けたもの
    f6 = np.einsum("ky,kny->ny", mul, Lm); f2 = np.tensordot(mul[:, 0], Lm, axes=1)
    assert np.allclose(f6, f2 * (mul[0] / mul[0, 0])[None, :])
    # 5. 割合をそろえる切片
    Lp = norm_log(rng.normal(size=(500, 4)) * 2 + np.array([1.0, 0.0, -1.0, 0.5]))
    tgt = np.array([0.3, 0.2, 0.25, 0.25])
    dl, rs = match_shift(Lp, tgt)
    assert rs < 1e-6 and abs(dl.sum()) < 1e-9
    # 6. 切片の最適化：片寄った点に切片を足すと対数尤度が上がる．重みつきの版も解ける
    yy = rng.integers(0, 4, 800)
    Zb = rng.normal(size=(800, 4)); Zb[np.arange(800), yy] += 1.5; Zb[:, 0] -= 2.0
    b = fit_wb(norm_log(Zb)[None], yy, np.ones(1), 0.0, fit_w=False, fit_b=True)[1]
    assert bal_ll(norm_log(Zb + b), yy) > bal_ll(norm_log(Zb), yy) + 0.05 and b[0] > 0.5
    w, _ = fit_wb(np.stack([centered(softmax(Zb)), centered(rng.dirichlet(np.ones(4), size=800))]), yy, np.ones(2), 0.01, fit_b=True, sw=np.full(800, 1 / 800))
    assert w[1] < 0.3
    # 7. 表の向き（F7）
    Ca = np.array([[0.70, 0.20, 0.05, 0.05], [0.05, 0.50, 0.40, 0.05], [0.10, 0.05, 0.80, 0.05], [0.30, 0.05, 0.05, 0.60]])
    y0 = rng.integers(0, 4, 200000)
    v0 = (rng.uniform(size=len(y0))[:, None] > Ca.cumsum(1)[y0]).sum(1)
    emp = np.stack([np.bincount(y0[v0 == j], minlength=4) / max((v0 == j).sum(), 1) for j in range(4)])
    assert np.abs(posterior_matrix(Ca) - emp).max() < 0.01 and np.abs(posterior_matrix(Ca.T) - emp).max() > 0.1
    # 8. 版の名前
    names = BASE + SELV + tuple(MIXSEL) + tuple(TIED) + REFS + tuple(f"sel_{v}" for v in SEL_MIX)
    assert len(set(names)) == len(names)
    known = set(names)
    for a_, b_, _ in contrasts():
        assert a_ in known and b_ in known, (a_, b_)
    for _, _, a_, b_, *_ml in CRITERIA:
        assert a_ in known and b_ in known, (a_, b_)
    # 9. fit_wb の勾配（差分との比べ）と，単位行列の表の読み替え
    Lg, yg = np.stack([centered(rng.dirichlet(np.ones(4), size=60)) for _ in range(3)]), rng.integers(0, 4, 60)
    fobj = lambda th: -bal_ll(norm_log(np.tensordot(th[:3], Lg, axes=1) + th[3:]), yg) + 0.5 * 0.2 * (((th[:3] - 1) ** 2).sum() + (th[3:] ** 2).sum())  # noqa: E731
    wg, bg = fit_wb(Lg, yg, np.ones(3), 0.2, fit_w=True, fit_b=True)
    th = np.concatenate([wg, bg])
    num = np.array([(fobj(th + e) - fobj(th - e)) / 2e-5 for e in np.eye(7) * 1e-5])
    assert np.all((np.abs(num) < 1e-3) | (th < 1e-6)), num
    Pi = rng.dirichlet(np.ones(4), size=40)
    assert np.abs(reread(Pi, np.eye(4)) - Pi).max() < 1e-6
    # 10. F8 の 12 個の重みの並び：(モダリティ j, 感情 i) の順に並べた 12 個の「見かけの分類器」の和は，F6 の式と同じ
    L12 = np.zeros((12, 60, 4))
    for j_ in range(3):
        for i_ in range(4):
            L12[j_ * 4 + i_][:, i_] = Lg[j_][:, i_]
    w12 = rng.uniform(0.2, 2.0, 12)
    assert np.allclose(np.tensordot(w12, L12, axes=1), np.einsum("ky,kny->ny", w12.reshape(3, 4), Lg))
    assert verdict(0.1, 2.0, 1.2) == "上回る" and verdict(0.1, 1.0, 0.5) == "言えない" and verdict(-3, -0.1, -1.5) == "下回る"
    print(f"selftest：通った（版 {len(BASE)}，差の定義 {len(contrasts())}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "units", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=None, help="流す K（既定は KS）．渡すと出力は <OUT_PROD>_k<K...> の置き場だけ（追補の設計）")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="試しの置き場だけ：単位のそろいの確かめを省く")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    global KS, K_MAIN
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    if args.ks:
        KS = tuple(sorted(set(args.ks)))
        K_MAIN = K_MAIN if K_MAIN in KS else KS[0]                          # 追補の回では，その K を規則の主の K として数える
        out_k = Path(f"{OUT_PROD}_k{'_'.join(map(str, KS))}").resolve()
        if prod:
            sys.exit("--ks を渡すときは，既定の置き場ではなく別の置き場に書く")
        prod = out == out_k
        if prod and not dai1.committed_clean(ADDENDUM):
            sys.exit("追補の設計の文書がコミットされていない")
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある（試走は一時の置き場で --allow-dirty）")
    if prod and not dai1.committed_clean(DESIGN):
        sys.exit("設計の文書がコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or set(args.targets) != set(DD.CORPORA) or args.allow_partial):
        sys.exit("本番の置き場では対象を絞らない")
    dai1.ENC["enc"] = DD.ENC1
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), torch=torch.__version__, numpy=np.__version__,
                scipy=scipy.__version__, python=platform.python_version(), pandas=pd.__version__,
                scripts={n: dai15.sha(HERE / n) for n in SCRIPTS}, design_sha256=dai15.sha(DESIGN), dai1_frozen_sha256=dai3.load_fz1(OUT1)[1],
                inputs_sha256={**{f"source/{q.name}": dai15.sha(q) for q in sorted((OUT1 / "source").glob("*.npz"))},
                               **{f"splits/{q.name}": dai15.sha(q) for q in sorted((ROOT / "splits").glob("*.csv"))}},
                KS=KS, NF=NF, D_MIN=D_MIN, EPS=EPS, AUC_CLIP=AUC_CLIP, TAU=TAU, TAU_F8=TAU_F8, W_MAX=W_MAX, N_ITER=N_ITER, GRID=[list(map(str, g)) for g in GRID],
                LAMS=LAMS, ADJ_TOL=ADJ_TOL, CHECK_TEACHER=CHECK_TEACHER, GRAD_TOL=GRAD_TOL, FLOOR=FLOOR, ASSUME_FLAG=ASSUME_FLAG, n_versions=len(BASE),
                EST=EST, MIXSEL={k: len(v) for k, v in MIXSEL.items()}, rng_seed=SEED)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "units":
        stage_units(out, args.targets, args.folds, args.seeds)
        if prod:
            (out / "units.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    else:
        stage_summary(out, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
