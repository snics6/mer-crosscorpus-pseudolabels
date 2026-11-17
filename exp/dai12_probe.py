"""DAI-12 の下調べ：疑似ラベルの出所の予測を，学び直す分類器の凍結した特徴から「非線形に読める分」（隠れ層 1 つのネット nl）は
「線形に読める分」（同じ損失・同じ格子で隠れ層だけを外した線形のネット lince）よりどれだけ大きいか（設計 docs/設計/実験12_下調べ_設計.md 第 6 版）。
nl は線形の項を持ち（点数 = Wl x + c + W2 relu(W1 x + c1)，W2 の初期値 0），選択は lince の選んだ版を候補に含む入れ子の形である（3.3 節）。

DAI-1 の仮のターゲットの回の保存（results/dai1/pseudo/units/），DAI-11 の保存（results/dai11_probe/rows/，enc2/source/）と特徴だけを読む．
新しく学ぶのは，出所の確率を目標にしたネット（ラベルを使わない），候補の全ラベルの分類器と 4×4 の読み出し（記述），リッジ回帰である．
評価の回の保存は読まない．

単位の段：  python exp/dai12_probe.py --out results/dai12_probe --stage units [--units AFEW_IEMOCAP_f0_r0]
集計の段：  python exp/dai12_probe.py --out results/dai12_probe --stage summarize [--allow-partial]
自己検査：  python exp/dai12_probe.py --stage selftest
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402   subset_masks（MOSEI の channel など）
import dai11_probe as P11  # noqa: E402   Unit2（6 つの特徴の標準化），none_wb，probs，representability，metrics，kappa_coef，unit_parts，load_unit_npz
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = HERE.parent
UD = ROOT / "results" / "dai1" / "pseudo" / "units"
D11 = ROOT / "results" / "dai11_probe"
FROZEN = ROOT / "frozen" / "dai1_frozen.json"
KS_ALL = (0, 1, 5, 10, 25)
KS_RULE = (1, 5, 10, 25)
MODS = ("A", "V", "T")                                                    # 学び直す分類器（融合 AVT は対象にしない．設計 2 節）
TEACHERS = ("self", "enc2", "fuse1", "other", "enc2fuse", "pair", "pair_shuf")   # 疑似ラベルの出所（設計 3.1 節）
RULE_TEACHERS = ("enc2", "fuse1", "other", "enc2fuse", "pair")           # R1 の対象（self は対照 R1c，pair_shuf は対照 R1s）
RET11_TEACHERS = ("self", "enc2", "fuse1", "other", "enc2fuse")          # DAI-11 に同じ出所の線形の版と pool_proj がある出所
N_UNITS = 120
SEED = 0
HIDDEN, STEPS = 256, 500                                                 # 隠れ層 1 つのネット（E30 と同じ形）
LRS = (1e-2,)                                                            # Adam の学習率（10⁻³ だけでは線形のネットが閉じた解に届かない．10⁻³ の 100 ステップは 10⁻² の 10 ステップの保存点が代わりになる．gate-probe 第 3 回）
WDS = (1e-4, 1e-2, 1e-1, 1.0, 10.0)                                      # 重みの L2 罰則の格子（gate-code 第 1 回で 10⁻² の端，gate-probe 第 3 回で 10⁻¹ の端で選ばれたので上に広げた）
CKPTS = (10, 25, 50, 100, 200, 300, 400, 500)                            # 保存点（gate-code 第 1 回：100 の端で選ばれたので 10・25・50 を足した）
FIXED_LR, FIXED_WD, FIXED_STEP = 1e-2, 1e-4, 500                        # 固定の版（選択の楽観の目安）
READ_WDS, READ_CKPTS, READ_STEPS = (0.0, 1e-4, 1e-2), (100, 200, 500, 1000, 2000), 2000   # 4×4 の読み出しの格子（入力 4 次元．罰則なし 0 を含む．試走で wd の下端に集まったため）
LAB_CKPTS, LAB_STEPS = (10, 25, 50, 100, 200, 300, 400, 500, 1000, 2000), 2000              # Q3 の全ラベルの分類器の保存点（試走で MOSEI が 500 の端に寄ったため延ばした．gate-probe 第 5 回）
CONV_REL = 0.005                                                         # 収束の目安：選んだ保存点が最後で，1 つ前からの予測する半分の損失の下がりがこの割合より大きければ未収束
BELOW_RIDGE_TOL = 1.0                                                    # lince が ridge（閉じた解）より低い行を数える幅（損失と解き方の違いの分の記述）
FLOOR_POOL, FLOOR_TEST = 1.0, 0.5                                        # 4.1 節の下限（候補の上の差，テストの差）
MIN_DELTA_SHARE = 0.5
MISMATCH_FRAC = 0.001                                                    # 判定の作り直しと保存の不一致の許し（max(1, 0.1%) 本）
SELF_TOL = 0.5                                                           # 対照 self の差の許し（4.3 節の (b)）
N_BOOT = ST.N_BOOT
METRICS = ("uar", "war", "waf", "mf1")

torch.backends.cuda.matmul.allow_tf32 = False                            # 再現性のため TF32 は使わない
torch.backends.cudnn.allow_tf32 = False


# ------------------------------------------------------------------ 2 つの半分（ラベルを使わない）

def balanced_halves(cl: np.ndarray) -> np.ndarray:
    """塊を本数の多い順（同数は名前の順）に並べ，先頭から順にその時点で本数の少ない半分へ入れる（同数なら半分 0）．設計 2 節（M3）。"""
    names, inv, cnt = np.unique(cl, return_inverse=True, return_counts=True)
    order = np.lexsort((names, -cnt))
    side = np.zeros(len(names), int)
    tot = [0, 0]
    for i in order:
        h = 0 if tot[0] <= tot[1] else 1
        side[i] = h; tot[h] += int(cnt[i])
    return side[inv].astype(int)


def name_order_halves(cl: np.ndarray) -> np.ndarray:
    """DAI-11 の分け方（名前の順に並べて交互に配る）．照合にだけ使う。"""
    return (pd.factorize(cl, sort=True)[0] % 2).astype(int)


# ------------------------------------------------------------------ 束ねたネットの学習（E30 の形．ネットどうしは重みを共有しない）

def _init(B: int, d: int, hidden: int, g: torch.Generator, zero: bool = False) -> list:
    """線形のネット：[W, c]．隠れ層 1 つのネット：[W1, c1, W2, c2, Wl]（点数 = Wl x + c2 + W2 relu(W1 x + c1)．線形の項を持ち，
    隠れ層の出力の重み W2 の初期値を 0 にするので，初期の点数は線形のネットと同じ分布になる．gate-code 第 2 回の 1）。"""
    dev = S.DEV
    if zero:                                                                                  # 初期値 0 の線形（記述の lince0）
        return [torch.zeros(B, d, 4, device=dev, requires_grad=True), torch.zeros(B, 4, device=dev, requires_grad=True)]
    b1 = 1 / np.sqrt(d)
    W = ((torch.rand(B, d, 4, generator=g, device=dev) * 2 - 1) * b1).requires_grad_(True)    # 線形の項（lince と同じ初期値の分布）
    c = ((torch.rand(B, 4, generator=g, device=dev) * 2 - 1) * b1).requires_grad_(True)
    if hidden > 0:
        W1 = ((torch.rand(B, d, hidden, generator=g, device=dev) * 2 - 1) * b1).requires_grad_(True)
        c1 = ((torch.rand(B, hidden, generator=g, device=dev) * 2 - 1) * b1).requires_grad_(True)
        W2 = torch.zeros(B, hidden, 4, device=dev, requires_grad=True)                        # 出力の重みの初期値 0
        return [W1, c1, W2, c, W]
    return [W, c]


def _forward(X: torch.Tensor, params: list) -> torch.Tensor:
    """X (n, d)（全ネットで共通）か (B, n, d)（ネットごと）→ 点数 (B, n, 4)。"""
    if len(params) == 5:
        W1, c1, W2, c, Wl = params
        H = torch.einsum("nd,bdh->bnh", X, W1) if X.dim() == 2 else torch.bmm(X, W1)
        L = torch.einsum("nd,bdk->bnk", X, Wl) if X.dim() == 2 else torch.bmm(X, Wl)
        return L + c[:, None, :] + torch.bmm(torch.relu(H + c1[:, None, :]), W2)
    W, c = params
    Z = torch.einsum("nd,bdk->bnk", X, W) if X.dim() == 2 else torch.bmm(X, W)
    return Z + c[:, None, :]


def _penalty(params: list) -> torch.Tensor:
    return sum((p ** 2).sum(tuple(range(1, p.dim()))) for p in params if p.dim() == 3)   # 重み行列だけ（切片は罰しない）


def _ce(Q: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """重みつきの交差エントロピー (B,)：Σ_i Σ_k Q[b,i,k]（−log softmax(L)[b,i,k]）÷ Σ Q。Q は出所の確率か，釣り合いの重みつきの one-hot。"""
    return -(Q * torch.log_softmax(L, -1)).sum((1, 2)) / Q.sum((1, 2)).clamp(min=1e-12)


def _rows(X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return X[idx] if X.dim() == 2 else X[:, idx]


def train_batched(Xtr, Qtr, Xte, Qte, Xtest, wds: tuple, seed: int, hidden: int, lr: float, steps: int = STEPS, ckpts: tuple = CKPTS, zero: bool = False) -> dict:
    """目標 J 個 × 罰則 len(wds) 個 = B 個の独立なネットを一括で学ぶ．Qtr (J, ntr, 4)．X は (n, d) か (B, n, d)．
    予測する半分 Xte の損失（Qte で測る）が最小の保存点の点数と重みを残す（選んだ版）．保存点ごとの点数も返す。"""
    J, nw = Qtr.shape[0], len(wds)
    B = J * nw
    Qtr_b, Qte_b = Qtr.repeat_interleave(nw, 0), Qte.repeat_interleave(nw, 0)
    wd = torch.tensor(wds, dtype=torch.float32, device=S.DEV).repeat(J)                 # 並びは j × nw + w
    g = torch.Generator(device=S.DEV).manual_seed(seed)
    d = Xtr.shape[-1]
    params = _init(B, d, hidden, g, zero)
    opt = torch.optim.Adam(params, lr=lr)
    ce_best = torch.full((B,), float("inf"), device=S.DEV)
    step_best = torch.zeros(B, dtype=torch.long, device=S.DEV)
    te_best = torch.zeros(B, Xte.shape[-2], 4, device=S.DEV)
    test_best = torch.zeros(B, Xtest.shape[-2], 4, device=S.DEV)
    best_params = [p.detach().clone() for p in params]
    te_at, test_at, ce_hist = {}, {}, []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        L = _forward(Xtr, params)
        loss_b = _ce(Qtr_b, L) + wd / 2 * _penalty(params)
        loss_b.sum().backward()
        opt.step()
        if step in ckpts:
            with torch.no_grad():
                Lte, Ltest = _forward(Xte, params), _forward(Xtest, params)
                ce_te = _ce(Qte_b, Lte)
                better = ce_te < ce_best
                ce_best = torch.where(better, ce_te, ce_best)
                step_best = torch.where(better, torch.full_like(step_best, step), step_best)
                te_best[better] = Lte[better]; test_best[better] = Ltest[better]
                for bp, p in zip(best_params, params):
                    bp[better] = p.detach()[better]
                te_at[step] = Lte.clone(); test_at[step] = Ltest.clone()
                ce_hist.append(ce_te.clone())
    ce_hist = torch.stack(ce_hist, 1)
    if not bool(torch.isfinite(ce_hist).all()):                                              # I11
        raise SystemExit("束ねたネットの損失に有限でない値がある")
    return dict(te_best=te_best, test_best=test_best, ce_best=ce_best, step_best=step_best, te_at=te_at, test_at=test_at,
                ce_hist=ce_hist, train_loss_final=loss_b.detach(), best_params=best_params)


def crossfit_nets(X, Q, half: np.ndarray, Xtest, seed: int, hidden: int, wds: tuple = WDS, lrs: tuple = LRS, keep_params: tuple = (), zero: bool = False,
                  ckpts: tuple = CKPTS, steps: int = STEPS) -> dict:
    """各半分を，もう一方の半分で学んだネットで予測する（交差の予測）．目標ごと・半分ごとに，予測する半分の損失が最小の (学習率, wd, 保存点) を選ぶ．
    X は (n, d)（共通）か (J, n, d)（目標ごと）．戻り：sel (J, n, 4)，fix (J, n, 4)，sel_test (2, J, ntest, 4)，fix_test，
    grid_pred (nL, nw, nck, J, n) int8，cfg {(h, j): (lr, wd, step)}，ce_sel (2, J)，ce_fix (2, J)，ce_hist (2, J, nL, nw, nck)，train_loss (2, J, nL, nw)，
    params：keep_params の目標の番号ごとに {(h, j): [float16 の重み]}。"""
    J, n = Q.shape[0], Q.shape[1]
    nL, nw, nck = len(lrs), len(wds), len(ckpts)
    il_f, iw_f, ick_f = lrs.index(FIXED_LR), wds.index(FIXED_WD), list(ckpts).index(FIXED_STEP)
    Xb = X if X.dim() == 2 else X.repeat_interleave(nw, 0)                                  # (B, n, d)
    Xtb = Xtest if Xtest.dim() == 2 else Xtest.repeat_interleave(nw, 0)
    sel = torch.zeros(J, n, 4, device=S.DEV); fix = torch.zeros(J, n, 4, device=S.DEV)
    sel_test = torch.zeros(2, J, Xtest.shape[-2], 4, device=S.DEV); fix_test = torch.zeros_like(sel_test)
    grid_pred = np.zeros((nL, nw, nck, J, n), np.int8)
    cfg, ce_sel, ce_fix = {}, np.zeros((2, J)), np.zeros((2, J))
    ce_hist, tl, params_out = np.zeros((2, J, nL, nw, nck)), np.zeros((2, J, nL, nw)), {}
    for h in (0, 1):
        tr = torch.as_tensor(np.flatnonzero(half != h), device=S.DEV); te = torch.as_tensor(np.flatnonzero(half == h), device=S.DEV)
        te_np = te.cpu().numpy()
        res_l = [train_batched(_rows(Xb, tr), Q[:, tr], _rows(Xb, te), Q[:, te], Xtb, wds, seed * 4 + h * 2 + il, hidden, lr, steps=steps, ckpts=ckpts, zero=zero) for il, lr in enumerate(lrs)]
        ce_all = torch.stack([res["ce_best"].view(J, nw) for res in res_l], 1)               # (J, nL, nw)
        flat = ce_all.view(J, -1).argmin(1)
        il_best, iw_best = (flat // nw).cpu().numpy(), (flat % nw).cpu().numpy()
        for il, res in enumerate(res_l):
            for ic, step in enumerate(ckpts):
                grid_pred[il, :, ic][:, :, te_np] = res["te_at"][step].argmax(-1).to(torch.int8).view(J, nw, -1).permute(1, 0, 2).cpu().numpy()
            ce_hist[h, :, il] = res["ce_hist"].view(J, nw, nck).cpu().numpy()
            tl[h, :, il] = res["train_loss_final"].view(J, nw).cpu().numpy()
        res_f = res_l[il_f]
        rows_fix = torch.arange(J, device=S.DEV) * nw + iw_f
        fix[:, te] = res_f["te_at"][FIXED_STEP][rows_fix]; fix_test[h] = res_f["test_at"][FIXED_STEP][rows_fix]
        ce_fix[h] = res_f["ce_hist"].view(J, nw, nck)[:, iw_f, ick_f].cpu().numpy()
        for j in range(J):
            res = res_l[int(il_best[j])]
            b = j * nw + int(iw_best[j])
            sel[j, te] = res["te_best"][b]; sel_test[h, j] = res["test_best"][b]
            ce_sel[h, j] = float(res["ce_best"][b])
            cfg[(h, j)] = (float(lrs[int(il_best[j])]), float(wds[int(iw_best[j])]), int(res["step_best"][b]))
            if j in keep_params:
                params_out[(h, j)] = [p[b].to(torch.float16).cpu().numpy() for p in res["best_params"]]
        del res_l
    return dict(sel=sel, fix=fix, sel_test=sel_test, fix_test=fix_test, grid_pred=grid_pred, cfg=cfg, ce_sel=ce_sel, ce_fix=ce_fix,
                ce_hist=ce_hist, train_loss=tl, params=params_out)


def nest_linear(nl: dict, lin: dict, half: np.ndarray) -> np.ndarray:
    """入れ子の選択（gate-code 第 2 回の 1）：半分 h・目標 j ごとに，線形のネット lin の選んだ版の損失が nl の選んだ版の損失以下なら，
    nl の選んだ版（候補の点数，テストの点数，損失，選んだ点）を lin のもので置き換える（隠れ層の項を使わない選択）．
    戻り：used_hidden (2, J) 真偽（隠れ層の項を使った版が選ばれたか）．固定の版と格子の全点の判定は変えない。"""
    J = nl["sel"].shape[0]
    used = np.ones((2, J), bool)
    for h in (0, 1):
        te = torch.as_tensor(np.flatnonzero(half == h), device=S.DEV)
        for j in range(J):
            if lin["ce_sel"][h, j] <= nl["ce_sel"][h, j]:
                used[h, j] = False
                nl["sel"][j, te] = lin["sel"][j, te]
                nl["sel_test"][h, j] = lin["sel_test"][h, j]
                nl["ce_sel"][h, j] = lin["ce_sel"][h, j]
                nl["cfg"][(h, j)] = lin["cfg"][(h, j)]
    nl["used_hidden"] = used
    return used


def soft_ce(Q: torch.Tensor, L: torch.Tensor) -> float:
    """出所の確率 Q (n, 4) と点数 L (n, 4) の交差エントロピー（行の平均）。"""
    return float(-(Q.float() * torch.log_softmax(L.float(), -1)).sum(1).mean())


def balanced_weights(y: np.ndarray, idx: np.ndarray, n: int) -> torch.Tensor:
    """idx の行に，感情の釣り合いの重み（len/(4 × その感情の件数)）を置いた one-hot (n, 4)（float32）。"""
    G = torch.zeros(n, 4, device=S.DEV, dtype=torch.float32)
    yy = y[idx]
    cnt = np.bincount(yy, minlength=4).astype(float)
    w = len(idx) / (4 * np.maximum(cnt[yy], 1))
    G[torch.as_tensor(idx, device=S.DEV), torch.as_tensor(yy, device=S.DEV)] = torch.as_tensor(w, device=S.DEV, dtype=torch.float32)
    return G


def argmax8(Z: torch.Tensor) -> np.ndarray:
    return Z.argmax(-1).to(torch.int8).cpu().numpy()


def f16(Z: torch.Tensor) -> np.ndarray:
    return Z.detach().to(torch.float16).cpu().numpy()


def a_matrix(P: torch.Tensor, y: np.ndarray) -> dict:
    """出所の確率 P (n, 4) を本当の感情 y ごとに平均した 4×4 の行列 A（行 y，列 k）の 16 個の値と，最小の特異値，条件数（設計 3.7 節）。"""
    Pn = P.float().cpu().numpy()
    A = np.stack([Pn[y == k].mean(0) if (y == k).any() else np.full(4, np.nan) for k in range(4)])
    out = {f"A_{yy}_{k}": float(A[yy, k]) for yy in range(4) for k in range(4)}
    if not np.isfinite(A).all():
        return out | dict(A_min_sv=float("nan"), A_cond=float("nan"))
    sv = np.linalg.svd(A, compute_uv=False)
    return out | dict(A_min_sv=float(sv.min()), A_cond=float(sv.max() / max(sv.min(), 1e-12)))


# ------------------------------------------------------------------ 候補の上の区間（分割をまたいで塊の重みを 1 回だけ引く．設計 4.1 節，S18）

def effect_stats_pool(cellA: dict, cellB: dict, g_eff: float, rng) -> dict:
    """cellA，cellB：{(f, r): counts}．g_eff ≥ 20 なら，引き直しの 1 回ごとに全分割の塊の和集合の上で重みを 1 組引き，各分割にはその候補に現れる塊の重みを当てる．
    g_eff < 20 なら dai1_stats.effect_stats（塊を全分割から同時に除く jackknife）をそのまま使う。"""
    if g_eff < 20:
        return ST.effect_stats(cellA, cellB, g_eff, rng)
    keys = sorted(cellA)
    folds = sorted({f for f, _ in keys})
    rs_of = {f: [r for ff, r in keys if ff == f] for f in folds}
    d = {f: [ST.uar_full(cellA[(f, r)]) - ST.uar_full(cellB[(f, r)]) for r in rs_of[f]] for f in folds}
    est = ST.theta(d)
    R = {f: len(d[f]) for f in folds}
    s2 = {f: (float(np.var(d[f], ddof=1)) if R[f] > 1 else 0.0) for f in folds}
    v_seed = float(sum(s2[f] / R[f] for f in folds) / len(folds) ** 2)
    all_cl = sorted({c for k in keys for c in cellA[k][0]})
    pos = {nm: i for i, nm in enumerate(all_cl)}
    G = len(all_cl)
    Wall = rng.multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(float)                # (B, G)：全分割で共通の 1 組
    boots = np.zeros(N_BOOT); nval = np.zeros(N_BOOT)
    for f in folds:
        ua, ub = [], []
        for r in rs_of[f]:                                                                    # (分割, 種) ごとに自分の塊の名前で重みを当てる（S18 の追記）
            names = cellA[(f, r)][0]
            if len(names) != len(cellB[(f, r)][0]) or not (names == cellB[(f, r)][0]).all():
                raise SystemExit("effect_stats_pool：cellA と cellB の塊の並びが違う")
            Wf = Wall[:, [pos[nm] for nm in names]]
            ua.append(ST.uar_rows(Wf @ cellA[(f, r)][1], Wf @ cellA[(f, r)][2])); ub.append(ST.uar_rows(Wf @ cellB[(f, r)][1], Wf @ cellB[(f, r)][2]))
        dm = np.nanmean(np.stack(ua) - np.stack(ub), axis=0)
        ok = np.isfinite(dm)
        boots[ok] += dm[ok]; nval[ok] += 1
    boots = np.where(nval > 0, boots / np.maximum(nval, 1), np.nan)
    boots = boots[np.isfinite(boots)]
    v_cl = float(np.var(boots, ddof=1))
    se = float(np.sqrt(v_cl + v_seed))
    p, ident = ST.pval(est, se, None)
    return dict(effect=est, n_folds=len(folds), n_units=sum(R.values()), lo=est - 1.96 * se, hi=est + 1.96 * se, p=p, se=se,
                method="boot_cl_shared+seed", df=np.nan, v_jk=v_cl, v_seed=v_seed, identical=ident, n_clusters_union=G)


# ------------------------------------------------------------------ 単位

def run_unit(p3: Path, out: Path, fz: dict, commit: str, commit11: str) -> dict:
    name, T, c, f, r = P11.unit_parts(p3)
    rows_dir = out / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for p in (rows_dir / f"{name}.npz", rows_dir / f"{name}_nl.csv", rows_dir / f"{name}_lab.csv"):
        if p.exists():
            raise SystemExit(f"{p} は既にある（重複を作らない．消すか別の --out にする）")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    srcs = [x for x in DD.CORPORA if x not in (T, c)]
    z1, z2, z3 = P11.load_unit_npz(name, fz, T)
    p11 = D11 / "rows" / f"{name}.npz"
    if not p11.exists():
        raise SystemExit(f"DAI-11 の保存 {p11} が無い")
    z11 = np.load(p11, allow_pickle=False)
    if str(z11["commit"]) != commit11:
        raise SystemExit(f"{name}：DAI-11 の保存のコミット {z11['commit']} が provenance.json の {commit11} と違う")
    tether = P11.parse_tether(z3["tether"])
    u = P11.Unit2(D11 / "enc2", T, c, srcs, f, r)
    n, nt = u.n, len(u.test)
    if int(z11["n_pool"]) != n or len(z11["test_y"]) != nt:
        raise SystemExit(f"{name}：候補・テストの本数が DAI-11 の保存と違う")
    tol_n, tol_t = max(1, int(MISMATCH_FRAC * n)), max(1, int(MISMATCH_FRAC * nt))
    store, mism, conv = {}, {}, {}
    nd = {m: n / u.Xp[m].shape[1] for m in MODS}

    def check(key: str, k: int, tol: int, what: str):
        mism[key] = k
        if k > tol:
            raise SystemExit(f"{name} {key}：{what}が保存と {k} 本違う")

    # --- none（第 1 エンコーダの K 本だけの版．DAI-1 の p1 の W・b）：テストと候補の判定を DAI-1・DAI-11 の保存と照合
    none_p = {}
    for m in MODS + ("AVT",):
        for K in KS_ALL:
            W, b = P11.none_wb(u, z1, m, K, tether)
            none_p[(m, K)] = P11.probs(u.Xp[m], W, b)
            if m == "AVT":
                continue
            pred = argmax8(S.logits(u.Xt[m], W, b))
            saved = z1[f"p1_{m}_start_pred"] if K == 0 else z1[f"p1_{m}_pred"][P11.p1_item(z1, m, P11.tether_tag(K, *tether[(m, K)]))]
            check(f"{m}_K{K}_test_none", int((pred != saved.astype(np.int8)).sum()), tol_t, "none のテストの判定")
            store[f"K{K}_{m}_none"] = pred
            store[f"K{K}_{m}_noneloco"] = z11[f"K{K}_{m}_noneloco"].astype(np.int8)
            pool_none = argmax8(none_p[(m, K)])
            check(f"{m}_K{K}_pool_none", int((pool_none != z11[f"pool_none_K{K}_{m}"].astype(np.int8)).sum()), tol_n, "none の候補の判定")
            store[f"pool_none_K{K}_{m}"] = pool_none
            for t_ in RET11_TEACHERS:                                                    # DAI-11 の線形の学び直しのテストの判定（R2 の相手．写し）
                store[f"K{K}_{m}_ret11_{t_}"] = z11[f"K{K}_{m}_{t_}"].astype(np.int8)

    # --- enc2・enc2fuse（第 2 エンコーダの K 本だけの版．DAI-11 の保存の W・b を作り直した特徴に当てる），other（DAI-1 の ret の 3 つの積）
    enc2_p, other_p = {}, {}
    for m in MODS + ("AVT",):
        for K in KS_ALL:
            W, b = S.t(z11[f"enc2_W_K{K}_{m}"]), S.t(z11[f"enc2_b_K{K}_{m}"])
            enc2_p[(m, K)] = P11.probs(u.Xp[m + "2"], W, b)
    for K in KS_ALL:
        L = 0
        for m in MODS:
            W, b = S.t(z2[f"p2_K{K}_ret_{m}_W"]), S.t(z2[f"p2_K{K}_ret_{m}_b"])
            L = L + P11.probs(u.Xp[m], W, b).clamp(min=1e-12).log()
        other_p[K] = torch.softmax(L, -1)

    # --- pair（ほかの 2 つのモダリティの K 本だけの版の確率の積．行ごとに正規化）
    pair_p = {}
    for m in MODS:
        for K in KS_ALL:
            L = 0
            for o in MODS:
                if o != m:
                    L = L + none_p[(o, K)].clamp(min=1e-12).log()
            pair_p[(m, K)] = torch.softmax(L, -1)

    cl_pool = u.cp.df["cluster"].to_numpy()[u.pool].astype(str)
    sid_pool = u.cp.df["sample_id"].to_numpy()[u.pool].astype(str)
    half = balanced_halves(cl_pool)                                                     # 主の分け方（M3）
    half11 = name_order_halves(cl_pool)                                                 # DAI-11 の分け方（照合にだけ）
    store["pool_half"], store["pool_half11"] = half.astype(np.int8), half11.astype(np.int8)
    half_counts = np.stack([np.bincount(u.ypool[half == h], minlength=4) for h in (0, 1)])
    store["half_counts"] = half_counts
    # 候補の部分集合の印（A1）
    subs_pool = dai3.subset_masks(c, sid_pool, cl_pool)
    if c == "MELD":
        dlg = u.cp.df["dialogue"].to_numpy()[u.pool].astype(str)
        free = np.ones(n, bool)
        for h in (0, 1):
            others = set(dlg[half != h])
            free[half == h] = ~np.isin(dlg[half == h], list(others))
        subs_pool["dlgfree"] = free
    for nm, mk in subs_pool.items():
        store[f"pool_sub_{nm}"] = np.asarray(mk, bool).astype(np.int8)

    # --- pair_shuf（入れ替えの対照．同じ感情の発話の間で，2 つの半分をまたいで入れ替える．候補のラベルを使う）
    seed_base = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % (2 ** 31)
    shuf_src = {}
    for mi, m in enumerate(MODS):
        rng_s = np.random.default_rng(seed_base + 100 + mi)
        src = np.arange(n)
        unchanged = 0
        for h in (0, 1):
            for k in range(4):
                rows_h = np.flatnonzero((half == h) & (u.ypool == k)); rows_o = np.flatnonzero((half != h) & (u.ypool == k))
                if len(rows_h) and len(rows_o):
                    src[rows_h] = rng_s.choice(rows_o, size=len(rows_h), replace=True)
                else:
                    unchanged += len(rows_h)
        mism[f"{m}_pair_shuf_unchanged_rows"] = unchanged                                       # もう一方の半分に同じ感情が無く入れ替えられなかった行（gate-code の記録）
        shuf_src[m] = src
        store[f"pool_shuf_src_{m}"] = src.astype(np.int32)

    def teacher(m: str, t_: str, K: int) -> torch.Tensor:
        if t_ == "self":
            return none_p[(m, K)]
        if t_ == "enc2":
            return enc2_p[(m, K)]
        if t_ == "fuse1":
            return none_p[("AVT", K)]
        if t_ == "other":
            return other_p[K]
        if t_ == "enc2fuse":
            return enc2_p[("AVT", K)]
        if t_ == "pair":
            return pair_p[(m, K)]
        if t_ == "pair_shuf":
            return pair_p[(m, K)][torch.as_tensor(shuf_src[m], device=S.DEV)]
        raise KeyError(t_)

    nl_rows, lab_rows = [], []
    for mi, m in enumerate(MODS):
        cols = [(t_, K) for t_ in TEACHERS for K in KS_ALL]
        J = len(cols)
        P_list = [teacher(m, t_, K) for t_, K in cols]
        for j, (t_, K) in enumerate(cols):                                               # 出所の候補の判定を DAI-11 の保存と照合（m2）
            pt = argmax8(P_list[j])
            store[f"pool_teacher_K{K}_{m}_{t_}"] = pt
            if t_ in ("enc2", "fuse1", "other", "enc2fuse"):
                check(f"{m}_K{K}_pool_teacher_{t_}", int((pt != z11[f"pool_teacher_K{K}_{m}_{t_}"].astype(np.int8)).sum()), tol_n, f"出所 {t_} の候補の判定")

        def centered_logs(P_sub):
            Y = torch.cat([P.clamp(min=1e-12).log() for P in P_sub], 1)
            return Y - Y.reshape(n, len(P_sub), 4).mean(-1, keepdim=True).expand(-1, -1, 4).reshape(n, -1)

        # 線形に読める分（リッジ）：(i) DAI-11 の分け方で DAI-11 の pool_proj と照合，(ii) 主の分け方で記述
        cols11 = [(t_, K) for t_ in RET11_TEACHERS for K in KS_ALL]
        Y11 = centered_logs([teacher(m, t_, K) for t_, K in cols11])
        Yhat11, _, _ = P11.representability(u.Xp[m], Y11, half11)
        for j, (t_, K) in enumerate(cols11):
            pp = argmax8(Yhat11[:, 4 * j: 4 * j + 4])
            check(f"{m}_K{K}_pool_proj_{t_}_vs_dai11", int((pp != z11[f"pool_proj_K{K}_{m}_{t_}"].astype(np.int8)).sum()), tol_n, f"線形に読める分（DAI-11 の分け方）の判定")
        Y = centered_logs(P_list)
        Yhat, lam, r2 = P11.representability(u.Xp[m], Y, half)
        if not bool(torch.isfinite(Yhat).all()):
            raise SystemExit(f"{name} {m}：リッジの交差の予測に有限でない値がある")

        # ネット nl と lince（同じ損失・同じ格子・同じ選び方．pair の nl の重みを残す）
        X32, Xt32 = u.Xp[m].float(), u.Xt[m].float()
        Q = torch.stack([P.float() for P in P_list])
        keep = tuple(j for j, (t_, K) in enumerate(cols) if t_ == "pair")
        nets = {"nl": crossfit_nets(X32, Q, half, Xt32, seed_base + mi, HIDDEN, keep_params=keep),
                "lince": crossfit_nets(X32, Q, half, Xt32, seed_base + 10 + mi, 0, keep_params=keep),
                "lince0": crossfit_nets(X32, Q, half, Xt32, seed_base + 50 + mi, 0, zero=True)}      # 初期値 0 の線形（記述．gate-code 第 1 回）
        pre_nl = dict(sel=nets["nl"]["sel"].clone(), sel_test=nets["nl"]["sel_test"].clone(), cfg=dict(nets["nl"]["cfg"]), ce_sel=nets["nl"]["ce_sel"].copy())   # 入れ子の前（隠れ層の版）の量（gate-probe 第 4 回の 2，gate-code 第 3 回の 5）
        used_hidden = nest_linear(nets["nl"], nets["lince"], half)                                  # 入れ子の選択（3.3 節）
        for kind, res in nets.items():
            conv[f"{kind}_{m}"] = dict(items=[f"{t_}_K{K}" for t_, K in cols], lrs=list(LRS), wds=list(WDS), ckpts=list(CKPTS),
                                       ce_hist=res["ce_hist"].tolist(), train_loss_final=res["train_loss"].tolist(),
                                       cfg={f"h{h}_{t_}_K{K}": list(res["cfg"][(h, j)]) for h in (0, 1) for j, (t_, K) in enumerate(cols)})
            store[f"grid_pred_{kind}_{m}"] = res["grid_pred"]                                   # 格子の全点の候補の判定（選ぶ規則を替えた版を後から出すため．R3）
        for (h, j), ps in nets["nl"]["params"].items():
            t_, K = cols[j]
            for pi, pn in enumerate(("W1", "c1", "W2", "c", "Wl")):
                store[f"w_nl_pair_K{K}_{m}_h{h}_{pn}"] = ps[pi]
        for (h, j), ps in nets["lince"]["params"].items():
            t_, K = cols[j]
            for pi, pn in enumerate(("W", "c")):
                store[f"w_lince_pair_K{K}_{m}_h{h}_{pn}"] = ps[pi]
        # 格子の全点の候補の上の UAR（K 本を除いた発話の上）
        for kind, res in nets.items():                                                      # 格子の全点の候補の上の UAR（K 本を除いた発話）
            grid = np.zeros((J, len(LRS), len(WDS), len(CKPTS)), np.float32)
            for j, (t_, K) in enumerate(cols):
                nk = u.nk(K)
                for il in range(len(LRS)):
                    for iw_ in range(len(WDS)):
                        for ic in range(len(CKPTS)):
                            grid[j, il, iw_, ic] = P11.metrics(u.ypool[nk], res["grid_pred"][il, iw_, ic, j].astype(int)[nk])["uar"]
            store[f"grid_uar_{kind}_{m}"] = grid

        # 候補の全ラベルの分類器（Q3）：同じ格子．選択は予測する半分の釣り合いの重みつきの損失（ラベルを使う．記述）
        Qlab = torch.zeros(1, n, 4, device=S.DEV)
        for h in (0, 1):
            te = np.flatnonzero(half == h)
            Qlab[0, torch.as_tensor(te, device=S.DEV)] = balanced_weights(u.ypool, te, n)[torch.as_tensor(te, device=S.DEV)]
        lab = {"nl": crossfit_nets(X32, Qlab, half, Xt32, seed_base + 20 + mi, HIDDEN, ckpts=LAB_CKPTS, steps=LAB_STEPS),
               "lince": crossfit_nets(X32, Qlab, half, Xt32, seed_base + 30 + mi, 0, ckpts=LAB_CKPTS, steps=LAB_STEPS)}
        pre_lab = dict(sel=lab["nl"]["sel"].clone(), sel_test=lab["nl"]["sel_test"].clone(), cfg=dict(lab["nl"]["cfg"]))
        lab_used_hidden = nest_linear(lab["nl"], lab["lince"], half)                                 # Q3 も同じ入れ子の選択
        lab_edge = {f"lab_{kind}_edge_{ax}_{side}_h{h}": (lab[kind]["cfg"][(h, 0)][1] == (min(WDS) if side == "low" else max(WDS)) if ax == "wd"
                                                           else lab[kind]["cfg"][(h, 0)][2] == (LAB_CKPTS[0] if side == "low" else LAB_CKPTS[-1]))
                    for kind in lab for ax in ("wd", "step") for side in ("low", "high") for h in (0, 1)}
        for kind in lab:                                                                    # Q3 の収束の目安（最後の保存点で，1 つ前からの下がりが 0.5% 超なら未収束）
            for h in (0, 1):
                lr_, wd_, st_ = lab[kind]["cfg"][(h, 0)]
                src = lab["lince"] if (kind == "nl" and not lab_used_hidden[h, 0]) else lab[kind]
                hist = src["ce_hist"][h, 0, LRS.index(lr_), WDS.index(wd_)]
                lab_edge[f"lab_{kind}_conv_h{h}"] = not (st_ == LAB_CKPTS[-1] and (hist[-2] - hist[-1]) > CONV_REL * hist[-2])
        conv[f"lab_{m}"] = {kind: dict(cfg={f"h{h}": list(res["cfg"][(h, 0)]) for h in (0, 1)}, ce_hist=res["ce_hist"].tolist()) for kind, res in lab.items()}
        for kind, res in lab.items():
            store[f"grid_pred_lab_{kind}_{m}"] = res["grid_pred"]                             # Q3 の格子の全点の候補の判定（延ばす前の格子で選んだ版を後から作るため．gate-code 第 5 回）
        p_lab = {kind: argmax8(res["sel"][0]) for kind, res in lab.items()}
        p_labfix = {kind: argmax8(res["fix"][0]) for kind, res in lab.items()}
        p_lab_hidden = argmax8(pre_lab["sel"][0])
        store[f"labtest_nlhid_{m}"] = argmax8(torch.log_softmax(pre_lab["sel_test"][:, 0], -1).mean(0))     # 入れ子の前の隠れ層の版のテストの判定（Q3）
        for h in (0, 1):
            _, wd_p, st_p = pre_lab["cfg"][(h, 0)]
            lab_edge[f"lab_nl_hidden_edge_wd_low_h{h}"] = wd_p == min(WDS); lab_edge[f"lab_nl_hidden_edge_wd_high_h{h}"] = wd_p == max(WDS)
            lab_edge[f"lab_nl_hidden_edge_step_low_h{h}"] = st_p == LAB_CKPTS[0]; lab_edge[f"lab_nl_hidden_edge_step_high_h{h}"] = st_p == LAB_CKPTS[-1]
        p_labtest = {kind: argmax8(torch.log_softmax(res["sel_test"][:, 0], -1).mean(0)) for kind, res in lab.items()}
        for kind in lab:
            store[f"pool_lab_{kind}_{m}"], store[f"pool_labfix_{kind}_{m}"] = p_lab[kind], p_labfix[kind]
            store[f"labtest_{kind}_{m}"] = p_labtest[kind]                                      # Q3 の分類器のテストの判定（選択に使っていないデータ．gate-code 第 1 回）
        for K in KS_ALL:
            nk = u.nk(K); yk = u.ypool[nk]
            ul = {kind: P11.metrics(yk, p_lab[kind].astype(int)[nk])["uar"] for kind in lab}
            ul["nl_hidden"] = P11.metrics(yk, p_lab_hidden.astype(int)[nk])["uar"]
            ulf = {kind: P11.metrics(yk, p_labfix[kind].astype(int)[nk])["uar"] for kind in lab}
            lab_rows.append(dict(unit=name, T=T, c=c, fold=f, seed=r, student=m, K=K, n_pool=n, n_nk=len(nk),
                                 uar_lab_lince=ul["lince"], uar_lab_nl=ul["nl"], lab_nl_minus_lince=ul["nl"] - ul["lince"], uar_lab_nl_hidden=ul["nl_hidden"],
                                 test_uar_lab_lince=P11.metrics(u.ytest, p_labtest["lince"].astype(int))["uar"], test_uar_lab_nl=P11.metrics(u.ytest, p_labtest["nl"].astype(int))["uar"], **lab_edge,
                                 lab_nl_used_hidden_h0=bool(lab_used_hidden[0, 0]), lab_nl_used_hidden_h1=bool(lab_used_hidden[1, 0]),
                                 uar_lab_lince_fixed=ulf["lince"], uar_lab_nl_fixed=ulf["nl"], lab_nl_minus_lince_fixed=ulf["nl"] - ulf["lince"],
                                 **{f"lab_{kind}_lr_h{h}": lab[kind]["cfg"][(h, 0)][0] for kind in lab for h in (0, 1)},
                                 **{f"lab_{kind}_wd_h{h}": lab[kind]["cfg"][(h, 0)][1] for kind in lab for h in (0, 1)},
                                 **{f"lab_{kind}_step_h{h}": lab[kind]["cfg"][(h, 0)][2] for kind in lab for h in (0, 1)}))

        # 4×4 の読み出しの上限（差分 C）：nl と lince の選んだ版の確率を入力に，候補の全ラベルで 4×4 の線形の読み出し（交差の予測．記述）
        Xread = torch.cat([torch.softmax(nets["nl"]["sel"], -1), torch.softmax(nets["lince"]["sel"], -1)], 0)   # (2J, n, 4)
        read = crossfit_nets(Xread, Qlab.expand(2 * J, -1, -1).contiguous(), half, torch.zeros(2 * J, 1, 4, device=S.DEV), seed_base + 40 + mi, 0,
                             wds=READ_WDS, ckpts=READ_CKPTS, steps=READ_STEPS)
        p_read = argmax8(read["sel"])                                                      # (2J, n)
        read_edge = {}
        for j in range(2 * J):
            for h in (0, 1):
                _, wd_r, st_r = read["cfg"][(h, j)]
                read_edge.setdefault("read_edge_wd_low", []).append(wd_r == min(READ_WDS) and min(READ_WDS) > 0)      # 罰則なし（0）は下端として数えない
                read_edge.setdefault("read_edge_wd_high", []).append(wd_r == max(READ_WDS))
                read_edge.setdefault("read_edge_step_low", []).append(st_r == READ_CKPTS[0]); read_edge.setdefault("read_edge_step_high", []).append(st_r == READ_CKPTS[-1])
        for k_, v_ in read_edge.items():
            mism[f"{m}_{k_}_frac"] = float(np.mean(v_))                                          # 4×4 の読み出しの格子の端の割合（単位ごと）
        for j, (t_, K) in enumerate(cols):
            store[f"pool_read_nl_K{K}_{m}_{t_}"], store[f"pool_read_lince_K{K}_{m}_{t_}"] = p_read[j], p_read[J + j]

        # 行（m, 出所, K）
        for j, (t_, K) in enumerate(cols):
            nk = u.nk(K); yk = u.ypool[nk]
            P = P_list[j]
            Lr = Yhat[:, 4 * j: 4 * j + 4]
            Nsel, Nfix, Lsel, Lfix, L0sel = nets["nl"]["sel"][j], nets["nl"]["fix"][j], nets["lince"]["sel"][j], nets["lince"]["fix"][j], nets["lince0"]["sel"][j]
            pt = store[f"pool_teacher_K{K}_{m}_{t_}"]
            pr, pnl, pnlf, pli, plif, pl0 = (argmax8(Z) for Z in (Lr, Nsel, Nfix, Lsel, Lfix, L0sel))
            ps = store[f"pool_none_K{K}_{m}"]
            store[f"pool_ridge_K{K}_{m}_{t_}"], store[f"pool_nl_K{K}_{m}_{t_}"], store[f"pool_nlfix_K{K}_{m}_{t_}"] = pr, pnl, pnlf
            store[f"pool_lince_K{K}_{m}_{t_}"], store[f"pool_lincefix_K{K}_{m}_{t_}"], store[f"pool_lince0_K{K}_{m}_{t_}"] = pli, plif, pl0
            store[f"score_ridge_K{K}_{m}_{t_}"], store[f"score_nl_K{K}_{m}_{t_}"], store[f"score_lince_K{K}_{m}_{t_}"] = f16(Lr), f16(Nsel), f16(Lsel)
            store[f"score_nl_test_K{K}_{m}_{t_}"], store[f"score_lince_test_K{K}_{m}_{t_}"] = f16(nets["nl"]["sel_test"][:, j]), f16(nets["lince"]["sel_test"][:, j])
            for kind, res in nets.items():
                store[f"K{K}_{m}_{kind}_{t_}"] = argmax8(torch.log_softmax(res["sel_test"][:, j], -1).mean(0))      # 2 つの半分のネットの平均
                if kind != "lince0":
                    store[f"K{K}_{m}_{kind}fix_{t_}"] = argmax8(torch.log_softmax(res["fix_test"][:, j], -1).mean(0))
            store[f"K{K}_{m}_nlhid_{t_}"] = argmax8(torch.log_softmax(pre_nl["sel_test"][:, j], -1).mean(0))     # 入れ子の前の隠れ層の版（gate-code 第 3 回の 5）
            mt, ms = P11.metrics(yk, pt.astype(int)[nk]), P11.metrics(yk, ps.astype(int)[nk])
            m_r, m_nl, m_nlf, m_li, m_lif, m_l0 = (P11.metrics(yk, Z.astype(int)[nk]) for Z in (pr, pnl, pnlf, pli, plif, pl0))
            flags = {}
            for h in (0, 1):                                                                    # 入れ子の前の隠れ層の版の端（gate-probe 第 4 回の 1）
                _, wd_p, st_p = pre_nl["cfg"][(h, j)]
                flags[f"nl_hidden_edge_wd_low_h{h}"] = wd_p == min(WDS); flags[f"nl_hidden_edge_wd_high_h{h}"] = wd_p == max(WDS)
                flags[f"nl_hidden_edge_step_low_h{h}"] = st_p == CKPTS[0]; flags[f"nl_hidden_edge_step_high_h{h}"] = st_p == CKPTS[-1]
                flags[f"ce_nl_hidden_h{h}"] = float(pre_nl["ce_sel"][h, j])
            for kind in ("nl", "lince", "lince0"):                                              # 収束の目安と格子の端の印（半分ごと．入れ子の後）
                for h in (0, 1):
                    lr_, wd_, st_ = nets[kind]["cfg"][(h, j)]
                    il, iw = LRS.index(lr_), WDS.index(wd_)
                    src = nets["lince"] if (kind == "nl" and not used_hidden[h, j]) else nets[kind]   # 置き換わった半分は lince の履歴（gate-probe 第 4 回の記録 7）
                    hist = src["ce_hist"][h, j, il, iw]
                    last, prev = hist[-1], hist[-2]
                    flags[f"{kind}_conv_h{h}"] = not (st_ == CKPTS[-1] and (prev - last) > CONV_REL * prev)
                    flags[f"{kind}_edge_wd_low_h{h}"] = (wd_ == min(WDS)); flags[f"{kind}_edge_wd_high_h{h}"] = (wd_ == max(WDS))        # 軸ごとの端（I18 の追記）
                    flags[f"{kind}_edge_step_low_h{h}"] = (st_ == CKPTS[0]); flags[f"{kind}_edge_step_high_h{h}"] = (st_ == CKPTS[-1])
                flags[f"{kind}_conv"] = flags[f"{kind}_conv_h0"] and flags[f"{kind}_conv_h1"]
            m_read_nl, m_read_li = P11.metrics(yk, p_read[j].astype(int)[nk]), P11.metrics(yk, p_read[J + j].astype(int)[nk])
            hk = half[nk]
            amat = a_matrix(P[torch.as_tensor(nk, device=S.DEV)], yk)
            row = dict(unit=name, T=T, c=c, fold=f, seed=r, student=m, teacher=t_, K=K, n_pool=n, n_nk=len(nk), n_over_d=nd[m],
                       n_half0=int((half == 0).sum()), n_half1=int((half == 1).sum()),
                       uar_s=ms["uar"], uar_t=mt["uar"], delta_uar=mt["uar"] - ms["uar"], agree=float((pt[nk] == ps[nk]).mean()),
                       uar_ridge=m_r["uar"], ridge_gain=m_r["uar"] - ms["uar"], r2_ridge=float(r2[j]), lam_rel=float(lam[j]), ce_ridge=soft_ce(P, Lr),
                       uar_lince=m_li["uar"], lince_gain=m_li["uar"] - ms["uar"], uar_lince_fixed=m_lif["uar"], ce_lince=soft_ce(P, Lsel), ce_lince_fixed=soft_ce(P, Lfix),
                       uar_nl=m_nl["uar"], nl_gain=m_nl["uar"] - ms["uar"], uar_nl_fixed=m_nlf["uar"], ce_nl=soft_ce(P, Nsel), ce_nl_fixed=soft_ce(P, Nfix),
                       uar_nl_hidden=P11.metrics(yk, argmax8(pre_nl["sel"][j]).astype(int)[nk])["uar"], ce_nl_hidden=soft_ce(P, pre_nl["sel"][j]),
                       uar_lince0=m_l0["uar"], lince0_gain=m_l0["uar"] - ms["uar"], ce_lince0=soft_ce(P, L0sel), nl_minus_lince0=m_nl["uar"] - m_l0["uar"],
                       nl_minus_lince=m_nl["uar"] - m_li["uar"], nl_minus_lince_fixed=m_nlf["uar"] - m_lif["uar"], lince_minus_ridge=m_li["uar"] - m_r["uar"],
                       lince_below_ridge=bool(m_li["uar"] < m_r["uar"] - BELOW_RIDGE_TOL), **flags,
                       nl_used_hidden_h0=bool(used_hidden[0, j]), nl_used_hidden_h1=bool(used_hidden[1, j]),
                       uar_nl_h0=P11.metrics(yk[hk == 0], pnl.astype(int)[nk][hk == 0])["uar"], uar_nl_h1=P11.metrics(yk[hk == 1], pnl.astype(int)[nk][hk == 1])["uar"],
                       uar_lince_h0=P11.metrics(yk[hk == 0], pli.astype(int)[nk][hk == 0])["uar"], uar_lince_h1=P11.metrics(yk[hk == 1], pli.astype(int)[nk][hk == 1])["uar"],
                       agree_ridge_t=float((pr[nk] == pt[nk]).mean()), agree_lince_t=float((pli[nk] == pt[nk]).mean()), agree_nl_t=float((pnl[nk] == pt[nk]).mean()),
                       agree_nl_lince=float((pnl[nk] == pli[nk]).mean()),
                       uar_read_nl=m_read_nl["uar"], uar_read_lince=m_read_li["uar"], **amat,
                       **{f"{kind}_lr_h{h}": nets[kind]["cfg"][(h, j)][0] for kind in nets for h in (0, 1)},
                       **{f"{kind}_wd_h{h}": nets[kind]["cfg"][(h, j)][1] for kind in nets for h in (0, 1)},
                       **{f"{kind}_step_h{h}": nets[kind]["cfg"][(h, j)][2] for kind in nets for h in (0, 1)},
                       **{f"ce_{kind}_h{h}": float(nets[kind]["ce_sel"][h, j]) for kind in nets for h in (0, 1)})
            if t_ == "pair":                                                              # pair の 2 つの成分の正誤の κ（記録 D8）
                comp = [argmax8(none_p[(o, K)]).astype(int)[nk] == yk for o in MODS if o != m]
                row["pair_comp_kappa"] = P11.kappa_coef(comp[0], comp[1])
            nl_rows.append(row)

    sid_t = u.cp.df["sample_id"].to_numpy()[u.test].astype(str)
    np.savez_compressed(rows_dir / f"{name}.npz", commit=commit, commit11=commit11, unit=name, T=T, c=c, fold=f, seed=r, n_pool=n,
                        n_over_d=json.dumps(nd), test_sample_id=sid_t, test_y=u.ytest.astype(np.int8),
                        test_cluster=u.cp.df["cluster"].to_numpy()[u.test].astype(str),
                        pool_sample_id=sid_pool, pool_y=u.ypool.astype(np.int8), pool_cluster=cl_pool, **{f"kpos_K{K}": u.kpos[K] for K in DD.KS},
                        mismatch=json.dumps(mism), conv=json.dumps(conv), **store)
    pd.DataFrame(nl_rows).to_csv(rows_dir / f"{name}_nl.csv", index=False)
    pd.DataFrame(lab_rows).to_csv(rows_dir / f"{name}_lab.csv", index=False)
    sec = time.time() - t0
    ints = [v for k, v in mism.items() if "frac" not in k]
    gpu_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float("nan")
    gpu_res = torch.cuda.max_memory_reserved() / 2 ** 30 if torch.cuda.is_available() else float("nan")
    print(f"単位 {name}: 候補 {n}（半分 {int((half == 0).sum())}/{int((half == 1).sum())}）テスト {nt} 保存 {len(store)} 行 {len(nl_rows)}+{len(lab_rows)} "
          f"判定の不一致の最大 {max(ints)} 本 GPU 最大 {gpu_gb:.1f} GB（予約 {gpu_res:.1f} GB）({sec:.0f} 秒)", flush=True)
    return dict(unit=name, sec=sec, gpu_max_gb=gpu_gb, gpu_reserved_gb=gpu_res)


# ------------------------------------------------------------------ 集計

def _add(d, key, cnt, label):
    if key in d:
        n0, c0, t0, m_ = d[key]
        if len(n0) != len(cnt[0]) or not (n0 == cnt[0]).all():
            raise SystemExit(f"{label}：塊の並びが同じ (f, r) のほかの T と違う")
        d[key] = (n0, c0 + cnt[1], t0 + cnt[2], m_ + 1)
    else:
        d[key] = (cnt[0], cnt[1].astype(float), cnt[2].astype(float), 1)


def _finish(cells):
    for k in cells:
        for kk, (n0, c0, t0, m_) in cells[k].items():
            cells[k][kk] = (n0, c0 / m_, t0 / m_)


def _diff(cells, c, ka, kb, geff, rng, extra: dict, pool: bool) -> dict | None:
    if (c, ka) not in cells or (c, kb) not in cells:
        return None
    A, B = cells[(c, ka)], cells[(c, kb)]
    if set(A) != set(B):
        raise SystemExit(f"{c} {ka} と {kb} の単位がそろわない")
    es = (effect_stats_pool if pool else ST.effect_stats)(A, B, geff, rng)
    return dict(c=c, **extra, a=ka, b=kb, est=es["effect"], lo=es["lo"], hi=es["hi"], se=es["se"], p=es["p"], method=es["method"], n_units=es["n_units"])


def _verdict(d: pd.DataFrame, floor: float):
    up = d[(d["lo"] > 0) & (d["est"] >= floor)]; down = d[(d["hi"] < 0) & (d["est"] <= -floor)]
    return ("上回った" if len(up) >= 3 else ("下回った" if len(down) >= 3 else "言えない")), up, down


def _robust(Dd, c, m, t_, q, sub, hit):
    """部分集合の頑健さ（4.2 節）：区間の端が 0 をまたがず，符号が全体と同じで，点推定が全体の半分以上の K の数。"""
    ds = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["teacher"] == t_) & (Dd["quantity"] == q) & (Dd["subset"] == sub) & (Dd["K"].isin(hit["K"]))]
    ds = ds.merge(hit[["K", "est"]].rename(columns={"est": "est_all"}), on="K")
    rob = ds[(np.sign(ds["est"]) == np.sign(ds["est_all"])) & ((ds["lo"] > 0) | (ds["hi"] < 0)) & (ds["est"].abs() >= 0.5 * ds["est_all"].abs())]
    return len(rob)


def summarize(out: Path, allow_partial: bool) -> None:
    files = sorted((out / "rows").glob("*.npz"))
    if len(files) != N_UNITS and not allow_partial:
        raise SystemExit(f"単位の出力が {len(files)} 個（{N_UNITS} のはず）．試走なら --allow-partial")
    rng = np.random.default_rng(SEED)
    commits = {str(np.load(p, allow_pickle=False)["commit"]) for p in files}
    if len(commits) != 1:
        raise SystemExit(f"行のコミットが 1 つでない：{commits}")
    missing = [p.name for p in files if not ((out / "rows" / f"{p.stem}_nl.csv").exists() and (out / "rows" / f"{p.stem}_lab.csv").exists())]
    if missing:
        raise SystemExit(f"npz と csv の対がそろわない：{missing[:5]}")
    pool_kinds = ("none", "teacher", "ridge", "lince", "lincefix", "lince0", "nl", "nlfix", "read_nl", "read_lince")
    cells, unit_rows, mism_rows, half_rows = {}, [], [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        T, c, f, r = str(z["T"]), str(z["c"]), int(z["fold"]), int(z["seed"])
        y, cl, sid = z["test_y"].astype(int), z["test_cluster"].astype(str), z["test_sample_id"].astype(str)
        yp, clp = z["pool_y"].astype(int), z["pool_cluster"].astype(str)
        kpos = {0: np.array([], int), **{K: z[f"kpos_K{K}"] for K in DD.KS}}
        nks = {K: np.setdiff1d(np.arange(len(yp)), kpos[K]) for K in KS_ALL}
        subs = dai3.subset_masks(c, sid, cl)
        psubs = {k[len("pool_sub_"):]: z[k].astype(bool) for k in z.files if k.startswith("pool_sub_")}
        hc = z["half_counts"]
        half_rows.append(dict(c=c, T=T, fold=f, seed=r, n_half0=int(hc[0].sum()), n_half1=int(hc[1].sum()), **{f"h{h}_e{e}": int(hc[h, e]) for h in (0, 1) for e in range(4)}))

        def add_pool(key, prd, K, label):
            nk = nks[K]
            _add(cells.setdefault((c, f"{key}|nk"), {}), (f, r), ST.counts(yp[nk], prd[nk], clp[nk]), label)
            for nm, mk in psubs.items():
                mk2 = nk[mk[nk]]
                if len(mk2) >= ST.MIN_TEST:
                    _add(cells.setdefault((c, f"{key}|nk|sub_{nm}"), {}), (f, r), ST.counts(yp[mk2], prd[mk2], clp[mk2]), f"{label} {nm}")

        for k in z.files:
            if z[k].dtype != np.int8:
                continue
            if z[k].shape == y.shape and k.startswith("K"):                                   # テストの判定
                prd = z[k].astype(int)
                _add(cells.setdefault((c, k), {}), (f, r), ST.counts(y, prd, cl), f"{p.name} {k}")
                unit_rows.append(dict(c=c, T=T, fold=f, seed=r, key=k, **P11.metrics(y, prd)))
                for nm, mk in subs.items():
                    mk = np.asarray(mk, bool)
                    if mk.any():
                        _add(cells.setdefault((c, f"{k}|sub_{nm}"), {}), (f, r), ST.counts(y[mk], prd[mk], cl[mk]), f"{p.name} {k} {nm}")
            elif z[k].shape == yp.shape and k.startswith("pool_") and any(k.startswith(f"pool_{kd}_K") for kd in pool_kinds):   # 候補の判定
                K = int(k.split("_K")[1].split("_")[0])
                add_pool(k, z[k].astype(int), K, f"{p.name} {k}")
            elif z[k].shape == yp.shape and (k.startswith("pool_lab_") or k.startswith("pool_labfix_")):
                for K in KS_ALL:
                    add_pool(f"{k}_K{K}", z[k].astype(int), K, f"{p.name} {k} K{K}")
        for k, v in json.loads(str(z["mismatch"])).items():
            mism_rows.append(dict(unit=str(z["unit"]), key=k, value=v))
    _finish(cells)
    U = pd.DataFrame(unit_rows); U.to_csv(out / "uar_units.csv", index=False)
    pd.DataFrame(mism_rows).to_csv(out / "checks_units.csv", index=False)
    pd.DataFrame(half_rows).to_csv(out / "halves_units.csv", index=False)
    R = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_nl.csv"))]); R.to_csv(out / "nl_units.csv", index=False)
    Lb = pd.concat([pd.read_csv(p) for p in sorted((out / "rows").glob("*_lab.csv"))]); Lb.to_csv(out / "lab_units.csv", index=False)
    num = [c_ for c_ in R.columns if c_ not in ("unit", "T", "c", "fold", "seed", "student", "teacher", "K")]
    Rc = R.groupby(["c", "student", "teacher", "K"])[num].mean().reset_index()
    Rc["n_units"] = R.groupby(["c", "student", "teacher", "K"]).size().to_numpy()
    if not allow_partial and (Rc["n_units"] != 30).any():
        raise SystemExit("単位の数が 30 でないセルがある")
    okd = Rc["delta_uar"] >= MIN_DELTA_SHARE
    for g_ in ("lince", "nl", "ridge"):
        Rc[f"share_{g_}"] = np.where(okd, Rc[f"{g_}_gain"] / Rc["delta_uar"].where(Rc["delta_uar"] != 0), np.nan)
    Rc.to_csv(out / "cells_nl.csv", index=False)
    lnum = [c_ for c_ in Lb.columns if c_ not in ("unit", "T", "c", "fold", "seed", "student", "K")]
    Lc = Lb.groupby(["c", "student", "K"])[lnum].mean().reset_index(); Lc.to_csv(out / "cells_lab.csv", index=False)

    # 差と区間（候補の上は effect_stats_pool，テストは effect_stats）
    geff = {c: ST.g_eff_of(c)[0] for c in DD.CORPORA}
    have_c = sorted(U["c"].unique())
    tsubs = sorted({k.split("|sub_")[1] for (c_, k) in cells if "|sub_" in k and "|nk" not in k})
    psubs_all = sorted({k.split("|sub_")[1] for (c_, k) in cells if "|nk|sub_" in k})
    rows = []
    for c in have_c:
        for m in MODS:
            for K in KS_ALL:
                for t_ in TEACHERS:
                    pre = f"K{K}_{m}"
                    pool_pairs = [(f"pool_nl_{pre}_{t_}", f"pool_lince_{pre}_{t_}", "nl_minus_lince"), (f"pool_nlfix_{pre}_{t_}", f"pool_lincefix_{pre}_{t_}", "nl_minus_lince_fixed"),
                                  (f"pool_nl_{pre}_{t_}", f"pool_none_{pre}", "nl_gain"), (f"pool_lince_{pre}_{t_}", f"pool_none_{pre}", "lince_gain"),
                                  (f"pool_nl_{pre}_{t_}", f"pool_lince0_{pre}_{t_}", "nl_minus_lince0"), (f"pool_lince0_{pre}_{t_}", f"pool_none_{pre}", "lince0_gain"),
                                  (f"pool_ridge_{pre}_{t_}", f"pool_none_{pre}", "ridge_gain"), (f"pool_teacher_{pre}_{t_}", f"pool_none_{pre}", "delta_uar"),
                                  (f"pool_read_nl_{pre}_{t_}", f"pool_nl_{pre}_{t_}", "read_nl_minus_nl"), (f"pool_read_nl_{pre}_{t_}", f"pool_read_lince_{pre}_{t_}", "read_nl_minus_read_lince")]
                    if t_ == "pair":
                        pool_pairs += [(f"pool_nl_{pre}_pair", f"pool_nl_{pre}_pair_shuf", "shuf_nl"), (f"pool_lince_{pre}_pair", f"pool_lince_{pre}_pair_shuf", "shuf_lince")]
                    for ka, kb, lab in pool_pairs:
                        d = _diff(cells, c, f"{ka}|nk", f"{kb}|nk", geff[c], rng, dict(student=m, teacher=t_, K=K, where="pool", quantity=lab, subset="all"), True)
                        if d:
                            rows.append(d)
                            if lab in ("nl_minus_lince", "nl_minus_lince_fixed", "nl_gain"):
                                for nm in psubs_all:
                                    d2 = _diff(cells, c, f"{ka}|nk|sub_{nm}", f"{kb}|nk|sub_{nm}", geff[c], rng, dict(student=m, teacher=t_, K=K, where="pool", quantity=lab, subset=nm), True)
                                    if d2:
                                        rows.append(d2)
                    test_pairs = [(f"{pre}_nl_{t_}", f"{pre}_lince_{t_}", "nl_vs_lince"), (f"{pre}_nlhid_{t_}", f"{pre}_lince_{t_}", "nlhid_vs_lince"), (f"{pre}_nl_{t_}", f"{pre}_lince0_{t_}", "nl_vs_lince0"), (f"{pre}_nl_{t_}", f"{pre}_none", "nl_vs_none"), (f"{pre}_nl_{t_}", f"{pre}_noneloco", "nl_vs_noneloco"),
                                  (f"{pre}_nlfix_{t_}", f"{pre}_lincefix_{t_}", "nlfix_vs_lincefix"), (f"{pre}_lince_{t_}", f"{pre}_none", "lince_vs_none")]
                    if t_ in RET11_TEACHERS:
                        test_pairs.append((f"{pre}_nl_{t_}", f"{pre}_ret11_{t_}", "nl_vs_ret11"))
                    for ka, kb, lab in test_pairs:
                        d = _diff(cells, c, ka, kb, geff[c], rng, dict(student=m, teacher=t_, K=K, where="test", quantity=lab, subset="all"), False)
                        if d:
                            Uc = U[U["c"] == c]
                            for met in ("war", "waf", "mf1"):
                                da = Uc[Uc["key"] == ka].set_index(["T", "fold", "seed"])[met]; db = Uc[Uc["key"] == kb].set_index(["T", "fold", "seed"])[met]
                                dd = (da - db).dropna()
                                ft = P11.fold_t({f_: dd.xs(f_, level="fold").to_numpy() for f_ in sorted({i[1] for i in dd.index})})
                                d.update({f"{met}_est": ft["est"], f"{met}_lo": ft["lo"], f"{met}_hi": ft["hi"]})
                            rows.append(d)
                            for nm in tsubs:
                                d2 = _diff(cells, c, f"{ka}|sub_{nm}", f"{kb}|sub_{nm}", geff[c], rng, dict(student=m, teacher=t_, K=K, where="test", quantity=lab, subset=nm), False)
                                if d2:
                                    rows.append(d2)
                for ka, kb, lab in ((f"pool_lab_nl_{m}_K{K}", f"pool_lab_lince_{m}_K{K}", "lab_nl_minus_lince"), (f"pool_labfix_nl_{m}_K{K}", f"pool_labfix_lince_{m}_K{K}", "lab_nl_minus_lince_fixed")):
                    d = _diff(cells, c, f"{ka}|nk", f"{kb}|nk", geff[c], rng, dict(student=m, teacher="labels", K=K, where="pool", quantity=lab, subset="all"), True)
                    if d:
                        rows.append(d)
                        for nm in psubs_all:
                            d2 = _diff(cells, c, f"{ka}|nk|sub_{nm}", f"{kb}|nk|sub_{nm}", geff[c], rng, dict(student=m, teacher="labels", K=K, where="pool", quantity=lab, subset=nm), True)
                            if d2:
                                rows.append(d2)
    Dd = pd.DataFrame(rows); Dd.to_csv(out / "diffs.csv", index=False)
    se_rows = []
    for (c, q), g in Dd[(Dd["where"] == "pool") & (Dd["subset"] == "all")].groupby(["c", "quantity"]):
        se_rows.append(dict(c=c, quantity=q, se_median=g["se"].median(), se_p90=g["se"].quantile(0.9), floor_over_se_median=FLOOR_POOL / g["se"].median() if g["se"].median() > 0 else np.nan, n=len(g)))
    pd.DataFrame(se_rows).to_csv(out / "se_summary.csv", index=False)

    # 数え上げ（4.2・4.3 節）
    crit = []
    pair_ok = {m: 0 for m in MODS}
    shuf_sub = {"MOSEI": "channel", "MELD": "dlgfree"}
    opt_bad = {}
    for c in have_c:                                                                     # 収束の目安（OPT）：未収束の行が 5% を超えたら R1 の行に印を付ける
        Rcc = R[R["c"] == c]
        for kind in ("nl", "lince", "lince0"):
            nc = int((~Rcc[f"{kind}_conv"].astype(bool)).sum())
            opt_bad[(c, kind)] = nc / max(len(Rcc), 1) > 0.05
            crit.append(dict(c=c, student="all", rule="OPT", teacher="all", quantity=f"{kind}_not_converged_rows", n_K=len(Rcc), n_K_up=nc,
                             verdict=("最適化の不足あり" if opt_bad[(c, kind)] else "最適化の不足は 5% 未満")))
    for c in have_c:
        for m in MODS:
            selfd = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["teacher"] == "self") & (Dd["quantity"] == "nl_minus_lince") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
            self_bad = int(((selfd["lo"] > 0) & (selfd["est"] >= FLOOR_POOL) | (selfd["hi"] < 0) & (selfd["est"] <= -FLOOR_POOL)).sum())
            for t_ in TEACHERS:
                d = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["teacher"] == t_) & (Dd["quantity"] == "nl_minus_lince") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                if len(d) == 0:
                    continue
                v, up, down = _verdict(d, FLOOR_POOL)
                dfx = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["teacher"] == t_) & (Dd["quantity"] == "nl_minus_lince_fixed") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                mg = d[["K", "est"]].merge(dfx[["K", "est"]], on="K", suffixes=("", "_fixed"))
                hit = up if v == "上回った" else (down if v == "下回った" else d.iloc[0:0])          # 判定の向きの K だけ（gate-probe 第 3 回の S2）
                mh = mg[mg["K"].isin(hit["K"])]
                fixed_support = int(((np.sign(mh["est"]) == np.sign(mh["est_fixed"])) & (mh["est_fixed"].abs() >= 0.5)).sum())   # 固定の版が同じ符号で絶対値 0.5 以上の K（支持する K）
                rule = {"self": "R1c", "pair_shuf": "R1shuf"}.get(t_, "R1")
                Rcm = R[(R["c"] == c) & (R["student"] == m) & (R["teacher"] == t_) & (R["K"].isin(KS_RULE))]
                row = dict(c=c, student=m, rule=rule, teacher=t_, quantity="nl_minus_lince", n_K=len(d), n_K_up=len(up), n_K_down=len(down), verdict=v,
                           K_up=",".join(map(str, up["K"])), K_down=",".join(map(str, down["K"])), est_min=d["est"].min(), est_max=d["est"].max(),
                           fixed_same_sign_n=int((np.sign(mg["est"]) == np.sign(mg["est_fixed"])).sum()), fixed_support_n=fixed_support,
                           note_optimism=bool(len(hit) and fixed_support <= len(hit) / 2), self_bad_K=self_bad,
                           note_opt=bool(opt_bad.get((c, "nl"), False) or opt_bad.get((c, "lince"), False)))
                for kind in ("nl", "lince", "nl_hidden"):                                        # 軸ごとの端の割合（半分 2 × K 4．入れ子の前の隠れ層の版も．I19）
                    for ax in ("wd", "step"):
                        for side in ("low", "high"):
                            row[f"edge_{ax}_{side}_frac_{kind}"] = float(Rcm[[f"{kind}_edge_{ax}_{side}_h0", f"{kind}_edge_{ax}_{side}_h1"]].to_numpy().mean()) if len(Rcm) else np.nan
                row["note_edge"] = bool(len(Rcm) and max(v_ for k_, v_ in row.items() if k_.startswith("edge_") and k_.endswith(("_nl", "_lince"))) > 0.5)
                row["note_edge_hidden"] = bool(len(Rcm) and max(v_ for k_, v_ in row.items() if k_.startswith("edge_") and k_.endswith("_nl_hidden")) > 0.5)
                if c in shuf_sub:
                    row["sub_name"] = shuf_sub[c]; row["sub_n_K_hit"] = len(hit); row["sub_n_K_robust"] = _robust(Dd, c, m, t_, "nl_minus_lince", shuf_sub[c], hit) if len(hit) else 0
                    row["note_shared"] = bool(len(hit) and row["sub_n_K_robust"] < 3)
                if t_ == "pair":
                    ds = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["teacher"] == "pair_shuf") & (Dd["quantity"] == "nl_minus_lince") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                    _, ups, _ = _verdict(ds, FLOOR_POOL)
                    row["shuf_n_K_up"] = len(ups)
                    row["shuf_sub_n_K_robust"] = _robust(Dd, c, m, "pair_shuf", "nl_minus_lince", shuf_sub[c], ups) if (c in shuf_sub and len(ups)) else np.nan
                    shuf_ok = len(ups) >= 3 and (c not in shuf_sub or row["shuf_sub_n_K_robust"] >= 3)             # pair_shuf にも部分集合の頑健さを当てる（gate-probe 第 3 回）
                    row["pair_minus_shuf_gap"] = float(d["est"].mean() - ds["est"].mean()) if len(ds) else np.nan   # nl−lince(pair) − nl−lince(pair_shuf)（記述）
                    row["note_label_route"] = bool(v == "上回った" and shuf_ok)              # 入れ替えた目標でも上回り部分集合でも頑健 → 少なくとも一部はラベルを通る経路でも出る（案に有利）
                    row["note_not_label_route"] = bool(v == "上回った" and len(ups) < 3)     # 入れ替えた目標では再現されない → ラベルを通らない結びつき（共有を含む）の疑い
                    row["note_shuf_shared"] = bool(v == "上回った" and len(ups) >= 3 and not shuf_ok)   # 再現されたが人物・場面の偏りと分けられない（有利とも不利とも書かない）
                    ok3 = (v == "上回った") and (not row.get("note_shared", False)) and shuf_ok                     # R3（第 3 条件の向きは gate-probe 第 2 回で直し，第 3 回で部分集合の頑健さを足した）
                    row["pair_r3_ok"] = bool(ok3)
                    if ok3:
                        pair_ok[m] += 1
                    for q in ("shuf_nl", "shuf_lince"):                                      # R1s：ラベルを通らない分（記述．規則の語は付けない）
                        dq = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["teacher"] == "pair") & (Dd["quantity"] == q) & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                        _, upq, downq = _verdict(dq, FLOOR_POOL)
                        crit.append(dict(c=c, student=m, rule="R1s", teacher="pair", quantity=q, n_K=len(dq), n_K_up=len(upq), n_K_down=len(downq), verdict="記述（規則の語なし）", est_min=dq["est"].min(), est_max=dq["est"].max()))
                    lab = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["quantity"] == "lab_nl_minus_lince") & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                    lm, pm = lab["est"].mean(), d["est"].mean()
                    Lcm = Lb[(Lb["c"] == c) & (Lb["student"] == m)]
                    crit.append(dict(c=c, student=m, rule="Q3", teacher="labels", quantity="lab_nl_minus_lince", n_K=len(lab), est_min=lab["est"].min(), est_max=lab["est"].max(),
                                     lab_mean=lm, pair_mean=pm, tie=bool(abs(lm) < 0.5 or abs(pm) < 0.5), sign_agree=bool(np.sign(lm) == np.sign(pm)),
                                     lab_nl_used_hidden_frac=float(Lcm[["lab_nl_used_hidden_h0", "lab_nl_used_hidden_h1"]].to_numpy().mean()) if len(Lcm) else np.nan,
                                     **{f"lab_{kd}_edge_{ax}_{sd}_frac": (float(Lcm[[f"lab_{kd}_edge_{ax}_{sd}_h0", f"lab_{kd}_edge_{ax}_{sd}_h1"]].to_numpy().mean()) if len(Lcm) else np.nan)
                                        for kd in ("nl", "lince", "nl_hidden") for ax in ("wd", "step") for sd in ("low", "high")},
                                     note_q3_edge=bool(len(Lcm) and max(float(Lcm[[f"lab_{kd}_edge_{ax}_{sd}_h0", f"lab_{kd}_edge_{ax}_{sd}_h1"]].to_numpy().mean())
                                                                        for kd in ("nl", "lince", "nl_hidden") for ax in ("wd", "step") for sd in ("low", "high")) > 0.5)))
                crit.append(row)
                for q in ("nl_vs_lince", "nl_vs_lince0", "nl_vs_none", "nl_vs_noneloco", "nlfix_vs_lincefix", "nl_vs_ret11"):   # R2（記述）
                    d2 = Dd[(Dd["c"] == c) & (Dd["student"] == m) & (Dd["teacher"] == t_) & (Dd["quantity"] == q) & (Dd["subset"] == "all") & (Dd["K"].isin(KS_RULE))]
                    if len(d2) == 0:
                        continue
                    v2, up2, down2 = _verdict(d2, FLOOR_TEST)
                    row2 = dict(c=c, student=m, rule="R2", teacher=t_, quantity=q, n_K=len(d2), n_K_up=len(up2), n_K_down=len(down2),
                                verdict=("作り方の違いを含む（規則の語は付けない）" if q == "nl_vs_ret11" else v2),
                                K_up=",".join(map(str, up2["K"])), K_down=",".join(map(str, down2["K"])), est_min=d2["est"].min(), est_max=d2["est"].max())
                    for met in ("war", "waf", "mf1"):
                        row2[f"{met}_same_sign_n"] = int((np.sign(d2[f"{met}_est"]) == np.sign(d2["est"])).sum())
                    if c == "MOSEI" and q != "nl_vs_ret11":
                        hit2 = pd.concat([up2, down2])
                        row2["chan_n_K_hit"] = len(hit2); row2["chan_n_K_robust"] = _robust(Dd, c, m, t_, q, "channel", hit2) if len(hit2) else 0
                    crit.append(row2)
    for c in have_c:                                                                     # 格子の端（EDGE．軸ごと・ネットごと），損失と解き方の違い（lince − ridge）
        Rcc = R[R["c"] == c]
        Lcc = Lb[Lb["c"] == c]
        for kind in ("nl", "lince", "lince0", "nl_hidden"):
            for ax in ("wd", "step"):
                for side in ("low", "high"):
                    fr = float(Rcc[[f"{kind}_edge_{ax}_{side}_h0", f"{kind}_edge_{ax}_{side}_h1"]].to_numpy().mean())
                    crit.append(dict(c=c, student="all", rule="EDGE", teacher="all", quantity=f"{kind}_edge_{ax}_{side}_frac", n_K=2 * len(Rcc), n_K_up=int(round(fr * 2 * len(Rcc))),
                                     verdict=("端で選ばれた割合が過半" if fr > 0.5 else "過半ではない"), est_min=fr))
        for kind in ("nl", "lince"):                                                     # Q3 の収束の目安
            nc = int((~Lcc[[f"lab_{kind}_conv_h0", f"lab_{kind}_conv_h1"]].to_numpy().astype(bool)).sum()) if len(Lcc) else 0
            crit.append(dict(c=c, student="all", rule="OPT", teacher="labels", quantity=f"lab_{kind}_not_converged_halves", n_K=2 * len(Lcc), n_K_up=nc,
                             verdict=("最適化の不足あり" if nc / max(2 * len(Lcc), 1) > 0.05 else "最適化の不足は 5% 未満")))
        for kind in ("nl", "lince", "nl_hidden"):
            for ax in ("wd", "step"):
                for side in ("low", "high"):
                    cols_ = [f"lab_{kind}_edge_{ax}_{side}_h0", f"lab_{kind}_edge_{ax}_{side}_h1"]
                    fr = float(Lcc[cols_].to_numpy().mean()) if len(Lcc) else np.nan
                    crit.append(dict(c=c, student="all", rule="EDGE", teacher="labels", quantity=f"lab_{kind}_edge_{ax}_{side}_frac", n_K=2 * len(Lcc), n_K_up=int(round(fr * 2 * len(Lcc))) if len(Lcc) else 0,
                                     verdict=("端で選ばれた割合が過半" if fr > 0.5 else "過半ではない"), est_min=fr))
        Mm = pd.DataFrame(mism_rows); Mm = Mm[Mm["key"].str.contains("read_edge")]
        Mm = Mm[Mm["unit"].isin(set(R[R["c"] == c]["unit"]))]
        for ax in ("wd", "step"):
            for side in ("low", "high"):
                v_ = Mm[Mm["key"].str.endswith(f"read_edge_{ax}_{side}_frac")]["value"].astype(float)
                fr = float(v_.mean()) if len(v_) else np.nan
                crit.append(dict(c=c, student="all", rule="EDGE", teacher="readout", quantity=f"read_edge_{ax}_{side}_frac", n_K=len(v_), verdict=("端で選ばれた割合が過半" if fr > 0.5 else "過半ではない"), est_min=fr))
        crit.append(dict(c=c, student="all", rule="LOSS", teacher="all", quantity="lince_below_ridge_rows", n_K=len(Rcc), n_K_up=int(Rcc["lince_below_ridge"].sum()),
                         verdict="記述（損失と解き方の違いの分）"))
        uh = Rcc[["nl_used_hidden_h0", "nl_used_hidden_h1"]].to_numpy().astype(bool)
        crit.append(dict(c=c, student="all", rule="NEST", teacher="all", quantity="nl_used_hidden_frac", n_K=uh.size, n_K_up=int(uh.sum()), est_min=float(uh.mean()) if uh.size else np.nan,
                         verdict="記述（隠れ層の項を使った版が選ばれた割合）"))
        for t_ in TEACHERS:
            uht = Rcc[Rcc["teacher"] == t_][["nl_used_hidden_h0", "nl_used_hidden_h1"]].to_numpy().astype(bool)
            crit.append(dict(c=c, student="all", rule="NEST", teacher=t_, quantity="nl_used_hidden_frac", n_K=uht.size, n_K_up=int(uht.sum()), est_min=float(uht.mean()) if uht.size else np.nan, verdict="記述"))
    q2a = Rc[Rc["teacher"].isin(("fuse1", "other", "enc2fuse"))]
    crit.append(dict(c="all", student="all", rule="Q2a", teacher="all", quantity="ce_nl_hidden_gt_ce_lince_cells_180", n_K=len(q2a), n_K_up=int((q2a["ce_nl_hidden"] > q2a["ce_lince"]).sum()), verdict="記述（入れ子の前の隠れ層の版）"))
    for m in MODS:
        crit.append(dict(c="all", student=m, rule="R3", teacher="pair", quantity="n_corpora_pair_ok", n_K_up=pair_ok[m],
                         verdict="進む価値がある" if pair_ok[m] >= 2 else "線形を超えて読める分で得が見込める条件は測れなかった（AFEW・IEMOCAP は区間が広く届きにくい）"))
    pd.DataFrame(crit).to_csv(out / "criteria.csv", index=False)

    # Q2：DAI-11 で Δ ≥ 5 なのに得 < 0.5 だったセルとの照合と，4.3 節の仮定の表による読み
    rv = pd.read_csv(D11 / "req_vs_gain.csv")
    big = rv[(rv["delta_uar"] >= P11.BIG_DELTA) & (rv["gain"] < P11.FLOOR)][["c", "student", "teacher", "K", "delta_uar", "gain", "proj_gain"]]
    q2 = big.merge(Rc[["c", "student", "teacher", "K", "nl_minus_lince", "nl_minus_lince_fixed", "nl_gain", "lince_gain", "ce_nl", "ce_lince", "ce_nl_hidden", "uar_nl_hidden",
                       "nl_used_hidden_h0", "nl_used_hidden_h1"]], on=["c", "student", "teacher", "K"], how="left")
    q2["K0"] = q2["K"] == 0
    q2 = q2.merge(Dd[(Dd["quantity"] == "nl_minus_lince") & (Dd["subset"] == "all")][["c", "student", "teacher", "K", "lo", "hi", "se"]], on=["c", "student", "teacher", "K"], how="left")
    selfc = Rc[Rc["teacher"] == "self"][["c", "student", "K", "nl_minus_lince"]].rename(columns={"nl_minus_lince": "self_nl_minus_lince"})
    q2 = q2.merge(selfc, on=["c", "student", "K"], how="left").merge(Lc[["c", "student", "K", "lab_nl_minus_lince"]], on=["c", "student", "K"], how="left")
    Cr = pd.DataFrame(crit)
    q3n = Cr[Cr["rule"] == "Q3"][["c", "student", "note_q3_edge"]] if "note_q3_edge" in Cr.columns else pd.DataFrame(columns=["c", "student", "note_q3_edge"])
    q2 = q2.merge(q3n, on=["c", "student"], how="left")
    r1n = Cr[Cr["rule"] == "R1"][["c", "student", "teacher", "note_edge_hidden"]] if "note_edge_hidden" in Cr.columns else pd.DataFrame(columns=["c", "student", "teacher", "note_edge_hidden"])
    q2 = q2.merge(r1n, on=["c", "student", "teacher"], how="left")
    rob = []
    for _, row in q2.iterrows():
        sub = shuf_sub.get(row["c"])
        if sub is None or pd.isna(row["nl_minus_lince"]):
            rob.append(np.nan); continue
        ds = Dd[(Dd["c"] == row["c"]) & (Dd["student"] == row["student"]) & (Dd["teacher"] == row["teacher"]) & (Dd["quantity"] == "nl_minus_lince") & (Dd["subset"] == sub) & (Dd["K"] == row["K"])]
        rob.append(bool(len(ds) and (np.sign(ds["est"].iloc[0]) == np.sign(row["nl_minus_lince"])) and ((ds["lo"].iloc[0] > 0) or (ds["hi"].iloc[0] < 0)) and abs(ds["est"].iloc[0]) >= 0.5 * abs(row["nl_minus_lince"])))
    q2["sub_robust"] = rob
    status = []
    for _, row in q2.iterrows():
        if pd.isna(row["nl_minus_lince"]):
            status.append("対応する出所が無い"); continue
        if row["lo"] > 0 and row["nl_minus_lince"] >= FLOOR_POOL:
            status.append("線形の壁" + ("" if (pd.isna(row["sub_robust"]) or row["sub_robust"]) else "（共有と分けられない）")); continue
        if row["ce_nl_hidden"] <= row["ce_lince"] and abs(row["self_nl_minus_lince"]) <= SELF_TOL:          # (a) は入れ子の前の隠れ層の版で確かめる
            txt = "この罰則のかけ方では，この規模のネットは線形を超えて読めない"
            txt += "（特徴には非線形の感情の情報がある" if row["lab_nl_minus_lince"] >= FLOOR_POOL else "（特徴にこの規模で取れる非線形の感情の情報が無い"
            if row["c"] == "MOSEI" and row["lab_nl_minus_lince"] >= FLOOR_POOL:
                txt += "．MOSEI の Neutral は消去法のラベル"
            if bool(row.get("note_q3_edge", False)):
                txt += "．Q3 の分類器は格子の端で選ばれた"
            if bool(row.get("note_edge_hidden", False)):
                txt += "．隠れ層の版は格子の端で選ばれた"
            txt += "）．否定の文は線形の項と同じ罰則で縮める作りのもとで実際より出やすい"
            status.append(txt); continue
        status.append("判定できない（" + ("隠れ層の版が出所を線形のネットほど再現していない" if row["ce_nl_hidden"] > row["ce_lince"] else "対照 self の差が大きい") + "）")
    q2["status"] = status
    crit.append(dict(c="all", student="all", rule="Q2a", teacher="all", quantity="ce_nl_hidden_gt_ce_lince_cells_30", n_K=int(q2["ce_nl_hidden"].notna().sum()),
                     n_K_up=int((q2["ce_nl_hidden"] > q2["ce_lince"]).sum()), verdict="記述（Q2 の 30 セルの中）"))
    pd.DataFrame(crit).to_csv(out / "criteria.csv", index=False)
    q2.to_csv(out / "q2_cells.csv", index=False)
    print(f"書いた：cells_nl.csv，cells_lab.csv，diffs.csv，se_summary.csv，criteria.csv，q2_cells.csv，uar_units.csv，nl_units.csv，lab_units.csv，halves_units.csv，checks_units.csv（{len(files)} 単位）", flush=True)


# ------------------------------------------------------------------ 自己検査

def selftest() -> None:
    """人工データ：(1) 実データに近い規模（3,000 行，1,024 次元，点数の標準偏差 2）の線形の出所で，lince と lince0 がリッジ（閉じた解）と 0.5 ポイント以内．
    (2) 排他的論理和の形の出所では nl が lince を 30 ポイント以上上回る．(3) balanced_halves の本数の差は最大の塊以下．
    (4) effect_stats_pool：同じ候補を 2 つの分割に置くと標準誤差が 1 分割と同じ．種で塊の集合が違っても動く．(5) 釣り合いの重み．(6) 目標ごとの入力。"""
    g = torch.Generator(device=S.DEV).manual_seed(0)
    n, d = 3000, 1024
    X = torch.randn(n, d, generator=g, device=S.DEV, dtype=S.DT)
    cl = np.array([f"c{i % 37:02d}" for i in range(n)])
    half = balanced_halves(cl)
    cnt = np.bincount(pd.factorize(cl)[0]).max()
    assert abs(int((half == 0).sum()) - int((half == 1).sum())) <= cnt, "半分の本数の差が最大の塊より大きい"
    Wl = torch.randn(d, 4, generator=g, device=S.DEV, dtype=S.DT) * (2.0 / np.sqrt(d))
    P_lin = torch.softmax(X @ Wl, -1)
    y_lin = P_lin.argmax(1).cpu().numpy()
    Y = P_lin.clamp(min=1e-12).log(); Y = Y - Y.mean(1, keepdim=True)
    Yhat, _, _ = P11.representability(X, Y, half)
    u_ridge = S.uar(y_lin, Yhat.argmax(1).cpu().numpy())
    Q1 = P_lin.float()[None]
    li = crossfit_nets(X.float(), Q1, half, X[:5].float(), 1, 0)
    l0 = crossfit_nets(X.float(), Q1, half, X[:5].float(), 2, 0, zero=True)
    nl1 = crossfit_nets(X.float(), Q1, half, X[:5].float(), 3, HIDDEN)
    u_hid = S.uar(y_lin, nl1["sel"][0].argmax(1).cpu().numpy())
    used1 = nest_linear(nl1, li, half)
    print(f"線形の出所の入れ子の前の隠れ層の版：{100 * u_hid:.1f}")
    u_li, u_l0, u_n1 = (S.uar(y_lin, r_["sel"][0].argmax(1).cpu().numpy()) for r_ in (li, l0, nl1))
    print(f"線形の出所で隠れ層の項を使った版が選ばれた半分：{int(used1.sum())}/2")
    print(f"線形の出所（1,024 次元）：リッジ {100 * u_ridge:.1f} lince {100 * u_li:.1f} lince0 {100 * u_l0:.1f} nl {100 * u_n1:.1f}（cfg {li['cfg'][(0, 0)]}, {l0['cfg'][(0, 0)]}, {nl1['cfg'][(0, 0)]}）")
    assert abs(u_li - u_ridge) < 0.005 and abs(u_l0 - u_ridge) < 0.005, "線形のネットがリッジと 0.5 ポイント以内でない"
    assert abs(u_n1 - u_ridge) < 0.005, "線形の項を持つ隠れ層のネットが線形の出所でリッジと 0.5 ポイント以内でない（gate-code 第 2 回の 1）"
    n2, d2 = 3000, 16
    X2 = torch.randn(n2, d2, generator=g, device=S.DEV, dtype=S.DT)
    half2 = balanced_halves(cl)
    bits = (X2[:, 0] * X2[:, 1] > 0).long() * 2 + (X2[:, 2] * X2[:, 3] > 0).long()               # 排他的論理和の形の 4 クラス
    P_xor = torch.softmax(4.0 * torch.nn.functional.one_hot(bits, 4).to(S.DT), -1)
    y_xor = bits.cpu().numpy()
    nl = crossfit_nets(X2.float(), P_xor.float()[None], half2, X2[:5].float(), 0, HIDDEN, keep_params=(0,))
    li2 = crossfit_nets(X2.float(), P_xor.float()[None], half2, X2[:5].float(), 1, 0)
    used2 = nest_linear(nl, li2, half2)
    assert used2.all(), "排他的論理和の出所で隠れ層の項が選ばれない"
    u_xn, u_xl = S.uar(y_xor, nl["sel"][0].argmax(1).cpu().numpy()), S.uar(y_xor, li2["sel"][0].argmax(1).cpu().numpy())
    print(f"排他的論理和の出所：lince {100 * u_xl:.1f} nl {100 * u_xn:.1f}（差 {100 * (u_xn - u_xl):+.1f}）")
    assert u_xl < 0.5 and u_xn - u_xl > 0.3, "排他的論理和の出所で nl が lince を上回らない"
    assert set(nl["params"]) == {(0, 0), (1, 0)} and nl["params"][(0, 0)][0].shape == (d2, HIDDEN) and nl["params"][(0, 0)][4].shape == (d2, 4), "重みの保存の形が違う"
    assert nl["grid_pred"].shape == (len(LRS), len(WDS), len(CKPTS), 1, n2)
    assert len(LRS) == 1, "学習率の格子は 1 つ（設計 3.3 節）"
    # (4) 候補の上の区間
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 4, 2000); cl2 = np.array([f"g{i % 60:02d}" for i in range(2000)])
    pa = np.where(rng.random(2000) < 0.6, yy, rng.integers(0, 4, 2000)); pb = np.where(rng.random(2000) < 0.55, yy, rng.integers(0, 4, 2000))
    cnt_a, cnt_b = ST.counts(yy, pa, cl2), ST.counts(yy, pb, cl2)
    one = effect_stats_pool({(0, 0): cnt_a}, {(0, 0): cnt_b}, 60.0, np.random.default_rng(1))
    two = effect_stats_pool({(0, 0): cnt_a, (1, 0): cnt_a}, {(0, 0): cnt_b, (1, 0): cnt_b}, 60.0, np.random.default_rng(1))
    assert abs(one["se"] - two["se"]) < 1e-9 and abs(one["effect"] - two["effect"]) < 1e-9, "分割をまたぐ引き直しの標準誤差が 1 分割と違う"
    indep = ST.effect_stats({(0, 0): cnt_a, (1, 0): cnt_a}, {(0, 0): cnt_b, (1, 0): cnt_b}, 60.0, np.random.default_rng(1))
    print(f"同じ候補を 2 分割に置いたときの標準誤差：共通の引き直し {two['se']:.3f}，独立の引き直し {indep['se']:.3f}，1 分割 {one['se']:.3f}")
    assert indep["se"] < two["se"], "独立の引き直しのほうが小さく出るはず"
    keep = cl2 != "g00"                                                                       # 種 1 では塊 g00 が K 本に入って候補から消えた，という形
    cnt_a1, cnt_b1 = ST.counts(yy[keep], pa[keep], cl2[keep]), ST.counts(yy[keep], pb[keep], cl2[keep])
    mixed = effect_stats_pool({(0, 0): cnt_a, (0, 1): cnt_a1}, {(0, 0): cnt_b, (0, 1): cnt_b1}, 60.0, np.random.default_rng(1))
    assert np.isfinite(mixed["se"]) and mixed["n_clusters_union"] == 60, "種で塊の集合が違う候補で動かない"
    # (5) 釣り合いの重み，(6) 目標ごとの入力
    G = balanced_weights(yy[:500], np.arange(500), 500)
    col = G.sum(0).cpu().numpy()
    assert np.allclose(col, col[0]), "釣り合いの重みの感情ごとの和が等しくない"
    Xj = torch.softmax(torch.randn(3, 500, 4, generator=g, device=S.DEV), -1)
    rd = crossfit_nets(Xj, G[None].expand(3, -1, -1).contiguous(), balanced_halves(cl2[:500]), torch.zeros(3, 1, 4, device=S.DEV), 2, 0)
    assert rd["sel"].shape == (3, 500, 4)
    print("selftest ok")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("units", "summarize", "selftest"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dai12_probe"))
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
    prov11 = json.load(open(D11 / "provenance.json"))
    commit11 = str(prov11["commit"])
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(), python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                time=time.strftime("%Y-%m-%d %H:%M:%S"),
                script_sha256=sha(Path(__file__)), dai11_probe_sha256=sha(HERE / "dai11_probe.py"), dai1_sha256=sha(HERE / "dai1.py"),
                solver_sha256=sha(HERE / "dai_solver.py"), data_sha256=sha(HERE / "dai_data.py"), frozen_sha256=sha(FROZEN),
                stats_sha256=sha(HERE / "dai1_stats.py"), dai3_sha256=sha(HERE / "dai3.py"),
                dai11_commit=commit11, dai11_script_sha256=prov11.get("script_sha256"),
                pseudo_commit={T: json.load(open(FROZEN))[T]["pseudo_commit"] for T in DD.CORPORA},
                features_sha256={p.name: sha(p) for p in sorted((ROOT / "data" / "cache").glob("*.npz"))
                                 if p.stem.split("_")[-1] in DD.CORPORA and any(p.stem.startswith(k + "_") for k in P11.BOTH.values())},
                constants=dict(KS_ALL=KS_ALL, KS_RULE=KS_RULE, MODS=MODS, TEACHERS=TEACHERS, RULE_TEACHERS=RULE_TEACHERS, HIDDEN=HIDDEN, STEPS=STEPS,
                               LRS=LRS, WDS=WDS, CKPTS=CKPTS, FIXED_LR=FIXED_LR, FIXED_WD=FIXED_WD, FIXED_STEP=FIXED_STEP, CONV_REL=CONV_REL, BELOW_RIDGE_TOL=BELOW_RIDGE_TOL,
                               READ_WDS=READ_WDS, READ_CKPTS=READ_CKPTS, READ_STEPS=READ_STEPS, LAB_CKPTS=LAB_CKPTS, LAB_STEPS=LAB_STEPS, LAM_REL=P11.LAM_REL, FLOOR_POOL=FLOOR_POOL, FLOOR_TEST=FLOOR_TEST,
                               MIN_DELTA_SHARE=MIN_DELTA_SHARE, MISMATCH_FRAC=MISMATCH_FRAC, SELF_TOL=SELF_TOL, SEED=SEED, N_BOOT=N_BOOT))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    fz = json.load(open(FROZEN))
    if args.stage == "units":
        files = sorted(UD.glob(f"{args.units}_p3.npz"))
        if args.units == "*":
            if len(files) != N_UNITS:
                sys.exit(f"入力の単位（DAI-1）が {len(files)} 個（{N_UNITS} のはず）")
            n11 = len(list((D11 / "rows").glob("*_req.csv")))
            if n11 != N_UNITS:
                sys.exit(f"入力の単位（DAI-11）が {n11} 個（{N_UNITS} のはず）")
        if args.resume:
            files = [p for p in files if not (out / "rows" / f"{p.name[:-len('_p3.npz')]}.npz").exists()]
            print(f"続き：残り {len(files)} 単位", flush=True)
        rows = [run_unit(p, out, fz, commit, commit11) for p in files]
        pd.DataFrame(rows).to_csv(out / f"units_time_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
        return
    if args.stage == "summarize":
        summarize(out, args.allow_partial)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
