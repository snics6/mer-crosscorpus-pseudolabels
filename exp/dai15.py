"""実験 15（DAI-15）：疑似ラベルの取り違えを K 本で推して直す（Gold Loss Correction の前向き補正）と，分析の予言は当たるか
（設計 `docs/設計/実験15_設計.md` 第 6 版）．

提案（実験 1 の主の版 `one`：学び直した声・顔・言葉の確率の積）の疑似ラベル q を，学び直す分類器に渡すときに，
疑似ラベルのモデルが本当の感情 i を感情 j と取り違える確率の表 C（C_ij ≈ P(疑似 = j | 本当 = i)）で前向きに補正する：
生徒の確率 p を C で混ぜた p̃ = Cᵀp（p̃_j = Σ_i p_i C_ij）を，疑似ラベルの項の目標 q に合わせる（Hendrycks ほか 2018）．
C は K 本で推す．K 本は疑似ラベルのモデルの学習にも入るので，K 本を感情ごとに 5 組に分け，組を抜いて疑似ラベルのモデルを
作り直し（K 本だけの融合の分類器 → 声・顔・言葉の学び直し → 積），抜いた組の上で数える（交差の推定．Dao ほか 2021 の型）．
縮小：C_λ = (1 − λ)Ĉ + λ S(Ĉ)．S(Ĉ) は対角の平均を保った対称の表．λ = 1 は「表の非対称な形の情報を持たない」対照．
提案は疑似ラベルの項を感情ごとの総量でそろえる（列 j に w_j = 1 / Σ_n q_nj を掛ける）ので，その目標に合う表は
C′ = 行の正規化(C_λ · diag(w)) である（第 2 版．関門の指摘）．補正にはこの C′ を使う．
生徒の K 本の項・疑似ラベルの総重み・出発点・引き戻しは，提案の選んだ設定と同じ．ただし C′ で補正すると，生徒が学ぶ問の
本当の感情の割合も変わる（第 3 版．査読想定の役の指摘）．そこで各版について，ラベルを使わずに切片だけをずらして候補の上の
予測の確率の平均を提案と同じにした版（_adj）も作る．

段：
  selftest  合成の数で，補正の式・C = 単位行列で提案と一致・縮小・組の分け方を確かめる
  pseudo    仮のターゲットの回（実験 1 の保存，120 単位）で，補正の版のテストの UAR を測る
  select    (評価のターゲット T, 生徒, K) ごとに，仮のターゲットの平均の UAR で「補正しない／λ = 0／λ = 0.5」から選び固定する
  eval      評価の回（200 単位）で，全部の版のテストの判定を保存する（固定のファイルがコミット済みであることを確かめる）
  summary   区間と予言の判定
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
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402
import dai14  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT_PROD = ROOT / "results" / "dai15"
DESIGN = ROOT / "docs" / "設計" / "実験15_設計.md"
FROZEN = ROOT / "frozen" / "dai15_frozen.json"
KS = (5, 25)
K_MAIN = 5
STU = DD.STUDENTS                       # ("A", "V", "T", "AVT")
LAMBDAS = (0.0, 0.5, 1.0)               # 1.0 は対称の対照
ADJ_ITERS = 3000                        # 割合をそろえる切片の反復の回数
ADJ_TOL = 1e-4                          # 割合をそろえる切片の残りの許し（確率の平均の差の最大）
D_MIN = 0.25                            # 交差の推定の表の対角の平均がこれ以下なら補正しない（でたらめと同じ当たり）
SEL_OPTS = ("off", "0.0", "0.5")        # 選ぶ候補（off は提案そのもの）
NF = 5                                  # K 本を感情ごとに分ける組の数
TOL = 1e-6
MAX_ITER = 20000
CHECK_TEACHER = 1e-4                    # K 本を全部使って作り直した疑似ラベルと保存の積の差の許し
CHECK_W = 1e-2                          # C = 単位行列の補正版と保存の提案の (W, b) の相対の差の許し（判定の入れ替わりは数を記録だけ）
SCRIPTS = ("dai15.py", "dai14.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
RUN = {"commit": "", "dirty": True, "partial": False}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ 表と補正

def kfold_of(y_k: np.ndarray, nf: int = NF) -> np.ndarray:
    """K 本の発話（u.kpos[K] の並び）を，感情ごとに並びの順で 0, 1, …, nf−1 の組に配る．"""
    out = np.empty(len(y_k), int)
    for c in np.unique(y_k):
        ii = np.flatnonzero(y_k == c)
        out[ii] = np.arange(len(ii)) % nf
    return out


def table_from(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """C_ij = (本当の感情 i の発話の上の P の j 列の平均)．無い感情の行は単位行列の行．"""
    C = np.eye(4)
    for i in range(4):
        m = y == i
        if m.any():
            C[i] = P[m].mean(0)
    return C


def shrink(C: np.ndarray, lam: float) -> np.ndarray:
    d = float(np.mean(np.diag(C)))
    Sm = np.full((4, 4), (1 - d) / 3.0)
    np.fill_diagonal(Sm, d)
    return (1 - lam) * C + lam * Sm


def match_prior(Zp: torch.Tensor, target: torch.Tensor, iters: int = ADJ_ITERS) -> tuple[torch.Tensor, float]:
    """候補の上の softmax(Zp + δ) の平均が target（4 つの感情の確率の平均）になる切片のずれ δ を不動点の反復で求める．ラベルを使わない．"""
    delta = torch.zeros(Zp.shape[-1], device=Zp.device, dtype=Zp.dtype)
    for _ in range(iters):
        m = torch.softmax(Zp + delta, -1).mean(0)
        delta = delta + (target.clamp(min=1e-12).log() - m.clamp(min=1e-12).log())
        delta = delta - delta.mean()
    resid = float((torch.softmax(Zp + delta, -1).mean(0) - target).abs().max())
    return delta, resid


def weight_table(C: np.ndarray, w: np.ndarray) -> np.ndarray:
    """感情ごとの総量をそろえる重み w（列 j の倍率）に合う前向きの表：C′_ij = C_ij w_j / Σ_k C_ik w_k．"""
    M = C * w[None, :]
    return M / M.sum(1, keepdims=True)


def fit_glc(X: torch.Tensor, GK: torch.Tensor, GQ: torch.Tensor, Cm: torch.Tensor, W0, b0, Cw: float, Cb: float) -> dict:
    """問 p の目的関数（S_p は GK と GQ の重みの和）：
        [ Σ_i Σ_k GK[p,i,k](−log p_ik) + Σ_i Σ_j GQ[p,i,j](−log (p_i Cm_p)_j) + ‖W − W0‖²/(2Cw) + ‖b − b0‖²/(2Cb) ] / S_p
    p_i = softmax(x_i Wᵀ + b)．Cm = 単位行列なら `dai_solver.fit` の G = GK + GQ と同じ問になる．"""
    X = S.t(X); GK = S.t(GK); GQ = S.t(GQ); Cm = S.t(Cm)
    P, n, k = GK.shape
    d = X.shape[1]
    W0 = S.t(W0).expand(P, k, d).contiguous() if S.t(W0).dim() == 2 else S.t(W0)
    b0 = S.t(b0).expand(P, k).contiguous() if S.t(b0).dim() == 1 else S.t(b0)
    Ssum = (GK.sum((1, 2)) + GQ.sum((1, 2))).clamp(min=1e-12)
    logC = Cm.clamp(min=1e-300).log()
    lw, lb = 1.0 / (2 * Cw), 1.0 / (2 * Cb)
    kd = k * d

    def f_and_g(th, ix=None):
        sel = (lambda v: v) if ix is None else (lambda v: v[ix])
        th = th.detach().requires_grad_(True)
        B = th.shape[0]
        W = th[:, :kd].view(B, k, d)
        b = th[:, kd:]
        Z = torch.einsum("nd,pkd->pnk", X, W) + b[:, None, :]
        lp = torch.log_softmax(Z, 2)
        lpt = torch.logsumexp(lp[:, :, :, None] + sel(logC)[:, None, :, :], dim=2)   # log Σ_i p_i C_ij
        nll = -(sel(GK) * lp).sum((1, 2)) - (sel(GQ) * lpt).sum((1, 2))
        reg = lw * ((W - sel(W0)) ** 2).sum((1, 2)) + lb * ((b - sel(b0)) ** 2).sum(1)
        f = (nll + reg) / sel(Ssum)
        (g,) = torch.autograd.grad(f.sum(), th)
        return f.detach(), g

    theta = torch.cat([W0.reshape(P, kd), b0], 1).clone()
    theta, f, grad, n_iter, stalled = S.lbfgs_batch(theta, f_and_g, TOL, MAX_ITER)
    gp = grad.abs().amax(1)
    finite = torch.isfinite(gp) & torch.isfinite(theta).all(1) & torch.isfinite(f)
    return {"W": theta[:, :kd].view(P, k, d).clone(), "b": theta[:, kd:].clone(), "grad_p": gp.cpu().numpy(),
            "stalled_p": stalled.cpu().numpy(), "finite_p": finite.cpu().numpy(), "f_p": f.cpu().numpy()}


# ------------------------------------------------------------------ 疑似ラベルのモデルの作り直し（交差の推定）

def teachers(u, tether: dict, K: int, kid_sets: list) -> torch.Tensor:
    """K 本の部分集合ごと（kid_sets：候補の中の位置の配列のリスト）に，実験 1 の段 1・2 と同じ手順で疑似ラベルのモデルを作る：
    K 本だけの融合の分類器（出発点から，(AVT, K) の引き戻し）→ その確率 F を疑似ラベルに声・顔・言葉を学び直す（G = K 本 + F の
    総量 n，(m, K) の引き戻し）→ 3 つの確率の積（指数 1, 1, 1）．戻り：(部分集合の数, n, 4)．"""
    n = u.n
    G0 = torch.stack([S.balanced_onehot(u.ypool, ks, n) for ks in kid_sets])
    W0, b0 = u.start["AVT"]
    cw, cb = tether[("AVT", K)]
    r = S.fit(u.Xp["AVT"], G0, W0, b0, Cw=cw, Cb=cb)
    if not r["finite_p"].all():
        raise SystemExit("疑似ラベルのモデルの作り直し（融合）が有限でない")
    conv = [float(r["grad_max"]), int(r["stalled"])]
    F = torch.softmax(S.logits(u.Xp["AVT"], r["W"], r["b"]), -1)
    L = 0
    for m in "AVT":
        G = G0 + torch.stack([S.soft_rows(F[j], float(n)) for j in range(len(kid_sets))])
        W0, b0 = u.start[m]
        cw, cb = tether[(m, K)]
        rm = S.fit(u.Xp[m], G, W0, b0, Cw=cw, Cb=cb)
        if not rm["finite_p"].all():
            raise SystemExit(f"疑似ラベルのモデルの作り直し（{m}）が有限でない")
        conv = [max(conv[0], float(rm["grad_max"])), conv[1] + int(rm["stalled"])]
        L = L + torch.log_softmax(S.logits(u.Xp[m], rm["W"], rm["b"]), -1)
    return torch.softmax(L, -1), conv


def tables(u, tether: dict, K: int, Q: torch.Tensor) -> dict:
    """交差の推定 Ĉ，K 本の上でそのまま数えた表（K 本の二重使用あり．記録），候補の全ラベルで数えた表（上限の参照）．"""
    kpos = u.kpos[K]
    yk = u.ypool[kpos]
    fold = kfold_of(yk)
    sets = [kpos] + [kpos[fold != j] for j in range(NF)]
    Qs, conv = teachers(u, tether, K, sets)
    dev = float((Qs[0] - Q).abs().max())
    if dev > CHECK_TEACHER:
        raise SystemExit(f"K 本を全部使って作り直した疑似ラベルが保存の積と {dev:.2e} 違う")
    qcf = np.empty((len(kpos), 4))
    for j in range(NF):
        held = np.flatnonzero(fold == j)
        qcf[held] = Qs[1 + j][torch.as_tensor(kpos[held], device=S.DEV)].cpu().numpy()
    Qn = Q.cpu().numpy()
    nk = np.setdiff1d(np.arange(u.n), kpos)
    acc_folds = [float(S.uar(u.ypool[nk], Qs[1 + j][torch.as_tensor(nk, device=S.DEV)].argmax(1).cpu().numpy())) for j in range(NF)]
    return dict(cf=table_from(qcf, yk), naive=table_from(Qn[kpos], yk), oracle=table_from(Qn[nk], u.ypool[nk]),
                qcf=qcf.astype(np.float32), teacher_dev=dev, teacher_grad=conv[0], teacher_stalled=conv[1],
                uar_teacher_full=float(S.uar(u.ypool[nk], Qn[nk].argmax(1))), uar_teacher_folds=acc_folds)


# ------------------------------------------------------------------ 1 単位

def run_unit(sv, fzT: dict, which: str) -> tuple[list, dict]:
    """sv：dai3.Saved．戻り：(行，保存する配列)．版：I（補正なしを解き直したもの），0.0・0.5・1.0（交差の推定の表の縮小），
    naive（K 本の上でそのまま数えた表．交差の推定を外した除去実験），oracle（候補の全ラベルで数えた表．参照）．"""
    u = sv.u
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    rows, arr = [], {}
    yt = u.ytest
    for K in KS:
        Q = dai1.product(TK[K], (1, 1, 1), 0)
        tb = tables(u, sv.teth, K, Q)
        w = (1.0 / Q.sum(0).clamp(min=1e-12)).cpu().numpy()             # soft_rows の balance と同じ列の倍率
        for nm in ("cf", "naive", "oracle"):
            arr[f"C_{nm}_K{K}"] = tb[nm]
        arr[f"w_K{K}"] = w
        arr[f"qcf_K{K}"] = tb["qcf"]
        arr[f"ykpos_K{K}"] = u.ypool[u.kpos[K]].astype(np.int8)
        dbar = float(np.mean(np.diag(tb["cf"])))
        skip = dbar <= D_MIN
        opts = [("I", np.eye(4))]
        for lam in LAMBDAS:
            opts.append((f"{lam}", np.eye(4) if skip else weight_table(shrink(tb["cf"], lam), w)))
        opts.append(("naive", weight_table(tb["naive"], w)))
        opts.append(("oracle", weight_table(tb["oracle"], w)))
        for nm, Cm in opts:
            arr[f"Cw_K{K}_{nm}"] = Cm
        rows.append(dict(K=K, option="_table", dbar_cf=dbar, skip=skip, teacher_dev=tb["teacher_dev"], teacher_grad=tb["teacher_grad"],
                         teacher_stalled=tb["teacher_stalled"], uar_teacher_full=100 * tb["uar_teacher_full"],
                         **{f"uar_teacher_fold{j}": 100 * v for j, v in enumerate(tb["uar_teacher_folds"])},
                         **{f"sv_{nm}_{i}": float(x) for nm in ("cf", "naive", "oracle") for i, x in enumerate(np.linalg.svd(tb[nm], compute_uv=False))}))
        Qt = dai1.product(TK[K], (1, 1, 1), 1)                             # 疑似ラベルのモデルのテストの確率
        Qtn = Qt.cpu().numpy()
        arr[f"prob_K{K}_teacher"] = Qtn.astype(np.float32)
        arr[f"pred_K{K}_teacher"] = Qtn.argmax(1).astype(np.int8)
        wq = Qtn * w[None, :]                                              # 生徒の目標と同じ列の倍率をかけた教師
        arr[f"pred_K{K}_teacher_w"] = wq.argmax(1).astype(np.int8)
        Cw0 = np.eye(4) if dbar <= D_MIN else weight_table(shrink(tb["cf"], 0.0), w)
        try:
            rj = np.linalg.solve(Cw0.T, wq.T).T                                # C′ の逆を掛けて判定し直す（記録．負の値もそのまま）
            arr[f"pred_K{K}_teacher_rj0"] = rj.argmax(1).astype(np.int8)
            rows.append(dict(K=K, option="_teacher_rj0", neg_share=float((rj < 0).any(1).mean()), singular=False))
        except np.linalg.LinAlgError:
            arr[f"pred_K{K}_teacher_rj0"] = arr[f"pred_K{K}_teacher_w"]
            rows.append(dict(K=K, option="_teacher_rj0", neg_share=np.nan, singular=True))
        arr[f"candcount"] = np.bincount(u.ypool, minlength=4)
        gk = u.gK(K)
        for s in STU:
            j = fzT["select"][f"{s}|{K}|joint|one"]
            rho, a = float(j["rho"]), float(j["a"])
            GK = (a * gk)[None].expand(len(opts), -1, -1)
            GQ = S.soft_rows(Q, dai1.mval(rho, u.n))[None].expand(len(opts), -1, -1)
            Cm = torch.stack([S.t(c) for _, c in opts])
            W0, b0 = u.start[s]
            cw, cb = sv.teth[(s, K)]
            res = fit_glc(u.Xp[s], GK, GQ, Cm, W0, b0, cw, cb)
            if not res["finite_p"].all():
                raise SystemExit(f"{sv.base.name}：補正の解が有限でない（{s}, K={K}）")
            Zt = S.logits(u.Xt[s], res["W"], res["b"])
            Zp = S.logits(u.Xp[s], res["W"], res["b"])
            pred = Zt.argmax(2).cpu().numpy()
            prob = torch.softmax(Zt, 2).cpu().numpy().astype(np.float32)
            Wp, bp = sv.prop(s, K)
            target = torch.softmax(S.logits(u.Xp[s], Wp, bp), -1).mean(0)            # 提案の候補の上の予測の確率の平均
            for (nm, Cx), Zpo, Zto, Wo, bo in zip(opts, Zp, Zt, res["W"], res["b"]):
                dl, rs = match_prior(Zpo, target)
                Za = Zto + dl
                arr[f"pred_K{K}_{s}_{nm}_adj"] = Za.argmax(1).cpu().numpy().astype(np.int8)
                arr[f"prob_K{K}_{s}_{nm}_adj"] = torch.softmax(Za, 1).cpu().numpy().astype(np.float32)
                arr[f"delta_K{K}_{s}_{nm}"] = dl.cpu().numpy()
                arr[f"W_K{K}_{s}_{nm}"] = Wo.cpu().numpy().astype(np.float32)
                arr[f"b_K{K}_{s}_{nm}"] = bo.cpu().numpy().astype(np.float32)
                arr[f"candmean_K{K}_{s}_{nm}"] = torch.softmax(Zpo, -1).mean(0).cpu().numpy()
                try:
                    pip = np.linalg.solve(np.asarray(Cx).T, np.full(4, 0.25))                # 生徒が学ぶ問の本当の割合（記録．範囲の外は旗）
                except np.linalg.LinAlgError:
                    pip = np.full(4, np.nan)
                rows.append(dict(K=K, student=s, option=f"_adj_{nm}", resid=rs, delta_max=float(dl.abs().max()),
                                 pi_ok=bool(np.all(np.isfinite(pip)) and np.all(pip >= 0) and np.all(pip <= 1)),
                                 **{f"pi_{i}": float(x) for i, x in enumerate(pip)}))
                if rs > ADJ_TOL:
                    raise SystemExit(f"{sv.base.name}：割合をそろえる切片が収束しない（{nm}，{s}，K={K}，{rs:.1e}）")
            Wn, bn = sv.none(s, K)
            Wk, bk = sv.partner(s, K)
            kj = fzT["select"][f"{s}|{K}|kd|one"]
            Wk1, bk1 = sv.p3_wb(s, K, sv.p3_index(s, K, method="kd", teacher="one", temp=kj["temp"], rho=kj["rho"]))
            refs = {"prop": (Wp, bp), "none": (Wn, bn), "partner": (Wk, bk), "kd_one": (Wk1, bk1)}
            arr[f"candmean_K{K}_{s}_prop"] = target.cpu().numpy()
            thp = torch.cat([S.t(Wp).flatten(), S.t(bp)])
            thI = torch.cat([res["W"][0].flatten(), res["b"][0]])
            th0 = torch.cat([S.t(W0).flatten(), S.t(b0)])
            wdiff = float((thI - thp).norm() / thp.norm().clamp(min=1e-12))
            wdiff_moved = float((thI - thp).norm() / (thp - th0).norm().clamp(min=1e-12))
            if wdiff > CHECK_W:
                raise SystemExit(f"{sv.base.name}：C = 単位行列の補正版と保存の提案の重みの相対の差が {wdiff:.2e}（{s}, K={K}）")
            for (nm, _), pr, pb, gp, stl, fv in zip(opts, pred, prob, res["grad_p"], res["stalled_p"], res["f_p"]):
                arr[f"pred_K{K}_{s}_{nm}"] = pr.astype(np.int8)
                arr[f"prob_K{K}_{s}_{nm}"] = pb
                rows.append(dict(K=K, student=s, option=nm, uar=100 * S.uar(yt, pr), grad=float(gp), stalled=bool(stl), fval=float(fv),
                                 rho=rho, a=a, Cw=cw, Cb=cb))
            for nm, (Wr, br) in refs.items():
                Zr = S.logits(u.Xt[s], Wr, br)
                pr = Zr.argmax(1).cpu().numpy()
                arr[f"pred_K{K}_{s}_{nm}"] = pr.astype(np.int8)
                arr[f"prob_K{K}_{s}_{nm}"] = torch.softmax(Zr, 1).cpu().numpy().astype(np.float32)
                rows.append(dict(K=K, student=s, option=nm, uar=100 * S.uar(yt, pr)))
            rows.append(dict(K=K, student=s, option="_I_vs_prop", value=wdiff, value_moved=wdiff_moved,
                             n_flip=int((pred[0] != arr[f"pred_K{K}_{s}_prop"]).sum())))
    arr["y"] = yt.astype(np.int8)
    arr["cl"] = sv.cl
    arr["sid"] = sv.sid
    return rows, arr


# ------------------------------------------------------------------ 段

def stage_units(out: Path, which: str, targets: list, folds=None, seeds=None) -> None:
    fz1, _ = dai3.load_fz1(OUT1)
    udir = out / which / "units"
    udir.mkdir(parents=True, exist_ok=True)
    for T in targets:
        fzT = fz1[T]
        if which == "pseudo":
            todo = dai3.pseudo_units(OUT1, T)
        else:
            todo = [(T, f, r) for f in range(DD.N_FOLDS) for r in range(int(fzT["n_seeds_eval"]))]
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
            sv = dai3.Saved(OUT1, which, T, c, f, r, fzT)
            rows, arr = run_unit(sv, fzT, which)
            df = pd.DataFrame(rows).assign(T=T, c=c, fold=f, seed=r, commit=RUN["commit"], dirty=RUN["dirty"], sec=time.time() - t0,
                                           dai1_commits=",".join(sorted(sv.commits)))
            np.savez_compressed(udir / f"{name}.npz", **arr)
            df.to_csv(udir / f"{name}.csv", index=False)
            print(f"{name}：{time.time() - t0:.0f} 秒．行 {len(df)}．勾配の最大 {df.grad.max():.1e}．止まり {int(df.stalled.fillna(False).sum())}", flush=True)
            del sv
        dai1.Corpus.cache.pop(T, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def frozen_path(out: Path) -> Path:
    return FROZEN if out.resolve() == OUT_PROD.resolve() else out / "frozen.json"


def stage_select(out: Path) -> None:
    rows = pd.concat([pd.read_csv(p) for p in sorted((out / "pseudo" / "units").glob("*.csv"))])
    if not RUN["partial"] and (rows.commit.nunique() != 1 or rows.dirty.astype(str).ne("False").any()):
        raise SystemExit("仮のターゲットの単位のコミットが 1 つでないか，汚れた作業木の単位がある")
    fz1, _ = dai3.load_fz1(OUT1)
    for T in DD.CORPORA:
        need = {(c, f, r) for c, f, r in dai3.pseudo_units(OUT1, T)}
        got = {(c, f, r) for c, f, r in rows[rows["T"] == T][["c", "fold", "seed"]].drop_duplicates().itertuples(index=False)}
        if need != got and not RUN["partial"]:
            raise SystemExit(f"{T}：仮のターゲットの単位がそろっていない（{len(got)} / {len(need)}）")
    r = rows[rows.option.isin(["I", "0.0", "0.5", "1.0"])].copy()
    r["option"] = r.option.replace({"I": "off"})
    mean = r.groupby(["T", "student", "K", "option"]).uar.mean().unstack("option")
    sel = {}
    mean = mean.dropna(how="all")
    for (T, s, K), x in mean.iterrows():
        best = "off"
        for o in ("0.0", "0.5"):                                         # 同点は off（補正しない）を残す
            if x[o] > x[best] + 1e-9:
                best = o
        sel[f"{T}|{s}|{K}"] = dict(option=best, **{f"pseudo_uar_{o}": float(x[o]) for o in ("off", "0.0", "0.5", "1.0")})
    frozen = dict(select=sel, pseudo_commit=RUN["commit"], n_rows=len(rows), made=time.strftime("%Y-%m-%d %H:%M:%S"))
    fp = frozen_path(out)
    fp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(frozen, open(fp, "w"), ensure_ascii=False, indent=1)
    mean.reset_index().to_csv(out / "select_means.csv", index=False)
    r.groupby(["T", "c", "student", "K", "option"]).uar.mean().reset_index().to_csv(out / "select_means_by_c.csv", index=False)
    print(f"固定のファイル {fp.name}：{len(sel)} セル．値は表示しない（評価の段の後まで開かない．設計 8 節）")


VERS = {"sel": None, "sel_adj": None, "I": "I", "lam0": "0.0", "lam05": "0.5", "sym": "1.0", "naive": "naive", "oracle": "oracle",
        "I_adj": "I_adj", "lam0_adj": "0.0_adj", "lam05_adj": "0.5_adj", "sym_adj": "1.0_adj", "naive_adj": "naive_adj",
        "oracle_adj": "oracle_adj", "prop": "prop", "none": "none", "kd": "partner", "kd_one": "kd_one"}
CONTRASTS = [("sel", "prop"), ("sel_adj", "prop"), ("sel", "I"), ("I", "prop"), ("sel", "none"), ("sel", "kd"), ("sel", "kd_one"),
             ("lam0", "sym"), ("lam0_adj", "sym_adj"), ("lam0_adj", "I_adj"), ("oracle", "I"), ("oracle", "prop"), ("oracle", "lam0"),
             ("oracle_adj", "lam0_adj"), ("oracle_adj", "I_adj"), ("lam0", "I"), ("lam05", "I"), ("sym", "I"), ("naive", "lam0"),
             ("naive_adj", "lam0_adj"), ("prop", "none"), ("sel", "sel_adj"), ("lam05_adj", "I_adj"), ("sel_adj", "I_adj")]
TEACHER_CONTRASTS = [("teacher_rj0", "teacher_w")]
LPI = {"sel_lpi": None, "lam0_lpi": "0.0"}                              # 記録の版：生徒が学ぶ問の割合 π′ の対数を引いて判定（予言に使わない）
LPI_CONTRASTS = [("sel_lpi", "prop"), ("lam0_lpi", "I")]


LPI_MIN_SHARE = 0.5                     # π′ が正の単位がこれ未満の割合なら，記録の差を作らない


def lpi_pred(a: dict, K: int, s_: str, opt: str) -> np.ndarray | None:
    """保存した確率と重みに合わせた表から，log π′ を引いた判定．π′ に 0 以下の成分があれば定義しないので None
    （その単位は記録の差の両側から外す．第 5 版）．"""
    try:
        pi = np.linalg.solve(a[f"Cw_K{K}_{opt}"].T, np.full(4, 0.25))
    except np.linalg.LinAlgError:
        return None
    if not np.all(pi > 0):
        return None
    return (np.log(np.clip(a[f"prob_K{K}_{s_}_{opt}"], 1e-12, None)) - np.log(pi)[None]).argmax(1)


def row_rng(T: str, K: int, s_: str, label: str) -> np.random.Generator:
    """区間の引き直しの乱数を行ごとに作る（ほかの行が作られたかどうかで変わらない．第 6 版）．"""
    h = int(hashlib.sha256(f"{s_}|{label}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng([20261113, DD.CORPUS_ID[T], K, h])


def stage_summary(out: Path) -> None:
    fz = json.load(open(frozen_path(out)))
    sdir = out / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    fz1, _ = dai3.load_fz1(OUT1)
    res, percls = [], []
    for T in DD.CORPORA:
        files = sorted((out / "eval" / "units").glob(f"{T}_{T}_f*_r*.npz"))
        need = DD.N_FOLDS * int(fz1[T]["n_seeds_eval"])
        if not files and RUN["partial"]:
            continue
        if len(files) != need and not RUN["partial"]:
            raise SystemExit(f"{T}：評価の単位が {len(files)} / {need}")
        csvs = pd.concat([pd.read_csv(str(p).replace(".npz", ".csv")) for p in files])
        if not RUN["partial"] and (csvs.commit.nunique() != 1 or csvs.dirty.astype(str).ne("False").any()):
            raise SystemExit(f"{T}：評価の単位のコミットが 1 つでないか，汚れた作業木の単位がある")
        U = {}
        for p in files:
            f, r = (int(x[1:]) for x in p.stem.split("_")[2:4])
            U[(f, r)] = dict(np.load(p, allow_pickle=False))
        g_eff, _ = ST.g_eff_of(T)
        for K in KS:
            for s in STU:
                if f"{T}|{s}|{K}" not in fz["select"]:
                    if RUN["partial"]:
                        continue
                    raise SystemExit(f"固定のファイルに {T}|{s}|{K} が無い")
                opt = fz["select"][f"{T}|{s}|{K}"]["option"]
                key = {**VERS, "sel": ("prop" if opt == "off" else opt),             # 補正しないなら提案そのもの
                       "sel_adj": ("prop" if opt == "off" else f"{opt}_adj")}

                def cell(v):
                    return {k: dai14.counts(a["y"], a[f"pred_K{K}_{s}_{key[v]}"], a["cl"]) for k, a in U.items()}
                for A_, B_ in CONTRASTS:
                    e = dai14.combo_stats([cell(A_), cell(B_)], [1, -1], g_eff, row_rng(T, K, s, f"{A_}-{B_}"))
                    res.append(dict(T=T, K=K, student=s, contrast=f"{A_}-{B_}", selected=opt) | e |
                               dict(verdict=dai14.verdict(e["lo"], e["hi"])))
                for A_, B_ in LPI_CONTRASTS:                                     # 記録の版（予言に使わない）
                    o = LPI[A_] if LPI[A_] is not None else (None if opt == "off" else opt)
                    rawA = "prop" if o is None else o                                # 同じ単位の上の生の版（sel なら選んだ版，λ0 なら 0.0）
                    cA_, cB_, cR_ = {}, {}, {}
                    for k, a in U.items():
                        pr_ = a[f"pred_K{K}_{s}_prop"] if o is None else lpi_pred(a, K, s, o)
                        if pr_ is None:
                            continue
                        cA_[k] = dai14.counts(a["y"], pr_, a["cl"])
                        cB_[k] = dai14.counts(a["y"], a[f"pred_K{K}_{s}_{key[B_]}"], a["cl"])
                        cR_[k] = dai14.counts(a["y"], a[f"pred_K{K}_{s}_{rawA}"], a["cl"])
                    n_used = len(cA_)
                    folds_used = sorted({f for f, _ in cA_})
                    min_seeds = min((sum(1 for f2, _ in cA_ if f2 == f) for f in folds_used), default=0)
                    row = dict(T=T, K=K, student=s, contrast=f"{A_}-{B_}", selected=opt, n_units_used=n_used, n_units_all=len(U),
                               n_folds_used=len(folds_used), min_seeds_per_fold=min_seeds)
                    if n_used < LPI_MIN_SHARE * len(U) or len(folds_used) < 2:          # 使える単位が半分未満か，分割が 2 つ未満なら作らない
                        res.append(row | dict(verdict="作らない"))
                        continue
                    e = dai14.combo_stats([cA_, cB_], [1, -1], g_eff, row_rng(T, K, s, f"{A_}-{B_}"))
                    er = dai14.combo_stats([cR_, cB_], [1, -1], g_eff, row_rng(T, K, s, f"{A_}-{B_}|raw_same_units"))
                    res.append(row | e | dict(verdict=dai14.verdict(e["lo"], e["hi"]), raw_effect=er["effect"], raw_lo=er["lo"],
                                              raw_hi=er["hi"], raw_verdict=dai14.verdict(er["lo"], er["hi"])))
                if s == "AVT":
                    for A_, B_ in TEACHER_CONTRASTS:
                        cA = {k: dai14.counts(a["y"], a[f"pred_K{K}_{A_}"], a["cl"]) for k, a in U.items()}
                        cB = {k: dai14.counts(a["y"], a[f"pred_K{K}_{B_}"], a["cl"]) for k, a in U.items()}
                        e = dai14.combo_stats([cA, cB], [1, -1], g_eff, row_rng(T, K, "teacher", f"{A_}-{B_}"))
                        res.append(dict(T=T, K=K, student="teacher", contrast=f"{A_}-{B_}", selected=opt) | e |
                                   dict(verdict=dai14.verdict(e["lo"], e["hi"])))
                for v in VERS:
                    e = dai14.combo_stats([cell(v)], [1], g_eff, row_rng(T, K, s, f"level|{v}"))
                    res.append(dict(T=T, K=K, student=s, contrast=f"{v}", selected=opt) | e | dict(verdict=""))
                    yy = np.concatenate([a["y"] for a in U.values()]).astype(int)
                    pp = np.concatenate([a[f"pred_K{K}_{s}_{key[v]}"] for a in U.values()]).astype(int)
                    for c in range(4):
                        percls.append(dict(T=T, K=K, student=s, version=v, emotion=c, recall=float((pp[yy == c] == c).mean()),
                                           pred_share=float((pp == c).mean()), true_share=float((yy == c).mean())))
    R = pd.DataFrame(res).assign(commit=RUN["commit"], dirty=RUN["dirty"])
    R.to_csv(sdir / "effects.csv", index=False)
    crit = []
    for (K, s, con), x in R[R.contrast.str.contains("-")].groupby(["K", "student", "contrast"]):
        crit.append(dict(K=K, student=s, contrast=con, n_above=int((x.verdict == "上回る").sum()), n_below=int((x.verdict == "下回る").sum()),
                         above=",".join(sorted(x[x.verdict == "上回る"]["T"])), below=",".join(sorted(x[x.verdict == "下回る"]["T"]))))
    pd.DataFrame(crit).assign(commit=RUN["commit"]).to_csv(sdir / "criteria.csv", index=False)
    pd.DataFrame(percls).assign(commit=RUN["commit"]).to_csv(sdir / "per_emotion.csv", index=False)
    # 予言の判定（設計 5 節の規則そのまま）
    V = {(r.T, r.K, r.student, r.contrast): r.verdict for r in R.itertuples()}
    up = lambda T, K, s_, c: V.get((T, K, s_, c)) == "上回る"
    skipn = {}
    for p in sorted((out / "eval" / "units").glob("*.csv")):
        x = pd.read_csv(p)
        for r in x[x.option == "_table"].itertuples():
            skipn[(r.T, int(r.K))] = skipn.get((r.T, int(r.K)), 0) + int(bool(r.skip))
    offc = lambda K, s_: ",".join(sorted(T for T in DD.CORPORA if fz["select"].get(f"{T}|{s_}|{K}", {}).get("option") == "off"))
    hyp = []
    for K in KS:
        Ts = sorted({t for (t, k, _, _) in V if k == K})
        both = lambda s_: [T for T in Ts if up(T, K, s_, "sel-prop") and up(T, K, s_, "sel_adj-prop")]
        a1, a2 = both("A"), both("AVT")
        hyp.append(dict(K=K, hypothesis="H1", n=len(a1), need=3, hit=len(a1) >= 3, corpora=",".join(a1), off=offc(K, "A"),
                        raw_only=",".join(T for T in Ts if up(T, K, "A", "sel-prop") and not up(T, K, "A", "sel_adj-prop")),
                        adj_only=",".join(T for T in Ts if up(T, K, "A", "sel_adj-prop") and not up(T, K, "A", "sel-prop"))))
        hyp.append(dict(K=K, hypothesis="H2", n=len(a2), need=2, hit=len(a2) >= 2, corpora=",".join(a2), off=offc(K, "AVT"),
                        raw_only=",".join(T for T in Ts if up(T, K, "AVT", "sel-prop") and not up(T, K, "AVT", "sel_adj-prop")),
                        adj_only=",".join(T for T in Ts if up(T, K, "AVT", "sel_adj-prop") and not up(T, K, "AVT", "sel-prop")),
                        rejudge=",".join(T for T in Ts if up(T, K, "teacher", "teacher_rj0-teacher_w")
                                         and fz["select"].get(f"{T}|AVT|{K}", {}).get("option") == "0.0"),       # 融合で λ = 0 が選ばれたコーパスだけ
                        rejudge_any=",".join(T for T in Ts if up(T, K, "teacher", "teacher_rj0-teacher_w"))))
        if "MOSEI" in Ts:
            d1, d2, d3 = (V.get(("MOSEI", K, "A", c), "") for c in ("oracle_adj-lam0_adj", "lam0_adj-I_adj", "oracle_adj-I_adj"))
            h3 = d1 == "上回る" and d3 == "上回る" and d2 != "上回る"
            hyp.append(dict(K=K, hypothesis="H3", n=int(h3), need=1, hit=h3, corpora="MOSEI" if h3 else "", oracle_adj_lam0_adj=d1,
                            lam0_adj_I_adj=d2, oracle_adj_I_adj=d3, n_skip_units=skipn.get(("MOSEI", K), 0)))
        base = [T for T in Ts if up(T, K, "A", "lam0_adj-I_adj")]
        c1 = [T for T in base if up(T, K, "A", "lam0_adj-sym_adj")]
        hyp.append(dict(K=K, hypothesis="C1", n=len(c1), need=3, hit=len(c1) >= 3, corpora=",".join(c1), base=",".join(base)))
    pd.DataFrame(hyp).assign(commit=RUN["commit"]).to_csv(sdir / "hypotheses.csv", index=False)
    print(len(R), "rows")


# ------------------------------------------------------------------ selftest

def selftest() -> None:
    rng = np.random.default_rng(0)
    # 1. 組の分け方：感情ごとに均等
    y = np.repeat(np.arange(4), 5)
    rng.shuffle(y)
    fo = kfold_of(y)
    for c in range(4):
        assert np.bincount(fo[y == c], minlength=NF).tolist() == [1] * NF
    # 2. 縮小：行の和 1，λ = 1 で対角の平均が保たれ非対角が等しい
    C = rng.dirichlet(np.ones(4) * 2, size=4)
    for lam in LAMBDAS:
        Cl = shrink(C, lam)
        assert np.allclose(Cl.sum(1), 1)
    S1 = shrink(C, 1.0)
    assert abs(np.mean(np.diag(S1)) - np.mean(np.diag(C))) < 1e-12 and np.allclose(S1[~np.eye(4, dtype=bool)], S1[0, 1])
    # 3. C = 単位行列なら dai_solver.fit と同じ解
    n, d = 400, 12
    X = torch.as_tensor(rng.normal(size=(n, d)))
    yy = rng.integers(0, 4, n)
    Q = torch.softmax(torch.as_tensor(rng.normal(size=(n, 4))) * 2, -1)
    gk = S.balanced_onehot(yy, np.arange(20), n)
    GQ = S.soft_rows(Q, float(n))
    W0 = torch.as_tensor(rng.normal(size=(4, d)) * 0.1); b0 = torch.zeros(4)
    r1 = fit_glc(X, gk[None], GQ[None], torch.eye(4)[None], W0, b0, 0.5, 1.0)
    r2 = S.fit(X, (gk + GQ)[None], W0, b0, Cw=0.5, Cb=1.0)
    assert float((r1["W"] - r2["W"]).abs().max()) < 1e-5, float((r1["W"] - r2["W"]).abs().max())
    # 4. 前向き補正で取り違えが戻る：本当の感情 y，疑似ラベル = 取り違えの表で混ぜた硬いラベル．補正した生徒は補正しない生徒より当たる
    n = 4000
    yv = rng.integers(0, 4, n)
    Xv = np.eye(4)[yv] * 2.0 + rng.normal(size=(n, 4)) * 0.8
    Ct = np.array([[0.55, 0.40, 0.03, 0.02], [0.05, 0.9, 0.03, 0.02], [0.02, 0.03, 0.9, 0.05], [0.02, 0.03, 0.05, 0.9]])
    qy = np.array([rng.choice(4, p=Ct[i]) for i in yv])
    Qv = torch.as_tensor(np.eye(4)[qy])
    GQv = S.soft_rows(Qv, float(n), balance=False)
    zero = torch.zeros(1, n, 4, dtype=S.DT, device=S.DEV)
    rr = fit_glc(torch.as_tensor(Xv), zero.expand(2, -1, -1), GQv[None].expand(2, -1, -1), torch.stack([torch.eye(4), torch.as_tensor(Ct)]),
                 torch.zeros(4, 4), torch.zeros(4), 100.0, 100.0)
    pr = S.logits(torch.as_tensor(Xv), rr["W"], rr["b"]).argmax(2).cpu().numpy()
    u0, u1 = S.uar(yv, pr[0]), S.uar(yv, pr[1])
    assert u1 > u0 + 0.05, (u0, u1)
    # 6. 感情の割合が偏り（Neutral 57%），疑似ラベルの項を感情ごとの総量でそろえる場合（本番の経路）：
    #    重みに合わせた表 C′ の補正は，表 C をそのまま掛けた補正より Neutral を潰さず，UAR も下回らない
    n = 6000
    yv = rng.choice(4, size=n, p=[0.57, 0.15, 0.14, 0.14])
    Xv = np.eye(4)[yv] * 2.0 + rng.normal(size=(n, 4)) * 0.8
    qy = np.array([rng.choice(4, p=Ct[i]) for i in yv])
    Qv = torch.as_tensor(np.eye(4)[qy] * 0.9 + 0.025)
    GQv = S.soft_rows(Qv, float(n))
    wv = (1.0 / Qv.sum(0)).cpu().numpy()
    Chat = table_from(Qv.cpu().numpy(), yv)
    Cms = torch.stack([torch.eye(4, dtype=S.DT), torch.as_tensor(Chat), torch.as_tensor(weight_table(Chat, wv))])
    zero3 = torch.zeros(3, n, 4, dtype=S.DT, device=S.DEV)
    rr = fit_glc(torch.as_tensor(Xv), zero3, GQv[None].expand(3, -1, -1), Cms, torch.zeros(4, 4), torch.zeros(4), 100.0, 100.0)
    pr = S.logits(torch.as_tensor(Xv), rr["W"], rr["b"]).argmax(2).cpu().numpy()
    uI, uC, uW = (S.uar(yv, x) for x in pr)
    recN = [float((x[yv == 0] == 0).mean()) for x in pr]
    assert recN[2] > recN[1] and uW >= uC - 1e-9, (uI, uC, uW, recN)
    assert np.allclose(weight_table(np.eye(4), wv), np.eye(4))
    # 7. 割合をそろえる切片：候補の上の確率の平均が目標に一致する
    Zr = torch.as_tensor(rng.normal(size=(3000, 4)) * 2, dtype=S.DT)
    tg = torch.as_tensor([0.4, 0.3, 0.2, 0.1], dtype=S.DT)
    dl, rs = match_prior(Zr, tg)
    assert rs < ADJ_TOL, rs
    # 5. table_from
    P = np.eye(4)[np.array([0, 1, 1, 2, 3])].astype(float)
    T5 = table_from(P, np.array([0, 0, 1, 2, 3]))
    assert np.allclose(T5[0], [0.5, 0.5, 0, 0])
    print(f"selftest：7 項目すべて通った（補正なし {100 * u0:.1f} → 補正 {100 * u1:.1f}．偏りあり：I {100 * uI:.1f}，C {100 * uC:.1f}，C′ {100 * uW:.1f}，Neutral の再現率 {recN[1]:.2f} → {recN[2]:.2f}）")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "pseudo", "select", "eval", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="試しの置き場だけ：単位のそろいとコミットの確かめを省く")
    args = ap.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある（試走は一時の置き場で --allow-dirty）")
    if prod and not dai1.committed_clean(DESIGN):
        sys.exit("設計の文書がコミットされていない")
    RUN["partial"] = bool(args.allow_partial)
    if prod and (args.folds is not None or args.seeds is not None or set(args.targets) != set(DD.CORPORA) or args.allow_partial):
        sys.exit("本番の置き場では対象を絞らない")
    if args.stage == "summary" and not frozen_path(out).exists():
        sys.exit("固定のファイルが無い")
    if args.stage in ("eval", "summary") and prod and not dai1.committed_clean(FROZEN):
        sys.exit("固定のファイルがコミットされていない（select の後にコミットしてから eval）")
    dai1.ENC["enc"] = DD.ENC1
    out.mkdir(parents=True, exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), torch=torch.__version__, numpy=np.__version__,
                scripts={n: sha(HERE / n) for n in SCRIPTS}, design_sha256=sha(DESIGN),
                frozen_sha256=(sha(frozen_path(out)) if frozen_path(out).exists() else ""), dai1_frozen_sha256=dai3.load_fz1(OUT1)[1],
                python=platform.python_version(), pandas=pd.__version__, KS=KS, LAMBDAS=LAMBDAS, NF=NF, TOL=TOL, MAX_ITER=MAX_ITER,
                SEL_OPTS=SEL_OPTS, D_MIN=D_MIN, LPI_MIN_SHARE=LPI_MIN_SHARE, CHECK_TEACHER=CHECK_TEACHER, CHECK_W=CHECK_W, ADJ_TOL=ADJ_TOL, ADJ_ITERS=ADJ_ITERS,
                tie="厳密な最大．同点は off", rng_seed=20261113)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage in ("pseudo", "eval"):
        stage_units(out, args.stage, args.targets, args.folds, args.seeds)
        if prod:
            (out / f"{args.stage}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    elif args.stage == "select":
        stage_select(out)
    else:
        stage_summary(out)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
