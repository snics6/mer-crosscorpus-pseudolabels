"""DAI 実験の学習器：重みつきの多クラスのロジスティック回帰を，多くの問を一度に GPU で解く。

問 p の目的関数（S_p は重みの和）：

    L_p(W, b) = [ Σ_i Σ_k G[p,i,k] · (−log softmax(x_i W_pᵀ + b_p)_k)
                  + ‖W_p − W0_p‖² / (2 Cw_p) + ‖b_p − b0_p‖² / (2 Cb_p) ] / S_p

- G[p,i,k] ≥ 0 は発話 i の感情 k の重み．K 本の正解（one-hot × 感情の釣り合いの重み × 倍率 a），
  疑似ラベル（確率 × 重み），の和で作る．発話ごとに複数の感情に重みを置いてよい（柔らかいラベル）．
- Cw = inf は W の罰則なし，Cb = inf は b の罰則なし．W0 = 0 なら普通の L2 正則化のロジスティック回帰である．
- 凸なので解は 1 つに決まる．問ごとに別々の履歴と歩幅を持つ L-BFGS（Armijo の後退．`exp/logreg.py` と同じ方式）で，
  問ごとの勾配の最大絶対値が tol（既定 1e-6）以下になるまで解く．
- fix_W=True なら W を W0 に固定して b だけを解く（役割を分けた版の切片の段）．

X は呼ぶ側で標準化してから渡す（計画 8.4 節の M9：同じ分割の中では全手法が同じ統計）。
`--selftest` で scikit-learn と同じ解になるかを確かめる。
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":                                # WSL2 では空きブロックが GPU の全量に触ると遅くなる（memory: wsl2-cuda-allocator）
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MERDA_GPU_FRACTION", "0.6")))
DT = torch.float64                               # 以前の学習器（exp/logreg.py）と同じく float64 で解く


def t(x, dtype=None):
    return x.to(DEV, dtype or DT) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), device=DEV, dtype=dtype or DT)


def balanced_onehot(y: np.ndarray, idx: np.ndarray, n: int, scale: float = 1.0) -> torch.Tensor:
    """idx の発話に，感情の釣り合いの重み（len/(感情の数 × その感情の件数)）× scale の one-hot を置いた (n, 4)。"""
    G = torch.zeros(n, 4, device=DEV, dtype=DT)
    if len(idx) == 0:
        return G
    yy = y[idx]
    cnt = np.bincount(yy, minlength=4).astype(float)
    present = (cnt > 0).sum()
    w = len(idx) / (present * cnt[yy])
    G[torch.as_tensor(idx, device=DEV), torch.as_tensor(yy, device=DEV)] = t(w * scale)
    return G


def soft_rows(P: torch.Tensor, total: float, balance: bool = True) -> torch.Tensor:
    """疑似ラベルの重み (n, 4)。balance=True なら感情ごとの総量をそろえてから，全体を total にする。"""
    P = t(P)
    if balance:
        P = P / P.sum(0, keepdim=True).clamp(min=1e-12)
    return P / P.sum().clamp(min=1e-12) * total


HISTORY, ARMIJO, MAX_HALVING = 10, 1e-4, 50


def lbfgs_batch(theta: torch.Tensor, f_and_g, tol: float, max_iter: int):
    """問ごとに別々の履歴と歩幅を持つ L-BFGS（Armijo の後退）の本体．theta: (P, D)．
    f_and_g(th, ix) は問の部分集合 ix（None なら全部）の目的関数 (B,) と勾配 (B, D) を返す．
    戻り：theta，f，grad，n_iter（問ごと），stalled（問ごと）。"""
    P = theta.shape[0]
    D = theta.shape[1]
    Sh = torch.zeros(HISTORY, P, D, dtype=DT, device=DEV)
    Yh = torch.zeros(HISTORY, P, D, dtype=DT, device=DEV)
    rho = torch.zeros(HISTORY, P, dtype=DT, device=DEV)
    gamma = torch.ones(P, dtype=DT, device=DEV)
    fresh = torch.ones(P, dtype=torch.bool, device=DEV)
    n_iter = torch.zeros(P, dtype=torch.long, device=DEV)
    stalled = torch.zeros(P, dtype=torch.bool, device=DEV)
    f, grad = f_and_g(theta)
    active = grad.abs().amax(1) > tol
    slot = 0
    for _ in range(max_iter):
        if not active.any():
            break
        q = grad.clone()
        order = [(slot - 1 - j) % HISTORY for j in range(HISTORY)]
        alpha = torch.zeros(HISTORY, P, dtype=DT, device=DEV)
        for j in order:
            alpha[j] = rho[j] * (Sh[j] * q).sum(1)
            q -= alpha[j][:, None] * Yh[j]
        q *= gamma[:, None]
        for j in reversed(order):
            beta = rho[j] * (Yh[j] * q).sum(1)
            q += Sh[j] * (alpha[j] - beta)[:, None]
        dvec = -q
        gtd = (grad * dvec).sum(1)
        reset = active & (gtd >= 0)
        if reset.any():
            rho[:, reset] = 0.0
            gamma[reset] = 1.0
            fresh |= reset
            dvec[reset] = -grad[reset]
            gtd = (grad * dvec).sum(1)
        dvec[~active] = 0.0
        tt = torch.ones(P, dtype=DT, device=DEV)
        tt[fresh] = torch.clamp(1.0 / grad[fresh].abs().sum(1).clamp(min=1e-30), max=1.0)
        pending = active.clone()
        f_new, g_new, th_new = f.clone(), grad.clone(), theta.clone()
        for _h in range(MAX_HALVING):
            ix = torch.nonzero(pending).squeeze(1)
            trial = theta[ix] + tt[ix, None] * dvec[ix]
            f_t, g_t = f_and_g(trial, ix)
            okx = f_t <= f[ix] + ARMIJO * tt[ix] * gtd[ix]
            io = ix[okx]
            f_new[io], g_new[io], th_new[io] = f_t[okx], g_t[okx], trial[okx]
            pending[io] = False
            if not pending.any():
                break
            tt[pending] *= 0.5
        stalled |= pending
        moved = active & ~pending
        s_vec, y_vec = th_new - theta, g_new - grad
        sy = (s_vec * y_vec).sum(1)
        good = moved & (sy > 1e-10 * s_vec.norm(dim=1) * y_vec.norm(dim=1))
        Sh[slot], Yh[slot] = s_vec, y_vec
        rho[slot] = torch.where(good, 1.0 / torch.where(good, sy, torch.ones_like(sy)), torch.zeros_like(sy))
        gamma = torch.where(good, sy / (y_vec * y_vec).sum(1).clamp_min(1e-30), gamma)
        fresh &= ~good
        slot = (slot + 1) % HISTORY
        theta, f, grad = th_new, f_new, g_new
        n_iter += moved.long()
        active = moved & (grad.abs().amax(1) > tol)
    return theta, f, grad, n_iter, stalled


def fit(X: torch.Tensor, G: torch.Tensor, W0: torch.Tensor, b0: torch.Tensor,
        Cw, Cb, fix_W: bool = False, max_iter: int = 20000, tol: float = 1e-6, G2=None, T2=None) -> dict:
    """G: (P, n, 4)．W0: (P, 4, d) か (4, d)．b0: (P, 4) か (4,)．Cw, Cb: 長さ P か数．
    問ごとに別々の履歴と歩幅を持つ L-BFGS（Armijo の後退）で，問ごとの勾配の最大絶対値が tol 以下になるまで解く
    （`exp/logreg.py` の fit_batch と同じ方式．1 本の L-BFGS でまとめると，曲がり方の違う問が履歴と歩幅を共有して収束が遅れる）．
    G2，T2 を渡すと，温度つき蒸留の項 T2² · Σ G2 · (−log softmax(z / T2)) を足す（Hinton ほか 2015．Albanie ほか 2018 は
    教師と生徒の両方に温度 2）．S は G と G2 の重みの和．
    戻り：W (P,4,d)，b (P,4)，n_iter（問ごとの反復の最大），grad_max（問ごとの勾配の最大絶対値の最大），grad_p（問ごと），
    stalled（Armijo の後退を MAX_HALVING 回しても下がらなかったことがある問の数），stalled_p・n_iter_p（問ごと）。"""
    X = t(X)
    G = t(G)
    P, n, k = G.shape
    d = X.shape[1]
    W0 = t(W0).expand(P, k, d).contiguous() if t(W0).dim() == 2 else t(W0)
    b0 = t(b0).expand(P, k).contiguous() if t(b0).dim() == 1 else t(b0)
    Cw = t(np.broadcast_to(np.asarray(Cw, float), (P,)).copy())
    Cb = t(np.broadcast_to(np.asarray(Cb, float), (P,)).copy())
    G2 = None if G2 is None else t(G2)
    T2 = None if T2 is None else t(np.broadcast_to(np.asarray(T2, float), (P,)).copy())
    S = (G.sum((1, 2)) + (0 if G2 is None else G2.sum((1, 2)))).clamp(min=1e-12)
    lw = torch.where(torch.isinf(Cw), torch.zeros_like(Cw), 1.0 / (2 * Cw))
    lb = torch.where(torch.isinf(Cb), torch.zeros_like(Cb), 1.0 / (2 * Cb))
    kd = k * d

    def f_and_g(th, ix=None):
        sel = (lambda v: v) if ix is None else (lambda v: v[ix])
        th = th.detach().requires_grad_(True)
        Bx = th.shape[0]
        W = sel(W0) if fix_W else th[:, :kd].view(Bx, k, d)
        b = th[:, kd:]
        Z = torch.einsum("nd,pkd->pnk", X, W) + b[:, None, :]
        nll = -(sel(G) * torch.log_softmax(Z, 2)).sum((1, 2))
        if G2 is not None:
            tt2 = sel(T2)[:, None, None]
            nll = nll - (sel(T2) ** 2) * (sel(G2) * torch.log_softmax(Z / tt2, 2)).sum((1, 2))
        reg = sel(lw) * ((W - sel(W0)) ** 2).sum((1, 2)) + sel(lb) * ((b - sel(b0)) ** 2).sum(1)
        f = (nll + reg) / sel(S)
        (g,) = torch.autograd.grad(f.sum(), th)
        if fix_W:
            g[:, :kd] = 0.0
        return f.detach(), g

    theta = torch.cat([W0.reshape(P, kd), b0], 1).clone()
    theta, f, grad, n_iter, stalled = lbfgs_batch(theta, f_and_g, tol, max_iter)
    gp = grad.abs().amax(1)
    finite = torch.isfinite(gp) & torch.isfinite(theta).all(1) & torch.isfinite(f)
    W = W0.clone() if fix_W else theta[:, :kd].view(P, k, d).clone()
    return {"W": W, "b": theta[:, kd:].clone(), "n_iter": int(n_iter.max()), "grad_max": float(gp.max()),
            "grad_p": gp.cpu().numpy(), "stalled": int(stalled.sum()), "stalled_p": stalled.cpu().numpy(),
            "n_iter_p": n_iter.cpu().numpy(), "finite_p": finite.cpu().numpy(),
            "ok": bool(float(gp.max()) <= 10 * tol) and bool(finite.all())}



def fit_coreg(Xs: list, G: torch.Tensor, W0s: list, b0s: list, cws: list, cbs: list, rhos, max_iter: int = 20000,
              tol: float = 1e-6) -> dict:
    """共正則化（Sindhwani & Rosenberg 2008 の多視点の考え）を，ρ の格子の問をまとめて lbfgs_batch で解く．
    Xs：モダリティごとの標準化した特徴 (n, d_m)．G：K 本の正解 (n, 4)．W0s・b0s：出発点．cws・cbs：モダリティごとの引き戻し．
    問 p の目的関数：[Σ_m (−Σ G·log softmax(Z_m) ＋ ‖W_m − W0_m‖²/(2C′_m) ＋ ‖b_m − b0_m‖²/(2C_b,m))
                    ＋ (ρ_p/2) Σ_i Σ_{a<b} ‖c_a,i − c_b,i‖²] / S_p，c は発話ごとに 4 感情の平均を引いた点数，S_p = 3ΣG ＋ ρ_p n．
    戻り：Ws・bs（モダリティごとに (P,4,d_m)・(P,4)），問ごとの勾配の最大，反復，行き詰まり，有限の印。"""
    Xs = [t(X) for X in Xs]
    G = t(G)
    rh = t(np.asarray(rhos, float))
    P, n = len(rhos), G.shape[0]
    dims = [X.shape[1] for X in Xs]
    S = (3 * G.sum() + rh * n).clamp(min=1e-12)
    offs = np.cumsum([0] + [4 * dm + 4 for dm in dims])
    W0s = [t(W) for W in W0s]
    b0s = [t(b) for b in b0s]

    def unpack(th, m):
        a = offs[m]
        dm = dims[m]
        return th[:, a:a + 4 * dm].view(th.shape[0], 4, dm), th[:, a + 4 * dm:a + 4 * dm + 4]

    def f_and_g(th, ix=None):
        rr = rh if ix is None else rh[ix]
        SS = S if ix is None else S[ix]
        th = th.detach().requires_grad_(True)
        tot = 0.0
        cen = []
        for m in range(len(Xs)):
            W, b = unpack(th, m)
            Z = torch.einsum("nd,pkd->pnk", Xs[m], W) + b[:, None, :]
            tot = tot - (G[None] * torch.log_softmax(Z, 2)).sum((1, 2)) \
                + ((W - W0s[m]) ** 2).sum((1, 2)) / (2 * cws[m]) + ((b - b0s[m]) ** 2).sum(1) / (2 * cbs[m])
            cen.append(Z - Z.mean(2, keepdim=True))
        dis = sum(((cen[a] - cen[c]) ** 2).sum((1, 2)) for a in range(len(Xs)) for c in range(a + 1, len(Xs)))
        f = (tot + 0.5 * rr * dis) / SS
        (g,) = torch.autograd.grad(f.sum(), th)
        return f.detach(), g

    theta = torch.cat([torch.cat([W0s[m].reshape(1, -1), b0s[m].reshape(1, -1)], 1) for m in range(len(Xs))], 1).repeat(P, 1)
    theta, f, grad, n_iter, stalled = lbfgs_batch(theta, f_and_g, tol, max_iter)
    gp = grad.abs().amax(1)
    finite = torch.isfinite(gp) & torch.isfinite(theta).all(1) & torch.isfinite(f)
    Ws, bs = zip(*[unpack(theta, m) for m in range(len(Xs))])
    return {"Ws": [W.clone() for W in Ws], "bs": [b.clone() for b in bs], "grad_p": gp.cpu().numpy(), "n_iter_p": n_iter.cpu().numpy(),
            "stalled_p": stalled.cpu().numpy(), "finite_p": finite.cpu().numpy(), "S": S.cpu().numpy()}

def logits(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(P,4,d),(P,4) → (P, n, 4)。W が (4,d) なら (n,4)。"""
    X = t(X)
    if W.dim() == 2:
        return X @ W.T + b
    return torch.einsum("nd,pkd->pnk", X, W) + b[:, None, :]


def uar(y: np.ndarray, pred: np.ndarray) -> float:
    rs = [float((pred[y == k] == k).mean()) for k in range(4) if (y == k).any()]
    return float(np.mean(rs))


def uar_batch(y: np.ndarray, Z: torch.Tensor) -> np.ndarray:
    """Z: (P, n, 4) → 問ごとの UAR（%）。"""
    pred = Z.argmax(2).cpu().numpy()
    return np.array([100.0 * uar(y, pr) for pr in pred])


def selftest() -> None:
    """scikit-learn の多項ロジスティック回帰（class_weight='balanced'）と同じ解になるか。"""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    n, d = 600, 20
    X = rng.normal(size=(n, d))
    Wt = rng.normal(size=(4, d))
    y = (X @ Wt.T + rng.gumbel(size=(n, 4))).argmax(1)
    y[:20] = 3                                   # 釣り合いを崩す
    Xs = (X - X.mean(0)) / X.std(0)
    for C in (0.01, 0.3, 10.0):
        sk = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, tol=1e-10).fit(Xs, y)
        G = balanced_onehot(y, np.arange(n), n)[None]
        # sklearn：½‖W‖² + C Σ w_i ℓ_i ↔ ここ：(Σ w_i ℓ_i + ‖W‖²/(2C')) / S は C' = C で同じ最小点
        r = fit(t(Xs), G, torch.zeros(4, d), torch.zeros(4), Cw=C, Cb=math.inf, tol=1e-7)
        dW = float((r["W"][0].cpu().numpy() - sk.coef_).__abs__().max())
        # 多項の切片は定数をずらしても同じなので，平均を引いて比べる
        bb = r["b"][0].cpu().numpy(); skb = sk.intercept_
        db = float(np.abs((bb - bb.mean()) - (skb - skb.mean())).max())
        print(f"C={C}: W の差の最大 {dW:.2e}，b の差の最大 {db:.2e}，反復 {r['n_iter']}，勾配 {r['grad_max']:.1e}")
        assert dW < 5e-5 and db < 5e-5, "scikit-learn と解が違う"
    # 引き戻し：Cw→0 で W0 に一致，fix_W で W が動かない
    W0 = torch.randn(4, d); b0 = torch.randn(4)
    G = balanced_onehot(y, np.arange(50), n)[None]
    r = fit(t(Xs), G, W0, b0, Cw=1e-8, Cb=1e-8)
    assert float((r["W"][0].cpu() - W0.to(r["W"].dtype)).abs().max()) < 1e-4
    r = fit(t(Xs), G, W0, b0, Cw=1.0, Cb=1.0, fix_W=True)
    assert float((r["W"][0].cpu() - W0.to(r["W"].dtype)).abs().max()) == 0.0
    # 温度の項：T2 = 1 なら G2 を G に足したのと同じ最小点（G2 の実装の検算）
    G1 = balanced_onehot(y, np.arange(100), n)[None]
    Q = torch.softmax(t(rng.normal(size=(n, 4))), -1)[None] * 0.5
    ra = fit(t(Xs), G1, W0, b0, Cw=0.3, Cb=1.0, G2=Q, T2=[1.0], tol=1e-8)
    rb = fit(t(Xs), G1 + Q, W0, b0, Cw=0.3, Cb=1.0, tol=1e-8)
    dd = float((ra["W"] - rb["W"]).abs().max())
    print(f"温度 1 の蒸留の項と足し合わせの差の最大 {dd:.1e}")
    assert dd < 1e-6
    # 温度 2 でも許容の勾配まで収束する
    rc = fit(t(Xs), G1, W0, b0, Cw=0.3, Cb=1.0, G2=Q, T2=[2.0], tol=1e-9)
    assert rc["grad_max"] < 1e-8
    # 共正則化：ρ = 0 なら各モダリティを別々に解いた解と一致する
    Xa, Xb = t(Xs[:, :10]), t(Xs[:, 10:])
    Gk = balanced_onehot(y, np.arange(80), n)
    Wa0, Wb0 = torch.zeros(4, 10), torch.zeros(4, 10)
    rc = fit_coreg([Xa, Xb], Gk, [Wa0, Wb0], [torch.zeros(4), torch.zeros(4)], [0.3, 0.3], [1.0, 1.0], [0.0, 1.0], tol=1e-9)
    ra_ = fit(Xa, Gk[None], Wa0, torch.zeros(4), Cw=0.3, Cb=1.0, tol=1e-9)
    dc = float((rc["Ws"][0][0] - ra_["W"][0]).abs().max())
    print(f"共正則化 ρ=0 と別々の解の差の最大 {dc:.1e}，勾配の最大 {rc['grad_p'].max():.1e}")
    assert dc < 1e-6 and rc["grad_p"].max() < 1e-8
    print("selftest: 通った")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
