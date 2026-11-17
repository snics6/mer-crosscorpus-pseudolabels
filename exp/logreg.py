"""多クラスのロジスティック回帰を GPU で解く（`exp/map.py` の学習器。`docs/methods.md` 1.2）。

sklearn の `LogisticRegression(C=C, class_weight="balanced")`（lbfgs，多クラス，L2，切片あり）
と**同じ目的関数**を，学習データで標準化した特徴の上で解く：

    min_{W,b}  (1/S) Σ_i s_i · CE(W z_i + b, y_i)  +  ‖W‖² / (2 C S)

s_i はクラスを均等にする重み n / (k · n_{y_i})，S = Σ s_i（均等重みなら S = n）。
切片 b には罰則をかけない。float64 の L-BFGS（強 Wolfe の直線探索）で，勾配の最大絶対値が
`TOL` 以下になるまで解く。sklearn の既定（1e-4）より厳しくし，最適解に近づける。

**温め直し**：`init` に別の学習データで解いた `Fitted` を渡すと，その解を，生の特徴の上で
**同じ出力になるよう**今の標準化へ写してから解き始める（`Fitted.moved_to`）。
W について狭義凸なので，出発点が変わっても行き着く W と予測は同じ（止める基準の範囲内で）。
b は全クラスに同じ数を足しても出力が変わらないので，その分だけ出発点に依る。

## 2 つの解き方

- `fit`：1 問ずつ torch の L-BFGS で解く。**比べる相手**（`exp/test_map.py`）
- `fit_batch`：同じ特徴行列の上の多数の問（学習に使う行の組 × C）を**束ねて一度に解く**。
  1 反復の時間がデータの大きさにほぼ依らず，GPU を呼ぶ手間が支配的だったため
  （`docs/researchvision.md` 6.3 の計測）。各問は自分の履歴・歩幅・止める判定を持ち，
  **互いに影響しない**。止まった問は動かさない。直線探索は Armijo 条件の後退法
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
N_CLASSES = 4
TOL = 1e-6               # 勾配の最大絶対値（sklearn の tol に当たる。既定 1e-4）
MAX_ITER = 20000
HISTORY = 10             # L-BFGS の記憶の本数（sklearn が使う scipy の既定と同じ）


def to_device(X: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(X, dtype=DTYPE, device=DEVICE)


@dataclass
class Fitted:
    """1 つの学習データについて，候補の C ごとに解いた分類器。"""
    mu: torch.Tensor                      # 標準化の平均 (d,)
    sd: torch.Tensor                      # 標準化の標準偏差 (d,)
    W: dict[float, torch.Tensor]          # (k, d)。標準化した特徴の上の係数
    b: dict[float, torch.Tensor]          # (k,)
    n_iter: dict[float, int]
    grad: dict[float, float]              # 止まったときの勾配の最大絶対値

    def logits(self, X: torch.Tensor, c: float) -> torch.Tensor:
        return ((X - self.mu) / self.sd) @ self.W[c].T + self.b[c]

    def predict(self, X: torch.Tensor, c: float) -> np.ndarray:
        return self.logits(X, c).argmax(1).cpu().numpy()

    def moved_to(self, mu: torch.Tensor, sd: torch.Tensor, c: float
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """生の特徴の上で同じ出力になる (W, b) を，標準化 (mu, sd) の上で返す（温め直し用）。"""
        A = self.W[c] / self.sd                                 # 生の特徴の上の係数
        a0 = self.b[c] - (self.mu / self.sd) @ self.W[c].T
        return A * sd, a0 + mu @ A.T


def objective(Z: torch.Tensor, y: torch.Tensor, sw: torch.Tensor, C: float,
              W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    S = sw.sum()
    ce = F.cross_entropy(Z @ W.T + b, y, reduction="none")
    return (sw * ce).sum() / S + (W * W).sum() / (2 * C * S)


def balanced_weights(y: torch.Tensor) -> torch.Tensor:
    cnt = torch.bincount(y, minlength=N_CLASSES)
    if (cnt == 0).any():
        raise ValueError(f"学習データに欠ける感情がある: {cnt.tolist()}")
    return (len(y) / (N_CLASSES * cnt.to(DTYPE)))[y]


def _solve(Z, y, sw, C, W0, b0) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    W = W0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=MAX_ITER, max_eval=2 * MAX_ITER,
                            tolerance_grad=TOL, tolerance_change=1e-14,
                            history_size=HISTORY, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective(Z, y, sw, C, W, b)
        loss.backward()
        return loss

    opt.step(closure)
    closure()                                   # 止まった点の勾配を記録する
    g = max(float(W.grad.abs().max()), float(b.grad.abs().max()))
    n_iter = int(opt.state[opt._params[0]]["n_iter"])
    return W.detach(), b.detach(), n_iter, g


def fit(X: torch.Tensor, y: torch.Tensor, Cs: tuple[float, ...],
        init: Fitted | None = None) -> Fitted:
    """X を標準化し（sklearn の StandardScaler と同じく母標準偏差），C ごとに解く。"""
    mu = X.mean(0)
    sd = X.std(0, correction=0)
    sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    Z = (X - mu) / sd
    sw = balanced_weights(y)
    k, d = N_CLASSES, X.shape[1]
    W, b, it, gr = {}, {}, {}, {}
    for c in Cs:
        if init is None:
            W0 = torch.zeros(k, d, dtype=DTYPE, device=X.device)
            b0 = torch.zeros(k, dtype=DTYPE, device=X.device)
        else:
            W0, b0 = init.moved_to(mu, sd, c)
        W[c], b[c], it[c], gr[c] = _solve(Z, y, sw, c, W0, b0)
    return Fitted(mu, sd, W, b, it, gr)


# ---------------------------------------------------------------- 束ねて解く

ARMIJO = 1e-4            # 直線探索の十分減少の係数
MAX_HALVING = 50         # 歩幅を半分にする回数の上限。超えた問は「行き詰まり」として止める


def fit_batch(X: torch.Tensor, y: torch.Tensor, masks: list[np.ndarray],
              Cs: tuple[float, ...], inits: list[Fitted | None] | None = None,
              anchor: Fitted | None = None, anchor_c: float | None = None,
              weights: list[torch.Tensor | None] | None = None,
              stats: tuple[torch.Tensor, torch.Tensor] | None = None,
              Cb: float | None | list = None, fix_b: bool = False,
              aug: np.ndarray | None = None, extra: tuple | None = None) -> list[Fitted]:
    """masks[g]（学習に使う行）ごとに，候補の C すべてを束ねて解く。`fit` と同じ問題を解く。

    **stats=(mu, sd) を渡すと，全問をその統計で標準化する**（既定は問ごとに学習に使う行の統計）。
    仮説転移で K 本の統計を使わず，ソースの統計の座標で解くためのもの（E28 の 1 節）。
    **Cb を渡すと，切片にも罰則 ‖b − b_anchor‖²/(2 Cb S) をかける**（anchor が無ければ b_anchor = 0）。
    Cb は masks と同じ長さの列でもよい（問ごとの値．None はその問で罰則なし）。
    **fix_b=True なら切片を出発点（anchor の切片）に固定する**。どちらも既定では何もしない。

    **aug（長さ n の真偽．真がソースの行）を渡すと，特徴の水増し（Daumé 2007）の問を解く**：係数は
    [共通, ソース用, ターゲット用] の 3 つ（W は (k, 3d)），ソースの行の出力は x(W_共通 + W_ソース)ᵀ + b，
    ターゲットの行は x(W_共通 + W_ターゲット)ᵀ + b。これは [x, x, 0]・[x, 0, x] に W を掛けたものと同じで，
    行列積を d 列で済ませる。X は標準化済みとみなし（stats・anchor・inits は使えない），返す Fitted は
    水増しした特徴 [x, 0, x] の上の分類器（mu = 0，sd = 1．長さ 3d）。

    問 p = g·len(Cs) + i は，行 masks[g] で標準化し，C = Cs[i] で解く。
    計算は生の特徴の上で行う：標準化した特徴 z = (x − μ)/σ に対する (W, b) の出力は
    x @ (W/σ)ᵀ + (b − (μ/σ) Wᵀ) なので，全問の係数を並べて**行列積 1 回**で出力を出す。

    **anchor を渡すと，罰則を ‖W − W_anchor‖² / (2CS) にする**（仮説転移 M5．Kuzborskij &
    Orabona 2017）。W_anchor は anchor の C=anchor_c の解を，各問の標準化へ写したもの
    （生の特徴の上で同じ出力．`moved_to`）。C が小さいほど anchor に強く引き寄せる。
    anchor を渡したときは，出発点も anchor にする（inits を一緒に渡したときだけ，罰則の中心は anchor，
    出発点は inits[g] の C=Cs[i] の解にする．E61 のエントロピー最小化が none の解から始めるためのもの）。

    weights[g] を渡すと，masks[g] の行の重みを感情の釣り合いの重み（`balanced_weights`）の
    代わりにそれにする（長さ n．使うのは masks[g] の行だけ）。罰則の S は重みの和。

    **extra=(fn, mass) を渡すと，目的関数に fn(L, ix)/S を足す**（E61 のエントロピー最小化．Cs は 1 つの
    ときだけ）。fn は全問の出力 L（n × 問 × 感情）と問の添字 ix（None なら全問）から問ごとの値（長さ 問）を
    返す。mass[g] は問 g で S に足す重みの総量（ラベル無しの項の「本数」）。既定（None）では何もしない。
    """
    n, d = X.shape
    k, nC, G = N_CLASSES, len(Cs), len(masks)
    explicit_inits = inits is not None
    if aug is not None:
        if anchor is not None or stats is not None or any(i is not None for i in (inits or [])):
            raise ValueError("aug と anchor・stats・inits は一緒に使えない")
        stats = (torch.zeros(d, dtype=DTYPE, device=X.device), torch.ones(d, dtype=DTYPE, device=X.device))
    D = 3 * d if aug is not None else d                       # 係数の長さ
    B, P = G * nC, k * D + k
    dev = X.device
    inits = inits or [None] * G

    mu = torch.empty(B, d, dtype=DTYPE, device=dev)
    sd = torch.empty(B, d, dtype=DTYPE, device=dev)
    Mw = torch.zeros(n, B, dtype=DTYPE, device=dev)          # 行の重み（学習に使わない行は 0）
    S = torch.empty(B, dtype=DTYPE, device=dev)
    C = torch.tensor([c for _ in range(G) for c in Cs], dtype=DTYPE, device=dev)
    theta = torch.zeros(B, P, dtype=DTYPE, device=dev)
    Wanc = torch.zeros(B, k, D, dtype=DTYPE, device=dev)     # 罰則の中心（既定は 0）
    banc = torch.zeros(B, k, dtype=DTYPE, device=dev)        # 切片の罰則の中心（Cb を渡したときだけ使う）
    if anchor is not None and not explicit_inits:
        inits = [anchor] * G
    for g, m in enumerate(masks):
        m = torch.as_tensor(np.array(m), device=dev)
        if stats is not None:
            mu_g, sd_g = stats[0].to(dev, DTYPE), stats[1].to(dev, DTYPE)
        else:
            Xm = X[m]                                                          # 統計を取る行だけ（stats を渡したときは複製しない）
            mu_g = Xm.mean(0)
            sd_g = Xm.std(0, correction=0)
            sd_g = torch.where(sd_g > 1e-12, sd_g, torch.ones_like(sd_g))
        sw = torch.zeros(n, dtype=DTYPE, device=dev)
        if weights is not None and weights[g] is not None:
            sw[m] = weights[g].to(dev, DTYPE)[m]
        else:
            sw[m] = balanced_weights(y[m])
        for i, c in enumerate(Cs):
            p = g * nC + i
            mu[p], sd[p], Mw[:, p], S[p] = mu_g, sd_g, sw, sw.sum() + (0.0 if extra is None else float(extra[1][g]))
            if anchor is not None:
                Wanc[p], banc[p] = anchor.moved_to(mu_g, sd_g, anchor_c)
            if inits[g] is not None:
                W0, b0 = inits[g].moved_to(mu_g, sd_g, anchor_c if (anchor is not None and not explicit_inits) else c)
                theta[p, :k * d], theta[p, k * d:] = W0.reshape(-1), b0
    yy = y.view(n, 1, 1).expand(n, B, 1)
    mu_sd = mu / sd
    if aug is not None:
        aug_t = torch.as_tensor(np.asarray(aug, bool), device=dev)
        i_s, i_t = torch.nonzero(aug_t).squeeze(1), torch.nonzero(~aug_t).squeeze(1)
        inv = torch.argsort(torch.cat([i_s, i_t]))                         # 並べ直した出力を元の行の順へ
        Xs_, Xt_ = X[i_s], X[i_t]
    cb_list = list(Cb) if isinstance(Cb, (list, tuple)) else [Cb] * G
    use_cb = any(c is not None for c in cb_list)
    Cbv = torch.tensor([float("inf") if c is None else float(c) for c in cb_list for _ in Cs], dtype=DTYPE, device=dev)

    def f_and_g(th: torch.Tensor, ix: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """th の問（ix を渡せば問 ix だけ．th はその行だけ）の目的関数と勾配。"""
        sel = (lambda v: v) if ix is None else (lambda v: v[ix])
        Bx = th.shape[0]
        th = th.detach().requires_grad_(True)
        W, b = th[:, :k * D].view(Bx, k, D), th[:, k * D:]
        if aug is not None:                                                # X は標準化済み（mu = 0，sd = 1）
            W3 = W.view(Bx, k, 3, d)
            As = (W3[:, :, 0] + W3[:, :, 1]).reshape(Bx * k, d)
            At = (W3[:, :, 0] + W3[:, :, 2]).reshape(Bx * k, d)
            L = torch.cat([Xs_ @ As.T, Xt_ @ At.T])[inv].view(n, Bx, k) + b
        else:
            A = (W / sel(sd)[:, None, :]).reshape(Bx * k, d)
            c0 = b - torch.einsum("bd,bkd->bk", sel(mu_sd), W)
            L = (X @ A.T).view(n, Bx, k) + c0
        nll = -torch.log_softmax(L, 2).gather(2, yy[:, :Bx]).squeeze(2)  # (n, Bx)
        Mx = Mw if ix is None else Mw[:, ix]
        Sx, Cx = sel(S), sel(C)
        f = (Mx * nll).sum(0) / Sx + ((W - sel(Wanc)) ** 2).sum((1, 2)) / (2 * Cx * Sx)
        if use_cb:
            f = f + ((b - sel(banc)) ** 2).sum(1) / (2 * sel(Cbv) * Sx)
        if extra is not None:
            f = f + extra[0](L, ix) / Sx
        (grad,) = torch.autograd.grad(f.sum(), th)
        if fix_b:
            grad[:, k * D:] = 0.0                                          # 切片を動かさない
        return f.detach(), grad

    m_hist = HISTORY
    Sh = torch.zeros(m_hist, B, P, dtype=DTYPE, device=dev)
    Yh = torch.zeros(m_hist, B, P, dtype=DTYPE, device=dev)
    rho = torch.zeros(m_hist, B, dtype=DTYPE, device=dev)                # 無効な履歴は 0
    gamma = torch.ones(B, dtype=DTYPE, device=dev)
    fresh = torch.ones(B, dtype=torch.bool, device=dev)                  # 履歴が空（初回・やり直し）
    n_iter = torch.zeros(B, dtype=torch.long, device=dev)
    stalled = torch.zeros(B, dtype=torch.bool, device=dev)

    f, grad = f_and_g(theta)
    active = grad.abs().amax(1) > TOL
    slot = 0
    for _ in range(MAX_ITER):
        if not active.any():
            break
        # ---- 方向：L-BFGS の 2 重ループ（新しい履歴から古い履歴へ，そして逆）
        q = grad.clone()
        order = [(slot - 1 - j) % m_hist for j in range(m_hist)]
        alpha = torch.zeros(m_hist, B, dtype=DTYPE, device=dev)
        for j in order:
            alpha[j] = rho[j] * (Sh[j] * q).sum(1)
            q -= alpha[j][:, None] * Yh[j]
        q *= gamma[:, None]
        for j in reversed(order):
            beta = rho[j] * (Yh[j] * q).sum(1)
            q += Sh[j] * (alpha[j] - beta)[:, None]
        dvec = -q
        gtd = (grad * dvec).sum(1)
        reset = active & (gtd >= 0)                   # 下り方向でなければ履歴を捨てて勾配方向へ
        if reset.any():
            rho[:, reset] = 0.0
            gamma[reset] = 1.0
            fresh |= reset
            dvec[reset] = -grad[reset]
            gtd = (grad * dvec).sum(1)
        dvec[~active] = 0.0
        t = torch.ones(B, dtype=DTYPE, device=dev)
        t[fresh] = torch.clamp(1.0 / grad[fresh].abs().sum(1), max=1.0)

        # ---- 歩幅：Armijo 条件を満たすまで半分にする（満たした問はそこで固定）
        pending = active.clone()
        f_new, g_new = f.clone(), grad.clone()
        th_new = theta.clone()
        for _h in range(MAX_HALVING):
            ix = torch.nonzero(pending).squeeze(1)                     # まだ歩幅の決まらない問だけを計算する
            trial = theta[ix] + t[ix, None] * dvec[ix]
            f_t, g_t = f_and_g(trial, ix)
            okx = f_t <= f[ix] + ARMIJO * t[ix] * gtd[ix]
            io = ix[okx]
            f_new[io], g_new[io], th_new[io] = f_t[okx], g_t[okx], trial[okx]
            pending[io] = False
            if not pending.any():
                break
            t[pending] *= 0.5
        stalled |= pending
        moved = active & ~pending

        # ---- 履歴の更新（曲率が正の組だけ）
        s_vec, y_vec = th_new - theta, g_new - grad
        sy = (s_vec * y_vec).sum(1)
        good = moved & (sy > 1e-10 * s_vec.norm(dim=1) * y_vec.norm(dim=1))
        Sh[slot], Yh[slot] = s_vec, y_vec
        rho[slot] = torch.where(good, 1.0 / torch.where(good, sy, 1.0), 0.0)
        gamma = torch.where(good, sy / (y_vec * y_vec).sum(1).clamp_min(1e-300), gamma)
        fresh &= ~good
        slot = (slot + 1) % m_hist

        theta, f, grad = th_new, f_new, g_new
        n_iter += moved.long()
        active = moved & (grad.abs().amax(1) > TOL)

    gmax = grad.abs().amax(1).cpu().numpy()
    its = n_iter.cpu().numpy()
    out = []
    for g in range(G):
        W, b, it, gr = {}, {}, {}, {}
        for i, c in enumerate(Cs):
            p = g * nC + i
            W[c], b[c] = theta[p, :k * D].view(k, D).clone(), theta[p, k * D:].clone()
            it[c], gr[c] = int(its[p]), float(gmax[p])
        if aug is not None:
            out.append(Fitted(torch.zeros(D, dtype=DTYPE, device=dev), torch.ones(D, dtype=DTYPE, device=dev), W, b, it, gr))
        else:
            out.append(Fitted(mu[g * nC], sd[g * nC], W, b, it, gr))
    if stalled.any():
        print(f"注意: 直線探索が行き詰まった問 {int(stalled.sum())} 個"
              f"（勾配 {gmax[stalled.cpu().numpy()]}）", flush=True)
    return out
