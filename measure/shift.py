"""ずれの指標の正本。**指標はすべてここを通す。実験コードで書き直さない。**

根拠と注意点は `docs/methods.md` 4 節（節番号を各関数に書く．共通の決まりは S1〜S10）。実装を変えたら md も直す。

## 共通の決まり（md の 0 節）

- 入力の指標は**両側を同じ本数に抜き出してから**計算する（C2）。本数の既定 `N_DEFAULT`
  は，ゼロ点を作るために RAVDESS（672 本）を話者で半分に割っても足りる 300。
- **ゼロ点**（同じ群を話者で 2 つに割った値）を `zero_point()` で必ず併記する（C1）。
- 距離を使う指標の前処理は「2 標本を合わせた標準化」（C4）。群ごとの標準化は
  ずれを消す操作なので `ablate.py` にある。
- 共分散を使う指標は PCA で `PCA_DIM` 次元に落とす（C6）。

## 使い方

    import shift
    m = shift.input_metrics(Xs, gs, Xt, gt, seed=0)          # 軸 1・2（ラベル不要）
    z = shift.zero_point(Xt, gt, seed=0)                      # 同じ手順のゼロ点
    l1 = shift.label_l1(ys, yt)                               # 軸 3
    C = shift.confusion(yt, clf.predict(Xt))                  # 軸 4
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

N_CLASSES = 4
N_DEFAULT = 300          # 抜き出す本数（片側）。md S2
R_DEFAULT = 20           # 抜き出しの繰り返し回数。md S2
PCA_DIM = 64             # 共分散を使う指標の次元。md S6（r ≪ n）
K_NN = 5                 # coverage の近傍数。md 4.4.1（Naeem 2020）
PAD_C_GRID = (0.01, 0.1, 1.0)
PAD_TOL = 1e-8           # 判別器の止める基準。sklearn の既定 1e-4 では判定が 1〜2 本ずれた。md 4.3.1
SPACES = ("raw", "pca64")   # 入力の指標を測る空間。対数オッズの空間は特徴として渡す。md 4.2
EIG_FLOOR = 1e-10        # 共分散の支配で除く方向：最大の固有値に対するこの比より小さい方向。md 4.4.2


# ---------------------------------------------------------------- 前処理

def subsample(n_total: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """0..n_total-1 から n 本を非復元で抜く。足りなければ例外（黙って減らさない）。"""
    if n_total < n:
        raise ValueError(f"標本が {n_total} 本しかなく {n} 本を抜けない")
    return np.sort(rng.choice(n_total, size=n, replace=False))


def subsample_balanced(y: np.ndarray, n: int, rng: np.random.Generator,
                       k: int = N_CLASSES) -> np.ndarray:
    """感情ごとに n/k 本ずつ抜く（md S9：割合をそろえた版）。足りなければ例外。"""
    if n % k:
        raise ValueError(f"n={n} が感情の数 {k} で割り切れない")
    y = np.asarray(y)
    idx = []
    for c in range(k):
        cand = np.flatnonzero(y == c)
        if len(cand) < n // k:
            raise ValueError(f"感情 {c} が {len(cand)} 本しかなく {n // k} 本を抜けない")
        idx.append(rng.choice(cand, size=n // k, replace=False))
    return np.sort(np.concatenate(idx))


def pooled_standardize(Xs: np.ndarray, Xt: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """2 標本を合わせた平均・標準偏差で各次元を標準化する（md S4）。"""
    sc = StandardScaler().fit(np.vstack([Xs, Xt]))
    return sc.transform(Xs), sc.transform(Xt)


def _rank(r: int, d: int, n: int) -> int:
    """PCA の次元。特徴の次元と標本数を超えない（対数オッズの 4 次元でも落ちないように）。"""
    return max(1, min(r, d, n - 1))


def pooled_pca(Xs: np.ndarray, Xt: np.ndarray, r: int = PCA_DIM
               ) -> tuple[np.ndarray, np.ndarray]:
    """合わせて標準化し，合わせた PCA で r 次元に落とす（md S6）。次元は `_rank` で切る。"""
    Xs, Xt = pooled_standardize(Xs, Xt)
    r = _rank(r, Xs.shape[1], len(Xs) + len(Xt))
    p = PCA(n_components=r, random_state=0).fit(np.vstack([Xs, Xt]))
    return p.transform(Xs), p.transform(Xt)


# ---------------------------------------------------------------- 軸 1：入力

def pad(Xs: np.ndarray, gs: np.ndarray, Xt: np.ndarray, gt: np.ndarray,
        n_folds: int = 5) -> float:
    """proxy A-distance = 2(1 − 2ε)（md 4.3.1。尺度は Ben-David 2010・Ganin 2016）。

    ε は「ソースかターゲットか」を当てる線形分類器の，**話者で分けた**交差検証の
    誤り率（md S3）。C は外側の評価データを見ずに内側の話者分割で選ぶ
    （Ganin 2016 は評価に使う半分で最良の C を選んでいて楽観的）。
    gs と gt は話者 id。両側で名前が衝突しないよう接頭辞を付ける。
    """
    X = np.vstack([Xs, Xt])
    d = np.r_[np.zeros(len(Xs), int), np.ones(len(Xt), int)]
    g = np.r_[np.char.add("s:", np.asarray(gs, str)),
              np.char.add("t:", np.asarray(gt, str))]
    wrong = 0
    # 話者を分けたまま，各袋に両側の話者が入るようにそろえる（md 4.3.1．GroupKFold では
    # IEMOCAP のゼロ点（片側 5 人）で学習側が片方の群だけになった）
    for tr, te in StratifiedGroupKFold(n_splits=n_folds).split(X, d, g):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        best_c, best_err = PAD_C_GRID[0], np.inf
        for c in PAD_C_GRID:                          # 内側：話者で 3 分割
            errs = []
            for itr, iva in StratifiedGroupKFold(n_splits=3).split(Xtr, d[tr], g[tr]):
                m = LogisticRegression(C=c, max_iter=100000, tol=PAD_TOL).fit(
                    Xtr[itr], d[tr][itr])
                errs.append(np.mean(m.predict(Xtr[iva]) != d[tr][iva]))
            if np.mean(errs) < best_err:
                best_c, best_err = c, float(np.mean(errs))
        m = LogisticRegression(C=best_c, max_iter=100000, tol=PAD_TOL).fit(Xtr, d[tr])
        wrong += int(np.sum(m.predict(Xte) != d[te]))
    eps = wrong / len(X)
    return max(0.0, 2.0 * (1.0 - 2.0 * eps))


def mmd2_unbiased(Xs: np.ndarray, Xt: np.ndarray) -> float:
    """不偏な MMD²（Gretton 2012 式 (3)）。ガウスカーネル，幅は合わせた点の距離の中央値。

    **負になりうる。0 に切らない**（md 4.3.2）。高次元では平均の差を見ているのに近い。
    """
    Xs, Xt = pooled_standardize(Xs, Xt)
    Z = np.vstack([Xs, Xt])
    sq = np.sum(Z ** 2, 1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    iu = np.triu_indices(len(Z), 1)
    sigma2 = np.median(D2[iu])                        # 距離の中央値の 2 乗
    Kz = np.exp(-D2 / (2 * sigma2))
    m = len(Xs)
    Kss, Ktt, Kst = Kz[:m, :m], Kz[m:, m:], Kz[:m, m:]
    n = len(Xt)
    return float((Kss.sum() - np.trace(Kss)) / (m * (m - 1))
                 + (Ktt.sum() - np.trace(Ktt)) / (n * (n - 1))
                 - 2 * Kst.mean())


def frechet(Xs: np.ndarray, Xt: np.ndarray, r: int = PCA_DIM) -> float:
    """PCA 後のフレシェ距離（md 4.3.3）。**有限標本の偏りがある**のでゼロ点と並べる。"""
    Xs, Xt = pooled_pca(Xs, Xt, r)
    mu = Xs.mean(0) - Xt.mean(0)
    Ss, St = np.cov(Xs, rowvar=False), np.cov(Xt, rowvar=False)
    cs = sqrtm(Ss @ St)
    cs = cs.real if np.iscomplexobj(cs) else cs
    return float(mu @ mu + np.trace(Ss + St - 2 * cs))


# ---------------------------------------------------------------- 軸 2：覆い方

def density_coverage(X_ref: np.ndarray, X_other: np.ndarray, k: int = K_NN
                     ) -> tuple[float, float]:
    """Naeem 2020 の density と coverage（md 4.4.1）。

    X_ref を「参照」（覆われる側）とし，X_other がどれだけ覆うかを返す。
    **向きがある**：ターゲットの覆われ方は `density_coverage(Xt, Xs)`。
    距離は 2 標本を合わせて標準化した特徴のユークリッド距離。
    """
    R, O = pooled_standardize(X_ref, X_other)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(R)
    radius = nn.kneighbors(R)[0][:, -1]               # 自分自身を除く k 番目
    D = np.sqrt(np.maximum(
        np.sum(O ** 2, 1)[:, None] + np.sum(R ** 2, 1)[None, :] - 2 * O @ R.T, 0.0))
    inside = D < radius[None, :]                      # (other, ref)
    density = float(inside.sum() / (k * len(O)))
    coverage = float(inside.any(axis=0).mean())
    return density, coverage


def expected_coverage(n_ref: int, n_other: int, k: int = K_NN) -> float:
    """同一分布のときの coverage の期待値（Naeem 2020 式 (9)）。分布にも次元にも依らない。"""
    num = np.prod([(n_ref - 1 - i) for i in range(k)], dtype=float)
    den = np.prod([(n_other + n_ref - 1 - i) for i in range(k)], dtype=float)
    return float(1.0 - num / den)


def cov_dominance(Xs: np.ndarray, Xt: np.ndarray, r: int = PCA_DIM) -> float:
    """Σ_s ⪰ c Σ_t となる最大の c（md 4.4.2。Du 2021）。ターゲットの PCA 上位 r 方向で測る。

    最小固有値は小さい固有値の誤差に弱い。r を変えて安定するか確かめること。
    """
    Xs, Xt = pooled_standardize(Xs, Xt)
    p = PCA(n_components=_rank(r, Xt.shape[1], len(Xt)), random_state=0).fit(Xt)
    Ps, Pt = p.transform(Xs), p.transform(Xt)
    Ss, St = np.cov(Ps, rowvar=False), np.cov(Pt, rowvar=False)
    Ss, St = np.atleast_2d(Ss), np.atleast_2d(St)
    w, V = np.linalg.eigh(St)
    keep = w > EIG_FLOOR * w.max()          # 広がりがほぼ 0 の方向は除く（md 4.4.2．落ちないように）
    V, w = V[:, keep], w[keep]
    St_ih = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    M = V.T @ (St_ih @ Ss @ St_ih) @ V       # 残した方向の中だけで比べる
    return float(np.linalg.eigvalsh((M + M.T) / 2).min())


# ---------------------------------------------------------------- まとめて測る

def input_metrics(Xs: np.ndarray, gs: np.ndarray, Xt: np.ndarray, gt: np.ndarray,
                  n: int = N_DEFAULT, reps: int = R_DEFAULT, seed: int = 0,
                  space: str = "raw", ys: np.ndarray | None = None,
                  yt: np.ndarray | None = None) -> dict:
    """軸 1・2 の指標を，両側 n 本の抜き出しを reps 回くり返して平均と標準偏差で返す。

    space：`raw`＝渡した特徴のまま，`pca64`＝抜き出した 2 標本を合わせて標準化し PCA で
    `PCA_DIM` 次元に落としてから測る（md 4.2）。対数オッズの空間は，それを特徴として渡す。
    ys・yt を渡すと感情ごとに同数を抜く（md S9。n は 4 の倍数）。
    """
    if space not in SPACES:
        raise ValueError(f"space は {SPACES} のどれか: {space}")
    rng = np.random.default_rng(seed)
    gs, gt = np.asarray(gs), np.asarray(gt)
    balanced = ys is not None and yt is not None
    rows = []
    for _ in range(reps):
        i = subsample_balanced(ys, n, rng) if balanced else subsample(len(Xs), n, rng)
        j = subsample_balanced(yt, n, rng) if balanced else subsample(len(Xt), n, rng)
        a, b = Xs[i], Xt[j]
        if space == "pca64":
            a, b = pooled_pca(a, b, PCA_DIM)
        dens_t, cov_t = density_coverage(b, a)        # ターゲットがソースに覆われる
        dens_s, cov_s = density_coverage(a, b)        # 逆向き
        rows.append({
            "pad": pad(a, gs[i], b, gt[j]),
            "mmd2": mmd2_unbiased(a, b),
            "frechet": frechet(a, b),
            "coverage_t_by_s": cov_t, "density_t_by_s": dens_t,
            "coverage_s_by_t": cov_s, "density_s_by_t": dens_s,
            "cov_dominance": cov_dominance(a, b),
        })
    out: dict = {}
    for key in rows[0]:
        v = np.array([r[key] for r in rows])
        out[key] = float(v.mean())
        out[key + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    out["coverage_expected"] = expected_coverage(n, n)
    out["n"], out["reps"] = n, reps
    out["space"], out["balanced"], out["dim"] = space, balanced, int(Xs.shape[1])
    return out


def speaker_halves(g: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """話者を 2 つに分け，それぞれの行番号を返す（ゼロ点用）。"""
    rng = np.random.default_rng(seed)
    spk = np.unique(np.asarray(g))
    rng.shuffle(spk)
    left = set(spk[: len(spk) // 2])
    mask = np.array([s in left for s in g])
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def zero_point(X: np.ndarray, g: np.ndarray, n: int = N_DEFAULT,
               reps: int = R_DEFAULT, seed: int = 0, space: str = "raw",
               y: np.ndarray | None = None) -> dict:
    """同じ群を話者で 2 つに割り，`input_metrics` と同じ手順で測った値（md S1）。"""
    g = np.asarray(g)
    a, b = speaker_halves(g, seed)
    ya, yb = (None, None) if y is None else (np.asarray(y)[a], np.asarray(y)[b])
    return input_metrics(X[a], g[a], X[b], g[b], n=n, reps=reps, seed=seed,
                         space=space, ys=ya, yt=yb)


# ---------------------------------------------------------------- 軸 3：割合

def label_prior(y: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    return np.bincount(np.asarray(y), minlength=k) / len(y)


def label_l1(ys: np.ndarray, yt: np.ndarray, k: int = N_CLASSES) -> float:
    """‖p_s(y) − p_t(y)‖₁（0〜2。md 4.5.1）。"""
    return float(np.abs(label_prior(ys, k) - label_prior(yt, k)).sum())


def fit_bcts(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """クラスごとのバイアスつき温度較正（BCTS。Alexandari 2020）。(温度, バイアス) を返す。

    ソースの**保留データ**（話者を分けたもの）で当てはめる。md 4.5.2。
    """
    k = logits.shape[1]

    def nll(p):
        z = logits / np.exp(p[0]) + p[1:]
        return -log_softmax(z, axis=1)[np.arange(len(y)), y].mean()

    res = minimize(nll, np.zeros(k + 1), method="L-BFGS-B")
    return float(np.exp(res.x[0])), res.x[1:]


def apply_bcts(logits: np.ndarray, temp: float, bias: np.ndarray) -> np.ndarray:
    return softmax(logits / temp + bias, axis=1)


def mlls_prior(probs_t: np.ndarray, prior_s: np.ndarray,
               iters: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """ラベルなしのターゲットで感情の割合を EM で推定する（Saerens 2002，md 4.5.2）。

    probs_t は**較正済み**の予測確率。prior_s は「ソース保留データでの較正後の予測の平均」
    にすること（Alexandari 2020 Lemma A：真のソース割合を使うと，ずれが無くてもずれを推定する）。
    ラベルシフトの仮定のもとの推定であり，コーパス間では近似にすぎない。
    """
    q = prior_s.copy()
    for _ in range(iters):
        w = probs_t * (q / prior_s)
        w /= w.sum(1, keepdims=True)
        q_new = w.mean(0)
        if np.abs(q_new - q).max() < tol:
            return q_new
        q = q_new
    return q


def kish_neff(w: np.ndarray) -> float:
    """重み付き学習の実効標本数 (Σw)²/Σw²（Kish）。クラス均等重みの損を見積もる（md 7.3）。"""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


# ---------------------------------------------------------------- 軸 4：付け方

def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    """行＝正解，列＝予測，行ごとの割合にした混同行列（md 4.6.1）。"""
    C = np.zeros((k, k))
    np.add.at(C, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return C / np.maximum(C.sum(1, keepdims=True), 1)


def uar(y_true: np.ndarray, y_pred: np.ndarray, k: int = N_CLASSES) -> float:
    return float(np.mean(np.diag(confusion(y_true, y_pred, k))))


def ber(conf: np.ndarray) -> float:
    """Tachet 2020 の BER：**クラスごとの誤り率の最大値**（平均ではない。md 4.6.2）。"""
    return float((1 - np.diag(conf)).max())


def delta_ce(conf_s: np.ndarray, conf_t: np.ndarray) -> tuple[float, np.ndarray]:
    """Tachet 2020 の Δ_CE と，非対角の差の行列全体（md 4.6.2）。

    最大値は推定誤差で上に偏るので，**行列全体も報告する**。
    """
    diff = np.abs(conf_s - conf_t)
    off = diff[~np.eye(len(diff), dtype=bool)]
    return float(off.max()), diff
