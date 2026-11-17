"""ずれを 1 種類ずつ消す操作と，その対照の正本。**操作はすべてここを通す。**

根拠と規則は `docs/methods.md` 5 節（操作の名前 P1・I1・D3 などを各関数に書く）。

## 規則（`docs/methods.md` 5.2）

- R1：**抜き出す操作には，同じ本数をランダムに抜いた対照を必ず付ける**
  （`random_subset_like`）。
- R2：操作の前後で `shift.py` の指標を測り，狙ったずれだけが下がったことを確かめる。
- R3：ターゲットについて使ってよいのは**ラベルなしデータと K 本だけ**。
  このモジュールの関数は，ターゲットのテストのラベルを引数に取らない。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.neighbors import NearestNeighbors

N_CLASSES = 4


# ---------------------------------------------------------------- 対照（R1）

def random_subset_like(n_total: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """操作で選んだのと同じ本数を，ランダムに選ぶ対照。"""
    if size > n_total:
        raise ValueError(f"{n_total} 本から {size} 本は選べない")
    return np.sort(rng.choice(n_total, size=size, replace=False))


# ---------------------------------------------------------------- 4.1 感情の割合

def balanced_subset(y: np.ndarray, rng: np.random.Generator,
                    k: int = N_CLASSES) -> np.ndarray:
    """P1：最小クラスの本数に合わせて，各クラスを同数に抜く。"""
    y = np.asarray(y)
    m = np.bincount(y, minlength=k).min()
    idx = [rng.choice(np.flatnonzero(y == c), size=m, replace=False) for c in range(k)]
    return np.sort(np.concatenate(idx))


def prior_matched_subset(y: np.ndarray, prior: np.ndarray, rng: np.random.Generator,
                         k: int = N_CLASSES) -> np.ndarray:
    """P2：クラスの割合が prior になるよう，できるだけ多く抜く。

    prior はターゲットの**推定**割合（`shift.mlls_prior`）。真の割合を渡すのはオラクル（P2-oracle）。
    """
    y = np.asarray(y)
    prior = np.asarray(prior, float) / np.sum(prior)
    counts = np.bincount(y, minlength=k)
    ok = prior > 0
    total = int(np.floor(np.min(counts[ok] / prior[ok])))   # 最も足りないクラスで決まる
    want = np.floor(prior * total).astype(int)
    idx = [rng.choice(np.flatnonzero(y == c), size=want[c], replace=False)
           for c in range(k) if want[c] > 0]
    return np.sort(np.concatenate(idx))


# ---------------------------------------------------------------- 4.2 入力の違い

def standardize_with(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """I1：ref（同じ群のラベルなしデータ）の平均・標準偏差で X を標準化する。

    ターゲットは，テストを含むラベルなしの全発話を ref にしてよい（ラベルを使わない）。
    """
    mu, sd = ref.mean(0), ref.std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def standardize_by_speaker(X: np.ndarray, speakers: np.ndarray,
                           min_count: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """I2：話者ごとに標準化する（Schuller 2010）。

    発話が min_count 本未満の話者は平均だけ引く（標準偏差を推定できない）。
    戻り値の 2 つ目は各行の話者の発話数（不安定さの記録用。`docs/methods.md` 5.3.2 の注意）。
    """
    X = X.astype(float).copy()
    speakers = np.asarray(speakers)
    n_per = np.zeros(len(X), int)
    for s in np.unique(speakers):
        m = speakers == s
        n_per[m] = m.sum()
        X[m] -= X[m].mean(0)
        if m.sum() >= min_count:
            sd = X[m].std(0)
            X[m] /= np.where(sd > 0, sd, 1.0)
    return X, n_per


def coral(Xs: np.ndarray, Xt: np.ndarray, reg: float = 1.0) -> np.ndarray:
    """I3：CORAL（Sun ほか 2016 算法 1）。ソースを白色化し，ターゲットの共分散で色付けし直す。

    **入力は PCA 後（`shift.pooled_pca`）で，群ごとに標準化済みであること**
    （単位行列を足す正則化は，標準化した尺度を前提にしている）。
    """
    def msqrt(C, inv):
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1e-12)
        return V @ np.diag(w ** (-0.5 if inv else 0.5)) @ V.T

    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + reg * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + reg * np.eye(d)
    return Xs @ msqrt(Cs, True) @ msqrt(Ct, False)


# ---------------------------------------------------------------- 4.3 覆い方

def near_target_subset(Xs: np.ndarray, Xt_unlabeled: np.ndarray, size: int) -> np.ndarray:
    """C1：ターゲット（ラベルなし）の最も近い点までの距離が小さい順に，ソースを size 本選ぶ。

    X は同じ前処理（`shift.pooled_standardize` など）を通してから渡す。
    対照は `random_subset_like(len(Xs), size, rng)`。
    """
    nn = NearestNeighbors(n_neighbors=1).fit(Xt_unlabeled)
    dist = nn.kneighbors(Xs)[0][:, 0]
    return np.sort(np.argsort(dist, kind="stable")[:size])


# ---------------------------------------------------------------- 4.4 付け方

def flip_labels(y: np.ndarray, rho: float, rng: np.random.Generator,
                k: int = N_CLASSES) -> np.ndarray:
    """D3s：ラベルの割合 rho を，**ほかの感情へ一様に**付け替える（Wang 2019 の εy に相当）。

    各発話で最も確率の高い感情は変わらない（rho < 3/4 のとき）ので，**付け方のずれは生じず，
    信号が弱まるだけ**（`docs/methods.md` 7.4）。付け方のずれを足すのは `flip_labels_pair`（D3a）。
    """
    y = np.asarray(y).copy()
    n_flip = int(round(rho * len(y)))
    idx = rng.choice(len(y), size=n_flip, replace=False)
    shift = rng.integers(1, k, size=n_flip)             # 0 を除くので必ず別のクラス
    y[idx] = (y[idx] + shift) % k
    return y


def flip_labels_pair(y: np.ndarray, src: int, dst: int, rho: float,
                     rng: np.random.Generator) -> np.ndarray:
    """D3a：感情 src の発話のうち割合 rho を，感情 dst へ付け替える（付け方のずれを足す）。

    src と dst の境界の対数オッズが log(1/(1−2ρ)) だけ動く（`docs/methods.md` 7.4）。
    一様な付け替え（`flip_labels`，D3s）と違い，理想的な分類器が変わる。
    """
    y = np.asarray(y).copy()
    cand = np.flatnonzero(y == src)
    idx = rng.choice(cand, size=int(round(rho * len(cand))), replace=False)
    y[idx] = dst
    return y


def fit_class_bias(logits_k: np.ndarray, y_k: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    """D1：ソースの分類器の対数オッズに足すクラスごとのずらしを，ターゲットの K 本で学ぶ。

    閾値のずれを消す操作（Reeve 2021 の閾値校正，Alexandari 2020 のバイアス補正）。
    K=1 では過学習するので弱い L2 をかける（l2 を変えたときの感度も報告する）。
    ずらしの和は 0 に固定する（全クラスに同じ数を足しても予測は変わらないため）。
    """
    k = logits_k.shape[1]
    y_k = np.asarray(y_k)

    def loss(b_free):
        b = np.r_[b_free, -b_free.sum()]
        lp = log_softmax(logits_k + b, axis=1)[np.arange(len(y_k)), y_k]
        return -lp.mean() + l2 * np.sum(b ** 2)

    res = minimize(loss, np.zeros(k - 1), method="L-BFGS-B")
    return np.r_[res.x, -res.x.sum()]
