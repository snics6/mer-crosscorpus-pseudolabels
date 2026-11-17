"""探索的な解析が共通に使う道具：組ごとの表と，「ターゲットの効果＋ソースの効果＋組に固有の部分」への分解。

2026-10-08 のコードの整理で，`explore_relation.py`（E1）・`explore_decomp.py`（E2）・`explore_modality.py`（E6）から
ここへ移した。計算の中身は移す前と同じ。

- 出力先：`OUT`（既定は `results/explore_relation/`）。環境変数 `MERDA_OUT_ROOT` を与えると，
  `results/` の代わりにその下へ書く（今の結果を上書きせずに試すため）。入力（地図・指標・学び直し）は
  常に `results/` から読む
- 組ごとの表：`per_pair`（得・K*・ずれの指標．E2 の入力）
- 分解：`design`・`r2`・`decompose`（ばらつきの内訳），`partial_rank`（ターゲットとソースの効果を除いた
  順位の偏相関と，その t 検定），`holm`，`fit`（効果と残りも返す版）
- 地図からの得：`gains`（反復ごと），`gains_all_k`（組 × K の平均）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"                                   # 入力はここから読む
OUT_ROOT = Path(os.environ["MERDA_OUT_ROOT"]) if os.environ.get("MERDA_OUT_ROOT") else RESULTS
OUT = OUT_ROOT / "explore_relation"
# 地図の場所．既定は results/map_<kind>/．MERDA_MAP_ROOT を指定すると <MERDA_MAP_ROOT>/<kind>/
# （C を固定した地図 results/fixc_map など．`docs/researchvision.md` 6.2 節，2026-10-09）
MAP_ROOT = Path(os.environ["MERDA_MAP_ROOT"]) if os.environ.get("MERDA_MAP_ROOT") else None


def map_dir(kind: str) -> Path:
    return MAP_ROOT / kind if MAP_ROOT else RESULTS / f"map_{kind}"

KINDS = ("audio_wavlm", "video", "text_roberta", "fuse-audio-video", "fuse-audio-text",
         "fuse-video-text", "fuse-audio-video-text")
SIX = ["IEMOCAP-impro", "IEMOCAP-script", "MOSEI", "MELD", "AFEW", "CREMA-D"]   # 映像の群
KSTAR_RANK = {"none": 0, "1": 1, "5": 2, "10": 3, "25": 4, "50": 5, "all": 6}
METRICS = {                               # 列名 → 表示名（軸）
    "pad": "PAD (axis 1)", "pad_bal": "PAD, class-balanced (axis 1)", "mmd2": "MMD² (axis 1)",
    "frechet": "Fréchet (axis 1)", "coverage_t_by_s": "coverage (axis 2)",
    "density_t_by_s": "density (axis 2)",
    "cov_dominance": "cov. dominance c (axis 2)", "l1_true": "label-prior L1 (axis 3)",
    "r_src": "R_src = 1 − UAR(src only) (axis 4)", "delta_ce": "Δ_CE (axis 4)",
}


def per_pair(kind: str) -> pd.DataFrame | None:
    m, s = map_dir(kind) / "summary", RESULTS / f"shift_{kind}"
    if not (m / "pairs.csv").exists() or not (s / "space_choice.json").exists():
        return None
    pairs = pd.read_csv(m / "pairs.csv", dtype={"K": str})
    piv = pairs.pivot_table(index=["target", "source"], columns="K", values="diff_mean")
    out = pd.DataFrame({"gain_K1": piv["1"], "gain_K5": piv["5"], "worst": piv.min(axis=1)})
    out["n_rep"] = pairs.groupby(["target", "source"])["n_rep"].min()
    ks = pd.read_csv(m / "kstar.csv", dtype=str).set_index(["target", "source"])
    out["kstar"] = [KSTAR_RANK.get(ks.loc[i, "K_star"], np.nan)
                    if ks.loc[i, "status"] not in ("single_rep", "iemocap_interval_undecided")
                    else np.nan for i in out.index]
    space = json.loads((s / "space_choice.json").read_text())["choice"]
    inp_all = pd.read_csv(s / "input.csv", comment="#")
    inp = inp_all[(inp_all.space == space) & (~inp_all.balanced)].set_index(["target", "source"])
    bal = inp_all[(inp_all.space == space) & (inp_all.balanced)].set_index(["target", "source"])
    lab = pd.read_csv(s / "labeling.csv", comment="#").set_index(["target", "source"])
    lb = pd.read_csv(s / "label.csv", comment="#").set_index(["target", "source"])
    for c in ("pad", "mmd2", "frechet", "coverage_t_by_s", "density_t_by_s", "cov_dominance"):
        out[c] = inp[c]
    out["pad_bal"] = bal["pad"]
    out["r_src"], out["delta_ce"], out["l1_true"] = lab["r_src"], lab["delta_ce"], lb["l1_true"]
    out["kind"], out["space"] = kind, space
    return out.reset_index()


def design(d: pd.DataFrame, use_t: bool, use_s: bool) -> np.ndarray:
    cols = [np.ones(len(d))]
    for use, col in ((use_t, "target"), (use_s, "source")):
        if use:
            levels = sorted(d[col].unique())
            cols += [(d[col] == lv).to_numpy(float) for lv in levels[1:]]
    return np.column_stack(cols)


def r2(y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return (1 - float((e ** 2).sum()) / sst if sst > 0 else np.nan), e


def decompose(d: pd.DataFrame, col: str) -> dict | None:
    x = d[["target", "source", col]].dropna()
    if len(x) < 8 or x[col].nunique() < 2:
        return None
    y = x[col].to_numpy(float)
    rt, _ = r2(y, design(x, True, False))
    rs, _ = r2(y, design(x, False, True))
    rts, e = r2(y, design(x, True, True))
    df_res = len(x) - (x.target.nunique() + x.source.nunique() - 1)
    return {"n_pairs": len(x), "df_resid": df_res, "r2_target": rt, "r2_source": rs,
            "r2_both": rts, "pair_specific": 1 - rts,
            "resid": pd.Series(e, index=pd.MultiIndex.from_frame(x[["target", "source"]]))}


def partial_rank(d: pd.DataFrame, resp: str, met: str) -> dict | None:
    """順位に直し，ターゲットとソースの指示変数を入れた回帰で，指標の係数を t 検定する。"""
    x = d[["target", "source", resp, met]].dropna()
    if len(x) < 8:
        return None
    y, m = stats.rankdata(x[resp]), stats.rankdata(x[met])
    X0 = design(x, True, True)
    X1 = np.column_stack([X0, m])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ beta
    df = len(y) - np.linalg.matrix_rank(X1)
    se = np.sqrt((e @ e) / df * np.linalg.inv(X1.T @ X1)[-1, -1])
    ry = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    rm = m - X0 @ np.linalg.lstsq(X0, m, rcond=None)[0]
    return {"r_partial_rank": float(np.corrcoef(ry, rm)[0, 1]), "t": float(beta[-1] / se),
            "p": float(2 * stats.t.sf(abs(beta[-1] / se), df)), "df": int(df), "n": len(y)}


def holm(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    adj = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]), 1)
    out = np.empty_like(adj)
    out[o] = adj
    return out


def fit(d: pd.DataFrame, col: str) -> dict:
    """v = μ + a_t + b_s + e を当てはめ，内訳・効果・残りを返す。"""
    y = d[col].to_numpy(float)
    sst = float(((y - y.mean()) ** 2).sum())

    def r2(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ beta
        return 1 - float(e @ e) / sst, beta, e
    rt, _, _ = r2(design(d, True, False))
    rts, beta, e = r2(design(d, True, True))
    tl, sl = sorted(d.target.unique()), sorted(d.source.unique())
    a = dict(zip(tl, np.r_[0.0, beta[1:len(tl)]]))
    b = dict(zip(sl, np.r_[0.0, beta[len(tl):len(tl) + len(sl) - 1]]))
    idx = pd.MultiIndex.from_frame(d[["target", "source"]])
    return {"target": rt, "source": rts - rt, "pair": 1 - rts,
            "resid": pd.Series(e, index=idx), "a": pd.Series(a), "b": pd.Series(b)}


def gains(kind: str) -> pd.DataFrame:
    """反復ごとの得（合併 − ターゲットのみ）。列：target, source, K, rep, gain。"""
    rows = []
    for p in sorted(map_dir(kind).glob("*.csv")):
        d = pd.read_csv(p, comment="#", dtype={"K": str, "source": str},
                        usecols=["target", "source", "method", "K", "rep", "uar"])
        t = d[d.method == "tgt_only"].set_index(["K", "rep"])["uar"]
        m = d[d.method == "merged"].copy()
        m["gain"] = m["uar"].to_numpy() - t.loc[list(zip(m.K, m.rep))].to_numpy()
        rows.append(m[["target", "source", "K", "rep", "gain"]])
    return pd.concat(rows, ignore_index=True)


def gains_all_k(kind: str) -> pd.DataFrame:
    p = pd.read_csv(map_dir(kind) / "summary" / "pairs.csv", dtype={"K": str})
    return p.pivot_table(index=["target", "source"], columns="K", values="diff_mean").reset_index()
