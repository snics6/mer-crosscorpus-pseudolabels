"""DAI 実験 3：説明の解析（設計は `docs/設計/実験3_設計.md` 第 3.1 版．計画は `docs/計画/計画_2026-11-04.tex` 8.3 節）．

実験 1 の保存（段 1〜3 の W・b（float32），テストの判定（int8），予測の確率（float16），sample_id）と特徴から，
位置の誤りと方向の誤り，族 5，再現できる度合い，代わりの説明の表を計算する．学び直す計算は，切片の当て直し
（`dai_solver.fit` の fix_W=True，C_b = ∞，感情の釣り合いの重み）と，再現できる度合いの線形の識別器だけである．

段（--stage）：
  validate：1.4 節の妥当性の確かめ（確かめ 1：交差エントロピーの b* と UAR を直接最大にする切片の差．確かめ 2：重みだけを
            既知の角度で回したときの方向の誤りの増え分 ÷ UAR の下がり分）．仮のターゲットの回の出力だけを読む．
  detect  ：1.1 節の族 5 の標準誤差の見込み，検出できる差，記述に回すセル（実験 1 の detect と同じ手順）．説明 1・2 の量の
            区間の半幅の見込み（「同じ水準」を出しうるか．統計の検査の 3 回目の記録 4）．1 節の W* の引き戻しの強さ．
            2 組・候補の 2 組・筋書きの対の件数の表（ラベルの件数だけ）．確かめの合否．`frozen.json` に書き，
            本番の置き場（results/dai3）のときだけ `frozen/dai3_frozen.json` に写す．
  eval    ：評価の回の出力から，1〜3 節の全部の量，族 5 の区間と Holm，3 節の判定の規則を機械的に当てた結果，4 節の件数の規則，
            5 節の仮定の表の量を計算する．

入力：実験 1 の置き場（--dai1-out．既定 results/dai1）の source/，pseudo/units/，eval/units/，select/frozen.json
（主の版・選んだ設定・族 3 の相手・引き戻し・指数・δ・種の数）．
出力：--out-dir（既定 results/dai3）の validate/，detect/，eval/ の csv と json．行にコミットと dirty を書く．
成績の値は画面に出さない（行の数，件数，収束，所要だけを出す）．
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
import platform
import re
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
PAIRS = list(itertools.combinations(range(4), 2))                    # 感情の対（0=Neutral 1=Happy 2=Sad 3=Angry）
VERSIONS = [(0, 1, 2, 3)] + PAIRS                                     # 版 0 は 4 感情，1〜6 は感情の対（2 値）
VNAME = ["four"] + [f"{i}-{j}" for i, j in PAIRS]
NEUTRAL_PAIRS = ("0-1", "0-2", "0-3")
MIN_GROUP = 10                                                        # 1 節：どちらかの組で感情が 10 本未満なら計算しない
MIN_TOTAL = 30                                                        # 4 節：合計が 30 本未満の組は付録だけ
N_GROUPS_BIG = 50                                                     # 1.3 節：MOSEI の delete-a-group jackknife の群の数
GRID = np.round(np.arange(-30, 31) * 0.1, 10)                         # 確かめ 1：b* のまわり ±3，刻み 0.1
SWEEPS = 3                                                            # 確かめ 1：座標の上昇の巡の数
CHECK1_MAX = 1.0                                                      # 確かめ 1：差の中央値がこれ以下なら合格（ポイント）
ANGLES = (15.0, 30.0, 45.0)                                           # 確かめ 2：回す角度
CHECK2_MIN_DROP = 1.0                                                 # 確かめ 2：UAR の下がり分がこれ以上の単位だけを使う
CHECK2_RATIO = 0.7                                                    # 確かめ 2：和の比がこれ以上なら合格
CHECK2_MIN_UNITS = 10                                                 # 確かめ 2：使える単位がこれ未満なら判定できない
N_PERM = 2000                                                         # 説明 3：並べ替えの回数
NEG_DIR_MAX = 0.25                                                    # 5 節：方向の誤りが負の単位の割合の上限
WIDTH_MAX = 20.0                                                      # 説明 5：区間の幅がこれを超える対は判定できない
SCEN = {"05": (("05",), (0, 3)), "02_06": (("02", "06"), (0, 2)), "03_07": (("03", "07"), (0, 1))}   # 説明 5 の同じ筋書きの中の対
RNG_SEED = 20261104
SPLIT_KEY = {"cluster": "塊の jackknife", "group": "50 群の delete-a-group jackknife"}
ABOVE, BELOW, SAME, UNDET = "上回る", "下回る", "同じ水準", "判定できない"


def ks_f() -> tuple:
    """説明と族 5 の K（5，25）．試走で K を絞ったときは DD.KS にあるものだけ．"""
    return tuple(K for K in (5, 25) if K in DD.KS)


def ks_m() -> tuple:
    return (0,) + ks_f()


def k_check2() -> int:
    """確かめ 2 の none[25]．試走で 25 が無いときだけ DD.KS の最大．"""
    return 25 if 25 in DD.KS else max(DD.KS)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


RUN = {"commit": "", "dirty": True, "encoder": 1}


PROD_OUT = {"dai3": ("dai1", 1), "dai3_enc2": ("dai1_enc2", 2)}           # 本番の置き場 → 読む実験 1 の置き場とエンコーダ（gate-code の I8）
SCRIPTS = ("dai3.py", "dai1_stats.py", "dai_solver.py", "dai1.py")        # detect と eval で sha256 を照合する台本（gate-code の記録 6）


def production(out: Path) -> bool:
    """本番の置き場（results/dai3 と，第 2 のエンコーダの results/dai3_enc2）なら真．どちらにも同じ守りをかける．"""
    return any(out.resolve() == (ROOT / "results" / n).resolve() for n in PROD_OUT)


def frozen_copy(out: Path) -> Path:
    """固定のファイルの写しの置き場（置き場ごとに名前を分ける：dai3_frozen.json，dai3_enc2_frozen.json）．"""
    return ROOT / "frozen" / f"{out.resolve().name}_frozen.json"


def script_shas() -> dict:
    return {n: sha(HERE / n) for n in SCRIPTS}


# ------------------------------------------------------------------ コーパスの表と部分集合

_DF: dict = {}


def corpus_df(c: str) -> pd.DataFrame:
    """コーパスの発話の表（塊，分割，対話，ラベル）．特徴を GPU に載せずに読む（ラベルの件数だけの表に使う）．"""
    if c in dai1.Corpus.cache:
        return dai1.Corpus.cache[c].df
    if c not in _DF:
        _DF[c] = DD.load_corpus(c, dai1.ENC["enc"])[0]
    return _DF[c]


def jk_mode(c: str) -> tuple[str, dict | None]:
    """区間の作り方（1.3 節）．実効の塊の数が 20 以上（MOSEI）なら塊を名前の順に 50 群に配る delete-a-group jackknife．"""
    g_eff, _ = ST.g_eff_of(c)
    if g_eff < 20:
        return "cluster", None
    names = sorted(corpus_df(c)["cluster"].astype(str).unique())
    return "group", {g: i % N_GROUPS_BIG for i, g in enumerate(names)}          # 名前の順に 0, 1, …, 49, 0, … と配る


_SUB: dict = {}


def subsets() -> pd.DataFrame:
    if "s" not in _SUB:
        _SUB["s"] = pd.read_csv(ROOT / "frozen" / "subsets.csv", keep_default_na=False)
    return _SUB["s"]


def subset_map(corpus: str, name: str) -> dict:
    s = subsets()
    s = s[(s.corpus == corpus) & (s.subset == name)]
    return dict(zip(s.sample_id.astype(str), s.value.astype(str)))


def mosei_channels() -> set:
    return set(pd.read_csv(ROOT / "data" / "metadata" / "mosei_channel_map.csv", dtype=str).channel_id)


def subset_masks(T: str, sid: np.ndarray, cl: np.ndarray) -> dict:
    """1.1 節の部分集合（族 5 の記述）と説明 4 の (b) のテストの印．成績は使わない．"""
    out = {}
    if T == "MOSEI":
        out["channel"] = np.isin(cl, list(mosei_channels()))
    if T == "IEMOCAP":
        mk = subset_map("IEMOCAP", "marker")
        v = np.array([mk.get(x, "") for x in sid])
        out["marker"], out["nomarker"] = v == "marker", v == "nomarker"
    if T == "AFEW":
        out["asr_ok"] = ~np.isin(sid, list(set(subset_map("AFEW", "asr_suspect"))))
    if T == "MELD":
        fc = subset_map("MELD", "face_compete")
        out["face_single"] = np.array([fc.get(x) == "single" for x in sid], bool)
    return out


def scenario(sid: np.ndarray) -> np.ndarray:
    """IEMOCAP 即興の筋書きの番号（2 桁．05a・05b は 05，08a・08b は 08）．"""
    return np.array([(re.search(r"impro(\d\d)", x) or re.search(r"(\d\d)", "xx")).group(1) for x in sid])


# ------------------------------------------------------------------ 2 組の分け方と複製

def test_halves(y: np.ndarray, cl: np.ndarray, sid: np.ndarray) -> np.ndarray:
    """テストの 2 組（1 節．D3，A1，M23，D26）：感情ごとに，発話を (塊の名前，sample_id) の順に 1 本の列に並べ，
    層をまたいで続けて A，B，A，B… と交互に配る（層ごとに A から数え直さない）．戻り：組の番号（0 = A，1 = B）．"""
    grp = np.zeros(len(y), int)
    for k in range(4):
        ix = np.flatnonzero(y == k)
        order = ix[np.lexsort((sid[ix].astype(str), cl[ix].astype(str)))]
        grp[order] = np.arange(len(order)) % 2
    return grp


def pool_halves(cl: np.ndarray) -> np.ndarray:
    """候補の 2 組（2 節）：塊を大きい順に，そのときの本数の少ない組に配る．同じ大きさは名前の順．ラベルは使わない．"""
    n = Counter(cl.tolist())
    tot, h = [0, 0], {}
    for g in sorted(n, key=lambda g: (-n[g], g)):
        k = 0 if tot[0] <= tot[1] else 1
        h[g] = k
        tot[k] += n[g]
    return np.array([h[g] for g in cl.tolist()], int)


def disjoint_halves(T: str, cl: np.ndarray, sid: np.ndarray) -> np.ndarray:
    """話者が重ならない 2 組（gate-code の R4．全体の複製だけで使う）：AFEW・MOSEI・MELD はテストの塊を候補の 2 組と同じ配り方
    （大きい順に，そのときの本数の少ない組へ）で分ける．IEMOCAP はテストが 1 つの塊（セッション）なので話者（F と M）で分ける．
    ラベルは使わない．"""
    if T == "IEMOCAP":
        return np.array([0 if x.split("_")[-1][:1] == "F" else 1 for x in sid.tolist()], int)
    return pool_halves(cl)


def groups_ok(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """1 節の 10 本の規則と 4 節の 30 本の規則：どちらの組でも，版の感情がそれぞれ 10 本以上で，合計が 30 本以上．
    c0・c1：2 組の感情ごとの件数（(…, k)）．偽の (単位，複製) は計算せず，その量の平均から外す（1.2 節と同じ扱い）．"""
    return ((c0 >= MIN_GROUP).all(-1) & (c1 >= MIN_GROUP).all(-1) & (c0.sum(-1) >= MIN_TOTAL) & (c1.sum(-1) >= MIN_TOTAL))


def rep_keep(cl: np.ndarray, mode: str, gmap: dict | None) -> tuple[list, np.ndarray]:
    """1.3 節の複製：最初は全体（"full"），続けて塊を 1 つ（MOSEI は 50 群の 1 つ）除いた残り．
    戻り：鍵の並びと残す印 (J, n)．この単位のテストに現れない塊・群は並べない（除いても値が変わらない）．"""
    keys, rows = ["full"], [np.ones(len(cl), bool)]
    if mode == "group":
        gid = np.array([gmap[c] for c in cl.tolist()])
        for j in sorted(set(gid.tolist())):
            keys.append(int(j)); rows.append(gid != j)
    else:
        for g in sorted(set(cl.tolist())):
            keys.append(str(g)); rows.append(cl != g)
    return keys, np.stack(rows)


# ------------------------------------------------------------------ 切片の当て直し

def local_labels(y: np.ndarray, cls) -> np.ndarray:
    yl = np.full(len(y), -1)
    for q, c in enumerate(cls):
        yl[y == c] = q
    return yl


def onehot(yl: np.ndarray, k: int) -> np.ndarray:
    Y = np.zeros((len(yl), k))
    ok = yl >= 0
    Y[np.flatnonzero(ok), yl[ok]] = 1.0
    return Y


def rec_of(hit: np.ndarray, mask: np.ndarray, Yk: np.ndarray) -> np.ndarray:
    """hit (P, n)，mask (P, n) → 印の中の感情ごとの再現率（%．(P, k)）．印に現れない感情は NaN．"""
    m = mask.astype(float)
    tot = m @ Yk
    cor = (m * hit) @ Yk
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 * np.where(tot > 0, cor / np.maximum(tot, 1), np.nan)


def nanmean_rows(r: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():                                    # 空の複製（IEMOCAP の塊を除いた残り）の警告だけを抑える
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(r, -1)


def uar_of(hit: np.ndarray, mask: np.ndarray, Yk: np.ndarray) -> np.ndarray:
    """hit (P, n)，mask (P, n) → 印の中の UAR（%）．印に現れる感情だけの再現率の平均（`dai_solver.uar` と同じ定義）．"""
    return nanmean_rows(rec_of(hit, mask, Yk))


def scores(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """切片を除いた点数 X Wᵀ（(n, 4)，float64）．"""
    return S.t(X) @ S.t(W).T


def solve_b(Zw: torch.Tensor, b: torch.Tensor, probs: list, ctx: str = "") -> tuple[list, dict]:
    """probs：[(版の感情 cls，学ぶ発話の印 (n,)，版の中のラベル yl (n,))]．W を固定し，切片だけを感情の釣り合いの重みつきの
    交差エントロピーで解く（`dai_solver.fit` の fix_W=True，C_b = ∞）．点数 Zw = X Wᵀ を特徴に，版の感情の列を選ぶ行列を W0 に
    渡すので，点数は X Wᵀ を毎回掛けたのと同じである．感情の数が同じ問をまとめて解く．戻り：b*（問ごとに (k,)）と収束の記録．
    解か勾配に NaN・Inf があれば，ctx（単位と識別器の名前）と版を書いて止まる（I11）．"""
    out = [None] * len(probs)
    conv = dict(n_problems=len(probs), grad_max=0.0, n_over_tol=0, stalled=0, finite=True)
    n = Zw.shape[0]
    for k in sorted({len(p[0]) for p in probs}):
        ix = [i for i, p in enumerate(probs) if len(p[0]) == k]
        G = torch.zeros(len(ix), n, k, device=S.DEV, dtype=S.DT)
        W0 = torch.zeros(len(ix), k, 4, device=S.DEV, dtype=S.DT)
        b0 = torch.zeros(len(ix), k, device=S.DEV, dtype=S.DT)
        for q, i in enumerate(ix):
            cls, m, yl = probs[i]
            sel = np.flatnonzero(m & (yl >= 0))
            ys = yl[sel]
            cnt = np.bincount(ys, minlength=k).astype(float)
            w = len(sel) / (k * cnt[ys])                                 # 感情の釣り合いの重み（合計は本数）
            G[q, torch.as_tensor(sel, device=S.DEV), torch.as_tensor(ys, device=S.DEV)] = S.t(w)
            W0[q, torch.arange(k, device=S.DEV), torch.as_tensor(list(cls), device=S.DEV)] = 1.0
            b0[q] = b[list(cls)]
        r = S.fit(Zw, G, W0, b0, Cw=math.inf, Cb=math.inf, fix_W=True)
        fin = r["finite_p"] & torch.isfinite(r["b"]).all(1).cpu().numpy()
        if not fin.all():
            bad = sorted({VNAME[VERSIONS.index(tuple(probs[ix[q]][0]))] for q in np.flatnonzero(~fin)})
            sys.exit(f"{ctx}：切片の当て直しの解か勾配に NaN・Inf がある（版 {bad}，{int((~fin).sum())} 問．I11）")
        for q, i in enumerate(ix):
            out[i] = r["b"][q].cpu().numpy()
        conv["grad_max"] = max(conv["grad_max"], float(r["grad_max"]))
        conv["n_over_tol"] += int((r["grad_p"] > 1e-5).sum())
        conv["stalled"] += int(r["stalled"])
        conv["finite"] &= bool(r["finite_p"].all())
    return out, conv


def refit_item(Zw: torch.Tensor, b: torch.Tensor, y: np.ndarray, grp: np.ndarray, keep: np.ndarray, refit: bool = True,
               ctx: str = "") -> tuple[dict, dict]:
    """1 つの識別器（点数 Zw = X Wᵀ，切片 b）の，版（4 感情と感情の対 6 つ）× 複製ごとの 2 組の上の UAR（1 節）．
      orig[v, j]  ：UAR(W, b) を組 A と組 B で測った平均
      refit[v, j] ：組 A で b* を解いて組 B で測り，入れ替えて平均した UAR(W, b*(W))（交差の当て直し）
      fullb[v, j] ：除いた後の全部の発話で解いた b* を 2 組で測った平均（refit − fullb が差の無いときの位置の誤りの水準）
      valid[v, j] ：どちらの組でも版の感情が 10 本以上で合計が 30 本以上（`groups_ok`．偽なら計算しない）
      rec[量][v]  ：orig・refit・fullb の感情ごとの再現率（(J, k)．2 組の平均．float32．gate-code の R2）
      bstar[v]    ：解いた b*（(J, 3, k)．組 A で解いた値，組 B で解いた値，全体で解いた値．float32．計算しない複製は NaN．R2）
    ほかに full[v]（全体のテストの UAR(W, b)），tot[v]・minc[v]（全体の複製での 2 組の本数と感情ごとの最少）．"""
    Zn, bn = Zw.cpu().numpy(), b.cpu().numpy()
    J, V = keep.shape[0], len(VERSIONS)
    R = {k: np.full((V, J), np.nan) for k in ("orig", "refit", "fullb")}
    R.update(valid=np.zeros((V, J), bool), full=np.full(V, np.nan), tot=np.zeros((V, 2), int), minc=np.zeros((V, 2), int),
             rec={k: [] for k in ("orig", "refit", "fullb")}, bstar=[])
    probs, where, cache = [], [], {}
    for vi, cls in enumerate(VERSIONS):
        k = len(cls)
        yl = local_labels(y, cls)
        inV = yl >= 0
        Yk = onehot(yl, k)
        A = [keep & (inV & (grp == g))[None, :] for g in (0, 1)]
        cnt = [a.astype(float) @ Yk for a in A]
        valid = groups_ok(cnt[0], cnt[1])
        R["valid"][vi] = valid
        R["tot"][vi] = [int(c[0].sum()) for c in cnt]
        R["minc"][vi] = [int(c[0].min()) for c in cnt]
        hit = np.argmax(Zn[:, list(cls)] + bn[list(cls)], 1) == yl
        R["full"][vi] = uar_of(hit[None], inV[None], Yk)[0]
        ro = 0.5 * (rec_of(np.broadcast_to(hit, A[0].shape), A[0], Yk) + rec_of(np.broadcast_to(hit, A[1].shape), A[1], Yk))
        ro[~valid] = np.nan
        R["orig"][vi] = nanmean_rows(ro)
        R["rec"]["orig"].append(ro.astype(np.float32))
        R["rec"]["refit"].append(np.full((J, k), np.nan, np.float32))
        R["rec"]["fullb"].append(np.full((J, k), np.nan, np.float32))
        R["bstar"].append(np.full((J, 3, k), np.nan, np.float32))
        cache[vi] = (cls, yl, Yk, A)
        if refit:
            for j in np.flatnonzero(valid):
                for tag, m in (("a", A[0][j]), ("b", A[1][j]), ("f", A[0][j] | A[1][j])):
                    probs.append((cls, m, yl)); where.append((vi, j, tag))
    conv = dict(n_problems=0, grad_max=0.0, n_over_tol=0, stalled=0, finite=True)
    if refit and probs:
        bs, conv = solve_b(Zw, b, probs, ctx)
        acc = {}
        for (vi, j, tag), bb in zip(where, bs):
            cls, yl, Yk, A = cache[vi]
            hit = (np.argmax(Zn[:, list(cls)] + bb, 1) == yl)[None]
            acc[(vi, j, tag)] = (rec_of(hit, A[0][j][None], Yk)[0], rec_of(hit, A[1][j][None], Yk)[0])
            R["bstar"][vi][j, "abf".index(tag)] = bb
        for vi in range(V):
            for j in np.flatnonzero(R["valid"][vi]):
                rr = 0.5 * (acc[(vi, j, "a")][1] + acc[(vi, j, "b")][0])           # A で解いて B で測る，入れ替える
                rf = 0.5 * (acc[(vi, j, "f")][0] + acc[(vi, j, "f")][1])
                R["rec"]["refit"][vi][j], R["rec"]["fullb"][vi][j] = rr, rf
                R["refit"][vi, j], R["fullb"][vi, j] = float(np.mean(rr)), float(np.mean(rf))
    return R, conv


# ------------------------------------------------------------------ 区間（1.3 節）と判定

def g_eff_from(n: Counter) -> float:
    a = np.array(list(n.values()), float)
    return float(a.sum() ** 2 / (a ** 2).sum()) if len(a) else float("nan")


NAN_STATS = dict(effect=np.nan, lo=np.nan, hi=np.nan, se=np.nan, df=np.nan, p=np.nan, identical=False, v_jk=np.nan,
                 v_seed=np.nan, v_seed_all=np.nan, g_eff=np.nan, n_reps=0, rep_fold_drops=0, folds_D="")


def jk_stats(vals: dict, mode: str, sizes: dict) -> dict:
    """1.3 節の区間．vals：{(f, r): {"full": x, 複製の鍵: x_(−g)}}（NaN は 10 本の規則で計算しなかった値）．
    sizes：{f: Counter(塊 → 発話の数)}（実効の塊の数を数える発話．部分集合ではその部分集合の発話）．
      mode="cluster"（AFEW，IEMOCAP，MELD）：塊を 1 つ除いた θ_(−g)（除いた塊の発話を両方の組から抜き，b* を解き直した値）の
        jackknife と種の項（実験 1 の 9.3 節）．自由度は残った分割で数えた実効の塊の数 G から max(1, ⌊G⌋ − 1)（1.2 節）．
      mode="group"（MOSEI）：塊を名前の順に 50 群に配った delete-a-group jackknife と種の項．自由度は 49．
    θ_(−g) の平均から外れた分割（除くと 10 本の規則で外れる）の数を返し，その分割の種の項は足さない（二重に数えない）．
    全体で 10 本の規則に当たって落ちた分割は，θ を残りの分割の平均にし，落ちた分割を返す（1.2 節）．"""
    folds = sorted({f for f, _ in vals})
    seeds = {f: sorted(r for ff, r in vals if ff == f and np.isfinite(vals[(ff, r)]["full"])) for f in folds}
    kept = [f for f in folds if seeds[f]]
    base = dict(n_units=int(sum(len(v) for v in seeds.values())), n_folds=len(kept),
                folds_dropped=",".join(str(f) for f in folds if not seeds[f]), method=SPLIT_KEY[mode])
    if not kept:
        return base | NAN_STATS
    full = {f: [vals[(f, r)]["full"] for r in seeds[f]] for f in kept}
    est = float(np.mean([np.mean(full[f]) for f in kept]))
    F = len(kept)
    s2 = {f: (float(np.var(full[f], ddof=1)) if len(full[f]) > 1 else 0.0) for f in kept}
    v_seed_all = float(sum(s2[f] / len(full[f]) for f in kept) / F ** 2)
    reps = list(range(N_GROUPS_BIG)) if mode == "group" else sorted({g for f in kept for g in sizes[f]})
    jk, D, n_out = [], set(), 0
    for g in reps:
        fv = []
        for f in kept:
            xs = [vals[(f, r)].get(g, vals[(f, r)]["full"]) for r in seeds[f]]
            xs = [x for x in xs if np.isfinite(x)]
            if xs:
                fv.append(float(np.mean(xs)))
            else:
                D.add(f); n_out += 1                                    # この塊を除くと分割 f が外れる
        if fv:
            jk.append(float(np.mean(fv)))
    jk = np.array(jk)
    G = len(jk)
    v_jk = float((G - 1) / G * ((jk - jk.mean()) ** 2).sum()) if G > 1 else np.nan
    v_seed = float(sum(s2[f] / len(full[f]) for f in kept if f not in D) / F ** 2)
    n_all = Counter()
    for f in kept:
        n_all.update(sizes[f])
    g_eff = g_eff_from(n_all)
    df = N_GROUPS_BIG - 1 if mode == "group" else max(1, int(np.floor(g_eff)) - 1)
    se = float(np.sqrt(v_jk + v_seed)) if np.isfinite(v_jk) else np.nan
    p, ident = ST.pval(est, se, df) if np.isfinite(se) else (np.nan, False)
    q = float(st.t.ppf(0.975, df))
    return base | dict(effect=est, lo=est - q * se, hi=est + q * se, se=se, df=df, p=p, identical=ident, v_jk=v_jk,
                       v_seed=v_seed, v_seed_all=v_seed_all, g_eff=g_eff, n_reps=G, rep_fold_drops=n_out,
                       folds_D=",".join(map(str, sorted(D))))


def verdict(lo, hi, delta=None) -> str:
    """「上回る」は下端 > 0，「下回る」は上端 < 0，「同じ水準」は区間が ±δ に収まる（δ を渡したときだけ），どれでもなければ判定できない．"""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return UNDET
    if lo > 0:
        return ABOVE
    if hi < 0:
        return BELOW
    if delta is not None and np.isfinite(delta) and -delta <= lo and hi <= delta:
        return SAME
    return UNDET


def fold_values(vals: dict) -> dict:
    """{(f, r): {"full": x}} → 分割ごとの種の平均（記述の「分割ごとの値の範囲」に使う）．"""
    out = {}
    for (f, r), v in vals.items():
        if np.isfinite(v["full"]):
            out.setdefault(f, []).append(v["full"])
    return {f: float(np.mean(v)) for f, v in out.items()}


# ------------------------------------------------------------------ 実験 1 の保存の読み込み

def tt_names(s: str, srcs: list) -> list:
    """実験 1 の段 3 の予測を付ける側の種類（`dai1.pass3` の tts と同じ並び）．"""
    n = [f"single_{m}" for m in "AVT" if m != s] + [f"pair_{k}" for k in ("AV", "AT", "VT")]
    n += ["fused_AVT", "poe_none", "noret_sel"] + [f"ret_{m}" for m in "AVT"] + ["start_AVT"]
    for kind in ["AVT"] + [m for m in "AVT" if m != s]:
        n += [f"src_{c}_{kind}" for c in srcs]
    return n


def repro_types(s: str, srcs: list) -> list:
    """2 節の除外の後の種類：生徒自身の特徴の線形の判定になる予測（単一の生徒では ret_<s>，AVT の生徒では積を除くすべて）を除く．"""
    names = tt_names(s, srcs)
    if s == "AVT":
        return [x for x in names if x in ("poe_none", "noret_sel")]
    return [x for x in names if x != f"ret_{s}"]


def noown_types(s: str, srcs: list) -> list:
    """生徒のモダリティを含まない予測の種類（実験 1 の設計 7 節の表の右列）．AVT の生徒には無い．"""
    if s == "AVT":
        return []
    out = [f"single_{m}" for m in "AVT" if m != s]
    out += [f"pair_{k}" for k in ("AV", "AT", "VT") if s not in k]
    out += [f"src_{c}_{m}" for m in "AVT" if m != s for c in srcs]
    return out


def tether_of(fz: dict) -> dict:
    return {(k.split("|")[0], int(k.split("|")[1])): tuple(v) for k, v in fz["tether"].items()}


class Saved:
    """1 単位（T, c, 分割, 種）の実験 1 の保存（段 1〜3）と，作り直した標準化の特徴と出発点（`dai1.Unit`）．"""

    def __init__(self, out1: Path, tag: str, T: str, c: str, f: int, r: int, fz1T: dict):
        self.tag, self.T, self.c, self.f, self.r, self.fz = tag, T, c, f, r, fz1T
        self.srcs = DD.others(T) if tag == "eval" else [x for x in DD.others(T) if x != c]
        cname = "ALL" if tag == "eval" else c
        self.base = out1 / tag / "units" / f"{T}_{cname}_f{f}_r{r}"
        self.p1, self.p2, self.p3 = (np.load(f"{self.base}_p{w}.npz") for w in (1, 2, 3))
        self.u = dai1.Unit(out1, T, c, self.srcs, f, r)
        df = self.u.cp.df
        self.sid = df["sample_id"].to_numpy().astype(str)[self.u.test]
        if not (self.p1["sample_test"].astype(str) == self.sid).all():
            sys.exit(f"{self.base.name}：保存のテストの sample_id が作り直した分割と違う")
        if not (self.p1["sample_pool"].astype(str) == df["sample_id"].to_numpy().astype(str)[self.u.pool]).all():
            sys.exit(f"{self.base.name}：保存の候補の sample_id が作り直した分割と違う")
        self.y = self.u.ytest
        self.cl = df["cluster"].to_numpy().astype(str)[self.u.test]
        self.cl_pool = df["cluster"].to_numpy().astype(str)[self.u.pool]
        self.dlg_pool = df["dialogue"].to_numpy().astype(str)[self.u.pool]
        self.sid_pool = df["sample_id"].to_numpy().astype(str)[self.u.pool]
        self.teth = tether_of(fz1T)
        self.wsel = {int(k): tuple(v) for k, v in fz1T["w"].items()}
        tag_t = [f"{k[0]}|{k[1]}|{v[0]}|{v[1]}" for k, v in sorted(self.teth.items())]
        tag_w = [f"{K}|{w}" for K, w in sorted(self.wsel.items())]
        if list(self.p3["tether"]) != tag_t or list(self.p3["wsel"]) != tag_w:
            sys.exit(f"{self.base.name}：保存の引き戻しか指数が実験 1 の固定のファイルと違う")
        self.commits = {str(z["commit"]) for z in (self.p1, self.p2, self.p3)}
        self.mode, self.gmap = jk_mode(c)
        self._items = {}

    def items(self, s: str, K: int) -> list:
        key = (s, K)
        if key not in self._items:
            self._items[key] = [json.loads(x) for x in self.p3[f"p3_{s}_K{K}_items"]]
        return self._items[key]

    def p3_index(self, s: str, K: int, **want) -> int:
        hits = [j for j, it in enumerate(self.items(s, K)) if all(ST.norm(it.get(k)) == ST.norm(v) for k, v in want.items())]
        if len(hits) != 1:
            sys.exit(f"{self.base.name}：段 3 の問 {s} K={K} {want} が {len(hits)} 個")
        return hits[0]

    def p3_wb(self, s: str, K: int, j: int):
        return S.t(self.p3[f"p3_{s}_K{K}_W"][j]), S.t(self.p3[f"p3_{s}_K{K}_b"][j])

    def none(self, s: str, K: int):
        return dai1.none_wb(self.u, self.p1, s, K, self.teth)

    def prop(self, s: str, K: int, fz: dict | None = None):
        """提案（同時に解く版．主の版．選んだ設定）の W・b．fz を渡すと，その選択（c を 1 つ除いて選び直した設定）で読む．"""
        fz = fz or self.fz
        v = fz["main"][str(K)]
        j = fz["select"][f"{s}|{K}|joint|{v}"]
        return self.p3_wb(s, K, self.p3_index(s, K, method="joint", teacher=v, rho=j["rho"], a=j["a"]))

    def partner(self, s: str, K: int):
        """族 3 の相手（固定の手法と版．選んだ設定）の W・b．AVT の生徒の共正則化は 3 つのモダリティの W をつなぎ，切片を足す．"""
        part = self.fz["partner"][f"{s}|{K}"]
        pm, pv = part["method"], part["version"]
        pp = {k: x for k, x in self.fz["select"][f"{s}|{K}|{pm}|{pv}"].items() if k != "pseudo_uar"}
        if pm == "cotrain" and s != "AVT":
            j = list(dai1.RHO_GRID).index(float(pp["rho"]))
            return S.t(self.p3[f"p3c_K{K}_{s}_W"][j]), S.t(self.p3[f"p3c_K{K}_{s}_b"][j])
        if pm == "coreg":
            j = list(dai1.COREG_RHO).index(float(pp["rho"]))
            if s != "AVT":
                return S.t(self.p3[f"p3g_K{K}_{s}_W"][j]), S.t(self.p3[f"p3g_K{K}_{s}_b"][j])
            W = np.concatenate([self.p3[f"p3g_K{K}_{m}_W"][j] for m in "AVT"], axis=1)   # AVT の特徴は A，V，T の順につないだもの
            b = sum(self.p3[f"p3g_K{K}_{m}_b"][j].astype(np.float64) for m in "AVT")
            return S.t(W), S.t(b)
        return self.p3_wb(s, K, self.p3_index(s, K, method=pm, teacher=pv, **pp))

    def tt(self, s: str, K: int, name: str):
        return self.p3_wb(s, K, self.p3_index(s, K, method="tt", teacher=name))

    def oracle(self, kind: str, cw: float, cb: float):
        items = [str(x) for x in self.p1[f"p1_{kind}_items"]]
        j = items.index(f"all|{cw}|{cb}")
        return S.t(self.p1[f"p1_{kind}_W"][j]), S.t(self.p1[f"p1_{kind}_b"][j])

    def zero_b(self):
        return torch.zeros(4, device=S.DEV, dtype=S.DT)


def load_fz1(out1: Path) -> tuple[dict, str]:
    p = out1 / "select" / "frozen.json"
    if not p.exists():
        sys.exit(f"実験 1 の固定のファイル {p} が無い")
    return json.load(open(p)), sha(p)


def pseudo_units(out1: Path, T: str) -> list:
    tw = json.load(open(out1 / "pseudo" / f"{T}_tether_w.json"))
    out = []
    for name in tw["units"]:
        c, f, r = re.fullmatch(r"(\w+?)_f(\d+)_r(\d+)", name).groups()
        out.append((c, int(f), int(r)))
    return out


def eval_units_present(out1: Path) -> bool:
    d = out1 / "eval" / "units"
    return d.exists() and any(d.iterdir())


# ------------------------------------------------------------------ 段 validate（1.4 節）

def coord_ascent(Z: np.ndarray, yl: np.ndarray, Yk: np.ndarray, b_star: np.ndarray) -> np.ndarray:
    """確かめ 1 の UAR を直接最大にする切片：b* から始め，切片の差（感情 0 を基準に 1..k−1）を 1 座標ずつ，b* のまわり ±3・
    刻み 0.1 の格子で UAR が最大の点に動かす座標の上昇を 3 巡する．同じ UAR の点が複数あるときは今の値に最も近い点
    （動かないことを優先する．同じ距離なら小さい値）．Z：学ぶ組の点数（版の列だけ，(n, k)）．"""
    k = Z.shape[1]
    b = b_star.astype(float).copy()
    ones = np.ones((len(GRID), 1), bool)
    for _ in range(SWEEPS):
        for j in range(1, k):
            cand = b_star[j] + GRID
            Zb = Z + b
            oth = [q for q in range(k) if q != j]
            om = Zb[:, oth].max(1)
            oarg = np.array(oth)[Zb[:, oth].argmax(1)]
            Sj = Z[:, j][None, :] + cand[:, None]
            isj = (Sj > om[None, :]) | ((Sj == om[None, :]) & (j < oarg[None, :]))   # argmax の同点は小さい番号（numpy と同じ）
            pred = np.where(isj, j, oarg[None, :])
            u = uar_of(pred == yl[None, :], ones & (yl >= 0)[None, :], Yk)
            best = np.nanmax(u)
            tie = np.flatnonzero(u >= best - 1e-9)
            pick = min(tie, key=lambda i: (abs(cand[i] - b[j]), cand[i]))
            b[j] = cand[pick]
    return b


def check1_item(Zw: torch.Tensor, b: torch.Tensor, y: np.ndarray, grp: np.ndarray, ctx: str = "") -> list:
    """確かめ 1（1.4 節の 1）：版ごとに，組 A で b*（交差エントロピー）と UAR を直接最大にする切片を決め，組 B で 2 つの UAR を測り，
    入れ替えて平均する．差 = 最適の切片の UAR − b* の UAR．"""
    Zn = Zw.cpu().numpy()
    probs, meta = [], []
    for vi, cls in enumerate(VERSIONS):
        yl = local_labels(y, cls)
        A = [(grp == g) & (yl >= 0) for g in (0, 1)]
        cnt = [np.bincount(yl[a], minlength=len(cls)) for a in A]
        if not groups_ok(cnt[0], cnt[1]):
            meta.append((vi, None)); continue
        for g in (0, 1):
            probs.append((cls, A[g], yl))
        meta.append((vi, (yl, A)))
    bs, conv = solve_b(Zw, b, probs, ctx) if probs else ([], {})
    rows, q = [], 0
    for vi, m in meta:
        if m is None:
            rows.append(dict(version=VNAME[vi], valid=False)); continue
        yl, A = m
        cls = VERSIONS[vi]
        Yk = onehot(yl, len(cls))
        us, uo = [], []
        for g in (0, 1):
            bstar = bs[q]; q += 1
            bopt = coord_ascent(Zn[A[g]][:, list(cls)], yl[A[g]], Yk[A[g]], bstar)
            B = A[1 - g]
            for bb, lst in ((bstar, us), (bopt, uo)):
                hit = np.argmax(Zn[:, list(cls)] + bb, 1) == yl
                lst.append(uar_of(hit[None], B[None], Yk)[0])
        rows.append(dict(version=VNAME[vi], valid=True, uar_bstar=float(np.mean(us)), uar_opt=float(np.mean(uo)),
                         diff=float(np.mean(uo) - np.mean(us)), grad_max=conv.get("grad_max", np.nan)))
    return rows


def rotate_rows(W: np.ndarray, deg: float, rng) -> np.ndarray:
    """確かめ 2：W の各行を，乱数で選んだその行に直交する向きへ deg 度だけ回す（行の長さを保つので行列全体の Frobenius ノルムもそろう）．"""
    th = np.deg2rad(deg)
    out = np.empty_like(W)
    for k in range(W.shape[0]):
        w = W[k]
        g = rng.standard_normal(len(w))
        g -= (g @ w) / (w @ w) * w
        g /= np.linalg.norm(g)
        out[k] = np.cos(th) * w + np.sin(th) * np.linalg.norm(w) * g
    out *= np.linalg.norm(W) / np.linalg.norm(out)
    return out


def stage_validate(args, out: Path, out1: Path) -> None:
    if production(out) and eval_units_present(out1):
        sys.exit("評価の回の出力がすでにある．妥当性の確かめは評価の回の前にだけ流す（1.4 節）")
    fz1, fz1_sha = load_fz1(out1)
    vdir = out / "validate"
    vdir.mkdir(parents=True, exist_ok=True)
    rows1, rows2, conv_rows = [], [], []
    t0 = time.time()
    for T in args.targets:
        units = pseudo_units(out1, T)
        orc = select_oracle(out1, T)["tether"]                          # 確かめ 1 の W* は detect と同じ選び方の点（gate-code の記録 2）
        for c in DD.others(T):
            for (cc, f, r) in [x for x in units if x[0] == c]:
                t1 = time.time()
                sv = Saved(out1, "pseudo", T, cc, f, r, fz1[T])
                grp = test_halves(sv.y, sv.cl, sv.sid)
                meta = dict(T=T, c=c, fold=f, seed=r)
                for si, s in enumerate(DD.STUDENTS):
                    Ws = {"start": sv.u.start[s]}
                    for K in ks_f():
                        Ws[f"none{K}"] = sv.none(s, K)
                        Ws[f"prop{K}"] = sv.prop(s, K)
                    Ws["oracle"] = sv.oracle(s, *orc[s])
                    for wt, (W, b) in Ws.items():
                        for row in check1_item(scores(sv.u.Xt[s], W), b, sv.y, grp, ctx=f"{T}_{c}_f{f}_r{r} {s} {wt}"):
                            rows1.append(meta | dict(student=s, wtype=wt) | row)
                    K2 = k_check2()
                    Wn, bn = sv.none(s, K2)
                    keep = np.ones((1, len(sv.y)), bool)
                    base, cv = refit_item(scores(sv.u.Xt[s], Wn), bn, sv.y, grp, keep, ctx=f"{T}_{c}_f{f}_r{r} {s} none{K2}")
                    conv_rows.append(meta | dict(student=s, what=f"none{K2}") | cv)
                    Wnp = Wn.cpu().numpy()
                    for ang in ANGLES:
                        rng = np.random.default_rng([RNG_SEED, DD.CORPUS_ID[T], DD.CORPUS_ID[c], f, r, si, int(ang)])
                        Wr = rotate_rows(Wnp, ang, rng)
                        rr, cv = refit_item(scores(sv.u.Xt[s], S.t(Wr)), bn, sv.y, grp, keep, ctx=f"{T}_{c}_f{f}_r{r} {s} rot{int(ang)}")
                        conv_rows.append(meta | dict(student=s, what=f"rot{int(ang)}") | cv)
                        rows2.append(meta | dict(student=s, K=K2, angle=ang, valid=bool(base["valid"][0, 0]),
                                                 uar_drop=float(base["orig"][0, 0] - rr["orig"][0, 0]),
                                                 dir_increase=float(base["refit"][0, 0] - rr["refit"][0, 0])))
                print(f"validate {T}_{c}_f{f}_r{r}: {time.time()-t1:.0f} 秒", flush=True)
            dai1.Corpus.cache.pop(c, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    d1, d2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
    summ = validity_summary(d1, d2)
    tag = dict(commit=RUN["commit"], dirty=RUN["dirty"], dai1_frozen_sha256=fz1_sha)
    d1.assign(**tag).to_csv(vdir / "check1_units.csv", index=False)
    d2.assign(**tag).to_csv(vdir / "check2_units.csv", index=False)
    pd.DataFrame(conv_rows).assign(**tag).to_csv(vdir / "convergence.csv", index=False)
    summ["_meta"] = dict(tag, time=time.strftime("%Y-%m-%d %H:%M:%S"), targets=list(args.targets), rule="docs/設計/実験3_設計.md 1.4 節")
    json.dump(summ, open(vdir / "summary.json", "w"), indent=1, ensure_ascii=False)
    nconv = int(pd.DataFrame(conv_rows)["n_over_tol"].sum()) if conv_rows else 0
    print(f"validate：確かめ 1 の行 {len(d1)}，確かめ 2 の行 {len(d2)}，勾配が 1e-5 を超えた切片の問 {nconv}，"
          f"{time.time()-t0:.0f} 秒．書いた：{vdir}", flush=True)


def validity_summary(d1: pd.DataFrame, d2: pd.DataFrame) -> dict:
    """1.4 節の合否．確かめ 1 は (コーパス c，生徒，W の種類，版（4 感情か対）) ごとに単位（対の版は単位 × 6 対）の差の中央値．
    確かめ 2 は (c，生徒) ごとに，下がり分が 1 ポイント以上の (単位，角度) をまとめた 1 つの比（角度ごとの比も記述で並べる）．"""
    out = {"check1": {}, "check2": {}}
    if len(d1):
        d = d1[d1.valid].copy()
        d["ver"] = np.where(d.version == "four", "four", "pair")
        for (c, s, wt, ver), g in d.groupby(["c", "student", "wtype", "ver"]):
            med = float(g["diff"].median())
            out["check1"][f"{c}|{s}|{wt}|{ver}"] = dict(median_diff=med, n=int(len(g)), passed=bool(med <= CHECK1_MAX),
                                                        per_pair_median={v: float(x["diff"].median()) for v, x in g.groupby("version")})
    if len(d2):
        for (c, s), g in d2[d2.valid].groupby(["c", "student"]):
            use = g[g.uar_drop >= CHECK2_MIN_DROP]
            ratio = float(use.dir_increase.sum() / use.uar_drop.sum()) if len(use) else np.nan
            per = {str(int(a)): (float(x.dir_increase.sum() / x.uar_drop.sum()) if len(x) else None)
                   for a, x in use.groupby("angle")}
            ok = None if len(use) < CHECK2_MIN_UNITS else bool(ratio >= CHECK2_RATIO)
            out["check2"][f"{c}|{s}"] = dict(ratio=ratio, n_usable=int(len(use)), n_all=int(len(g)), per_angle=per,
                                             verdict=UNDET if ok is None else ("合格" if ok else "不合格"))
    return out


# ------------------------------------------------------------------ 段 detect（1.1 節．実験 1 の detect と同じ手順）

def pred_rows(sv: Saved, up: ST.UnitPreds, s: str, K: int, startp, fz: dict | None = None) -> dict:
    """予測の判定（int8）から作る行：prop，none，rolesep，default，partner（実験 1 の族の行），kbias，tt の全種類．"""
    fz = fz or sv.fz
    P = ST.unit_preds(fz, up, s, K, startp)
    if K > 0:
        cb = float(fz["select"][f"{s}|{K}|kbias|-"]["Cb"])
        items = [str(x) for x in sv.p1[f"p1_{s}_kbias_items"]]
        P["kbias"] = sv.p1[f"p1_{s}_kbias_pred"][items.index(f"{K}|{cb}")]
    for name in tt_names(s, sv.srcs):
        P[f"tt:{name}"] = sv.p3[f"p3_{s}_K{K}_pred"][sv.p3_index(s, K, method="tt", teacher=name)]
    return P


def detect_unit(sv: Saved, alt: dict | None) -> dict:
    """仮のターゲットの 1 単位：族 5 と説明 1 の (iii)・(iv) の切片の当て直しと，説明 1 の (ii)・説明 2 の判定の数．
    alt：評価のターゲット T = この単位の T の効果の見込みに使う，c を 1 つ除いて選び直した設定（族 5 の提案だけ）．"""
    keys, keep = rep_keep(sv.cl, sv.mode, sv.gmap)
    grp = test_halves(sv.y, sv.cl, sv.sid)
    R = {"keys": keys, "sizes": Counter(sv.cl.tolist()), "refit": {}, "counts": {}, "conv": []}
    up = ST.UnitPreds(sv.base)
    for s in DD.STUDENTS:
        its = {"start": sv.u.start[s]}
        for K in ks_f():
            its[f"none|{K}"] = sv.none(s, K)
            its[f"prop|{K}"] = sv.prop(s, K)
        if alt is not None and 5 in DD.KS:
            its["prop_alt|5"] = sv.prop(s, 5, fz=alt)
        for nm, (W, b) in its.items():
            R["refit"][f"{s}|{nm}"], cv = refit_item(scores(sv.u.Xt[s], W), b, sv.y, grp, keep, ctx=f"{sv.base.name} {s}|{nm}")
            R["conv"].append(dict(item=f"{s}|{nm}") | cv)
        startp = up.none(s, 0, None)
        for K in ks_f():
            P = pred_rows(sv, up, s, K, startp)
            for row in ("none", "kbias", "default", "tt:poe_none", "tt:fused_AVT"):
                R["counts"][(s, K, row)] = ST.counts(sv.y, P[row], sv.cl)
    return R


def to_vals(U: dict, fn) -> dict:
    """U：{(f, r): 単位の結果}．fn(結果) → 複製の並びの値 (J,)（NaN は計算しない）．戻り：{(f, r): {鍵: 値}}．"""
    out = {}
    for fr, R in U.items():
        a = fn(R)
        if a is None:
            continue
        out[fr] = {k: float(x) for k, x in zip(R["keys"], a)}
    return out


def sizes_of(U: dict) -> dict:
    return {f: R["sizes"] for (f, _), R in U.items()}


def refit_diff(A: str, B: str, vi: int, what=("refit", "refit")):
    def fn(R):
        if A not in R["refit"] or B not in R["refit"]:
            return None
        a, b = R["refit"][A], R["refit"][B]
        return np.where(a["valid"][vi], a[what[0]][vi] - b[what[1]][vi], np.nan)
    return fn


def cells_of(U: dict, key: tuple) -> dict:
    return {fr: R["counts"][key] for fr, R in U.items() if key in R["counts"]}


DETECT_CELLS = ("5", "e1ii", "e1iii", "e1iv", "e2a", "e2b")


def detect_cell_stats(U: dict, mode: str, g_eff: float, rng) -> dict:
    """仮のターゲットの (T, c) の 10 単位で，族 5 と説明 1・2 の各セルの区間を作る（標準誤差の見込みに使う）．"""
    out = {}
    sz = sizes_of(U)
    for s in DD.STUDENTS:
        if 5 in DD.KS:
            for vi in range(1, len(VERSIONS)):
                out[f"5|{s}|{VNAME[vi]}"] = jk_stats(to_vals(U, refit_diff(f"{s}|prop|5", f"{s}|none|5", vi)), mode, sz)
        for K in ks_f():
            out[f"e1iii|{s}|{K}"] = jk_stats(to_vals(U, refit_diff(f"{s}|none|{K}", f"{s}|start", 0)), mode, sz)
            out[f"e1iv|{s}|{K}"] = jk_stats(to_vals(U, refit_diff(f"{s}|prop|{K}", f"{s}|none|{K}", 0)), mode, sz)
            for name, (a, b) in (("e1ii", ("kbias", "none")), ("e2a", ("default", "tt:poe_none")), ("e2b", ("tt:poe_none", "tt:fused_AVT"))):
                e = ST.effect_stats(cells_of(U, (s, K, a)), cells_of(U, (s, K, b)), g_eff, rng)
                out[f"{name}|{s}|{K}"] = e
    return out


def crit(X: str, kind: str, m: int) -> float:
    """両側 α/m の基準．切片を当て直す量は 1.3 節（MOSEI は自由度 49 の t），判定から作る量は実験 1 の 9.3 節（MOSEI は正規）．"""
    g_eff, _ = ST.g_eff_of(X)
    q = 1 - 0.05 / m / 2
    if g_eff >= 20:
        return float(st.t.ppf(q, N_GROUPS_BIG - 1)) if kind == "refit" else float(st.norm.ppf(q))
    return float(st.t.ppf(q, max(1, int(np.floor(g_eff)) - 1)))


def select_oracle(out1: Path, T: str) -> dict:
    """W* の引き戻しの強さ（1 節．m29，D17）：(T，種類) ごとに，T の仮のターゲットの回（c ≠ T）の W* のテストの UAR の平均が最大の点．
    同点は小さい C′，小さい C_b．T 自身のテストのラベルは使わない．"""
    import dai1_select as SEL
    p1 = SEL.load(out1, T, 1)
    d = p1[(p1.stage == 1) & (p1.method == "oracle")]
    out, conv = {}, {}
    for kind, g in d.groupby("kind"):
        m = g.groupby(["Cw", "Cb"])["uar"].mean().reset_index().sort_values(["uar", "Cw", "Cb"], ascending=[False, True, True])
        cw, cb = float(m.iloc[0].Cw), float(m.iloc[0].Cb)
        out[kind] = [cw, cb]
        h = g[np.isclose(g.Cw, cw) & np.isclose(g.Cb, cb)]
        conv[kind] = dict(n=int(len(h)), n_grad_over=int((h.grad > 1e-5).sum()), n_stalled=int(h.stalled.astype(str).eq("True").sum()),
                          at_edge=bool(cw in (min(x[0] for x in dai1.TETHER), max(x[0] for x in dai1.TETHER))))
    return {"tether": out, "convergence": conv}


def label_tables(X: str) -> tuple[list, list, list, list, list]:
    """評価の回の前にラベルの件数だけから作る表：テストの 2 組の感情ごとの件数，塊を除くと件数の規則（10 本と 30 本）で外れる回の数，
    候補の 2 組の本数（MELD は共有する対話と顔が競らない発話に限った本数），同じ筋書きの中の対の件数，
    件数の規則で外れる (コーパス，部分集合，版，分割)（全体のテストと 1.1 節の部分集合．gate-code の L4）．"""
    df = corpus_df(X)
    y = df["label_4"].to_numpy().astype(int)
    cl = df["cluster"].to_numpy().astype(str)
    sid = df["sample_id"].to_numpy().astype(str)
    gc, drops, ph, sc, cr = [], [], [], [], []
    mode, gmap = jk_mode(X)
    face = subset_map("MELD", "face_compete") if X == "MELD" else {}
    for f in range(DD.N_FOLDS):
        te = np.flatnonzero(df["fold"].to_numpy() == f)
        grp = test_halves(y[te], cl[te], sid[te])
        for k in range(4):
            for g in (0, 1):
                gc.append(dict(corpus=X, fold=f, emotion=dai3_emo(k), group="AB"[g], n=int(((y[te] == k) & (grp == g)).sum())))
        keys, keep = rep_keep(cl[te], mode, gmap)
        for vi, cls in enumerate(VERSIONS):
            yl = local_labels(y[te], cls)
            bad = 0
            for j in range(1, len(keys)):
                cnt = [np.bincount(yl[keep[j] & (grp == g) & (yl >= 0)], minlength=len(cls)) for g in (0, 1)]
                bad += int(not groups_ok(cnt[0], cnt[1]))
            cnt0 = [np.bincount(yl[(grp == g) & (yl >= 0)], minlength=len(cls)) for g in (0, 1)]
            drops.append(dict(corpus=X, fold=f, version=VNAME[vi], n_reps=len(keys) - 1, n_reps_dropping=bad,
                              full_valid=bool(groups_ok(cnt0[0], cnt0[1])),
                              min_group_total=int(min(c.sum() for c in cnt0))))
        subs = {"all": np.ones(len(te), bool)} | subset_masks(X, sid[te], cl[te])
        for sname, m in subs.items():
            ix = np.flatnonzero(m)
            g2 = test_halves(y[te][ix], cl[te][ix], sid[te][ix])
            for vi, cls in enumerate(VERSIONS):
                yl = local_labels(y[te][ix], cls)
                cnt0 = [np.bincount(yl[(g2 == g) & (yl >= 0)], minlength=len(cls)) for g in (0, 1)]
                if not groups_ok(cnt0[0], cnt0[1]):
                    cr.append(dict(corpus=X, subset=sname, version=VNAME[vi], fold=f, min_group_total=int(min(c.sum() for c in cnt0)),
                                   min_emotion=int(min(c.min() for c in cnt0))))
        pool = np.flatnonzero(DD.pool_mask(df, f))
        sets = [("all", pool)]
        if X == "MELD":
            sets.append(("face_single", pool[np.array([face.get(x) == "single" for x in sid[pool]], bool)]))
        for tag, idx in sets:
            h = pool_halves(cl[idx])
            row = dict(corpus=X, fold=f, subset=tag, n_half0=int((h == 0).sum()), n_half1=int((h == 1).sum()))
            if X == "MELD":
                dl = df["dialogue"].to_numpy().astype(str)[idx]
                shared = set(dl[h == 0]) & set(dl[h == 1])
                row |= dict(n_dialogues=len(set(dl)), n_shared_dialogues=len(shared),
                            n_nonshared_utt=int((~np.isin(dl, list(shared))).sum()))
            ph.append(row)
    if X == "IEMOCAP":
        scen = scenario(sid)
        for pn, (scs, cls) in SCEN.items():
            for f in range(DD.N_FOLDS):
                te = (df["fold"].to_numpy() == f) & np.isin(scen, scs)
                n = [int((te & (y == k)).sum()) for k in cls]
                sc.append(dict(pair=pn, fold=f, **{f"n_{dai3_emo(k)}": v for k, v in zip(cls, n)}, kept=bool(min(n) >= MIN_GROUP)))
    return gc, drops, ph, sc, cr


def dai3_emo(k: int) -> str:
    return ("Neutral", "Happy", "Sad", "Angry")[k]


def stage_detect(args, out: Path, out1: Path) -> None:
    import dai1_select as SEL
    if eval_units_present(out1):
        sys.exit("評価の回の出力がすでにある（族 5 の検出できる差は評価の回の前にだけ作る）")
    fz1, fz1_sha = load_fz1(out1)
    ddir = out / "detect"
    ddir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    se_acc, exp_acc, conv_rows, oracle = {}, {}, [], {}
    for T in args.targets:
        fzT = fz1[T]
        oracle[T] = select_oracle(out1, T)
        tw = json.load(open(out1 / "pseudo" / f"{T}_tether_w.json"))
        p1, p2, p3 = (SEL.load(out1, T, w) for w in (1, 2, 3))
        units = pseudo_units(out1, T)
        for c in DD.others(T):
            keep_rows = [d[d.c != c] for d in (p1, p2, p3)]
            r_ = SEL.select_T(tw, *keep_rows, n_units=len([u for u in tw["units"] if not u.startswith(c + "_")]))
            alt = {"select": r_["select"], "main": r_["main"], "partner": r_["partner"], "tether": tw["tether"], "w": tw["w"]}
            U = {}
            for (cc, f, r) in [x for x in units if x[0] == c]:
                t1 = time.time()
                sv = Saved(out1, "pseudo", T, cc, f, r, fzT)
                U[(f, r)] = detect_unit(sv, alt)
                conv_rows += [dict(T=T, c=c, fold=f, seed=r) | x for x in U[(f, r)]["conv"]]
                print(f"detect {T}_{c}_f{f}_r{r}: {time.time()-t1:.0f} 秒", flush=True)
            mode, _ = jk_mode(c)
            g_eff, _ = ST.g_eff_of(c)
            for key, e in detect_cell_stats(U, mode, g_eff, rng).items():
                se_acc.setdefault(c, {}).setdefault(key, []).append(dict(T=T) | e)
            if 5 in DD.KS:                                                   # 効果の見込み（評価のターゲット X = T．X のテストを使わない）
                for s in DD.STUDENTS:
                    for vi in range(1, len(VERSIONS)):
                        v = to_vals(U, refit_diff(f"{s}|prop_alt|5", f"{s}|none|5", vi))
                        d = [x["full"] for x in v.values() if np.isfinite(x["full"])]
                        exp_acc.setdefault(T, {}).setdefault(f"5|{s}|{VNAME[vi]}", []).append(float(np.mean(d)) if d else np.nan)
            dai1.Corpus.cache.pop(c, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    det, se_rows, expected = {}, [], {}
    for X, cells in se_acc.items():
        fX = fz1.get(X, {})
        if "n_seeds_eval" not in fX or "detect" not in fX:
            print(f"{X}：実験 1 の固定のファイルに種の数か δ が無い（実験 1 の detect の後に流す）", flush=True)
            continue
        R = int(fX["n_seeds_eval"])
        delta = float(fX["detect"]["delta_same"])
        out_cells = {}
        for key, lst in cells.items():
            fam = key.split("|")[0]
            kind = "refit" if fam in ("5", "e1iii", "e1iv") else "pred"
            v_cl = [max(0.0, x["v_jk"] - (x["v_seed_all"] - x["v_seed"])) for x in lst if np.isfinite(x["v_jk"])]
            v_sd = [x["v_seed_all"] for x in lst if np.isfinite(x["v_seed_all"])]
            vjk = float(np.mean(v_cl)) if v_cl else np.nan
            vseed = float(np.mean(v_sd)) * 2 / R if v_sd else np.nan          # 仮のターゲットは種 2 つ
            se = float(np.sqrt(vjk + vseed))
            c95 = crit(X, kind, 1)
            cell = dict(se=se, v_cl_or_jk=vjk, v_seed=vseed, halfwidth=c95 * se, same_level_possible=bool(c95 * se < delta),
                        n_T=len(lst))
            if fam == "5":
                ex = exp_acc.get(X, {}).get(key)
                ee = float(np.nanmean(ex)) if ex is not None and np.isfinite(ex).any() else np.nan
                cell |= dict(detectable_first=crit(X, kind, 24) * se, detectable_last=crit(X, kind, 1) * se)
                cell["descriptive"] = bool(not np.isfinite(ee) or cell["detectable_first"] > abs(ee))
                expected.setdefault(X, {})[key] = ee                           # 効果の見込みは固定のファイルに入れない（gate-code の記録 3）
            out_cells[key] = cell
            se_rows += [dict(X=X, cell=key) | {k: x.get(k) for k in ("T", "v_jk", "v_seed", "v_seed_all", "n_units", "n_folds",
                                                                     "folds_dropped", "rep_fold_drops", "df")} for x in lst]
        det[X] = dict(n_seeds_eval=R, delta_same=delta, g_eff=ST.g_eff_of(X)[0], crit_first_family5=crit(X, "refit", 24),
                      cells=out_cells,
                      descriptive_cells=[k for k, v in out_cells.items() if v.get("descriptive")],
                      same_level_cells=[k for k, v in out_cells.items() if v["same_level_possible"]])
        print(f"{X}：族 5 のセル {sum(k.startswith('5|') for k in out_cells)}，記述に回すセル {len(det[X]['descriptive_cells'])}，"
              f"同じ水準を出しうるセル {len(det[X]['same_level_cells'])}", flush=True)
    for X, d in exp_acc.items():                                         # 標準誤差の見込みが無い X（試走だけ）の効果の見込み
        if X not in det:
            for k, v in d.items():
                expected.setdefault(X, {})[k] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    tabs = {"group_counts": [], "rep_drops": [], "pool_halves": [], "scenario_counts": [], "count_rule": []}
    for X in DD.CORPORA:
        for name, rows in zip(tabs, label_tables(X)):
            tabs[name] += rows
    tag = dict(commit=RUN["commit"], dirty=RUN["dirty"])
    for name, rows in tabs.items():
        pd.DataFrame(rows).assign(**tag).to_csv(ddir / f"{name}.csv", index=False)
    pd.DataFrame(se_rows).assign(**tag).to_csv(ddir / "se_rows.csv", index=False)
    pd.DataFrame(conv_rows).assign(**tag).to_csv(ddir / "convergence.csv", index=False)
    vpath = out / "validate" / "summary.json"
    ep = out / "detect_expected.json"                                   # 台本だけが読む．人は評価の回の前に開かない
    json.dump({"_meta": tag, "expected_effect": expected}, open(ep, "w"), indent=1)
    frozen = {"_meta": dict(tag, time=time.strftime("%Y-%m-%d %H:%M:%S"), rule="docs/設計/実験3_設計.md 1 節・1.1 節・1.4 節・9 節",
                            dai1_frozen_sha256=fz1_sha, targets=list(args.targets), encoder=dai1.ENC["enc"],
                            validate_sha256=sha(vpath) if vpath.exists() else None, script_sha256=script_shas(),
                            detect_expected_sha256=sha(ep)),
              "oracle_tether": {T: v["tether"] for T, v in oracle.items()},
              "oracle_convergence": {T: v["convergence"] for T, v in oracle.items()},
              "detect": det,
              "validity": json.load(open(vpath)) if vpath.exists() else None,
              "scenario_folds": {pn: [r["fold"] for r in tabs["scenario_counts"] if r["pair"] == pn and r["kept"]] for pn in SCEN},
              "count_rule": dict(rule="どちらかの組で版の感情が 10 本未満か合計が 30 本未満の分割は，その量の平均から外す（全体の複製）",
                                 n_dropped=len(tabs["count_rule"]), dropped=tabs["count_rule"])}
    if list(args.targets) != list(DD.CORPORA):
        frozen["_meta"]["partial_targets"] = list(args.targets)            # 試走だけ．本番は 4 つ全部
    if frozen["validity"] is None:
        print("注意：validate/summary.json が無い（妥当性の確かめの前に detect を流した）", flush=True)
    fp = out / "frozen.json"
    json.dump(frozen, open(fp, "w"), indent=1, ensure_ascii=False)
    print(f"detect：{time.time()-t0:.0f} 秒．書いた：{fp} sha256 {sha(fp)}", flush=True)
    if production(out):                                                  # 本番の置き場のときだけ写しを置く（置き場ごとに名前を分ける）
        shutil.copy(fp, frozen_copy(out))
        print("写し：", frozen_copy(out), flush=True)


# ------------------------------------------------------------------ 段 eval：単位ごとの計算

def pool_teachers(sv: Saved, tk: dict, s: str, K: int, names: list) -> dict:
    """予測を付ける側の候補の上の確率を，段 1・段 2 の W・b（`dai1.build_tk`．段 3 が受け取ったのと同じ作り）とソースの W・b から作り直す．"""
    u = sv.u
    Q = {}
    for name in names:
        if name.startswith("single_"):
            Q[name] = tk[f"none_{name[7:]}"][0]
        elif name.startswith("pair_"):
            Q[name] = tk[f"none_{name[5:]}"][0]
        elif name == "fused_AVT":
            Q[name] = tk["none_AVT"][0]
        elif name == "poe_none":
            Q[name] = dai1.poe(tk, "AVT", 0)
        elif name == "noret_sel":
            Q[name] = dai1.product(tk, sv.wsel[K], 0, src="none")
        elif name.startswith("ret_"):
            Q[name] = tk[name][0]
        elif name == "start_AVT":
            Q[name] = dai1.probs(u.Xp["AVT"], *u.start["AVT"])
        elif name.startswith("src_"):
            c, kind = name[4:].rsplit("_", 1)
            Wc, bc = [(W, b) for cc, W, b in u.src[kind] if cc == c][0]
            Q[name] = dai1.probs(u.Xp[kind], Wc, bc)
        elif name == "prop_sel":
            Q[name] = dai1.product(tk, sv.wsel[K], 0)
        elif name == "prop_one":
            Q[name] = dai1.product(tk, (1.0, 1.0, 1.0), 0)
        else:
            raise ValueError(name)
    return Q


def repro_fit(sv: Saved, tk: dict, s: str, K: int, names: list, pool_sel: np.ndarray | None = None, tag: str = "all") -> list:
    """再現できる度合い（2 節）：候補を塊の単位で 2 組に分け，片方で生徒と同じ形の線形の識別器（出発点から，Q の判定を硬いラベル
    （感情の釣り合いの重み，重みの合計はその組の本数）にし，(生徒，K) の実験 1 の引き戻しで学ぶ）を学び，もう片方で Q の判定との
    釣り合いを取った一致（Q の判定の感情ごとの一致の割合の平均）を測り，入れ替えて平均する．pool_sel を渡すと候補をそこに限る．"""
    u = sv.u
    idx = np.arange(u.n) if pool_sel is None else np.flatnonzero(pool_sel)
    h = pool_halves(sv.cl_pool[idx])
    halves = [idx[h == 0], idx[h == 1]]
    dl = sv.dlg_pool
    shared = set(dl[halves[0]]) & set(dl[halves[1]]) if sv.T == "MELD" or sv.c == "MELD" else set()
    nonshared = ~np.isin(dl, list(shared)) if shared else np.ones(u.n, bool)
    Q = pool_teachers(sv, tk, s, K, names)
    qy = {nm: Q[nm].argmax(1).cpu().numpy() for nm in names}
    G = torch.stack([S.balanced_onehot(qy[nm], halves[hh], u.n) for nm in names for hh in (0, 1)])
    W0, b0 = u.start[s]
    cw, cb = sv.teth[(s, K)]
    r = S.fit(u.Xp[s], G, W0, b0, Cw=cw, Cb=cb)
    pred = S.logits(u.Xp[s], r["W"], r["b"]).argmax(2).cpu().numpy()
    nk = np.setdiff1d(idx, u.kpos[K])
    rows = []
    for i, nm in enumerate(names):
        ag, agn = [], []
        for hh in (0, 1):
            other = halves[1 - hh]
            pr = pred[2 * i + hh]
            ag.append(100 * S.uar(qy[nm][other], pr[other]))
            o2 = other[nonshared[other]]
            agn.append(100 * S.uar(qy[nm][o2], pr[o2]) if len(o2) else np.nan)
        rows.append(dict(student=s, K=K, teacher=nm, subset=tag, agree=float(np.mean(ag)), agree_nonshared=float(np.mean(agn)),
                         acc=100 * S.uar(u.ypool[nk], qy[nm][nk]), n_half0=len(halves[0]), n_half1=len(halves[1]),
                         n_shared_dialogues=len(shared), grad=float(max(r["grad_p"][2 * i], r["grad_p"][2 * i + 1])),
                         stalled=bool(r["stalled_p"][2 * i] or r["stalled_p"][2 * i + 1]),
                         finite=bool(r["finite_p"][2 * i] and r["finite_p"][2 * i + 1])))
    return rows


def decomp_items(names) -> list:
    """位置の誤りと方向の誤りを分ける識別器（start，none，prop，partner，ソースのコーパスごとの識別器）．"""
    return [n for n in names if re.fullmatch(r"\w+\|(start|none\|\d+|prop\|\d+|partner\|\d+|src\|\w+)", n)]


def eval_items(sv: Saved, fz3: dict, first_seed: bool) -> dict:
    """切片を当て直す識別器：名前 → (種類，W，b)．"""
    it = {}
    for s in DD.STUDENTS:
        it[f"{s}|start"] = (s, *sv.u.start[s])
        for K in ks_f():
            it[f"{s}|none|{K}"] = (s, *sv.none(s, K))
        for K in ks_m():
            it[f"{s}|prop|{K}"] = (s, *sv.prop(s, K))
        if 5 in DD.KS:
            it[f"{s}|partner|5"] = (s, *sv.partner(s, 5))
        for K in ks_f():
            for nm in noown_types(s, sv.srcs):
                it[f"{s}|tt|{K}|{nm}"] = (s, *sv.tt(s, K, nm))
    for kind in DD.STUDENTS:
        it[f"{kind}|oracle"] = (kind, *sv.oracle(kind, *fz3["oracle_tether"][sv.T][kind]))
    if first_seed:                                                       # 説明 6：ソースのコーパスごとの識別器（種に依らない）
        for kind in ("AVT", "A", "V", "T"):
            for c, Wc, bc in sv.u.src[kind]:
                it[f"{kind}|src|{c}"] = (kind, Wc, bc)
    return it


def eval_subsets(sv: Saved) -> dict:
    """1.1 節の部分集合（族 5 の記述）と説明 4 の (b)．名前 → (テストの印，切片を当て直す識別器の名前の並び)．"""
    out = {}
    for k, mask in subset_masks(sv.T, sv.sid, sv.cl).items():
        if k == "face_single":
            names = [f"{s}|{w}" for s in "AVT" for w in ["oracle"] + [f"none|{K}" for K in ks_f()]]
        else:
            studs = ("T", "AVT") if k == "asr_ok" else DD.STUDENTS
            names = [f"{s}|{w}|5" for s in studs for w in ("prop", "none")]
        out[k] = (mask, names)
    if 5 not in DD.KS:
        out = {k: (m, [n for n in names if not n.endswith("|5")]) for k, (m, names) in out.items()}
    return out


def rebuilt_check(sv: Saved) -> list:
    """5 節：段 1・段 3 の W・b と特徴から作り直した判定と保存した int8 の判定の一致の割合（段，生徒，K，単位ごと）．
    AVT の生徒の共正則化（3 つの W をつないだもの）も同じく照合する．"""
    rows = []
    for s in DD.STUDENTS:
        items = [str(x) for x in sv.p1[f"p1_{s}_items"]]
        Z = S.logits(sv.u.Xt[s], S.t(sv.p1[f"p1_{s}_W"]), S.t(sv.p1[f"p1_{s}_b"])).argmax(2).cpu().numpy()
        ag = (Z == sv.p1[f"p1_{s}_pred"]).mean(1)
        for K in sorted({x.split("|")[0] for x in items}):
            m = np.array([x.split("|")[0] == K for x in items])
            rows.append(dict(stage=1, student=s, K=K, n_items=int(m.sum()), n_not_full=int((ag[m] < 1).sum()), min_agree=float(ag[m].min())))
        for K in dai1.KS_ALL:
            Z = S.logits(sv.u.Xt[s], S.t(sv.p3[f"p3_{s}_K{K}_W"]), S.t(sv.p3[f"p3_{s}_K{K}_b"])).argmax(2).cpu().numpy()
            ag = (Z == sv.p3[f"p3_{s}_K{K}_pred"]).mean(1)
            rows.append(dict(stage=3, student=s, K=str(K), n_items=len(ag), n_not_full=int((ag < 1).sum()), min_agree=float(ag.min())))
    def agree(stage, s, K, X, W, b, pred):
        Z = S.logits(X, S.t(W), S.t(b)).argmax(2).cpu().numpy()
        ag = (Z == pred).mean(1)
        rows.append(dict(stage=stage, student=s, K=str(K), n_items=len(ag), n_not_full=int((ag < 1).sum()), min_agree=float(ag.min())))
    nA = len(dai1.A_GRID)
    for K in dai1.KS_ALL:
        W = np.concatenate([sv.p3[f"p3g_K{K}_{m}_W"] for m in "AVT"], axis=2)
        b = sum(sv.p3[f"p3g_K{K}_{m}_b"].astype(np.float64) for m in "AVT")
        agree("3g", "AVT", K, sv.u.Xt["AVT"], W, b, sv.p3[f"p3g_K{K}_AVT_pred"])
        for m in "AVT":                                                  # 単一の生徒の共訓練と共正則化（gate-code の I14）
            agree("3c", m, K, sv.u.Xt[m], sv.p3[f"p3c_K{K}_{m}_W"], sv.p3[f"p3c_K{K}_{m}_b"], sv.p3[f"p3c_K{K}_{m}_pred"])
            agree("3g", m, K, sv.u.Xt[m], sv.p3[f"p3g_K{K}_{m}_W"], sv.p3[f"p3g_K{K}_{m}_b"], sv.p3[f"p3g_K{K}_{m}_pred"])
        for s in DD.STUDENTS:                                            # 役割を分ける版の段 a と段 b（段 b の W は段 a の解．並びは (版, ρ, a)）
            Wa, ba = sv.p3[f"p3r_{s}_K{K}_Wa"], sv.p3[f"p3r_{s}_K{K}_ba"]
            agree("3r_a", s, K, sv.u.Xt[s], Wa, ba, sv.p3[f"p3r_{s}_K{K}_pred_a"])
            if K > 0:
                bb = sv.p3[f"p3r_{s}_K{K}_bb"]
                agree("3r_b", s, K, sv.u.Xt[s], np.repeat(Wa, nA, axis=0), bb, sv.p3[f"p3r_{s}_K{K}_pred_b"])
    return rows


def eval_unit(sv: Saved, fz3: dict, first_seed: bool) -> dict:
    u = sv.u
    keys, keep = rep_keep(sv.cl, sv.mode, sv.gmap)
    grp = test_halves(sv.y, sv.cl, sv.sid)
    R = {"meta": dict(T=sv.T, fold=sv.f, seed=sv.r, n_test=len(sv.y), n_pool=u.n, dai1_commits=sorted(sv.commits),
                      dims={k: int(u.Xt[k].shape[1]) for k in DD.STUDENTS}),
         "keys": keys, "sizes": Counter(sv.cl.tolist()), "refit": {}, "sub": {}, "counts": {}, "counts_sub": {}, "preds": {},
         "repro": [], "scen": {}, "checks": [], "conv": [], "tstats": [], "oracle": [], "rowsum": []}
    items = eval_items(sv, fz3, first_seed)
    Z = {}
    for name, (kind, W, b) in items.items():
        Zw = scores(u.Xt[kind], W)
        Z[name] = (kind, Zw, b, W)
        R["refit"][name], cv = refit_item(Zw, b, sv.y, grp, keep, ctx=f"{sv.base.name} {name}")
        R["conv"].append(dict(item=name) | cv)
        Wn = W.cpu().numpy()
        R["rowsum"].append(dict(item=name, rel=float(np.abs(Wn.sum(0)).max() / max(np.abs(Wn).max(), 1e-30))))
    for name in decomp_items(items):                                     # 順序を入れ替えた分け方（W* を W と同じ大きさにして元の b と組む）
        kind, Zw, b, W = Z[name]
        _, Zo, _, Wo = Z[f"{kind}|oracle"]
        scale = float(torch.linalg.norm(W) / torch.linalg.norm(Wo))
        al, _ = refit_item(Zo * scale, b, sv.y, grp, keep, refit=False)
        R["refit"][name]["align"] = al["orig"]
    # gate-code の R4（全体の複製 j = 0 だけ）：(i) W* の引き戻しの格子の全部の点の UAR(W*, b*(W*))，
    # (ii) 話者が重ならない 2 組（`disjoint_halves`）での当て直し（分解に使う識別器と W*）．
    keep0 = keep[:1]
    R["oracle_grid"] = {}
    for kind in DD.STUDENTS:
        for itm in [str(x) for x in sv.p1[f"p1_{kind}_items"] if str(x).startswith("all|")]:
            _, cw, cb = itm.split("|")
            Wg, bg = sv.oracle(kind, cw, cb)
            R["oracle_grid"][f"{kind}|{cw}|{cb}"], cv = refit_item(scores(u.Xt[kind], Wg), bg, sv.y, grp, keep0,
                                                                   ctx=f"{sv.base.name} {kind}|oracle_grid|{cw}|{cb}")
            R["conv"].append(dict(item=f"oracle_grid:{kind}|{cw}|{cb}") | cv)
    gdis = disjoint_halves(sv.T, sv.cl, sv.sid)
    R["dis_meta"] = dict(n_group0=int((gdis == 0).sum()), n_group1=int((gdis == 1).sum()),
                         n_clusters_shared=len(set(sv.cl[gdis == 0]) & set(sv.cl[gdis == 1])))
    R["refit_dis"] = {}
    for name in decomp_items(items) + [f"{k}|oracle" for k in DD.STUDENTS]:
        kind, Zw, b, _ = Z[name]
        R["refit_dis"][name], cv = refit_item(Zw, b, sv.y, gdis, keep0, ctx=f"{sv.base.name} {name}（話者が重ならない 2 組）")
        R["conv"].append(dict(item=f"disjoint:{name}") | cv)
    for sname, (mask, names) in eval_subsets(sv).items():
        idx = np.flatnonzero(mask)
        ys, cs, ss = sv.y[idx], sv.cl[idx], sv.sid[idx]
        k2, keep2 = rep_keep(cs, sv.mode, sv.gmap)
        g2 = test_halves(ys, cs, ss)
        blk = {"keys": k2, "sizes": Counter(cs.tolist()), "n": len(idx), "refit": {}}
        for name in names:
            kind, Zw, b, _ = Z[name]
            blk["refit"][name], cv = refit_item(Zw[torch.as_tensor(idx, device=S.DEV)], b, ys, g2, keep2, ctx=f"{sv.base.name} {sname}:{name}")
            R["conv"].append(dict(item=f"{sname}:{name}") | cv)
        R["sub"][sname] = blk
    up = ST.UnitPreds(sv.base)
    face = None
    if sv.T == "MELD":
        fc = subset_map("MELD", "face_compete")
        face = np.array([fc.get(x) == "single" for x in sv.sid])
    for s in DD.STUDENTS:
        startp = up.none(s, 0, None)
        for K in ks_m():
            P = pred_rows(sv, up, s, K, startp)
            for row, pr in P.items():
                R["counts"][(s, K, row)] = ST.counts(sv.y, pr, sv.cl)
            for row in ("prop", "none"):
                R["preds"][(s, K, row)] = P[row].astype(np.int8)
            if face is not None:
                for row in ("prop", "none"):
                    R["counts_sub"][("face_single", s, K, row)] = ST.counts(sv.y[face], P[row][face], sv.cl[face])
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    fpool = None
    if sv.T == "MELD":
        fc = subset_map("MELD", "face_compete")
        fpool = np.array([fc.get(x) == "single" for x in sv.sid_pool])
    for s in DD.STUDENTS:
        for K in ks_f():
            R["repro"] += repro_fit(sv, TK[K], s, K, repro_types(s, sv.srcs) + ["prop_sel", "prop_one"])
            if fpool is not None and s != "AVT":
                main = "prop_" + sv.fz["main"][str(K)]
                R["repro"] += repro_fit(sv, TK[K], s, K, [main], pool_sel=fpool, tag="face_single")
    if sv.T == "IEMOCAP" and 5 in DD.KS:                                 # 説明 5：同じ筋書きの中の対（元の切片．当て直さない）
        scen = scenario(sv.sid)
        for pn, (scs, cls) in SCEN.items():
            base_m = np.isin(scen, scs) & np.isin(sv.y, cls)
            yl = local_labels(sv.y, cls)
            Yk = onehot(yl, 2)
            for s in ("T", "AVT"):
                hit = {}
                for w in ("prop", "none"):
                    _, Zw, b, _ = Z[f"{s}|{w}|5"]
                    zz = Zw.cpu().numpy()[:, list(cls)] + b.cpu().numpy()[list(cls)]
                    hit[w] = np.argmax(zz, 1) == yl
                vals = []
                for j in range(len(keys)):
                    m = base_m & keep[j]
                    cnt = np.bincount(yl[m], minlength=2) if m.any() else np.zeros(2, int)
                    if cnt.min() < MIN_GROUP:
                        vals.append(np.nan); continue
                    vals.append(float(uar_of(hit["prop"][None], m[None], Yk)[0] - uar_of(hit["none"][None], m[None], Yk)[0]))
                R["scen"][(s, pn)] = np.array(vals)
    R["checks"] = rebuilt_check(sv)
    p3c = pd.read_csv(f"{sv.base}_p3.csv", low_memory=False)
    ts = p3c[p3c.method == "_teacher_stats"]
    R["tstats"] = ts[["student", "teacher", "K", "maxp", "entropy"]].to_dict("records") if "student" in ts else []
    p1c = pd.read_csv(f"{sv.base}_p1.csv", low_memory=False)
    for kind in DD.STUDENTS:
        cw, cb = fz3["oracle_tether"][sv.T][kind]
        h = p1c[(p1c.method == "oracle") & (p1c.kind == kind) & np.isclose(p1c.Cw, cw) & np.isclose(p1c.Cb, cb)]
        _, Zo, bo, Wo = Z[f"{kind}|oracle"]
        pz = S.logits(u.Xp[kind], Wo, bo).argmax(1).cpu().numpy()
        R["oracle"].append(dict(kind=kind, pool_uar=100 * S.uar(u.ypool, pz), n_pool=u.n, dim=int(u.Xp[kind].shape[1]),
                                grad=float(h.grad.iloc[0]) if len(h) else np.nan,
                                stalled=bool(str(h.stalled.iloc[0]) == "True") if len(h) else None))
    return R


# ------------------------------------------------------------------ 段 eval：集計

def cell_flags_refit(U: dict, name: str, vi: int, sub: str | None = None) -> dict:
    """4 節と 1.2 節（gate-code の L4）：2 組のどちらかで感情が 10 本未満か合計が 30 本未満の分割は，その量の平均から外す
    （`groups_ok`．refit_item が計算しない）．全部の分割が外れたセルだけを付録に回す（区間を示さない）．"""
    n_out, n_all, folds = 0, 0, set()
    for (f, r), R in U.items():
        blk = R["refit"] if sub is None else R["sub"].get(sub, {}).get("refit", {})
        if name not in blk:
            continue
        n_all += 1
        if not blk[name]["valid"][vi][0]:
            n_out += 1; folds.add(f)
    return dict(n_units_count_rule=n_out, folds_count_rule=",".join(map(str, sorted(folds))),
                appendix_only=bool(n_all > 0 and n_out == n_all))


def sub_vals(U: dict, sub: str, fn) -> tuple[dict, dict]:
    vals, sizes = {}, {}
    for fr, R in U.items():
        blk = R["sub"].get(sub)
        if blk is None:
            continue
        a = fn(blk)
        if a is None:
            continue
        vals[fr] = {k: float(x) for k, x in zip(blk["keys"], a)}
        sizes[fr[0]] = blk["sizes"]
    return vals, sizes


def validity_ok(fz3: dict, c: str, s: str, wts: list, ver: str) -> bool | None:
    """確かめ 1 の合否（不合格の W の種類を使う量は記述に回す）．確かめの結果が無いときは None．呼ぶ側は True のときだけ
    判定に使い，False と None はどちらも「判定できない」にする（欠けを合格として扱わない．gate-code の I11-a）．"""
    v = (fz3.get("validity") or {}).get("check1", {})
    res = [v.get(f"{c}|{s}|{w}|{ver}") for w in wts]
    if any(x is None for x in res):
        return None
    return all(x["passed"] for x in res)


def check2_ok(fz3: dict, c: str, s: str) -> str | None:
    v = (fz3.get("validity") or {}).get("check2", {}).get(f"{c}|{s}")
    return None if v is None else v["verdict"]


def stats_row(e: dict, with_p: bool = False) -> dict:
    """区間の行．p 値は族 5 にだけ付ける（6 節．ほかの数には区間と単位の数を添える）．"""
    keys = ("effect", "lo", "hi", "se", "df") + (("p",) if with_p else ()) + ("n_units", "n_folds", "folds_dropped", "rep_fold_drops",
                                                                            "folds_D", "g_eff", "n_reps", "method", "identical", "v_jk", "v_seed")
    return {k: e.get(k) for k in keys}


def aggregate(T: str, U: dict, fz1T: dict, fz3: dict, rng, tabs: dict) -> dict:
    """1 つの評価のターゲット T の全部の表と判定．tabs に行を足し，判定の要約を返す．"""
    mode, _ = jk_mode(T)
    g_eff, _ = ST.g_eff_of(T)
    delta = float(fz1T["detect"]["delta_same"]) if "detect" in fz1T else np.nan
    det5 = (fz3.get("detect", {}).get(T) or {}).get("cells", {})
    sz = sizes_of(U)
    first = {fr: R for fr, R in U.items() if any(k.split("|")[1] == "src" for k in R["refit"])}
    srcs = DD.others(T)
    summ = {}

    # 1 節：位置の誤りと方向の誤り（記述．区間は 1.3 節）
    names = sorted({n for R in U.values() for n in decomp_items(R["refit"])})
    negs = {}
    for name in names:
        kind = name.split("|")[0]
        oname = f"{kind}|oracle"
        UU = first if "|src|" in name else U
        szz = sizes_of(UU)
        for vi, vn in enumerate(VNAME):
            def q(fn):
                def g(R):
                    if name not in R["refit"]:
                        return None
                    a, o = R["refit"][name], R["refit"][oname]
                    return np.where(a["valid"][vi], fn(a, o), np.nan)
                return g
            defs = {"position": lambda a, o: a["refit"][vi] - a["orig"][vi],
                    "direction": lambda a, o: o["refit"][vi] - a["refit"][vi],
                    "sum": lambda a, o: o["refit"][vi] - a["orig"][vi],
                    "position_null": lambda a, o: a["refit"][vi] - a["fullb"][vi],
                    "position_minus_null": lambda a, o: a["fullb"][vi] - a["orig"][vi],
                    "direction_swap": lambda a, o: a["align"][vi] - a["orig"][vi],
                    "position_swap": lambda a, o: (o["refit"][vi] - a["orig"][vi]) - (a["align"][vi] - a["orig"][vi]),
                    "swap_minus_main": lambda a, o: (a["align"][vi] - a["orig"][vi]) - (o["refit"][vi] - a["refit"][vi])}
            for qn, fn in defs.items():
                vals = to_vals(UU, q(fn))
                e = jk_stats(vals, mode, szz)
                row = dict(T=T, item=name, kind=kind, version=vn, quantity=qn) | stats_row(e)
                if qn == "direction":
                    v0 = [x["full"] for x in vals.values() if np.isfinite(x["full"])]
                    row["n_negative_units"] = int(sum(x < 0 for x in v0))
                    if vi == 0 and "|src|" not in name:
                        negs.setdefault(kind, []).extend(v0)
                if qn == "position_null":
                    v0 = [x["full"] for x in vals.values() if np.isfinite(x["full"])]
                    row["median_units"] = float(np.median(v0)) if v0 else np.nan
                tabs["decomp"].append(row | cell_flags_refit(UU, name, vi))
            gap = [R["refit"][name]["orig"][vi, 0] - R["refit"][name]["full"][vi] for R in UU.values() if name in R["refit"]]
            tabs["twogroup_vs_full"].append(dict(T=T, item=name, version=vn, median_gap=float(np.nanmedian(gap)) if gap else np.nan,
                                                 n_units=len(gap)))
    neg_share = {k: (float(np.mean(np.array(v) < 0)) if v else np.nan) for k, v in negs.items()}
    r4_tables(T, U, first, names, fz3, tabs)
    # 差が無いときの位置の誤りの水準（1 節）：コーパスと版ごとに，単位 × 識別器の中央値（組の大きさで変わる）
    for vi, vn in enumerate(VNAME):
        v0 = [R["refit"][n]["refit"][vi, 0] - R["refit"][n]["fullb"][vi, 0] for R in U.values() for n in decomp_items(R["refit"])
              if R["refit"][n]["valid"][vi, 0]]
        tabs["null_level"].append(dict(T=T, version=vn, median=float(np.median(v0)) if v0 else np.nan, n=len(v0)))

    # 1.1 節：族 5（コーパスごとに Holm）と，族 3 の相手との対ごとの差（記述），部分集合（記述）
    fam = []
    if 5 in DD.KS:
        for s in DD.STUDENTS:
            for vi in range(1, len(VERSIONS)):
                key = f"5|{s}|{VNAME[vi]}"
                e = jk_stats(to_vals(U, refit_diff(f"{s}|prop|5", f"{s}|none|5", vi)), mode, sz)
                ep = jk_stats(to_vals(U, refit_diff(f"{s}|prop|5", f"{s}|partner|5", vi)), mode, sz)
                fl = cell_flags_refit(U, f"{s}|prop|5", vi)
                ok1 = validity_ok(fz3, T, s, ["prop5", "none5"], "pair")
                ok2 = check2_ok(fz3, T, s)
                fam.append(dict(T=T, student=s, pair=VNAME[vi], partner=fz1T["partner"][f"{s}|5"]["method"]) | stats_row(e, with_p=True) | fl |
                           dict(descriptive=bool(det5.get(key, {}).get("descriptive", True)), check1_pair_ok=ok1, check2=ok2,
                                wording_direction=bool(ok2 == "合格"),
                                partner_effect=ep["effect"], partner_lo=ep["lo"], partner_hi=ep["hi"], partner_n_units=ep["n_units"]))
        ps = np.array([x["p"] if np.isfinite(x["p"]) else 1.0 for x in fam])     # 落ちたセルも族に含める（1.2 節）
        for x, a in zip(fam, ST.holm(ps) if len(ps) else []):
            x["p_holm"] = float(a)                                     # 確かめ 1 に不合格のセルも Holm の族に残す
            x["verdict_raw"] = verdict(x["lo"], x["hi"], delta) if not x["appendix_only"] else UNDET
            x["verdict"] = x["verdict_raw"] if x["check1_pair_ok"] is True else UNDET    # 確かめ 1 の不合格・欠けは判定できない（I1-a）
        tabs["family5"] += fam
        for sub in sorted({k for R in U.values() for k in R["sub"]}):
            for s in DD.STUDENTS:
                for vi in range(1, len(VERSIONS)):
                    vals, szs = sub_vals(U, sub, refit_diff(f"{s}|prop|5", f"{s}|none|5", vi))
                    if not vals:
                        continue
                    e = jk_stats(vals, mode, szs)
                    tabs["family5_subsets"].append(dict(T=T, subset=sub, student=s, pair=VNAME[vi]) | stats_row(e) |
                                                   cell_flags_refit(U, f"{s}|prop|5", vi, sub))
        summ["family5"] = dict(n_cells=len(fam), n_holm_sig=int(sum(x["p_holm"] < 0.05 for x in fam)),
                               verdicts=dict(Counter(x["verdict"] for x in fam)))

    # 説明 1
    C = {}
    for fr, R in U.items():
        for key, cnt in R["counts"].items():
            C.setdefault(key, {})[fr] = cnt
    e1 = []
    for s in DD.STUDENTS:
        for K in ks_f():
            q_i = ST.effect_stats(C[(s, K, "rolesep")], C[(s, K, "prop")], g_eff, rng)
            q_ii = ST.effect_stats(C[(s, K, "kbias")], C[(s, K, "none")], g_eff, rng)
            q_iii = jk_stats(to_vals(U, refit_diff(f"{s}|none|{K}", f"{s}|start", 0)), mode, sz)
            q_iv = jk_stats(to_vals(U, refit_diff(f"{s}|prop|{K}", f"{s}|none|{K}", 0)), mode, sz)
            q_34 = jk_stats(to_vals(U, refit_diff(f"{s}|prop|{K}", f"{s}|start", 0)), mode, sz)
            ok_iii = validity_ok(fz3, T, s, [f"none{K}", "start"], "four")
            ok_iv = validity_ok(fz3, T, s, [f"prop{K}", f"none{K}"], "four")
            row = dict(T=T, student=s, K=K, check2=check2_ok(fz3, T, s), check1_iii_ok=ok_iii, check1_iv_ok=ok_iv)
            for tag, e, ok in (("i", q_i, True), ("ii", q_ii, True), ("iii", q_iii, ok_iii), ("iv", q_iv, ok_iv),
                               ("iii_iv", q_34, ok_iii is True and ok_iv is True)):
                v = verdict(e["lo"], e["hi"], delta)
                row |= {f"{tag}_{k}": e.get(k) for k in ("effect", "lo", "hi", "n_units", "df")}
                row[f"{tag}_verdict_raw"] = v
                row[f"{tag}_verdict"] = v if ok is True else UNDET                   # 確かめ 1 の不合格・欠けの量は記述だけ（1.4 節の 3）
            # 確かめ 1 に不合格（か欠け）の量を含むセルは，規則で支持にも退けるにも数えない「判定できないセル」（I1-b）．
            # (i)・(ii) は判定から作る量で b* を使わないので，(iii)・(iv) の W の種類だけで決まる
            row["cell_undetermined"] = bool(not (ok_iii is True and ok_iv is True))
            e1.append(row)
            if s != "AVT":
                for nm in noown_types(s, srcs):
                    e = jk_stats(to_vals(U, refit_diff(f"{s}|tt|{K}|{nm}", f"{s}|none|{K}", 0)), mode, sz)
                    tabs["expl1_tt"].append(dict(T=T, student=s, K=K, teacher=nm) | stats_row(e))
    tabs["expl1"] += e1

    # 説明 2
    e2 = []
    ts = pd.DataFrame([x for R in U.values() for x in R["tstats"]])
    for s in DD.STUDENTS:
        for K in ks_f():
            qa = ST.effect_stats(C[(s, K, "default")], C[(s, K, "tt:poe_none")], g_eff, rng)
            qb = ST.effect_stats(C[(s, K, "tt:poe_none")], C[(s, K, "tt:fused_AVT")], g_eff, rng)
            qd = ST.effect_stats(C[(s, K, "default")], C[(s, K, "tt:fused_AVT")], g_eff, rng)
            row = dict(T=T, student=s, K=K)
            for tag, e in (("a", qa), ("b", qb), ("desc_one_minus_fused", qd)):
                row |= {f"{tag}_{k}": e.get(k) for k in ("effect", "lo", "hi", "n_units")}
                row[f"{tag}_verdict"] = verdict(e["lo"], e["hi"], delta)
            if len(ts):
                for tv, lab in (("one", "maxp_joint_one"), ("tt:poe_none", "maxp_poe_none"), ("tt:fused_AVT", "maxp_fused_AVT")):
                    h = ts[(ts.teacher == tv) & (ts.K == K) & ((ts.student == s) if tv.startswith("tt:") else ts.student.isna())]
                    row[lab] = float(h.maxp.mean()) if len(h) else np.nan
            e2.append(row)
        q0 = ST.effect_stats(C[(s, 0, "tt:poe_none")], C[(s, 0, "tt:start_AVT")], g_eff, rng)
        e2.append(dict(T=T, student=s, K=0) | {f"k0_poe_minus_start_{k}": q0.get(k) for k in ("effect", "lo", "hi", "n_units")})
    tabs["expl2"] += e2

    # 2 節：再現できる度合い
    rp = pd.DataFrame([x for R in U.values() for x in R["repro"]])
    if len(rp):
        g = rp.groupby(["student", "K", "teacher", "subset"])
        tabs["repro_units"] += [dict(T=T, fold=fr[0], seed=fr[1]) | x for fr, R in U.items() for x in R["repro"]]   # 組の大きさと共有する対話
        tabs["repro"] += [dict(T=T, student=s, K=K, teacher=tv, subset=sb, agree=float(x.agree.mean()),
                               agree_nonshared=float(x.agree_nonshared.mean()), acc=float(x.acc.mean()), chance=25.0,
                               n_units=len(x), n_grad_over=int((x.grad > 1e-5).sum()), n_stalled=int(x.stalled.sum()),
                               n_shared_dialogues_mean=float(x.n_shared_dialogues.mean()))
                          for (s, K, tv, sb), x in g]

    # 説明 3（単一の生徒 6 セル．Freedman–Lane の並べ替え）
    e3 = expl3(T, U, rp, srcs, tabs)
    # 説明 4（記述だけ）
    expl4(T, U, rp, fz1T, mode, tabs)
    # 説明 5（IEMOCAP だけ）
    e5 = None
    if T == "IEMOCAP" and 5 in DD.KS:
        e5 = {}
        for s in ("T", "AVT"):
            for pn in SCEN:
                vals = {fr: {k: float(x) for k, x in zip(R["keys"], R["scen"][(s, pn)])} for fr, R in U.items()}
                e = jk_stats(vals, mode, sz)
                width = e["hi"] - e["lo"]
                v = UNDET if not np.isfinite(width) or width > WIDTH_MAX else verdict(e["lo"], e["hi"], None)
                tabs["expl5"].append(dict(T=T, student=s, pair=pn, width=width, verdict=v) | stats_row(e))
                e5[(s, pn)] = v
    # 説明 6（記述）と顔の行の補足
    for kind in ("AVT", "A", "V", "T"):
        for c in srcs:
            name = f"{kind}|src|{c}"
            for vi, vn in enumerate(VNAME):
                rowsel = [r for r in tabs["decomp"] if r["T"] == T and r["item"] == name and r["version"] == vn]
                d = {r["quantity"]: r for r in rowsel}
                if not d:
                    continue
                tabs["expl6"].append(dict(T=T, source=c, kind=kind, version=vn,
                                          position=d["position"]["effect"], position_minus_null=d["position_minus_null"]["effect"],
                                          position_minus_null_lo=d["position_minus_null"]["lo"],
                                          position_minus_null_hi=d["position_minus_null"]["hi"],
                                          direction=d["direction"]["effect"], n_units=d["position"]["n_units"],
                                          mosei_involved=bool("MOSEI" in (c, T)), neutral_pair=bool(vn in NEUTRAL_PAIRS),
                                          face_meld_row=bool(kind == "V" and "MELD" in srcs), trained_on_meld=bool(c == "MELD")))
    # D7：ラベルの中身ごとの再現率（5 つのテストを合わせ，各発話を 1 回ずつ数える．種ごとに出して種で平均）
    if T in ("IEMOCAP", "MOSEI"):
        recall_content(T, U, tabs)
    # 5 節：仮定の表
    orc = pd.DataFrame([x for R in U.values() for x in R["oracle"]])
    for kind, x in orc.groupby("kind"):
        tabs["assumptions"].append(dict(T=T, assumption="W* が参照として働く", key=kind,
                                        n_pool_over_dim=float((x.n_pool / x.dim).mean()), oracle_pool_uar=float(x.pool_uar.mean()),
                                        oracle_n_grad_over=int((x.grad > 1e-5).sum()), neg_direction_share=neg_share.get(kind, np.nan),
                                        flag=bool(neg_share.get(kind, 0) > NEG_DIR_MAX)))
    ck = pd.DataFrame([x | dict(fold=fr[0], seed=fr[1]) for fr, R in U.items() for x in R["checks"]])
    for (stg, s, K), x in ck.groupby(["stage", "student", "K"]):
        tabs["assumptions"].append(dict(T=T, assumption="作り直した判定が保存と一致する", key=f"{stg}|{s}|{K}",
                                        n_units=len(x), n_units_not_full=int((x.n_not_full > 0).sum()), min_agree=float(x.min_agree.min()),
                                        flag=bool((x.n_not_full > 0).any())))
    tg = pd.DataFrame([r for r in tabs["twogroup_vs_full"] if r["T"] == T])
    for vk, x in tg.assign(vk=np.where(tg.version == "four", "four", "pair")).groupby("vk"):
        med = float(x.median_gap.median())
        tabs["assumptions"].append(dict(T=T, assumption="2 組の UAR の平均が全体の UAR に近い", key=vk, median_gap=med,
                                        flag=bool(abs(med) > 1.0)))
    if len(rp):
        x = rp[rp.subset == "all"]
        tabs["assumptions"].append(dict(T=T, assumption="再現できる度合いが偶然より高い", key="all", agree_mean=float(x.agree.mean()),
                                        chance=25.0, n_rows=len(x)))
    rs = pd.DataFrame([x for R in U.values() for x in R["rowsum"]])
    tabs["assumptions"].append(dict(T=T, assumption="W と W* の行の和が 0", key="rel_max", value=float(rs.rel.max()),
                                    flag=bool(rs.rel.max() > 1e-4)))
    cv = pd.DataFrame([x for R in U.values() for x in R["conv"]])
    tabs["assumptions"].append(dict(T=T, assumption="切片の当て直しの収束", key="n_over_tol", value=int(cv.n_over_tol.sum()),
                                    n_problems=int(cv.n_problems.sum()), n_stalled=int(cv.stalled.sum()), flag=bool(cv.n_over_tol.sum() > 0)))
    summ.update(expl1=e1, expl2=e2, expl3=e3, expl5=e5, neg_share=neg_share)
    return summ


def theta_units(vals: dict) -> tuple[float, int]:
    """{(f, r): x} → 分割ごとの種の平均の分割の平均（NaN は外す）と単位の数．全体の複製だけの記述に使う（区間を付けない）．"""
    fv = {}
    for (f, r), x in vals.items():
        if np.isfinite(x):
            fv.setdefault(f, []).append(x)
    return (float(np.mean([np.mean(v) for v in fv.values()])) if fv else np.nan), int(sum(len(v) for v in fv.values()))


def r4_tables(T: str, U: dict, first: dict, names: list, fz3: dict, tabs: dict) -> None:
    """gate-code の R4 の記述（全体の複製 j = 0 だけ．区間は付けない）：
      oracle_grid      ：W* の引き戻しの格子の各点で測った方向の誤り UAR(W*_点, b*) − UAR(W, b*(W)) と，その点の UAR(W*, b*(W*))
      speaker_disjoint ：同じ話者が両方の組に入る 2 組（1 節）と，話者が重ならない 2 組（`disjoint_halves`）での位置・方向の誤りと，
                         その差（同じ話者 − 重ならない．漏れの L9 の大きさ）．両方の分け方で計算できた単位だけで比べる．"""
    for name in names:
        kind = name.split("|")[0]
        oname = f"{kind}|oracle"
        UU = first if "|src|" in name else U
        sel = "|".join(str(x) for x in fz3["oracle_tether"][T][kind])
        tks = sorted({k.split("|", 1)[1] for R in UU.values() for k in R.get("oracle_grid", {}) if k.split("|")[0] == kind})
        for vi, vn in enumerate(VNAME):
            for tk in tks:
                d, o = {}, {}
                for fr, R in UU.items():
                    a, g = R["refit"].get(name), R["oracle_grid"].get(f"{kind}|{tk}")
                    if a is None or g is None or not a["valid"][vi, 0]:
                        continue
                    d[fr] = g["refit"][vi, 0] - a["refit"][vi, 0]
                    o[fr] = g["refit"][vi, 0]
                th, n = theta_units(d)
                tabs["oracle_grid"].append(dict(T=T, item=name, kind=kind, version=vn, tether=tk,
                                                selected=bool(np.isclose([float(x) for x in tk.split("|")], [float(x) for x in sel.split("|")]).all()),
                                                direction=th, oracle_refit=theta_units(o)[0], n_units=n))
            q = {k: {} for k in ("position_same", "position_disjoint", "direction_same", "direction_disjoint")}
            for fr, R in UU.items():
                a, ao = R["refit"].get(name), R["refit"].get(oname)
                b, bo = R.get("refit_dis", {}).get(name), R.get("refit_dis", {}).get(oname)
                if a is None or b is None or not (a["valid"][vi, 0] and b["valid"][vi, 0]):
                    continue
                q["position_same"][fr] = a["refit"][vi, 0] - a["orig"][vi, 0]
                q["position_disjoint"][fr] = b["refit"][vi, 0] - b["orig"][vi, 0]
                q["direction_same"][fr] = ao["refit"][vi, 0] - a["refit"][vi, 0]
                q["direction_disjoint"][fr] = bo["refit"][vi, 0] - b["refit"][vi, 0]
            row = dict(T=T, item=name, kind=kind, version=vn)
            for k, v in q.items():
                row[k], row["n_units"] = theta_units(v)
            row["position_same_minus_disjoint"] = row["position_same"] - row["position_disjoint"]
            row["direction_same_minus_disjoint"] = row["direction_same"] - row["direction_disjoint"]
            n_dis = sum(1 for R in UU.values() if name in R.get("refit_dis", {}))
            row["n_units_disjoint_not_computed"] = int(n_dis - row["n_units"])
            row["disjoint_split"] = "話者（F と M）" if T == "IEMOCAP" else "テストの塊（大きい順に少ない組へ）"
            tabs["speaker_disjoint"].append(row)


def spearman_rows(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A (B, n) の各行と b (n,) の Spearman の順位相関．"""
    ra = st.rankdata(A, axis=1)
    rb = st.rankdata(b)
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra @ rb) / den


def expl3(T: str, U: dict, rp: pd.DataFrame, srcs: list, tabs: dict) -> dict:
    """説明 3：単一の生徒 A・V・T × K（5，25）の 6 セル．セルごとに，種類の得（tt の行の UAR − 比較の基準）と再現できる度合いを
    単位で平均し，それぞれを当たり方に 1 次で回帰した残差の Spearman を取る．生徒ごとに，その生徒の種類の集合の中で「得を当たり方に
    回帰した残差」を並べ替える Freedman–Lane の形（K = 5 と 25 に同じ並べ替え），2,000 回．集合は 2 通り．"""
    out = {}
    if not len(rp):
        return out
    for si, s in enumerate("AVT"):
        for set_name, types in (("all", repro_types(s, srcs)), ("noown", noown_types(s, srcs))):
            per_K = []
            for K in ks_f():
                gain, rep, acc = [], [], []
                for tv in types:
                    d = [ST.uar_full(R["counts"][(s, K, f"tt:{tv}")]) - ST.uar_full(R["counts"][(s, K, "none")]) for R in U.values()]
                    x = rp[(rp.student == s) & (rp.K == K) & (rp.teacher == tv) & (rp.subset == "all")]
                    gain.append(float(np.mean(d))); rep.append(float(x.agree.mean())); acc.append(float(x.acc.mean()))
                per_K.append((K, np.array(gain), np.array(rep), np.array(acc)))
                for tv, g_, r_, a_ in zip(types, gain, rep, acc):
                    tabs["expl3_types"].append(dict(T=T, student=s, K=K, set=set_name, teacher=tv, gain=g_, repro=r_, acc=a_))
            n = len(types)
            if n < 3 or not per_K:
                out[(s, set_name)] = dict(n_types=n, stat=np.nan, p_upper=np.nan, p_lower=np.nan)
                continue
            rng = np.random.default_rng([RNG_SEED, DD.CORPUS_ID[T], si, 0 if set_name == "all" else 1])
            perms = np.stack([rng.permutation(n) for _ in range(N_PERM)])
            obs, null = [], []
            for K, g_, r_, a_ in per_K:
                X = np.column_stack([np.ones(n), a_])
                M = np.eye(n) - X @ np.linalg.pinv(X)
                e, f_ = M @ g_, M @ r_
                obs.append(spearman_rows(e[None], f_)[0])
                null.append(spearman_rows((M @ e[perms].T).T, f_))
                tabs["expl3_cells"].append(dict(T=T, student=s, K=K, set=set_name, n_types=n, spearman=obs[-1]))
            stat = float(np.mean(obs))
            nd = np.mean(np.stack(null), 0)
            pu = float((1 + np.sum(nd >= stat - 1e-12)) / (N_PERM + 1))
            pl = float((1 + np.sum(nd <= stat + 1e-12)) / (N_PERM + 1))
            out[(s, set_name)] = dict(n_types=n, stat=stat, p_upper=pu, p_lower=pl)
            tabs["expl3"].append(dict(T=T, student=s, set=set_name, n_types=n, stat=stat, p_upper=pu, p_lower=pl, n_perm=N_PERM,
                                      upper5=bool(pu <= 0.05), lower5=bool(pl <= 0.05)))
    return out


def expl4(T: str, U: dict, rp: pd.DataFrame, fz1T: dict, mode: str, tabs: dict) -> None:
    """説明 4（記述だけ）：生徒 A・V・T の顔の得（族 1 の効果：提案 − 比較の基準），再現できる度合い（提案の主の版の予測），
    直す余地（UAR(W*, b*(W*)) − 比較の基準．比較の基準は元の切片の 2 組の平均）と，「V − A」「V − T」の点の値と分割ごとの範囲．
    MELD は候補とテストの両方を `face_compete` が `single` の発話に限った同じ量も出す（(b)）．"""
    for K in ks_f():
        main = "prop_" + fz1T["main"][str(K)]
        for tag in ("all",) + (("face_single",) if T == "MELD" else ()):
            per = {}
            for s in "AVT":
                if tag == "all":
                    gain_f = {}
                    for (f, r), R in U.items():
                        gain_f.setdefault(f, []).append(ST.uar_full(R["counts"][(s, K, "prop")]) - ST.uar_full(R["counts"][(s, K, "none")]))
                    room = to_vals(U, refit_diff(f"{s}|oracle", f"{s}|none|{K}", 0, what=("refit", "orig")))
                else:
                    gain_f = {}
                    for (f, r), R in U.items():
                        gain_f.setdefault(f, []).append(ST.uar_full(R["counts_sub"][("face_single", s, K, "prop")]) -
                                                        ST.uar_full(R["counts_sub"][("face_single", s, K, "none")]))
                    room, _ = sub_vals(U, "face_single", refit_diff(f"{s}|oracle", f"{s}|none|{K}", 0, what=("refit", "orig")))
                rep_f = {}
                for (f, r), R in U.items():
                    y_ = [z["agree"] for z in R["repro"] if z["student"] == s and z["K"] == K and z["teacher"] == main and z["subset"] == tag]
                    if y_:
                        rep_f.setdefault(f, []).append(y_[0])
                per[s] = {"gain": {f: float(np.mean(v)) for f, v in gain_f.items()},
                          "repro": {f: float(np.mean(v)) for f, v in rep_f.items()},
                          "room": fold_values(room)}
            for qn in ("gain", "repro", "room"):
                row = dict(T=T, K=K, subset=tag, quantity=qn)
                for s in "AVT":
                    fv = per[s][qn]
                    row[s] = float(np.mean(list(fv.values()))) if fv else np.nan
                for o in "AT":
                    fs = sorted(set(per["V"][qn]) & set(per[o][qn]))
                    dv = [per["V"][qn][f] - per[o][qn][f] for f in fs]
                    row[f"V_minus_{o}"] = row["V"] - row[o]
                    row[f"V_minus_{o}_fold_min"] = float(min(dv)) if dv else np.nan
                    row[f"V_minus_{o}_fold_max"] = float(max(dv)) if dv else np.nan
                row["dims"] = json.dumps(next(iter(U.values()))["meta"]["dims"])
                row["second_encoder"] = ("この表が第 2 のエンコーダの回の値" if RUN["encoder"] == 2 else
                                         "確かめていない（第 2 のエンコーダの回は results/dai3_enc2 の同じ表）")   # (c)．I8
                tabs["expl4"].append(row)


def recall_content(T: str, U: dict, tabs: dict) -> None:
    """D7：IEMOCAP の Happy を happy と excited に分けた再現率（筋書き 03・07 の中でも），MOSEI の Happy・Sad・Angry を強さ 1.0 未満と
    以上に分けた再現率．5 つのテストを合わせ，種ごとに出して種で平均する．"""
    df = corpus_df(T)
    if T == "IEMOCAP":
        man = pd.read_csv(ROOT / "data" / "metadata" / "iemocap_utterance_manifest.csv", usecols=["sample_id", "emotion_raw"])
        raw = dict(zip(man.sample_id.astype(str), man.emotion_raw.astype(str)))
    else:
        man = pd.read_csv(ROOT / "data" / "metadata" / "mosei_utterance_manifest.csv",
                          usecols=["sample_id", "int_happy", "int_sad", "int_anger"])
        inten = {k: dict(zip(man.sample_id.astype(str), man[c].astype(float))) for k, c in ((1, "int_happy"), (2, "int_sad"), (3, "int_anger"))}
    sid_all = df["sample_id"].to_numpy().astype(str)
    fold = df["fold"].to_numpy()
    y_all = df["label_4"].to_numpy().astype(int)
    ymap = dict(zip(sid_all, y_all))
    seeds = sorted({r for _, r in U})
    keys = sorted({k for R in U.values() for k in R["preds"]})
    for (s, K, row) in keys:
        per_seed = {}
        for r in seeds:
            sid, pred = [], []
            for f in range(DD.N_FOLDS):
                R = U.get((f, r))
                if R is None:
                    continue
                sid.append(sid_all[fold == f]); pred.append(R["preds"][(s, K, row)])
            sid, pred = np.concatenate(sid), np.concatenate(pred)
            ys = np.array([ymap[x] for x in sid])
            groups = {}
            if T == "IEMOCAP":
                sc = scenario(sid)
                rw = np.array([raw.get(x, "") for x in sid])
                for lab in ("hap", "exc"):
                    groups[f"Happy:{lab}"] = (ys == 1) & (rw == lab)
                    for scn in ("03", "07"):
                        groups[f"Happy:{lab}:scenario{scn}"] = (ys == 1) & (rw == lab) & (sc == scn)
                ks = {k: 1 for k in groups}
            else:
                ks = {}
                for k in (1, 2, 3):
                    iv = np.array([inten[k].get(x, np.nan) for x in sid])
                    groups[f"{dai3_emo(k)}:lt1"] = (ys == k) & (iv < 1.0)
                    groups[f"{dai3_emo(k)}:ge1"] = (ys == k) & (iv >= 1.0)
                    ks[f"{dai3_emo(k)}:lt1"] = ks[f"{dai3_emo(k)}:ge1"] = k
            for gname, m in groups.items():
                if m.sum():
                    per_seed.setdefault(gname, []).append((float((pred[m] == ks[gname]).mean() * 100), int(m.sum())))
        for gname, v in per_seed.items():
            tabs["recall_content"].append(dict(T=T, student=s, K=K, row=row, group=gname, recall=float(np.mean([a for a, _ in v])),
                                               n=v[0][1], n_seeds=len(v)))


def rules(summs: dict) -> dict:
    """3 節の判定の規則を機械的に当てる．「過半のコーパス」は 4 つのうち 3 つ以上，「過半のセル」はセルの数の半分を超える数．
    言葉と言葉を含む生徒の文のため，IEMOCAP 即興を除いた 3 つのうち 2 つ以上でも判定する．"""
    out = {}
    corp = [T for T in DD.CORPORA if T in summs]

    def major_corpora(flag: dict, excl_iemocap=False) -> bool:
        cs = [c for c in corp if not (excl_iemocap and c == "IEMOCAP")]
        need = 3 if not excl_iemocap else 2
        return sum(bool(flag[c]) for c in cs) >= need

    # 説明 1
    per = {}
    for T in corp:
        all_cells = summs[T]["expl1"]
        n = len(all_cells)                                                # 「過半のセル」の分母は行のセルの数（8）のまま
        half = n // 2 + 1
        cells = [c for c in all_cells if not c.get("cell_undetermined")]  # 判定できないセルはどの条件にも数えない（I1-b）
        iii_below = sum(c["iii_verdict"] == BELOW for c in cells) >= half
        first = [c["i_verdict"] in (SAME, ABOVE) and c["ii_verdict"] in (SAME, ABOVE) and c["iii_verdict"] != ABOVE for c in cells]
        full = [f and c["iv_verdict"] == ABOVE and (not iii_below or c["iii_iv_verdict"] == ABOVE) for f, c in zip(first, cells)]
        per[T] = dict(n_cells=n, n_undetermined_cells=n - len(cells), n_first=int(sum(first)), n_full=int(sum(full)),
                      n_iii_above=int(sum(c["iii_verdict"] == ABOVE for c in cells)), iii_below_majority=bool(iii_below),
                      n_ii_supportive=int(sum(c["ii_verdict"] in (SAME, ABOVE) for c in cells)),
                      corpus_first=bool(sum(first) >= half), corpus_full=bool(sum(full) >= half),
                      corpus_reject_a=bool(sum(c["iii_verdict"] == ABOVE for c in cells) >= half),
                      corpus_note_b=bool(sum(c["ii_verdict"] in (SAME, ABOVE) for c in cells) >= half and iii_below),
                      per_student={s: dict(first=int(sum(f for f, c in zip(first, cells) if c["student"] == s)),
                                           full=int(sum(f for f, c in zip(full, cells) if c["student"] == s)))
                                   for s in DD.STUDENTS})

    def e1_verdict(excl):
        if major_corpora({c: per[c]["corpus_full"] for c in corp}, excl):
            return "支持"
        if major_corpora({c: per[c]["corpus_reject_a"] for c in corp}, excl):
            return "退ける（(a)）"
        if major_corpora({c: per[c]["corpus_first"] for c in corp}, excl):
            return "前半だけ（K 本は主に位置を直す．後半は書かない）"
        return UNDET
    out["expl1"] = dict(per_corpus=per, verdict=e1_verdict(False), verdict_excl_IEMOCAP=e1_verdict(True),
                        note_b=bool(major_corpora({c: per[c]["corpus_note_b"] for c in corp})),
                        n_corpora=len(corp))
    # 説明 2
    per = {}
    for T in corp:
        all_cells = [c for c in summs[T]["expl2"] if c.get("K", 0) > 0]
        half = len(all_cells) // 2 + 1
        cells = [c for c in all_cells if not c.get("cell_undetermined")]  # 説明 2 の量は判定から作り，確かめ 1 の対象を含まない
        per[T] = dict(n_cells=len(all_cells), n_a_above=int(sum(c["a_verdict"] == ABOVE for c in cells)),
                      n_b_above=int(sum(c["b_verdict"] == ABOVE for c in cells)),
                      corpus_a=bool(sum(c["a_verdict"] == ABOVE for c in cells) >= half),
                      corpus_b=bool(sum(c["b_verdict"] == ABOVE for c in cells) >= half))

    def e2_verdict(excl):
        a = major_corpora({c: per[c]["corpus_a"] for c in corp}, excl)
        b = major_corpora({c: per[c]["corpus_b"] for c in corp}, excl)
        return "両方" if a and b else ("2a を支持" if a else ("2b を支持" if b else UNDET))
    out["expl2"] = dict(per_corpus=per, verdict=e2_verdict(False), verdict_excl_IEMOCAP=e2_verdict(True))
    # 説明 3
    per = {}
    for T in corp:
        e3 = summs[T]["expl3"]
        ok = all((s, st_) in e3 for s in "AVT" for st_ in ("all", "noown"))
        sup = ok and all(e3[(s, st_)]["p_upper"] <= 0.05 for s in "AVT" for st_ in ("all", "noown"))
        rej = ok and all(e3[(s, st_)]["p_lower"] <= 0.05 for s in "AVT" for st_ in ("all", "noown"))
        per[T] = dict(support=bool(sup), reject=bool(rej))
    v = ("支持" if major_corpora({c: per[c]["support"] for c in corp}) else
         ("退ける" if major_corpora({c: per[c]["reject"] for c in corp}) else UNDET))
    out["expl3"] = dict(per_corpus=per, verdict=v)
    # 説明 5
    if "IEMOCAP" in summs and summs["IEMOCAP"].get("expl5"):
        e5 = summs["IEMOCAP"]["expl5"]
        n_above = sum(e5[("T", pn)] == ABOVE for pn in SCEN)
        out["expl5"] = dict(T_pairs={pn: e5[("T", pn)] for pn in SCEN}, AVT_pairs={pn: e5[("AVT", pn)] for pn in SCEN},
                            n_above_T=int(n_above),
                            verdict="筋書きの話題の違いだけでは説明できない" if n_above >= 2 else
                            "言葉についての文は IEMOCAP 即興を除いて書く")
    out["descriptive_only"] = ["説明 4", "説明 6"]
    return out


def stage_eval(args, out: Path, out1: Path) -> None:
    fz1, fz1_sha = load_fz1(out1)
    fp = out / "frozen.json"
    if not fp.exists():
        sys.exit(f"{fp} が無い（detect を先に流す）")
    fz3 = json.load(open(fp))
    if fz3["_meta"]["dai1_frozen_sha256"] != fz1_sha:
        sys.exit("実験 3 の固定のファイルが作られたときの実験 1 の固定のファイルと，今の実験 1 の固定のファイルが違う")
    if fz3["_meta"].get("script_sha256") != script_shas():                 # detect と eval で同じ台本か（gate-code の記録 6）
        sys.exit(f"固定のファイルを作ったときの台本（{', '.join(SCRIPTS)}）の sha256 と今の台本が違う（detect から流し直す）")
    if fz3["_meta"].get("encoder") != dai1.ENC["enc"]:
        sys.exit("固定のファイルを作ったときのエンコーダと --encoder が違う")
    if production(out):
        cp = frozen_copy(out)
        if not cp.exists() or sha(cp) != sha(fp) or not dai1.committed_clean(cp):
            sys.exit(f"{cp.relative_to(ROOT)} が無いか，frozen.json と一致しないか，コミットされていない")
        if RUN["dirty"] or fz3["_meta"].get("dirty", True):
            sys.exit("本番の評価の段は未コミットの変更があると流さない（固定のファイルも未コミットの台本で作ったものは使わない）")
        if "partial_targets" in fz3["_meta"]:
            sys.exit("固定のファイルが一部の評価のターゲットだけで作られている（detect を 4 つ全部で流す）")
        if fz3.get("validity") is None:
            sys.exit("固定のファイルに妥当性の確かめの結果が無い（validate → detect の順に流す）")
        wts = ["start"] + [f"{x}{K}" for K in ks_f() for x in ("none", "prop")] + ["oracle"]
        need1 = [f"{c}|{s}|{w}|{v}" for c in DD.CORPORA for s in DD.STUDENTS for w in wts for v in ("four", "pair")]
        need2 = [f"{c}|{s}" for c in DD.CORPORA for s in DD.STUDENTS]
        miss = [k for k in need1 if k not in fz3["validity"].get("check1", {})] + \
               [k for k in need2 if k not in fz3["validity"].get("check2", {})]
        if miss:
            sys.exit(f"確かめの結果の鍵が欠けている（{len(miss)} 個．例 {miss[:3]}．欠けを合格として扱わない．I11-a）")
    edir = out / "eval"
    (edir / "units").mkdir(parents=True, exist_ok=True)
    tabs = {k: [] for k in ("decomp", "twogroup_vs_full", "family5", "family5_subsets", "expl1", "expl1_tt", "expl2", "repro",
                            "expl3_types", "expl3_cells", "expl3", "expl4", "expl5", "expl6", "recall_content", "assumptions",
                            "null_level", "repro_units", "oracle_grid", "speaker_disjoint")}
    summs, conv = {}, []
    t0 = time.time()
    for T in args.targets:
        fzT = fz1[T]
        R_ = int(fzT["n_seeds_eval"]) if args.seeds is None else len(args.seeds)
        seeds = list(range(R_)) if args.seeds is None else list(args.seeds)
        U, commits = {}, set()
        for f in range(DD.N_FOLDS):
            for r in seeds:
                cache = edir / "units" / f"{T}_f{f}_r{r}.pkl"
                if cache.exists():
                    R = pickle.load(open(cache, "rb"))
                    if R["_run"]["commit"] != RUN["commit"] or R["_run"]["frozen3"] != sha(fp):
                        sys.exit(f"保存済みの {cache.name} のコミットか固定のファイルが今の実行と違う（消してから流し直す）")
                else:
                    t1 = time.time()
                    sv = Saved(out1, "eval", T, T, f, r, fzT)
                    R = eval_unit(sv, fz3, first_seed=(r == seeds[0]))
                    R["_run"] = dict(commit=RUN["commit"], dirty=RUN["dirty"], frozen3=sha(fp), sec=time.time() - t1)
                    pickle.dump(R, open(cache, "wb"))
                    print(f"eval {T}_f{f}_r{r}: {time.time()-t1:.0f} 秒", flush=True)
                U[(f, r)] = R
                commits |= set(R["meta"]["dai1_commits"])
                conv += [dict(T=T, fold=f, seed=r) | x for x in R["conv"]]
        if len(commits) != 1:
            sys.exit(f"{T}：評価の回の保存のコミットが 1 つでない {commits}")
        rng = np.random.default_rng([RNG_SEED, DD.CORPUS_ID[T]])
        summs[T] = aggregate(T, U, fzT, fz3, rng, tabs)
        dai1.Corpus.cache.pop(T, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    e6 = pd.DataFrame(tabs["expl6"])
    e6sum = []
    if len(e6):
        for kind, x in e6.groupby("kind"):
            pr = x[x.version != "four"]
            g = pr.groupby(["T", "source", "neutral_pair"]).position_minus_null.mean().unstack()
            if True in g and False in g:
                e6sum.append(dict(kind=kind, part="(i) 対の位置の誤り − 水準", neutral=float(g[True].mean()), other=float(g[False].mean()),
                                  n_pairs=len(g)))
            fo = x[x.version == "four"]
            for col in ("position", "position_minus_null"):
                m = fo.groupby("mosei_involved")[col].mean()
                e6sum.append(dict(kind=kind, part=f"(ii) 4 感情の {col}", mosei_involved=float(m.get(True, np.nan)),
                                  not_involved=float(m.get(False, np.nan)), n_pairs=len(fo)))
    tabs["expl6_summary"] = e6sum
    fc = subsets()
    fc = fc[(fc.corpus == "MELD") & (fc.subset == "face_compete")]
    tabs["face_compete_share"] = [dict(n=len(fc), n_compete=int((fc.value == "compete").sum()), n_single=int((fc.value == "single").sum()),
                                       n_undetermined=int((fc.value == "undetermined").sum()))]
    tag = dict(commit=RUN["commit"], dirty=RUN["dirty"], dai1_frozen_sha256=fz1_sha, dai3_frozen_sha256=sha(fp))
    for name, rows in tabs.items():
        pd.DataFrame(rows).assign(**tag).to_csv(edir / f"{name}.csv", index=False)
    pd.DataFrame(conv).assign(**tag).to_csv(edir / "convergence.csv", index=False)
    vr = rules(summs) if len(summs) == len(DD.CORPORA) or args.allow_partial else {"note": "評価のターゲットが 4 つそろっていない"}
    vr["_meta"] = tag | dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), targets=list(args.targets),
                             rule="docs/設計/実験3_設計.md 3 節（判定の規則）．族 5 の外の数に p 値を付けない")
    vr["family5"] = {T: summs[T].get("family5") for T in summs}
    vr["notes"] = [                                                    # 設計 3 節の書き方の約束（判定の語と並べて残す）
        "族 5 は「対の切片を当て直した後の UAR の差」を検定している．確かめ 2 が合格でない (コーパス，生徒) では「方向」の語を使わない（wording_direction）",
        "説明 1 の (iv) が上回っても「別のモダリティの予測が方向を直した」とは書かず「積の予測で学ぶと方向が良くなった」と書く",
        "説明 1 の (ii) は切片だけの版が 2 点から，比較の基準が 14 点から選ぶ（支持しにくい向きの偏り）",
        "説明 4・6 は記述だけ．「支持」「退ける」の語を使わない．説明 4 の (c) は第 2 のエンコーダの評価の回で同じ段を流すまで「確かめていない」",
        "IEMOCAP の顔の再現できる度合いにはマーカーの手がかりが入りうる",
        "族 5 の p 値と説明の判定は確認ではない（同じ 4 コーパスの前の結果から立てた仮説）"]
    json.dump(vr, open(edir / "verdicts.json", "w"), indent=1, ensure_ascii=False, default=str)
    nov = int(pd.DataFrame(conv).n_over_tol.sum()) if conv else 0
    print(f"eval：表 {len(tabs)} 本（行 {sum(len(v) for v in tabs.values())}），勾配が 1e-5 を超えた切片の問 {nov}，"
          f"{time.time()-t0:.0f} 秒．書いた：{edir}", flush=True)


# ------------------------------------------------------------------ 進め方

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("validate", "detect", "eval"))
    ap.add_argument("--dai1-out", default=str(ROOT / "results" / "dai1"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai3"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="eval の種（既定は実験 1 の固定のファイルの n_seeds_eval．試走だけ）")
    ap.add_argument("--allow-dirty", action="store_true", help="試走だけ（本番の置き場では受け付けない）")
    ap.add_argument("--allow-partial", action="store_true", help="試走だけ：評価のターゲットが 4 つそろわなくても判定の規則を当てる")
    ap.add_argument("--encoder", type=int, default=1, choices=(1, 2), help="1：WavLM・ResNet-50・RoBERTa，2：HuBERT・CLIP・DeBERTa（説明 4 の (c)）")
    args = ap.parse_args()
    dai1.ENC["enc"] = DD.ENC1 if args.encoder == 1 else DD.ENC2
    out, out1 = Path(args.out_dir), Path(args.dai1_out)
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    RUN["encoder"] = args.encoder
    if dirty and (production(out) or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある（試走は一時の置き場で --allow-dirty）")
    if production(out):                                                  # 本番の置き場は，読む実験 1 の置き場とエンコーダが決まっている（I8）
        d1name, enc = PROD_OUT[out.resolve().name]
        if out1.resolve() != (ROOT / "results" / d1name).resolve() or args.encoder != enc:
            sys.exit(f"本番の置き場 {out.name} は --dai1-out results/{d1name} と --encoder {enc} で流す")
    out.mkdir(parents=True, exist_ok=True)                                # 守りを通ってから置き場を作る
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                numpy=np.__version__, pandas=pd.__version__, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                script_sha256=sha(Path(__file__)), dai1_sha256=sha(HERE / "dai1.py"), stats_sha256=sha(HERE / "dai1_stats.py"),
                solver_sha256=sha(HERE / "dai_solver.py"), data_sha256=sha(HERE / "dai_data.py"),
                subsets_sha256=sha(ROOT / "frozen" / "subsets.csv"), dai1_out=str(out1), encoder=dai1.ENC["enc"])
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    {"validate": stage_validate, "detect": stage_detect, "eval": stage_eval}[args.stage](args, out, out1)


if __name__ == "__main__":
    main()
