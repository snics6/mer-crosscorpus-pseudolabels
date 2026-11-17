"""実験 14（DAI-14）：どのモダリティの特徴が，ほかのモダリティの判断を再現できるか（設計 `docs/設計/実験14_設計.md` 第 5 版）．

本番の評価の回（実験 1．results/dai1/eval）の保存から，各モダリティの K 本だけの分類器の判断 J_m と，提案の主の版の疑似ラベル
J_prop（prop_one）を候補の上で作り直し，生徒 s（A・V・T）の特徴から線形に再現できる度合い R(s ← m) を 2 つの読み出しで測る．
  R1（そろえた読み出し．主）：特徴を候補の上の主成分で 512 次元にそろえ，出発点 0，罰則の強さ C′ を目標の判断だけを使う内側の
      交差の予測で格子から選ぶ（本当の感情のラベルは使わない）．入れ替えの対照には同じ C′ を当てる．R1fixC は C′ を固定した感度．
  R0（実験 1 の作り．副）：生徒の出発点と実験 1 の引き戻し（実験 3 の 2 節と同じ．保存の repro_units.csv と照合する）．
対照の目標：本当の感情の中で入れ替えた #shuf，（感情，塊，性別）の中で入れ替えた #shufcl（予言 P3・P4 の主），#shufcl の層を
さらに細かく分けた #shuffine（IEMOCAP は元のラベル × 覚醒度の感情ごとの 3 区分，MOSEI は感情の強さの最大の感情ごとの 3 区分）と
#shufage（AFEW だけ．俳優の年齢の 3 区分）．
K = 5（主）と 25（副）．r = 0 の単位では，K = 5・R1 について候補の 2 組の別の分け方 4 通りでも測り，区間に分け方の揺れを足す．

段：
  selftest  合成の数で区間の関数と一致の関数と対照の作りを確かめる
  units     単位ごとの計算．results/dai14/units/<T>_f<f>_r<r>.npz と .csv（同じコミットの既存の単位は飛ばす）
  summary   区間と予言の判定．results/dai14/summary/*.csv
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
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT_PROD = ROOT / "results" / "dai14"
KS = (5, 25)
K_MAIN = 5
STU = ("A", "V", "T")
GRID_C = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)   # R1 の C′ の格子（小さいほど強い罰．第 3 版で下へ広げた．gate-code の G1）
EDGE_SHARE = 0.25                               # 格子の端に選ばれた割合がこれを超えた（コーパス，生徒，目標）は限界として書く
FRAC_LOW = 0.40                                 # 入れ替えで目標が変わった割合の平均がこれ未満の（コーパス，目標）は「判定できない」の読みを弱める
MARKER_RE = r"[\[\]\(\)\{\}!]"                 # 書き起こしの記号（角・丸・波括弧と「!」）．「?」は語順と疑問詞で言葉の中身に入るので含めない
CB_R1 = 1.0                                    # R1 の切片の C_b（共通）
DIM_R1 = 512                                   # R1 でそろえる次元（顔の特徴の次元）
N_ALT = 4                                      # r = 0 で足す別の分け方の数
ALT_TRIES = 400
N_BOOT = ST.N_BOOT
MIN_N = ST.MIN_TEST
MEM_FRAC = 0.6
RNG_SEED = 20261112
VERS = ("true", "shuf", "shufcl", "shuffine", "shufage")
SCRIPTS = ("dai14.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
DESIGN = ROOT / "docs" / "設計" / "実験14_設計.md"
REPRO_SAVED = ROOT / "results" / "dai3" / "eval" / "repro_units.csv"
TOL_REPRO = 1e-3
RUN = {"commit": "", "dirty": True}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def script_shas() -> dict:
    return {n: sha(HERE / n) for n in SCRIPTS}


def code(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ 一致の割合と区間（設計 4 節）

def counts(y, pred, cl):
    return ST.counts(np.asarray(y, int), np.asarray(pred, int), np.asarray(cl).astype(str))


def uar_present(cor, tot):
    """(…, 4) の数 → 現れた感情（件数 > 0）だけの再現率の平均（%）．件数の合計が MIN_N 未満なら NaN．全体・塊を除いた値・引き直しで同じ式．"""
    cor = np.asarray(cor, float); tot = np.asarray(tot, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(tot > 0, cor / np.where(tot > 0, tot, 1.0), np.nan)
        u = 100.0 * np.nanmean(rec, axis=-1)
    return np.where(tot.sum(-1) < MIN_N, np.nan, u)


def theta(d: dict) -> float:
    vals = [np.nanmean(v) for v in d.values() if np.isfinite(v).any()]
    return float(np.mean(vals)) if vals else float("nan")


def combo_stats(cells: list, coefs: list, g_eff: float, rng, split_vals: dict | None = None) -> dict:
    """Σ coefs[i] · UAR(cells[i]) の区間．cells[i]：{(f, r): counts}．g_eff < 20 は塊の jackknife ＋ 種の項（自由度 max(1, ⌊g_eff⌋ − 1)），
    ≥ 20 は全分割で共通の塊の重みの引き直し ＋ 種の項．split_vals：{f: [分け方ごとの値]} を渡すと，分け方の揺れを SE² に足す．
    分割の間で分け方の誤差がそろいうるので，足すのは保守的な上限 ((1/F) Σ_f √Var_f)²（第 3 版．gate-stats の S1，dai-review-ml の M2）．
    独立とみなした値 (1/F²) Σ_f Var_f は v_split_indep に残す．分け方の揺れを足さない区間を lo_ns・hi_ns に残す
    （主と副の判定を同じ作りの区間で比べるため．gate-stats の S3）．"""
    keys = sorted(cells[0])
    for c in cells[1:]:
        if sorted(c) != keys:
            raise SystemExit("combo_stats：cells の鍵がそろっていない")
    for k in keys:
        n0 = cells[0][k][0]
        for c in cells[1:]:
            if len(c[k][0]) != len(n0) or not (c[k][0] == n0).all():
                raise SystemExit(f"combo_stats：単位 {k} で塊の並びが違う")
    folds = sorted({f for f, _ in keys})
    rs_of = {f: [r for ff, r in keys if ff == f] for f in folds}
    F = len(folds)

    def val_full(k):
        return float(sum(a * uar_present(c[k][1].sum(0), c[k][2].sum(0)) for c, a in zip(cells, coefs)))

    d = {f: [val_full((f, r)) for r in rs_of[f]] for f in folds}
    est = theta(d)
    R = {f: int(np.isfinite(d[f]).sum()) for f in folds}
    s2 = {f: (float(np.nanvar(d[f], ddof=1)) if R[f] > 1 else 0.0) for f in folds}
    v_seed = float(sum(s2[f] / max(R[f], 1) for f in folds) / F ** 2)
    v_split = v_split_indep = np.nan
    n_split_folds = 0
    if split_vals is not None:
        vs = [float(np.nanvar(v, ddof=1)) for f, v in split_vals.items() if np.isfinite(v).sum() > 1]
        n_split_folds = len(vs)
        if vs:
            v_split_indep = float(sum(vs) / F ** 2)
            v_split = float((sum(np.sqrt(vs)) / F) ** 2)
    extra = v_split if np.isfinite(v_split) else 0.0
    base = dict(effect=est, n_folds=F, n_units=int(sum(R.values())), g_eff=g_eff, v_seed=v_seed, v_split=v_split,
                v_split_indep=v_split_indep, n_split_folds=n_split_folds)
    if not np.isfinite(est):
        return base | dict(lo=np.nan, hi=np.nan, se=np.nan, df=np.nan, method="", v_cl=np.nan, n_clusters=0, lo_ns=np.nan, hi_ns=np.nan)
    if g_eff >= 20:
        all_cl = sorted({nm for k in keys for nm in cells[0][k][0]})
        pos = {nm: i for i, nm in enumerate(all_cl)}
        G = len(all_cl)
        Wall = rng.multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(float)
        boots = np.zeros(N_BOOT); nval = np.zeros(N_BOOT)
        for f in folds:
            per_r = []
            for r in rs_of[f]:
                k = (f, r)
                Wf = Wall[:, [pos[nm] for nm in cells[0][k][0]]]
                per_r.append(sum(a * uar_present(Wf @ c[k][1], Wf @ c[k][2]) for c, a in zip(cells, coefs)))
            dm = np.nanmean(np.stack(per_r), axis=0)
            ok = np.isfinite(dm)
            boots[ok] += dm[ok]; nval[ok] += 1
        boots = np.where(nval > 0, boots / np.maximum(nval, 1), np.nan)
        boots = boots[np.isfinite(boots)]
        v_cl = float(np.var(boots, ddof=1))
        se = float(np.sqrt(v_cl + v_seed + extra)); se_ns = float(np.sqrt(v_cl + v_seed))
        return base | dict(lo=est - 1.96 * se, hi=est + 1.96 * se, se=se, df=np.nan, method="boot_cl_shared+seed", v_cl=v_cl, n_clusters=G,
                           lo_ns=est - 1.96 * se_ns, hi_ns=est + 1.96 * se_ns)
    all_cl = sorted({nm for k in keys for nm in cells[0][k][0]})
    jk = []
    for g in all_cl:
        dg = {}
        for f in folds:
            vals = []
            for r in rs_of[f]:
                k = (f, r)
                hit = np.flatnonzero(cells[0][k][0] == g)
                if len(hit):
                    i = int(hit[0])
                    v = sum(a * uar_present(c[k][1].sum(0) - c[k][1][i], c[k][2].sum(0) - c[k][2][i]) for c, a in zip(cells, coefs))
                else:
                    v = val_full(k)
                vals.append(float(v))
            dg[f] = vals
        jk.append(theta(dg))
    jk = np.array(jk, float); jk = jk[np.isfinite(jk)]
    G = len(jk)
    v_cl = float((G - 1) / G * ((jk - jk.mean()) ** 2).sum()) if G > 1 else np.nan
    se = float(np.sqrt(v_cl + v_seed + extra)) if np.isfinite(v_cl) else np.nan
    se_ns = float(np.sqrt(v_cl + v_seed)) if np.isfinite(v_cl) else np.nan
    df = max(1, int(np.floor(g_eff)) - 1)
    q = float(st.t.ppf(0.975, df))
    return base | dict(lo=est - q * se, hi=est + q * se, se=se, df=df, method="jk+seed", v_cl=v_cl, n_clusters=G,
                       lo_ns=est - q * se_ns, hi_ns=est + q * se_ns)


def verdict(lo, hi) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "判定できない"
    return "上回る" if lo > 0 else ("下回る" if hi < 0 else "判定できない")


# ------------------------------------------------------------------ 層と部分集合（発話一覧から．ラベル以外の注釈）

_META: dict = {}


def corpus_meta(T: str) -> pd.DataFrame:
    """sample_id → 細かい層の元の値（IEMOCAP は元のラベル emotion_raw と覚醒度 arousal，MOSEI は感情の強さ int_* の最大．
    候補のラベルの一部なので，入れ替えの層を作るだけに使う），性別 sex（IEMOCAP は utterance_speaker，AFEW は gender．
    MELD・MOSEI は空），書き起こしの記号の有無 marker．"""
    if T in _META:
        return _META[T]
    sp = pd.read_csv(ROOT / "splits" / f"{T}.csv", keep_default_na=False, usecols=["sample_id"])
    fn = {"IEMOCAP": "iemocap", "MOSEI": "mosei", "MELD": "meld", "AFEW": "afew"}[T]
    m = pd.read_csv(ROOT / "data" / "metadata" / f"{fn}_utterance_manifest.csv", keep_default_na=False)
    m = m[m.sample_id.isin(sp.sample_id)].set_index("sample_id")
    out = pd.DataFrame(index=sp.sample_id)
    out["fine_cat"], out["fine_val"], out["sex"] = "", np.nan, ""
    if T == "IEMOCAP":
        out["fine_cat"] = m["emotion_raw"].astype(str).reindex(out.index)
        out["fine_val"] = pd.to_numeric(m["arousal"], errors="coerce").reindex(out.index)
        out["sex"] = m["utterance_speaker"].astype(str).reindex(out.index)
    elif T == "MOSEI":
        ic = [c for c in m.columns if c.startswith("int_")]
        out["fine_val"] = m[ic].apply(pd.to_numeric, errors="coerce").max(axis=1).reindex(out.index)
    elif T == "AFEW":
        out["sex"] = m["gender"].astype(str).reindex(out.index)
        out["age"] = pd.to_numeric(m["actor_age"], errors="coerce").reindex(out.index)
    if T == "AFEW":
        w = pd.read_csv(ROOT / "data" / "metadata" / "afew_transcript_whisper.csv", keep_default_na=False).set_index("sample_id")
        tr = w["transcript_asr"].reindex(out.index).fillna("")
    else:
        tr = m["transcript"].reindex(out.index).fillna("")
    out["marker"] = tr.astype(str).str.contains(MARKER_RE, regex=True)
    if T in ("IEMOCAP", "AFEW") and (out["sex"].isin(["", "nan"]) | out["sex"].isna()).any():
        raise SystemExit(f"{T}：性別の欠けた発話がある")
    _META[T] = out
    return out


def tertile_within(v: np.ndarray, group: np.ndarray) -> np.ndarray:
    """group（本当の感情）ごとに，候補の上の 3 分位で v を 0・1・2 に分ける（同じ値が多いので 3 等分にはならない）．欠けは "na"．"""
    out = np.full(len(v), "na", dtype=object)
    for g in np.unique(group):
        ii = np.flatnonzero((group == g) & np.isfinite(v))
        if len(ii):
            q = np.nanquantile(v[ii], [1 / 3, 2 / 3])
            out[ii] = np.digitize(v[ii], q).astype(str)
    return out.astype(str)


def fine_strata(meta: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """細かい層：元のラベル（IEMOCAP）と，元の値の感情ごとの候補の上の 3 分位の区分（第 4 版．全体の 3 分位では MOSEI の
    Happy・Sad・Angry がほぼ 1 つの区分に入るため．dai-review-affect の第 3 版の A1）．テストの発話を使わない．"""
    return np.char.add(np.char.add(meta["fine_cat"].to_numpy().astype(str), "|"), tertile_within(meta["fine_val"].to_numpy(float), y))


# ------------------------------------------------------------------ 段 units

def shuffle_within(qy: np.ndarray, strata: np.ndarray, seed: list) -> tuple[np.ndarray, float]:
    """層 strata が同じ候補の間で目標 qy を入れ替える．戻り：入れ替えた目標と，目標が実際に変わった発話の割合．"""
    rng = np.random.default_rng(seed)
    out = qy.copy()
    for g in np.unique(strata):
        ii = np.flatnonzero(strata == g)
        if len(ii) > 1:
            out[ii] = qy[rng.permutation(ii)]
    return out, float((out != qy).mean())


def alt_halves(cl: np.ndarray, seed: int) -> np.ndarray:
    """塊を乱数の順に並べ，そのときの本数の少ない組に配る（pool_halves と同じ規則で順だけを変える．ラベルを使わない）．"""
    rng = np.random.default_rng(seed)
    names, cnt = np.unique(cl, return_counts=True)
    order = rng.permutation(len(names))
    tot, h = [0, 0], {}
    for i in order:
        k = 0 if tot[0] <= tot[1] else 1
        h[names[i]] = k; tot[k] += cnt[i]
    return np.array([h[g] for g in cl], int)


def same_split(a: np.ndarray, b: np.ndarray) -> bool:
    return bool((a == b).all() or (a == 1 - b).all())


def aligned_features(X: torch.Tensor) -> torch.Tensor:
    """候補の特徴を中心化し，主成分の上位 DIM_R1 個の得点にする（ラベルを使わない）．次元あたりの分散の平均が 1 になるよう全体を一律に割る．"""
    X = S.t(X)
    Xc = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    k = min(DIM_R1, Vh.shape[0])
    sv_ = torch.linalg.svdvals(Xc)
    ev = (sv_ ** 2).cpu().numpy()
    Z = Xc @ Vh[:k].T
    p = ev / ev.sum()
    info = dict(pca_kept=float(ev[:k].sum() / ev.sum()), eff_rank=float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())), dim_in=int(X.shape[1]))
    return Z / Z.var(0).mean().sqrt(), info


def fit_r0(sv, s, K, tg: dict, halves) -> dict:
    """R0：生徒の出発点と実験 1 の引き戻し．tg：{名前: 目標}．戻り：{名前: (交差の予測, 2 組の平均の一致, 学びの記録)}．"""
    u = sv.u
    n = u.n
    keys = list(tg)
    G = torch.stack([S.balanced_onehot(tg[k], halves[hh], n) for k in keys for hh in (0, 1)])
    W0, b0 = u.start[s]
    cw, cb = sv.teth[(s, K)]
    res = S.fit(u.Xp[s], G, W0, b0, Cw=cw, Cb=cb)
    lg = S.logits(u.Xp[s], res["W"], res["b"])
    pred = lg.argmax(2).cpu().numpy()
    prob = torch.softmax(lg, 2).cpu().numpy().astype(np.float32)
    return collect(keys, tg, pred, res, halves, n, dict(Cw=cw, Cb=cb, dim=int(u.Xp[s].shape[1])), prob)


def collect(keys, tg, pred, res, halves, n, info, prob=None) -> dict:
    """交差の予測をそろえる．prob（問, n, 4）を渡すと，交差の予測の確率（float32．下端の C′ では確率の差が小さく float16 では丸めで消えるため．gate-code の第 5 版の K1）を info["prob"] に入れる．"""
    out = {}
    for i, k in enumerate(keys):
        full = np.empty(n, np.int8)
        pf = np.empty((n, 4), np.float32) if prob is not None else None
        ag = []
        for hh in (0, 1):
            other = halves[1 - hh]
            full[other] = pred[2 * i + hh][other]
            if pf is not None:
                pf[other] = prob[2 * i + hh][other]
            ag.append(100 * S.uar(tg[k][other], pred[2 * i + hh][other]))
        fin = bool(res["finite_p"][2 * i] and res["finite_p"][2 * i + 1])
        if not fin:
            raise SystemExit(f"解が有限でない（{k}）")
        out[k] = (full, float(np.mean(ag)), info | dict(grad=float(max(res["grad_p"][2 * i], res["grad_p"][2 * i + 1])),
                                                         stalled=bool(res["stalled_p"][2 * i] or res["stalled_p"][2 * i + 1]),
                                                         prob=pf))
    return out


def fit_r1(Z: torch.Tensor, cl: np.ndarray, halves, tg_true: dict, tg_all: dict, grid_keys=()) -> tuple:
    """1 つの分け方の R1．fit_r1_multi の 1 本の版．"""
    outs, curves, grid = fit_r1_multi(Z, cl, [halves], tg_true, tg_all, grid_keys)
    return outs[0], curves, grid


def fit_r1_multi(Z: torch.Tensor, cl: np.ndarray, splits: list, tg_true: dict, tg_all: dict, grid_keys=()) -> tuple:
    """R1：出発点 0，C_b = CB_R1，C′ は学ぶ側の組を塊で 2 つに分けた内側の交差の予測で，目標の判断との一致が最大の格子の点
    （同点は小さい C′）．tg_true：{名前: 目標}（C′ を選ぶ目標），tg_all：{(名前, 版): 目標}（同じ名前の C′ を当てる）．
    splits：2 組の分け方の並び．すべての分け方の内側の問を 1 回，本体の問を 1 回の L-BFGS の束で解く（速さのため．解は問ごとに独立）．
    grid_keys：tg_all の鍵のうち，最初の分け方で格子の全点でも外側を解いて交差の予測を残すもの（C′ を固定した感度．gate-compare の C1）．
    戻り：(分け方ごとの結果, 内側の一致の曲線 {(分け方, 組, 名前): (格子の点,)}, {(鍵, 格子の番号): 交差の予測})．"""
    n, d = Z.shape
    zeroW = torch.zeros(4, d, device=S.DEV, dtype=S.DT); zerob = torch.zeros(4, device=S.DEV, dtype=S.DT)
    names = list(tg_true)
    probs, cws, index = [], [], []
    subs_of = {}
    for isp, halves in enumerate(splits):
        for hh in (0, 1):
            tr = halves[hh]
            sub = dai3.pool_halves(cl[tr])
            subs = [tr[sub == 0], tr[sub == 1]]
            if min(len(subs[0]), len(subs[1])) == 0:
                raise SystemExit("R1 の内側の組が空")
            subs_of[(isp, hh)] = subs
            for nm in names:
                for c in GRID_C:
                    for ss in (0, 1):
                        index.append((isp, hh, nm, c, ss))
                        probs.append(S.balanced_onehot(tg_true[nm], subs[ss], n)); cws.append(c)
    res = S.fit(Z, torch.stack(probs), zeroW, zerob, Cw=cws, Cb=CB_R1)
    if not all(res["finite_p"]):
        raise SystemExit("R1 の内側の解が有限でない")
    inner_grad = float(np.max(res["grad_p"])); inner_stalled = int(np.sum(res["stalled_p"]))
    pr = S.logits(Z, res["W"], res["b"]).argmax(2).cpu().numpy()
    del res, probs
    ag = {}
    for j, (isp, hh, nm, c, ss) in enumerate(index):
        o = subs_of[(isp, hh)][1 - ss]
        ag.setdefault((isp, hh, nm, c), []).append(100 * S.uar(tg_true[nm][o], pr[j][o]))
    chosen, curves = {}, {}
    for isp in range(len(splits)):
        for hh in (0, 1):
            for nm in names:
                both = np.array([ag[(isp, hh, nm, c)] for c in GRID_C]).T          # (内側の 2 つの小さい組, 格子)
                cur = both.mean(0)
                curves[(isp, hh, nm)] = both
                best, bc = -1.0, None
                for c, v in zip(GRID_C, cur):
                    if v > best + 1e-12:
                        best, bc = v, c
                chosen[(isp, nm, hh)] = bc
    keys = list(tg_all)
    G, cws = [], []
    for isp, halves in enumerate(splits):
        for k in keys:
            for hh in (0, 1):
                G.append(S.balanced_onehot(tg_all[k], halves[hh], n)); cws.append(chosen[(isp, k[0], hh)])
    n_main = len(G)
    for k in grid_keys:
        for c in GRID_C:
            for hh in (0, 1):
                G.append(S.balanced_onehot(tg_all[k], splits[0][hh], n)); cws.append(c)
    res = S.fit(Z, torch.stack(G), zeroW, zerob, Cw=cws, Cb=CB_R1)
    lg = S.logits(Z, res["W"], res["b"])
    pred = lg.argmax(2).cpu().numpy()
    prob = torch.softmax(lg[:n_main], 2).cpu().numpy().astype(np.float32)
    del lg
    outs = []
    P1 = 2 * len(keys)
    for isp, halves in enumerate(splits):
        sl = slice(isp * P1, (isp + 1) * P1)
        sub_res = {kk: (v[sl] if hasattr(v, "__len__") and len(v) == len(G) else v) for kk, v in res.items()}
        out = collect(keys, dict(tg_all), pred[sl], sub_res, halves, n, dict(Cb=CB_R1, dim=int(d)), prob[sl])
        for k in keys:
            out[k][2]["C0"], out[k][2]["C1"] = chosen[(isp, k[0], 0)], chosen[(isp, k[0], 1)]
            out[k][2]["inner_grad"], out[k][2]["inner_stalled"] = inner_grad, inner_stalled
        outs.append(out)
    grid = {}
    j = n_main
    for k in grid_keys:
        for ic, c in enumerate(GRID_C):
            full = np.empty(n, np.int8)
            for hh in (0, 1):
                if not res["finite_p"][j]:
                    raise SystemExit("R1 の格子の解が有限でない")
                other = splits[0][1 - hh]
                full[other] = pred[j][other]
                j += 1
            grid[(k, ic)] = full
    return outs, curves, grid


def run_unit(T: str, f: int, r: int, fz1T: dict, out: Path) -> pd.DataFrame | None:
    base = out / "units" / f"{T}_f{f}_r{r}"
    if Path(f"{base}.csv").exists() and Path(f"{base}.npz").exists():
        old = pd.read_csv(f"{base}.csv")
        if (old.commit == RUN["commit"]).all():
            return None                                               # 同じコミットで済んだ単位は飛ばす（再開）
        raise SystemExit(f"{base.name} が別のコミットで既にある（消してから流し直す）")
    t0 = time.time()
    sv = dai3.Saved(OUT1, "eval", T, T, f, r, fz1T)
    u = sv.u
    TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)
    n = u.n
    y = np.asarray(u.ypool, int)
    cl = sv.cl_pool.astype(str)
    h = dai3.pool_halves(cl)
    halves = [np.flatnonzero(h == 0), np.flatnonzero(h == 1)]
    meta = corpus_meta(T).reindex(sv.sid_pool)
    if meta["marker"].isna().any():
        raise SystemExit(f"{base.name}：発話一覧に無い候補がある")
    has_fine = T in ("IEMOCAP", "MOSEI")
    sex = meta["sex"].to_numpy().astype(str)
    st_e = y.astype(str)
    st_cl = np.char.add(np.char.add(np.char.add(np.char.add(st_e, "|"), cl), "|"), sex)     # （感情，塊，性別）．MELD・MOSEI は性別が空
    st_f = np.char.add(np.char.add(st_cl, "|"), fine_strata(meta, y)) if has_fine else st_cl   # #shuffine は #shufcl の中に入れ子
    has_age = T == "AFEW"
    st_a = (np.char.add(np.char.add(st_cl, "|"), tertile_within(meta["age"].to_numpy(float), np.zeros(n, int))) if has_age else st_cl)
    nst = {nm_: pd.Series(x).value_counts() for nm_, x in (("cl", st_cl), ("fine", st_f), ("age", st_a))}
    shared = set(sv.dlg_pool[halves[0]]) & set(sv.dlg_pool[halves[1]]) if T == "MELD" else set()
    nonshared = ~np.isin(sv.dlg_pool, list(shared)) if shared else np.ones(n, bool)
    arr = dict(y=y.astype(np.int8), cl=cl, sid=sv.sid_pool.astype(str), half=h.astype(np.int8), nonshared=nonshared,
               marker=meta["marker"].to_numpy().astype(bool), sex=sex)
    Z, PCA = {}, {}
    for s in STU:
        Z[s], PCA[s] = aligned_features(u.Xp[s])
    for s in STU:
        arr[f"start__{s}"] = S.logits(u.Xp[s], *u.start[s]).argmax(1).cpu().numpy().astype(np.int8)
    rows = []

    def add_rows(K, ro, s, res, split, tgt_frac):
        for (nm, ver), (full, ag, info) in res.items():
            arr[f"K{K}__{split}__{ro}__{s}__{nm}__{ver}"] = full
            tg = TGT[(K, nm, ver)]
            cnt = np.bincount(tg, minlength=4)
            if K == K_MAIN and split == "main" and info.get("prob") is not None:          # 全部の版（gate-code の第 3 版の H1）
                arr[f"prob__K{K}__{split}__{ro}__{s}__{nm}__{ver}"] = info["prob"]         # 版を鍵に入れる（第 5 版．J1）
            info = {k_: v_ for k_, v_ in info.items() if k_ != "prob"}
            rows.append(dict(T=T, fold=f, seed=r, K=K, split=split, readout=ro, student=s, target=nm, version=ver,
                             agree_halfmean=ag, agree_pooled=float(uar_present(*[x.sum(0) for x in counts(tg, full, cl)[1:]])),
                             frac_changed=tgt_frac.get((K, nm, ver), 0.0), min_class=int(cnt.min()),
                             n_strata_cl=int(len(nst["cl"])), share_singleton_cl=float((nst["cl"] == 1).sum() / n),
                             n_strata_fine=int(len(nst["fine"])), share_singleton_fine=float((nst["fine"] == 1).sum() / n),
                             share_singleton_age=(float((nst["age"] == 1).sum() / n) if has_age else np.nan),
                             **(PCA[s] if ro == "R1" else {}), **info))

    TGT, FRAC, R1_TRUE, R1_ALL = {}, {}, {}, {}
    for K in KS:
        tk = TK[K]
        main = "prop_" + fz1T["main"][str(K)]
        nk = np.ones(n, bool); nk[u.kpos[K]] = False
        arr[f"nk__K{K}"] = nk
        for s in STU:
            arr[f"none__K{K}__{s}"] = S.logits(u.Xp[s], *sv.none(s, K)).argmax(1).cpu().numpy().astype(np.int8)
            names = [f"single_{m}" for m in "AVT" if m != s] + [main]
            Q = dai3.pool_teachers(sv, tk, s, K, names)
            for nm in names:
                qy = Q[nm].argmax(1).cpu().numpy().astype(int)
                if (K, nm, "true") in TGT:
                    if not (TGT[(K, nm, "true")] == qy).all():
                        raise SystemExit(f"{base.name}：目標 {nm} が生徒 {s} で作り直すと違う")
                    continue
                TGT[(K, nm, "true")] = qy
                arr[f"maxp__K{K}__{nm}"] = Q[nm].max(1).values.cpu().numpy().astype(np.float16)
                for ver, strata in (("shuf", st_e), ("shufcl", st_cl), ("shuffine", st_f), ("shufage", st_a)):
                    if (ver == "shuffine" and not has_fine) or (ver == "shufage" and not has_age):
                        continue
                    TGT[(K, nm, ver)], FRAC[(K, nm, ver)] = shuffle_within(qy, strata, [RNG_SEED, DD.CORPUS_ID[T], f, r, K, code(nm), code(ver)])
            vers = [v for v in VERS if (v != "shuffine" or has_fine) and (v != "shufage" or has_age)]
            tg0 = {(nm, v): TGT[(K, nm, v)] for nm in names for v in vers}
            add_rows(K, "R0", s, fit_r0_keys(sv, s, K, tg0, halves), "main", FRAC)
            R1_TRUE.update({(K, nm): TGT[(K, nm, "true")] for nm in names})
            R1_ALL.setdefault(s, {}).update({((K, nm), v): TGT[(K, nm, v)] for nm in names for v in vers})
    # R1：K = 5 と 25 の問を生徒ごとに 1 回の束で解く（速さのため．解は問ごとに独立）
    for s in STU:
        tg_all = R1_ALL[s]
        tg_true = {k[0]: R1_TRUE[k[0]] for k in tg_all}
        gk = [k for k in tg_all if k[0][0] == K_MAIN and k[0][1].startswith("single_") and k[1] in ("true", "shufcl", "shufage")]   # R1fixC の格子（AFEW は #shufage も．J5）
        res, curves, grid = fit_r1(Z[s], cl, halves, tg_true, tg_all, gk)
        for (isp, hh, (K, nm)), cur in curves.items():
            arr.setdefault(f"inner__main__{s}__K{K}__{nm}", np.zeros((2, 2, len(GRID_C))))[hh] = cur
        for ((K, nm), ver), ic in grid:
            arr[f"K{K}__grid{ic}__R1__{s}__{nm}__{ver}"] = grid[(((K, nm), ver), ic)]
        for K in KS:
            add_rows(K, "R1", s, {(k[0][1], k[1]): v for k, v in res.items() if k[0][0] == K}, "main", FRAC)
    for (K, nm, ver), tg in TGT.items():
        arr[f"tgt__K{K}__{nm}__{ver}"] = tg.astype(np.int8)
    # r = 0：K = 5，R1，判断 J_m の真と #shuf・#shufcl を，別の分け方 N_ALT 通りでも測る（分け方の揺れ．gate-stats の 1）
    if r == 0:
        alts, j = [], 0
        while len(alts) < N_ALT and j < ALT_TRIES:
            a = alt_halves(cl, RNG_SEED + 1000 * DD.CORPUS_ID[T] + 100 * f + j)
            j += 1
            if same_split(a, h) or any(same_split(a, b) for b in alts):
                continue
            alts.append(a)
        arr["n_alt"] = np.array(len(alts))
        hvs = []
        for ia, a in enumerate(alts):
            hvs.append([np.flatnonzero(a == 0), np.flatnonzero(a == 1)])
            arr[f"althalf__{ia}"] = a.astype(np.int8)
        if hvs:
            for s in STU:
                names = [f"single_{m}" for m in "AVT" if m != s]
                tga = {(nm, v): TGT[(K_MAIN, nm, v)] for nm in names for v in ("true", "shuf", "shufcl")}
                outs, curves, _ = fit_r1_multi(Z[s], cl, hvs, {nm: TGT[(K_MAIN, nm, "true")] for nm in names}, tga)
                for (isp, hh, nm), cur in curves.items():
                    arr.setdefault(f"inner__alt{isp}__{s}__K{K_MAIN}__{nm}", np.zeros((2, 2, len(GRID_C))))[hh] = cur
                for ia, o in enumerate(outs):
                    add_rows(K_MAIN, "R1", s, o, f"alt{ia}", FRAC)
    for kk in [k for k in arr if k.startswith("prob__")]:            # 確率の鍵の数と，argmax が同じ名前の予測と一致すること
        pf = arr[kk].astype(np.float32); pk = arr["K" + kk[len("prob__K"):]].astype(int)
        gap = pf.max(1) - pf[np.arange(len(pk)), pk]                 # 丸めで同点になった発話は許す
        if not (gap <= 1e-3).all():
            raise SystemExit(f"{base.name}：{kk} の確率の argmax が予測と合わない")
    n_prob = len([k for k in arr if k.startswith("prob__")])
    n_prob_exp = 2 * len(STU) * 3 * (3 + int(has_fine) + int(has_age))
    if n_prob != n_prob_exp:
        raise SystemExit(f"{base.name}：確率の鍵の数 {n_prob}（見込み {n_prob_exp}）")
    df = pd.DataFrame(rows)
    df["commit"], df["dirty"], df["sec"] = RUN["commit"], RUN["dirty"], time.time() - t0
    df["dai1_commits"] = ",".join(sorted(sv.commits))
    df["n_shared_dialogues"] = len(shared)
    np.savez_compressed(f"{base}.npz", **arr)
    df.to_csv(f"{base}.csv", index=False)
    return df


def fit_r0_keys(sv, s, K, tg0: dict, halves) -> dict:
    """fit_r0 の鍵を (名前, 版) にしたもの．"""
    keys = list(tg0)
    flat = {f"{k[0]}#{k[1]}": tg0[k] for k in keys}
    res = fit_r0(sv, s, K, flat, halves)
    return {k: res[f"{k[0]}#{k[1]}"] for k in keys}


def check_against_saved(df: pd.DataFrame) -> pd.DataFrame:
    """R0 の真の目標の 2 組の平均の一致を，実験 3 の保存（subset=all）と照合する（K = 5・25．判断 single_m と主の版の疑似ラベル）．"""
    sv_ = pd.read_csv(REPRO_SAVED, low_memory=False)
    sv_ = sv_[sv_.K.isin(KS) & (sv_.subset == "all")]
    x = df[(df.readout == "R0") & (df.version == "true") & (df.split == "main")]
    m = x.merge(sv_[["T", "fold", "seed", "student", "K", "teacher", "agree"]].rename(columns={"teacher": "target"}),
                on=["T", "fold", "seed", "student", "K", "target"], how="left")
    m["abs_diff"] = (m.agree_halfmean - m.agree).abs()
    return m


# ------------------------------------------------------------------ 段 summary

def load_units(out: Path, T: str) -> dict:
    U = {}
    for p in sorted((out / "units").glob(f"{T}_f*_r*.npz")):
        f, r = (int(x[1:]) for x in p.stem.split("_")[1:3])
        U[(f, r)] = dict(np.load(p, allow_pickle=False))
    return U


def mask_of(a: dict, subset: str, K: int | None = None) -> np.ndarray:
    m = np.ones(len(a["y"]), bool)
    if subset == "nonshared":
        m &= a["nonshared"].astype(bool)
    elif subset == "nomarker":
        m &= ~a["marker"].astype(bool)
    if K is not None:
        m &= a[f"nk__K{K}"].astype(bool)
    return m


def cell(U: dict, pred_key, tgt_key, subset="all", K_nk=None) -> dict:
    out = {}
    for k, a in U.items():
        m = mask_of(a, subset, K_nk)
        out[k] = counts(a[tgt_key][m], a[pred_key][m], a["cl"][m])
    return out


def Rcell(U, K, ro, s, nm, ver, subset="all", split="main"):
    return cell(U, f"K{K}__{split}__{ro}__{s}__{nm}__{ver}", f"tgt__K{K}__{nm}__{ver}", subset)


def split_values(U: dict, K: int, terms: list, subset: str = "all") -> dict | None:
    """r = 0 の単位で，元の分け方と別の分け方ごとの値 Σ coef · UAR．terms：[(読み出し, 生徒, 名前, 版, 係数)]．
    別の分け方の予測が無い項（R0，K = 25，#shuffine，J_prop）を含むなら None（分け方の揺れを足さない）．
    部分集合 nonshared の割り当ては元の分け方のものを使う（別の分け方では共有の対話が変わるので近似．設計 4 節）．"""
    sv = {}
    for (f, r), a in U.items():
        if r != 0 or "n_alt" not in a:
            continue
        vals = []
        m = mask_of(a, subset)
        for sp in ["main"] + [f"alt{i}" for i in range(int(a["n_alt"]))]:
            v = 0.0
            for ro, s, nm, ver, co in terms:
                pk = f"K{K}__{sp}__{ro}__{s}__{nm}__{ver}"
                if pk not in a:
                    return None
                c = counts(a[f"tgt__K{K}__{nm}__{ver}"][m], a[pk][m], a["cl"][m])
                v += co * float(uar_present(c[1].sum(0), c[2].sum(0)))
            vals.append(v)
        sv[f] = vals
    return sv


PRED_DEF = {   # 予言（設計 5 節）：[(生徒, 判断の生徒, 版, 係数)]．版 cl は #shufcl を対照にした差（主），e は #shuf（副）
    "P1": [("A", "T", "true", 1), ("V", "T", "true", -1)],
    "P2": [("A", "V", "true", 1), ("T", "V", "true", -1)],
    "P3": [("A", "T", "true", 1), ("A", "T", "shufcl", -1)],
    "P3b": [("A", "V", "true", 1), ("A", "V", "shufcl", -1)],          # P4b の前提（第 4 版．gate-define の第 3 版の 1）
    "P4a": [("A", "T", "true", 1), ("A", "T", "shufcl", -1), ("V", "T", "true", -1), ("V", "T", "shufcl", 1)],
    "P4b": [("A", "V", "true", 1), ("A", "V", "shufcl", -1), ("T", "V", "true", -1), ("T", "V", "shufcl", 1)],
}
PRIMARY_3 = ("P1", "P3", "P4a")         # 言葉の判断 J_T を使う予言：主の数えは AFEW を除く 3 コーパスで 2 以上（gate-stats の 2）


CTRL_VER = {"cl": "shufcl", "e": "shuf", "fine": "shuffine", "age": "shufage"}
PRECOND = {"P4a": "P3", "P4b": "P3b"}   # 差の予言は，同じコーパスで声の D そのものが上回るときだけ「上回る」に数える
PRED_CTL = {   # 記述：対照どうしの生徒の間の差（gate-compare の 4）
    "P1ctl": [("A", "T", "shufcl", 1), ("V", "T", "shufcl", -1)],
    "P2ctl": [("A", "V", "shufcl", 1), ("T", "V", "shufcl", -1)],
}


def pred_terms(pn: str, ro: str, ctrl: str) -> list:
    out = []
    for s, m, ver, co in {**PRED_DEF, **PRED_CTL}[pn]:
        v = ver if ver in ("true", "shufcl") and (ver == "true" or pn in PRED_CTL) else CTRL_VER[ctrl]
        out.append((ro, s, f"single_{m}", v, co))
    return out


def fixed_c_index(csvs: pd.DataFrame) -> tuple:
    """C′ を固定した感度の版で使う格子の点：そのコーパスの R1・K = 5・元の分け方・判断 J_m の真の目標の行で，組 0 で学んだときに
    選ばれた C′（C0）の最頻値 i0 と，組 1 の C1 の最頻値 i1（同数は小さい C′）．組 1 の発話は組 0 で学んだ識別器が予測するので i0 を，
    組 0 の発話には i1 を当てる（同じ単位の中では予測する組の判断が C′ に入らない．最頻値は全単位から取るので，別の分割で学ぶ側に
    入った同じ発話の判断はわずかに入る．設計 3 節）．"""
    x = csvs[(csvs.readout == "R1") & (csvs.split == "main") & (csvs.K == K_MAIN) & csvs.target.str.startswith("single_") & (csvs.version == "true")]
    out = []
    for col in ("C0", "C1"):
        vc = x[col].value_counts()
        top = vc[vc == vc.max()].index.min()
        out.append(int(np.argmin(np.abs(np.log10(np.array(GRID_C)) - np.log10(top)))))
    return tuple(out)


def fixc_cell(U: dict, ic: tuple, s: str, nm: str, ver: str, subset: str) -> dict:
    """R1fixC の数：組 1 の発話は格子の点 ic[0]，組 0 の発話は ic[1] で学んだ交差の予測．"""
    out = {}
    for k, a in U.items():
        pred = np.where(a["half"] == 1, a[f"K{K_MAIN}__grid{ic[0]}__R1__{s}__{nm}__{ver}"], a[f"K{K_MAIN}__grid{ic[1]}__R1__{s}__{nm}__{ver}"])
        m = mask_of(a, subset)
        out[k] = counts(a[f"tgt__K{K_MAIN}__{nm}__{ver}"][m], pred[m], a["cl"][m])
    return out


def stage_summary(out: Path, targets: list) -> None:
    sdir = out / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    fz1, _ = dai3.load_fz1(OUT1)
    desc, preds, cond, conv, robust, ncand, plateau = [], [], [], [], [], {}, []
    for T in targets:
        U = load_units(out, T)
        need = {(f, r) for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))}
        if RUN.get("partial"):
            need = set(U)
        if set(U) != need:
            raise SystemExit(f"{T}：単位の格子がそろっていない（{len(U)} / {len(need)}）")
        csvs = pd.concat([pd.read_csv(out / "units" / f"{T}_f{f}_r{r}.csv") for f, r in sorted(U)])
        if not RUN.get("partial") and (csvs.commit.nunique() != 1 or csvs.dirty.astype(str).ne("False").any()):
            raise SystemExit(f"{T}：単位のコミットが 1 つでないか，汚れた作業木の単位がある")
        conv.append(csvs)
        for (f, r), a in U.items():                                   # 行の数のそろい（gate-code の G6）
            nv = 3 + int(T in ("IEMOCAP", "MOSEI")) + int(T == "AFEW")
            n_alt = int(a["n_alt"]) if "n_alt" in a else 0
            if r == 0 and n_alt == 0:
                raise SystemExit(f"{T}_f{f}_r0：別の分け方が 1 つも作れていない")
            exp_rows = len(KS) * len(STU) * 3 * nv * 2 + n_alt * len(STU) * 2 * 3
            got = int(((csvs.fold == f) & (csvs.seed == r)).sum())
            if got != exp_rows or (r == 0) != ("n_alt" in a):
                raise SystemExit(f"{T}_f{f}_r{r}：行の数 {got}（見込み {exp_rows}）か n_alt の有無が合わない")
        ic_fix = fixed_c_index(csvs)
        ncand[T] = dict(n_candidates=[int(len(a["y"])) for a in U.values()], n_clusters=[int(len(np.unique(a["cl"]))) for a in U.values()],
                        fixC=[GRID_C[ic_fix[0]], GRID_C[ic_fix[1]]])
        for (f, r), a in U.items():                                   # 下端の C′ が「罰を強めた極限に届いた」だけかの印（内側の曲線の下端の 2 点が同じ）
            for kk in [k for k in a if k.startswith("inner__main__")]:
                cur = a[kk].mean(1)
                for hh in (0, 1):
                    j = int(np.argmax(cur[hh] > cur[hh].max() - 1e-12))
                    plateau.append(dict(T=T, key=kk, fold=f, seed=r, half=hh, chosen=GRID_C[j],
                                        low_plateau=bool(j == 0 and abs(cur[hh][0] - cur[hh][1]) < 1e-9)))
        g_eff, _ = ST.g_eff_of(T)
        rng = np.random.default_rng([RNG_SEED, DD.CORPUS_ID[T]])
        subsets = ["all", "nonshared"] if T == "MELD" else ["all"]
        for K in KS:
            main = "prop_" + fz1[T]["main"][str(K)]
            # 記述：R，対照，差（読み出し 2 つ）．学び直す前の一致（出発点の判定）
            for ro in ("R1", "R0"):
                for s in STU:
                    for nm in [f"single_{m}" for m in "AVT" if m != s] + [main]:
                        for subset in subsets:
                            cR = Rcell(U, K, ro, s, nm, "true", subset)
                            q = [("R", [cR], [1])]
                            for ver in ("shuf", "shufcl", "shuffine", "shufage"):
                                if f"tgt__K{K}__{nm}__{ver}" not in next(iter(U.values())):
                                    continue
                                cS = Rcell(U, K, ro, s, nm, ver, subset)
                                q += [(f"R_{ver}", [cS], [1]), (f"D_{ver}", [cR, cS], [1, -1])]
                            for qn, cells_, co in q:
                                e = combo_stats(cells_, co, g_eff, rng)
                                desc.append(dict(T=T, K=K, readout=ro, student=s, target=nm, subset=subset, quantity=qn) | e |
                                            dict(verdict=verdict(e["lo"], e["hi"])))
            for s in STU:
                for nm in [f"single_{m}" for m in "AVT" if m != s] + [main]:
                    cs = cell(U, f"start__{s}", f"tgt__K{K}__{nm}__true")
                    e = combo_stats([cs], [1], g_eff, rng)
                    desc.append(dict(T=T, K=K, readout="start", student=s, target=nm, subset="all", quantity="R") | e |
                                dict(verdict=verdict(e["lo"], e["hi"])))
            # 予言（主：K = 5，R1，#shufcl，部分集合 all．副：R0，K = 25，#shuf，#shuffine，nonshared，nomarker，C′ の固定）
            ctrls = ("cl", "e") + (("fine",) if T in ("IEMOCAP", "MOSEI") else ()) + (("age",) if T == "AFEW" else ())
            ro_list = ("R1", "R0") + ((f"R1fixC",) if K == K_MAIN else ())
            for ro in ro_list:
                for ctrl in ctrls:
                    for subset in subsets + ["nomarker"]:
                        if ro == "R1fixC" and (ctrl not in ("cl", "age") or subset not in subsets):
                            continue
                        for pn in list(PRED_DEF) + list(PRED_CTL):
                            if ctrl != "cl" and pn in ("P1", "P2", "P1ctl", "P2ctl"):
                                continue
                            if ro == "R1fixC":
                                terms = pred_terms(pn, "R1", ctrl)
                                cells_ = [fixc_cell(U, ic_fix, s, nm, ver, subset) for _, s, nm, ver, co in terms]
                                spv = None
                            else:
                                terms = pred_terms(pn, ro, ctrl)
                                cells_ = [Rcell(U, K, ro_, s, nm, ver, subset) for ro_, s, nm, ver, co in terms]
                                spv = split_values(U, K, terms, subset) if (K == K_MAIN and ro == "R1") else None
                            e = combo_stats(cells_, [t[-1] for t in terms], g_eff, rng, split_vals=spv)
                            preds.append(dict(T=T, K=K, readout=ro, control=ctrl, subset=subset, prediction=pn,
                                              fixC0=(GRID_C[ic_fix[0]] if ro == "R1fixC" else np.nan),
                                              fixC1=(GRID_C[ic_fix[1]] if ro == "R1fixC" else np.nan)) | e |
                                         dict(verdict=verdict(e["lo"], e["hi"]), verdict_ns=verdict(e["lo_ns"], e["hi_ns"]),
                                              primary=bool(K == K_MAIN and ro == "R1" and ctrl == "cl" and subset == "all" and pn in PRED_DEF and pn != "P3b")))
            # 作りの違いの差の区間（判定が主と副で違うとき，原因を名指しするのは差の区間が 0 を含まないときだけ．gate-stats の第 3 版の 4）
            if K == K_MAIN:
                pairs = [("R1-R0", "R1", "cl", "R0", "cl"), ("R1-R1fixC", "R1", "cl", "R1fixC", "cl")]
                if T in ("IEMOCAP", "MOSEI"):
                    pairs.append(("cl-fine", "R1", "cl", "R1", "fine"))
                if T == "AFEW":
                    pairs.append(("cl-age", "R1", "cl", "R1", "age"))
                for pn in PRED_DEF:
                    for lab, ra, ca, rb, cb in pairs:
                        if pn in ("P1", "P2") and lab in ("cl-fine", "cl-age"):
                            continue
                        cells_, coefs = [], []
                        for ro_, ct_, sg in ((ra, ca, 1), (rb, cb, -1)):
                            for _, s, nm, ver, co in pred_terms(pn, "R1" if ro_ == "R1fixC" else ro_, ct_):
                                cells_.append(fixc_cell(U, ic_fix, s, nm, ver, "all") if ro_ == "R1fixC" else Rcell(U, K, ro_, s, nm, ver, "all"))
                                coefs.append(sg * co)
                        e = combo_stats(cells_, coefs, g_eff, rng)
                        robust.append(dict(T=T, K=K, prediction=pn, contrast=lab) | e | dict(verdict=verdict(e["lo"], e["hi"])))
            # 記述：3 条件の表の評価の回の版（R0 は生徒の作り，R1 はそろえた読み出し）と判断の当たり方
            for s in STU:
                cN = {k: counts(a["y"][a[f"nk__K{K}"]], a[f"none__K{K}__{s}"][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]]) for k, a in U.items()}
                cJ = {k: counts(a["y"][a[f"nk__K{K}"]], a[f"tgt__K{K}__{main}__true"][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]]) for k, a in U.items()}
                q = [("uar_none", [cN], [1]), ("uar_pseudo", [cJ], [1]), ("delta", [cJ, cN], [1, -1])]
                for ro in ("R0", "R1"):
                    cP = {k: counts(a["y"][a[f"nk__K{K}"]], a[f"K{K}__main__{ro}__{s}__{main}__true"][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]])
                          for k, a in U.items()}
                    q += [(f"uar_imitation_{ro}", [cP], [1]), (f"imitable_{ro}", [cP, cN], [1, -1])]
                for qn, cells_, co in q:
                    e = combo_stats(cells_, co, g_eff, rng)
                    cond.append(dict(T=T, K=K, student=s, quantity=qn) | e | dict(verdict=verdict(e["lo"], e["hi"])))
            for m in "AVT":
                nm = f"single_{m}"
                cJm = {k: counts(a["y"][a[f"nk__K{K}"]], a[f"tgt__K{K}__{nm}__true"][a[f"nk__K{K}"]], a["cl"][a[f"nk__K{K}"]]) for k, a in U.items()}
                e = combo_stats([cJm], [1], g_eff, rng)
                cond.append(dict(T=T, K=K, student=m, quantity="uar_judge") | e | dict(verdict=verdict(e["lo"], e["hi"])))
        dai1.Corpus.cache.pop(T, None)
    D = pd.DataFrame(desc); P = pd.DataFrame(preds); C = pd.DataFrame(cond); CV = pd.concat(conv)
    crit = []
    PP = P[P.prediction.isin(list(PRED_DEF))]

    def raw(K, ro, ctrl, subset, pn, T, col):
        x = PP[(PP.K == K) & (PP.readout == ro) & (PP.control == ctrl) & (PP.subset == subset) & (PP.prediction == pn) & (PP["T"] == T)]
        if len(x) != 1:
            raise SystemExit(f"数えに要る行が無い：K={K} {ro} {ctrl} {subset} {pn} {T}（{len(x)} 行）")
        return bool(x[col].iloc[0] == "上回る")

    def counted(K, ro, ctrl, subset, pn, T, col, notes):
        """数えで「上回る」とする判定（第 5 版）：素の判定に，MELD の共有の対話の規則，AFEW の #shufage の規則，
        差の予言の前提（前提にも同じ規則を当てた判定）を当てる．規則が使う行が無ければ止める（黙って飛ばさない）．"""
        if not raw(K, ro, ctrl, subset, pn, T, col):
            return False
        if T == "MELD" and subset == "all" and not raw(K, ro, ctrl, "nonshared", pn, "MELD", col):
            notes.append(f"{pn}:MELD:共有の対話を除くと上回らない"); return False
        if T == "AFEW" and ctrl == "cl" and pn in ("P3", "P3b", "P4a", "P4b") and not raw(K, ro, "age", subset, pn, "AFEW", col):
            notes.append(f"{pn}:AFEW:#shufage で上回らない"); return False
        if pn in PRECOND and not counted(K, ro, ctrl, subset, PRECOND[pn], T, col, notes):
            notes.append(f"{pn}:{T}:前提 {PRECOND[pn]} が数えで上回らない"); return False
        return True

    for (K, ro, ctrl), x in PP[PP.subset == "all"].groupby(["K", "readout", "control"]):     # 数えは候補の全体と記号の無い版
        for subset in ("all", "nomarker"):
            for pn in PRED_DEF:
                xx = PP[(PP.K == K) & (PP.readout == ro) & (PP.control == ctrl) & (PP.subset == subset) & (PP.prediction == pn)]
                if not len(xx):
                    continue
                row = dict(K=K, readout=ro, control=ctrl, subset=subset, prediction=pn,
                           primary=bool(K == K_MAIN and ro == "R1" and ctrl == "cl" and subset == "all" and pn != "P3b"))
                for col, sfx in (("verdict", ""), ("verdict_ns", "_ns")):
                    notes = []
                    ab = {T for T in xx["T"] if counted(K, ro, ctrl, subset, pn, T, col, notes)}
                    if pn in PRIMARY_3:
                        pool = [t for t in xx["T"] if t != "AFEW"]; need = 2
                    else:
                        pool = list(xx["T"]); need = 3
                    if ctrl in ("fine", "age") or pn == "P3b":                 # 2 コーパス以下の対照と前提の量は記述（数えの規則を当てない）
                        need = np.nan
                    n_ab = len([t for t in pool if t in ab])
                    row |= {f"n_above{sfx}": n_ab, f"n_below{sfx}": int((xx[xx["T"].isin(pool)][col] == "下回る").sum()),
                            f"hit{sfx}": (bool(n_ab >= need) if np.isfinite(need) else None), f"above{sfx}": ",".join(sorted(t for t in pool if t in ab)),
                            f"n_above_all4{sfx}": len(ab), f"above_all4{sfx}": ",".join(sorted(ab)), f"notes{sfx}": ";".join(notes)}
                row |= dict(n_pool=len(pool), need=need)
                crit.append(row)
    CR = pd.DataFrame(crit)
    tag = dict(commit=RUN["commit"], dirty=RUN["dirty"])
    for name, df in (("descriptive", D), ("predictions", P), ("criteria", CR), ("conditions", C), ("robustness", pd.DataFrame(robust))):
        df.assign(**tag).to_csv(sdir / f"{name}.csv", index=False)
    conv_s = dict(n_rows=len(CV), n_units=int(CV[["T", "fold", "seed"]].drop_duplicates().shape[0]), grad_max=float(CV.grad.max()),
                  n_stalled=int(CV.stalled.astype(str).eq("True").sum()), min_class_min=int(CV.min_class.min()),
                  frac_changed_by_version={v: [float(CV[CV.version == v].frac_changed.min()), float(CV[CV.version == v].frac_changed.max())]
                                           for v in ("shuf", "shufcl", "shuffine", "shufage") if (CV.version == v).any()},
                  C_chosen_counts={str(k): int(v) for k, v in pd.concat([CV.get("C0", pd.Series(dtype=float)),
                                                                          CV.get("C1", pd.Series(dtype=float))]).value_counts().items()})
    conv_s["candidates_clusters_fixC"] = ncand
    PL = pd.DataFrame(plateau)
    conv_s["low_edge"] = {T: dict(share_chosen_low=float((x.chosen == GRID_C[0]).mean()), share_low_plateau=float(x.low_plateau.mean()))
                          for T, x in PL.groupby("T")} if len(PL) else {}
    conv_s["inner_grad_max"] = float(CV.inner_grad.max()) if "inner_grad" in CV else np.nan
    conv_s["inner_stalled_max"] = int(CV.inner_stalled.max()) if "inner_stalled" in CV else 0
    json.dump(conv_s, open(sdir / "convergence.json", "w"), ensure_ascii=False, indent=1)
    # 仮定の印（設計 8 節）：C′ が格子の端に選ばれた割合，入れ替えで目標が変わった割合の平均
    asm = []
    if len(PL):                                                       # 端の印（第 5 版．8 節の基準 flag と，第 3 版の基準 flag_old を並べる）
        PL[["_", "_2", "student", "Kc", "target"]] = PL.key.str.split("__", expand=True)
        for (T, Kc, st_, tg), x in PL.groupby(["T", "Kc", "student", "target"]):
            lo = float((x.chosen == GRID_C[0]).mean()); hi = float((x.chosen == GRID_C[-1]).mean())
            lo_np = float(((x.chosen == GRID_C[0]) & ~x.low_plateau).mean())
            asm.append(dict(kind="C_edge", T=T, K=int(Kc[1:]), split="main", student=st_, target=tg, share_low=lo, share_high=hi,
                            share_low_nonplateau=lo_np, share_low_plateau=lo - lo_np,
                            flag=bool(max(hi, lo_np) > EDGE_SHARE), flag_old=bool(max(lo, hi) > EDGE_SHARE)))
    sh = CV[(CV.version != "true") & (CV.readout == "R0")].drop_duplicates(["T", "fold", "seed", "K", "target", "version"])
    for (T, K, tg, ver), x in sh.groupby(["T", "K", "target", "version"]):
        asm.append(dict(kind="frac_changed", T=T, K=K, target=tg, version=ver, frac_mean=float(x.frac_changed.mean()),
                        frac_min=float(x.frac_changed.min()), flag=bool(x.frac_changed.mean() < FRAC_LOW)))
    pd.DataFrame(asm).assign(**tag).to_csv(sdir / "assumptions.csv", index=False)
    print(json.dumps({k: conv_s[k] for k in ("n_rows", "n_units", "grad_max", "n_stalled")}, ensure_ascii=False))


# ------------------------------------------------------------------ 段 selftest

def selftest() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400); cl = np.array([f"g{i % 13}" for i in range(400)])
    c = counts(y, y, cl)
    assert abs(uar_present(c[1].sum(0), c[2].sum(0)) - 100) < 1e-12
    pr = rng.integers(0, 4, 400)
    c2 = counts(y, pr, cl)
    assert abs(uar_present(c2[1].sum(0), c2[2].sum(0)) - ST.uar_full(c2)) < 1e-9
    y3 = np.where(y == 3, 0, y)
    c3 = counts(y3, pr, cl)
    assert np.isfinite(uar_present(c3[1].sum(0), c3[2].sum(0)))
    cellA = {(f, r): counts(rng.integers(0, 4, 300), rng.integers(0, 4, 300), np.array([f"g{i % 11}" for i in range(300)]))
             for f in range(5) for r in range(3)}
    for g in (8.0, 30.0):
        e = combo_stats([cellA, cellA], [1, -1], g, np.random.default_rng(1))
        assert abs(e["effect"]) < 1e-12 and abs(e["lo"]) < 1e-9 and abs(e["hi"]) < 1e-9, e
    e = combo_stats([cellA], [1], 8.0, np.random.default_rng(1))
    ref = np.mean([np.mean([ST.uar_full(cellA[(f, r)]) for r in range(3)]) for f in range(5)])
    assert abs(e["effect"] - ref) < 1e-9
    cellB = {k: counts(rng.integers(0, 4, 300), rng.integers(0, 4, 300), np.array([f"g{i % 11}" for i in range(300)])) for k in cellA}
    a_ = combo_stats([cellA, cellB], [1, -1], 8.0, np.random.default_rng(2))
    b_ = ST.effect_stats(cellA, cellB, 8.0, np.random.default_rng(2))
    assert abs(a_["effect"] - b_["effect"]) < 1e-9 and abs(a_["se"] - b_["se"]) < 1e-6, (a_, b_)
    # 分け方の揺れを足すと SE は広がる
    e1 = combo_stats([cellA], [1], 8.0, np.random.default_rng(1), split_vals={f: [50.0, 52.0, 48.0] for f in range(5)})
    assert e1["se"] > e["se"]
    qy = rng.integers(0, 4, 500); yy = rng.integers(0, 4, 500)
    qs, fr = shuffle_within(qy, yy.astype(str), [1, 2, 3])
    for e_ in range(4):
        assert (np.bincount(qy[yy == e_], minlength=4) == np.bincount(qs[yy == e_], minlength=4)).all()
    assert 0 < fr < 1
    # 別の分け方は元と違い，2 組とも空でない
    cl2 = np.array([f"g{i % 9}" for i in range(300)])
    a2 = alt_halves(cl2, 5)
    assert 0 < a2.sum() < len(a2)
    # 合成の対照（GPU の学び．R1 の作り）：特徴が感情だけを持つとき D ≈ 0，目標の発話ごとの違いを持つとき D > 0
    n, d = 1200, 24
    yv = rng.integers(0, 4, n); clv = np.array([f"c{i % 40}" for i in range(n)])
    noise = rng.integers(0, 4, n); flip = rng.random(n) < 0.4
    J = np.where(flip, noise, yv)                                      # 判断：感情 ＋ 発話ごとのずれ
    E = np.eye(4)
    hv = dai3.pool_halves(clv)
    halves = [np.flatnonzero(hv == 0), np.flatnonzero(hv == 1)]
    for case, X_ in (("neg", np.c_[3 * E[yv], rng.normal(size=(n, d - 4))]), ("pos", np.c_[3 * E[yv], 3 * E[J], rng.normal(size=(n, d - 8))])):
        Zt, _ = aligned_features(torch.as_tensor(X_))
        Js, _ = shuffle_within(J, yv.astype(str), [9, 9])
        res, _, _ = fit_r1(Zt, clv, halves, {"j": J}, {("j", "true"): J, ("j", "shuf"): Js})
        D = res[("j", "true")][1] - res[("j", "shuf")][1]
        if case == "neg":
            assert abs(D) < 3, D
        else:
            assert D > 15, D
    # 合成の対照（#shufcl）：判断が（感情と，塊をまたいで共通な属性 g（性別の代わり））だけで決まり，特徴も g を持つとき，
    # 感情だけの入れ替えでは D > 0，（感情，塊，g）の中の入れ替えでは D ≈ 0
    gv = rng.integers(0, 2, n)
    Jg = np.where(rng.random(n) < 0.7, np.where(gv == 1, 3, 0), yv)
    Xg = np.c_[3 * E[yv], 3 * gv[:, None], rng.normal(size=(n, d - 5))]
    Zt, _ = aligned_features(torch.as_tensor(Xg))
    Je, _ = shuffle_within(Jg, yv.astype(str), [7, 7])
    stg = np.char.add(np.char.add(np.char.add(yv.astype(str), "|"), clv), np.char.add("|", gv.astype(str)))
    Jc, fr_ = shuffle_within(Jg, stg, [7, 8])
    res, _, _ = fit_r1(Zt, clv, halves, {"j": Jg}, {("j", "true"): Jg, ("j", "shuf"): Je, ("j", "shufcl"): Jc})
    De = res[("j", "true")][1] - res[("j", "shuf")][1]
    Dc = res[("j", "true")][1] - res[("j", "shufcl")][1]
    assert De > 3 and abs(Dc) < 1.5 and fr_ > 0.15, (De, Dc, fr_)
    print("selftest：12 項目すべて通った")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "units", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="試しの置き場だけ：格子のそろいと汚れの確かめを省いて summary を作る")
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
    if args.stage == "selftest":
        selftest()
        return
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"] = commit, dirty
    RUN["partial"] = bool(args.allow_partial)
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある（試走は一時の置き場で --allow-dirty）")
    if prod and not dai1.committed_clean(DESIGN):
        sys.exit("設計の文書がコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or args.allow_partial):
        sys.exit("本番の置き場では分割と種を絞らない")
    dai1.ENC["enc"] = DD.ENC1
    fz1, fz1_sha = dai3.load_fz1(OUT1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "units").mkdir(exist_ok=True)
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                host=platform.node(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""),
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                scripts=script_shas(), design_sha256=sha(DESIGN), edge_share=EDGE_SHARE, frac_low=FRAC_LOW, marker_re=MARKER_RE,
                alt_tries=ALT_TRIES, min_n=MIN_N, tol_repro=TOL_REPRO, mem_frac=MEM_FRAC, dai1_frozen_sha256=fz1_sha, repro_saved_sha256=sha(REPRO_SAVED),
                encoder=dai1.ENC["enc"], KS=KS, grid_c=GRID_C, cb_r1=CB_R1, dim_r1=DIM_R1, n_alt=N_ALT, rng_seed=RNG_SEED, n_boot=N_BOOT)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "units":
        allrows = []
        for T in args.targets:
            fzT = fz1[T]
            seeds = list(range(int(fzT["n_seeds_eval"]))) if args.seeds is None else args.seeds
            folds = list(range(DD.N_FOLDS)) if args.folds is None else args.folds
            for f in folds:
                for r in seeds:
                    df = run_unit(T, f, r, fzT, out)
                    if df is None:
                        print(f"{T}_f{f}_r{r}：済み（飛ばす）", flush=True)
                        continue
                    chk = check_against_saved(df)
                    bad = chk[chk.abs_diff.isna() | (chk.abs_diff > TOL_REPRO)]
                    print(f"{T}_f{f}_r{r}：{df.sec.iloc[0]:.0f} 秒．保存との照合 {len(chk)} 行，最大差 {chk.abs_diff.max():.2e}，許しの外 {len(bad)}", flush=True)
                    if len(bad):
                        bad.to_csv(out / f"check_fail_{T}_f{f}_r{r}.csv", index=False)
                        sys.exit(f"{T}_f{f}_r{r}：保存の再現できる度合い（実験 3）と {TOL_REPRO} を超えて違う行がある")
                    allrows.append(chk)
            dai1.Corpus.cache.pop(T, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if allrows:
            pd.concat(allrows).to_csv(out / f"check_saved_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
        if prod and set(args.targets) == set(DD.CORPORA):
            (out / "units.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    else:
        stage_summary(out, args.targets)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)
        if prod and set(args.targets) == set(DD.CORPORA):
            (out / "summary.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
