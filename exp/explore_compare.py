"""【探索的】E31：論文 2 の比較実験の解析（設定の選び方 3 規則，主な比較，予言 R1〜R11 のうち段 1 で判定できるもの）。新しい学習なし。

規則は `docs/exploratory-analyses.md` E31「段 1 の解析の細部」。入力は `results/compare_<kind>/*.csv`，
地図（`results/map_<kind>/`：ターゲットのみ・ソースのみ・合併の C の交差検証，`results/fixc_map/<kind>/`：C を固定した合併）。

    ./.venv/bin/python exp/explore_compare.py [--kinds …]

出力：`results/explore_relation/e31_pairs.csv`（組 × K × 手法と選び方の UAR．点），`e31_tests.csv`（予言の判定），
`e31_main.csv`（主な比較），`e31_summary.csv`（種類 × K の平均）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC  # noqa: E402

KINDS = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video",
         "text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
TEXT = KINDS[4:]
K_ORDER = ["1", "5", "10", "25", "50", "all"]
REPS = list(range(10))
OUT = ROOT / "results" / "explore_relation"
DEF_CP, DEF_CB, DEF_ALPHA = 0.01, "8", 0.5
CB_ORDER = {"8": 0, "2": 1, "free": 2}


def corpus(g: str) -> str:
    return GROUP_SPEC[g][0]


def same(a: str, b: str) -> bool:
    return corpus(a) == corpus(b)


# ---------------------------------------------------------------- 読み込み

KS_USE: list[str] | None = None          # --ks で絞る（途中の確認用）


def load_compare(kind: str, reps=REPS) -> pd.DataFrame:
    fs = sorted((ROOT / "results" / f"compare_{kind}").glob("*.csv"))
    d = pd.concat([pd.read_csv(f, dtype={"K": str, "cb": str, "variant": str}) for f in fs], ignore_index=True)
    d["cb"], d["variant"] = d.cb.fillna(""), d.variant.fillna("")
    d["uar"], d["cv_uar"] = 100 * d.uar, 100 * d.cv_uar
    d = d[d.rep.isin(reps)]
    return d if KS_USE is None else d[d.K.isin(KS_USE)]


def load_map(kind: str, targets, reps=REPS) -> pd.DataFrame:
    out = []
    for t in targets:
        for path, tag in ((ROOT / "results" / f"map_{kind}" / f"{t}.csv", "cv"), (ROOT / "results" / "fixc_map" / kind / f"{t}.csv", "fix")):
            d = pd.read_csv(path, comment="#", dtype={"K": str, "source": str}, usecols=["target", "source", "method", "K", "rep", "uar"])
            d["K"], d["source"], d["src"] = d.K.fillna(""), d.source.fillna(""), tag
            out.append(d[d.rep.isin(reps)])
    m = pd.concat(out, ignore_index=True)
    m["uar"] = 100 * m.uar
    return m


# ---------------------------------------------------------------- 選び方

class Family:
    """手法の族：行の絞り込み，設定の列，既定，同点の順。"""

    def __init__(self, name, method, keys, cond, default, tie_key):
        self.name, self.method, self.keys, self.cond, self.default, self.tie_key = name, method, keys, cond, default, tie_key

    def rows(self, d):
        x = d[(d.method == self.method)]
        x = x[self.cond(x)].copy()
        x["setting"] = list(zip(*[x[k] for k in self.keys])) if len(self.keys) > 1 else list(x[self.keys[0]])
        return x


def ht_tie(s):
    """同点の順：|log10(C′/0.01)|，C′，C_b（8，2，自由）。s は C′ か（C′, C_b）。"""
    cp, cb = s if isinstance(s, tuple) else (s, DEF_CB)
    return (round(abs(np.log10(cp / DEF_CP)), 9), cp, CB_ORDER[cb])


FAMILIES = [
    Family("P1", "HT", ["cp"], lambda x: (x.cen == 0) & (x.cb == "free"), DEF_CP, ht_tie),
    Family("P2", "HT", ["cp"], lambda x: (x.cen == 1) & (x.cb == "free"), DEF_CP, ht_tie),
    Family("P3", "HT", ["cp", "cb"], lambda x: (x.cen == 1) & x.cb.isin(["2", "8"]), (DEF_CP, DEF_CB), ht_tie),
    Family("P3c", "HT", ["cp"], lambda x: (x.cen == 1) & (x.cb == "8"), DEF_CP, ht_tie),
    Family("HT24", "HT", ["cen", "cp", "cb"], lambda x: x.cen.isin([0, 1]), (1, DEF_CP, DEF_CB),
           lambda s: ((0 if s[0] == 1 else 1),) + ht_tie((s[1], s[2]))),
    Family("O1", "O1", ["alpha"], lambda x: x.alpha.notna(), DEF_ALPHA, lambda a: (round(abs(a - DEF_ALPHA), 9), a)),
]
P4FAM = Family("P4", "HT_ALL", ["cp", "cb"], lambda x: (x.cen == 1) & x.cb.isin(["2", "8"]), (DEF_CP, DEF_CB), ht_tie)


def pick(scores: dict, fam: Family):
    """scores：設定 → 値。最大の設定（同点は既定，次に fam.tie_key の順）。"""
    best = max(scores.values())
    tied = [s for s, v in scores.items() if abs(v - best) <= 1e-12]
    if fam.default in tied:
        return fam.default
    return sorted(tied, key=fam.tie_key)[0]


def select_family(x: pd.DataFrame, fam: Family) -> pd.DataFrame:
    """1 ソースの族：組 × K ごとに 既定・メタ検証・K 本の交差検証・オラクルの UAR。"""
    pm = x.groupby(["target", "source", "K", "setting"]).uar.mean()
    tab = pm.unstack("setting")
    out = []
    pairs = tab.index.to_frame(index=False)
    for (t, s, K), row in tab.iterrows():
        rec = {"target": t, "source": s, "K": K}
        rec[f"{fam.name}_def"] = row.get(fam.default, np.nan)
        rec[f"{fam.name}_orc"] = row.max()
        # メタ検証：S をほかのターゲット T′ に使った組（T・S のコーパス・T のコーパスを除く）
        tp = pairs[(pairs.source == s) & (pairs.K == K)].target
        tp = [u for u in tp if u != t and not same(u, s) and not same(u, t)]
        if not tp:
            raise ValueError(f"メタ検証に使える T′ が無い: {s} → {t}，K={K}")
        sc = tab.loc[[(u, s, K) for u in tp]].mean(axis=0)
        ch = pick(sc.dropna().to_dict(), fam)
        rec[f"{fam.name}_meta"], rec[f"{fam.name}_meta_set"] = row[ch], str(ch)
        out.append(rec)
    R = pd.DataFrame(out)
    # K 本の交差検証：反復ごとに選ぶ
    cvv = []
    for (t, s, K, r), g in x.groupby(["target", "source", "K", "rep"]):
        sc = g.dropna(subset=["cv_uar"])
        if len(sc):
            ch = pick(dict(zip(sc.setting, sc.cv_uar)), fam)
        else:
            ch = fam.default
        cvv.append({"target": t, "source": s, "K": K, "rep": r, "u": float(g[g.setting.apply(lambda v: v == ch)].uar.iloc[0])})
    C = pd.DataFrame(cvv).groupby(["target", "source", "K"]).u.mean().rename(f"{fam.name}_cv").reset_index()
    return R.merge(C, on=["target", "source", "K"])


def select_p4(d: pd.DataFrame, ht: pd.DataFrame) -> pd.DataFrame:
    """P4：ターゲット × K ごとに 既定・メタ検証（1 ソースの疑似組で代用）・交差検証・オラクル。"""
    x = P4FAM.rows(d)
    tab = x.groupby(["target", "K", "setting"]).uar.mean().unstack("setting")
    h = next(f for f in FAMILIES if f.name == "P3").rows(ht)      # 1 ソースの P3（同じ 8 設定）
    h = h[~h.apply(lambda r: same(r.source, r.target), axis=1)]
    hp = h.groupby(["target", "source", "K", "setting"]).uar.mean().reset_index()
    out = []
    for (t, K), row in tab.iterrows():
        q = hp[(hp.K == K) & ~hp.target.map(lambda u: same(u, t)) & ~hp.source.map(lambda u: same(u, t))]
        sc = q.groupby("setting").uar.mean().to_dict()
        ch = pick(sc, P4FAM)
        out.append({"target": t, "K": K, "P4_def": row.get(P4FAM.default, np.nan), "P4_orc": row.max(),
                    "P4_meta": row[ch], "P4_meta_set": str(ch)})
    R = pd.DataFrame(out)
    cvv = []
    for (t, K, r), g in x.groupby(["target", "K", "rep"]):
        sc = g.dropna(subset=["cv_uar"])
        ch = pick(dict(zip(sc.setting, sc.cv_uar)), P4FAM) if len(sc) else P4FAM.default
        cvv.append({"target": t, "K": K, "u": float(g[g.setting.apply(lambda v: v == ch)].uar.iloc[0])})
    C = pd.DataFrame(cvv).groupby(["target", "K"]).u.mean().rename("P4_cv").reset_index()
    return R.merge(C, on=["target", "K"])


# ---------------------------------------------------------------- 組の表

def pair_table(kind: str) -> pd.DataFrame:
    d = load_compare(kind)
    targets = sorted(d.target.unique())
    mp = load_map(kind, targets)
    T = None
    for fam in FAMILIES:
        S = select_family(fam.rows(d), fam)
        T = S if T is None else T.merge(S, on=["target", "source", "K"], how="outer")
    ht = d[d.method == "HT"]
    for cen in (0, 1):
        for cb in ("8", "free"):
            v = ht[(ht.cen == cen) & (ht.cp == DEF_CP) & (ht.cb == cb)].groupby(["target", "source", "K"]).uar.mean()
            T = T.merge(v.rename(f"HT_c{cen}_{cb}").reset_index(), on=["target", "source", "K"], how="left")
    for cen, name in ((1, "H3"), (2, "H3t")):
        h3 = d[(d.method == "H3") & (d.cen == cen)].groupby(["target", "source", "K"]).uar.mean().rename(name).reset_index()
        T = T.merge(h3, on=["target", "source", "K"], how="left")
    p3t = ht[(ht.cen == 2) & (ht.cp == DEF_CP) & (ht.cb == DEF_CB)].groupby(["target", "source", "K"]).uar.mean()
    T = T.merge(p3t.rename("P3t_def").reset_index(), on=["target", "source", "K"], how="left")
    o2 = d[(d.method == "O2") & (d.variant == "Cs")].groupby(["target", "source", "K"]).uar.mean().rename("O2").reset_index()
    T = T.merge(o2, on=["target", "source", "K"], how="left")
    for variant in ("none", "src_mean", "tgt_mean"):
        v = d[(d.method == "O6") & (d.variant == variant)].groupby(["target", "K"]).uar.mean().rename(f"O6_{variant}").reset_index()
        T = T.merge(v, on=["target", "K"], how="left")
    o7 = d[d.method == "O7"].groupby(["target", "K"]).uar.mean().rename("O7").reset_index()
    T = T.merge(o7, on=["target", "K"], how="left")
    T = T.merge(select_p4(d, ht), on=["target", "K"], how="left")
    # 地図の基準
    b1 = mp[(mp.src == "cv") & (mp.method == "tgt_only")].groupby(["target", "K"]).uar.mean().rename("B1").reset_index()
    b2 = mp[(mp.src == "cv") & (mp.method == "src_only")].groupby(["target", "source"]).uar.mean().rename("B2").reset_index()
    b3 = mp[(mp.src == "cv") & (mp.method == "merged")].groupby(["target", "source", "K"]).uar.mean().rename("B3").reset_index()
    b4 = mp[(mp.src == "fix") & (mp.method == "merged")].groupby(["target", "source", "K"]).uar.mean().rename("B4").reset_index()
    T = T.merge(b1, on=["target", "K"], how="left").merge(b2, on=["target", "source"], how="left")
    T = T.merge(b3, on=["target", "source", "K"], how="left").merge(b4, on=["target", "source", "K"], how="left")
    lam = pd.read_csv(ROOT / "results" / "explore_relation" / "e15_lambda.csv", comment="#", dtype={"K": str})
    lam = lam[(lam.kind == kind) & (lam.K == "5")][["target", "source", "G"]]
    T = T.merge(lam, on=["target", "source"], how="left")
    T.insert(0, "kind", kind)
    T["same"] = [same(a, b) for a, b in zip(T.target, T.source)]
    return T


# ---------------------------------------------------------------- 判定

def wil(x, alternative="two-sided"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return np.nan
    return float(stats.wilcoxon(x, alternative=alternative).pvalue)


def tci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def holm(ps):
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    ok = ~np.isnan(ps)
    idx = np.argsort(np.where(ok, ps, np.inf))
    m, run = ok.sum(), 0.0
    for rank, i in enumerate(idx[:m]):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


def unsafe(P: pd.DataFrame, col: str) -> pd.Series:
    return P[col] < np.maximum(P.B1, P.B2) - 1


def judge(P: pd.DataFrame):
    rows = []
    X = P[~P.same]
    for kind in X.kind.unique():
        A = X[X.kind == kind]
        aud = kind in ("audio_wavlm", "audio_hubert")
        k1, k5, k10, k25 = (A[A.K == K] for K in ("1", "5", "10", "25"))
        # R1
        db = (k1.HT_c1_8 - k1.HT_c0_8).mean()
        df = (k1.HT_c1_free - k1.HT_c0_free).mean()
        rows.append({"kind": kind, "R": "R1", "value": f"縛る {db:+.2f}，自由 {df:+.2f}", "holds": (db >= 1 and df < 0.5) if aud else np.nan})
        # R2
        d5, d10, d1 = (g.P3_def - g.H3 for g in (k5, k10, k1))
        p5, p10 = wil(d5, "greater"), wil(d10, "greater")
        ok = d5.mean() >= 1 and p5 < 0.05 and d10.mean() >= 1 and p10 < 0.05 and abs(d1.mean()) <= 0.5
        rows.append({"kind": kind, "R": "R2", "value": f"K5 {d5.mean():+.2f}(p={p5:.3g}) K10 {d10.mean():+.2f}(p={p10:.3g}) K1 {d1.mean():+.2f}",
                     "holds": ok if aud else np.nan})
        # R3（ターゲットが単位）
        g = k5.groupby("target").agg(P4=("P4_def", "first"), P3m=("P3_def", "mean"), P3b=("P3_def", "max"))
        r3 = (g.P4 - g.P3m).mean()
        rows.append({"kind": kind, "R": "R3", "value": f"P4 − P3 平均 {r3:+.2f}（記述：P4 − 最良 {(g.P4 - g.P3b).mean():+.2f}，n={len(g)}）",
                     "holds": (r3 >= 3) if aud else np.nan})
        # R6 前半
        f25 = (k25.P3_def > k25.B1).mean()
        rows.append({"kind": kind, "R": "R6a", "value": f"K25 で P3 > ターゲットのみ {f25:.2f}（n={len(k25)}）", "holds": (f25 >= 0.7) if aud else np.nan})
        # R7
        L = A[A.K.isin(["1", "5", "10"])]
        mc, om = (L.P3c_meta - L.P3c_cv).mean(), (L.P3c_orc - L.P3c_meta).mean()
        per = "，".join(f"K{K} {(L[L.K == K].P3c_meta - L[L.K == K].P3c_cv).mean():+.2f}/{(L[L.K == K].P3c_orc - L[L.K == K].P3c_meta).mean():.2f}"
                       for K in ("1", "5", "10"))
        rows.append({"kind": kind, "R": "R7", "value": f"メタ − 交差 {mc:+.2f}，オラクル − メタ {om:.2f}（{per}）",
                     "holds": (mc >= 0.5 and om <= 1) if aud else np.nan})
        # R8
        a6, a7 = (k5.P3_def - k5.O6_tgt_mean).mean(), (k5.P3_def - k5.O7).mean()
        rows.append({"kind": kind, "R": "R8", "value": f"P3 − O6 {a6:+.2f}，P3 − O7 {a7:+.2f}", "holds": (a6 >= 2 and a7 >= 0) if aud else np.nan})
        # R10
        u1, u5 = unsafe(k1, "P3_def").mean(), unsafe(k5, "P3_def").mean()
        rows.append({"kind": kind, "R": "R10", "value": f"K1 {u1:.3f}，K5 {u5:.3f}（n={len(k1)}）",
                     "holds": (u1 <= 0.05 and u5 <= 0.075) if aud else (u1 <= 0.2 and u5 <= 0.2)})
        rows += judge_stage2(kind, A, aud)
    rows += judge_r9(X)
    return pd.DataFrame(rows)


def judge_stage2(kind, A, aud):
    """段 2 の予言（R4，R5，R6 の後半，R11）。計算が済んでいないものは飛ばす。"""
    rows = []
    ka = A[A.K == "all"]
    if len(ka) and ka.O1_meta.notna().any():
        gain, bad = (ka.O1_meta - ka.B1).mean(), (ka.O1_meta < ka.B1 - 0.5).mean()
        med = [A[A.K == K].O1_meta_set.astype(float).median() for K in K_ORDER if (A.K == K).any()]
        mono = all(b >= a for a, b in zip(med, med[1:])) and len(med) == len(K_ORDER) and med[-1] > med[0]
        rows.append({"kind": kind, "R": "R4", "value": f"得 {gain:+.2f}，B1 − 0.5 未満 {bad:.2f}，α の中央値 {med}",
                     "holds": (gain >= 0 and bad <= 0.2 and mono) if aud else np.nan})
    k5 = A[A.K == "5"]
    if "G" in k5 and k5.O2.notna().any():
        r = {c: stats.spearmanr(k5.G, k5[c] - k5.B2, nan_policy="omit").statistic for c in ("B4", "O2", "P1_def", "P3_def", "P2_def")}
        rows.append({"kind": kind, "R": "R5", "value": "，".join(f"{c} {v:+.2f}" for c, v in r.items()),
                     "holds": (r["B4"] >= 0.4 and r["O2"] >= 0.4 and r["P1_def"] < 0.2 and r["P3_def"] < 0.2) if aud else np.nan})
    if len(ka) and ka.P3c_meta.notna().any():
        f = (ka.P3c_meta < ka.B1 - 1).mean()
        rows.append({"kind": kind, "R": "R6b", "value": f"全件で P3（メタ）< B1 − 1 の割合 {f:.2f}（n={len(ka)}）",
                     "holds": (f <= 0.3) if aud else np.nan})
    L = A[A.K.isin(["1", "5", "10", "25"])]
    if L.P3t_def.notna().any():
        dd = (L.P3t_def - L.P3_def).mean()
        per = "，".join(f"K{K} {(L[L.K == K].P3t_def - L[L.K == K].P3_def).mean():+.2f}" for K in ("1", "5", "10", "25"))
        rows.append({"kind": kind, "R": "R11", "value": f"推移 − 帰納 {dd:+.2f}（{per}．H3 {(L.H3t - L.H3).mean():+.2f}）",
                     "holds": (abs(dd) < 0.5) if aud else np.nan})
    return rows


def judge_r9(X):
    T = X[X.kind.isin(TEXT)]
    if not len(T):
        return []
    per, good = [], 0
    for kind in TEXT:
        A = T[T.kind == kind]
        if not len(A):
            continue
        v = [(A[A.K == K].HT_c1_8 - A[A.K == K].HT_c0_8).mean() for K in K_ORDER if (A.K == K).any()]
        ok = len(v) == len(K_ORDER) and all(x < 0.5 for x in v)
        good += ok
        per.append(f"{kind} [{', '.join(f'{x:+.2f}' for x in v)}]")
    return [{"kind": "text", "R": "R9", "value": "；".join(per), "holds": good >= 3 if len(per) == 4 else np.nan}]


MAIN_PAIR = [("P3_def", "B3"), ("P3_def", "B4"), ("P3_def", "H3"), ("P3_def", "P1_def"),
             ("P3_meta", "B3"), ("P3_meta", "B4"), ("P3_meta", "H3"), ("P3_meta", "P1_def")]
MAIN_TGT = [("P4_def", "P3_srcmean"), ("P4_def", "P3_srcbest"), ("P4_meta", "P3_srcmean"), ("P4_meta", "P3_srcbest")]


def main_comparisons(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    X = P[~P.same]
    for kind in X.kind.unique():
        for K in sorted(X.K.unique(), key=lambda k: (k == "all", float(k) if k != "all" else 0)):
            A = X[(X.kind == kind) & (X.K == K)]
            G = A.groupby("target").agg(P4_def=("P4_def", "first"), P4_meta=("P4_meta", "first"),
                                        P3_srcmean=("P3_def", "mean"), P3_srcbest=("P3_def", "max"))
            fam = []
            for a, b in MAIN_PAIR:
                dd = (A[a] - A[b]).to_numpy()
                fam.append({"unit": "pair", "a": a, "b": b, "dd": dd, "safe_a": 1 - unsafe(A, a).mean()})
            for a, b in MAIN_TGT:
                fam.append({"unit": "target", "a": a, "b": b, "dd": (G[a] - G[b]).to_numpy(), "safe_a": np.nan})
            ps = [wil(f["dd"]) for f in fam]
            for f, p, ph in zip(fam, ps, holm(ps)):
                lo, hi = tci(f["dd"])
                rows.append({"kind": kind, "K": K, "unit": f["unit"], "a": f["a"], "b": f["b"], "n": int(np.sum(~np.isnan(f["dd"]))),
                             "diff": np.nanmean(f["dd"]), "lo": lo, "hi": hi, "wins": int(np.sum(f["dd"] > 0)),
                             "p": p, "p_holm": ph, "safe_a": f["safe_a"]})
    return pd.DataFrame(rows)


def finals(P: pd.DataFrame, kind: str) -> pd.DataFrame:
    """段 3：反復 0〜29 の組の平均で最終候補の表を作る。メタ検証の設定は反復 0〜9 で選んだもの（P の *_meta_set）。"""
    import ast
    reps = list(range(30))
    d = load_compare(kind, reps)
    mp = load_map(kind, sorted(d.target.unique()), reps)
    Q = P[P.kind == kind].copy()
    key = ["target", "source", "K"]
    ht = d[d.method == "HT"]

    def mean_of(x, name, by=key):
        g = x.groupby(by).agg(**{name: ("uar", "mean"), f"n_{name}": ("rep", "nunique")}).reset_index()
        return g

    F = Q[key + ["same", "P3_meta_set", "P4_meta_set", "O1_meta_set"]].copy()
    F = F.merge(mean_of(ht[(ht.cen == 1) & (ht.cp == DEF_CP) & (ht.cb == DEF_CB)], "P3_def"), on=key, how="left")
    F = F.merge(mean_of(ht[(ht.cen == 0) & (ht.cp == DEF_CP) & (ht.cb == "free")], "P1_def"), on=key, how="left")
    F = F.merge(mean_of(d[(d.method == "H3") & (d.cen == 1)], "H3"), on=key, how="left")
    hp = ht[ht.cen == 1].groupby(key + ["cp", "cb"]).agg(u=("uar", "mean"), n=("rep", "nunique"))
    ha = d[(d.method == "HT_ALL") & (d.cen == 1)].groupby(["target", "K", "cp", "cb"]).agg(u=("uar", "mean"), n=("rep", "nunique"))
    o1 = d[d.method == "O1"].groupby(key + ["alpha"]).agg(u=("uar", "mean"), n=("rep", "nunique"))
    p4d = d[(d.method == "HT_ALL") & (d.cen == 1) & (d.cp == DEF_CP) & (d.cb == DEF_CB)]
    F = F.merge(mean_of(p4d, "P4_def", ["target", "K"]), on=["target", "K"], how="left")
    pm, p4m, om, n_pm = [], [], [], []
    for r in F.itertuples():
        cp, cb = ast.literal_eval(r.P3_meta_set)
        v = hp.loc[(r.target, r.source, r.K, cp, cb)]
        pm.append(v.u); n_pm.append(v.n)
        cp4, cb4 = ast.literal_eval(r.P4_meta_set)
        p4m.append(ha.loc[(r.target, r.K, cp4, cb4)].u)
        om.append(o1.loc[(r.target, r.source, r.K, float(r.O1_meta_set))].u)
    F["P3_meta"], F["n_P3_meta"], F["P4_meta"], F["O1_meta"] = pm, n_pm, p4m, om
    for name, src, m in (("B1", "cv", "tgt_only"), ("B3", "cv", "merged"), ("B4", "fix", "merged")):
        x = mp[(mp.src == src) & (mp.method == m)]
        by = ["target", "K"] if name == "B1" else key
        F = F.merge(x.groupby(by).uar.mean().rename(name).reset_index(), on=by, how="left")
    b2 = mp[(mp.src == "cv") & (mp.method == "src_only")].groupby(["target", "source"]).uar.mean().rename("B2").reset_index()
    F = F.merge(b2, on=["target", "source"], how="left")
    F.insert(0, "kind", kind)
    return F


COLS = ["B1", "B2", "B3", "B4", "H3", "H3t", "P1_def", "P2_def", "P3_def", "P3t_def", "P3_meta", "P3_cv", "P3_orc", "HT24_meta",
        "HT24_orc", "P4_def", "P4_meta", "P4_cv", "P4_orc", "O1_meta", "O1_cv", "O1_orc", "O2", "O6_tgt_mean", "O6_none", "O7"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--ks", nargs="+", help="K を絞る（途中の確認用．出力は results/explore_relation/partial/）")
    args = ap.parse_args()
    global KS_USE, OUT
    if args.ks:
        KS_USE = args.ks
        OUT = OUT / "partial"
        OUT.mkdir(exist_ok=True)
    kinds = [k for k in args.kinds if (ROOT / "results" / f"compare_{k}").exists() and any((ROOT / "results" / f"compare_{k}").glob("*.csv"))]
    P = pd.concat([pair_table(k) for k in kinds], ignore_index=True)
    P.to_csv(OUT / "e31_pairs.csv", index=False)
    T, Mn = judge(P), main_comparisons(P)
    T.to_csv(OUT / "e31_tests.csv", index=False)
    Mn.to_csv(OUT / "e31_main.csv", index=False)
    S = P[~P.same].groupby(["kind", "K"])[[c for c in COLS if c in P]].mean().reset_index()
    S.to_csv(OUT / "e31_summary.csv", index=False)
    fk = [k for k in kinds if any((ROOT / "results" / f"compare_{k}").glob("*_final*.csv"))]
    if fk:
        F = pd.concat([finals(P, k) for k in fk], ignore_index=True)
        F.to_csv(OUT / "e31_finals30.csv", index=False)
        M30 = main_comparisons(F)
        M30.to_csv(OUT / "e31_main30.csv", index=False)
        cols = ["B1", "B3", "B4", "H3", "P1_def", "P3_def", "P3_meta", "P4_def", "P4_meta", "O1_meta"]
        print(F[~F.same].groupby(["kind", "K"])[cols].mean().round(2).to_string())
        print("反復の数（P3_def）：", sorted(F.n_P3_def.dropna().unique()))
        print(M30.round(3).to_string(index=False))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(T.to_string(index=False))
    print(S.round(2).to_string(index=False))
    print(Mn.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
