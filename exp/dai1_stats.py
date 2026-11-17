"""DAI 実験 1 の区間・p 値・Holm（設計 `docs/設計/実験1_設計.md` 9 節，計画 8.2 節）。

2 つの使い方がある．
  python exp/dai1_stats.py detect  ：選ぶ段の後・評価の回の前に，仮のターゲットの出力に 9 節の区間の規則を当て，
                                        評価のターゲットごとの標準誤差の見込み，検出できる差，種の数，同じ水準の δ を書く（統計の検査 S2）。
                                        固定のファイル（frozen.json と frozen/<置き場の名前>_frozen.json）に足す。
  python exp/dai1_stats.py eval    ：評価の回の出力から族 1〜4 を計算する（評価の回の後）。

族（コーパスごと．族 5 は実験 3）：
  族 1：提案（同時に解く版．主の版．選んだ設定）− 比較の基準（K 本だけ．K=0 は出発点）．生徒 4 × K 3（0，5，25）
  族 2：提案（役割を分ける版）− 提案（同時に解く版）．どちらも主の版・選んだ設定．生徒 4 × K 2（5，25）
  族 3：提案（同時に解く版）− 族 3 の相手（固定のファイルの partner の手法と版．選んだ設定）．生徒 4 × K 3
  族 4：提案（同時に解く版．主の版・選んだ設定）− 提案（同時に解く版．選択なし：予測は指数 (1,1,1)，ρ = 1，a = 1）．生徒 4 × K 3

効果：単位（分割 f，種 r）ごとにテストの UAR の差 d_fr を取り，θ = (1/|F|) Σ_f d̄_f，d̄_f = (1/R) Σ_r d_fr．
判定は保存したテストの発話ごとの判定（int8）から作る．rows.csv の UAR と全部の問で一致するかを数えて残す（統計の検査 S3）。

区間（実効の塊の数 G_eff = (Σ n_g)² / Σ n_g²．コーパスの全発話で数える）：
  G_eff ≥ 20（MOSEI）：テストの塊の引き直し B = 2,000 回の分散 V_cl と種の項 V_seed の和．1 回ごとに，分割ごとにテストの塊を
    1 回だけ復元抽出し，その分割の全部の種に同じ組を当て，種で平均した差の θ を作る（統計の検査 S1）．種は引き直さず，種の項は
    解析の式で足す（引き直しで種を数えると (R−1)/R 倍に縮むため．統計の検査の 2 回目の 3）．区間は θ ± 1.96√(V_cl + V_seed)，p は両側の正規。
  G_eff < 20（AFEW，IEMOCAP，MELD）：テストの塊の jackknife と種の項の和．
    θ_(−g)：塊 g を除いたテストで UAR を作り直す．除いた後の分割 f のテストが 30 本未満か，感情が 1 つでも欠けるなら，その分割を
      θ_(−g) の平均から外す（IEMOCAP は塊 = 分割なのでいつもそうなる．統計の検査 S3・S16）。
    V_jk = (G − 1)/G · Σ_g (θ_(−g) − θ̄_(·))²，G はテストに現れる塊の数．
    V_seed = (1/|F|²) Σ_{f ∉ D} s_f² / R，s_f² は分割 f の種の間の d_fr の不偏分散（統計の検査 S3 の 2）．D は，ある塊を除くと
      分割ごと外れる分割の集合．その分割の種の揺れは V_jk に入っているので V_seed に足さない（二重に数えない．統計の検査の 2 回目の 2．
      IEMOCAP は D が全部の分割なので V_seed = 0）．
    区間は θ ± t_{0.975, df} √(V_jk + V_seed)，df = max(1, ⌊G_eff⌋ − 1)．p は両側の t．
  K=0 の手法は種に依らない（統計の検査 S9）．K=0 の単位の数は分割の数と書き，V_seed は 0 になる。
  標準誤差が 0（2 つの行の判定が全部の単位で同じ）なら p = 1 とし，`identical` の印を付ける（統計の検査の 2 回目の 1）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai_data as DD  # noqa: E402

KS_MAIN = (0, 5, 25)
FAMS = {1: ("prop", "none", KS_MAIN), 2: ("rolesep", "prop", (5, 25)), 3: ("prop", "partner", KS_MAIN), 4: ("prop", "default", KS_MAIN)}
N_BOOT = 2000
MIN_TEST = 30
ALPHA = 0.05


def norm(v) -> str:
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else repr(f)
    except (TypeError, ValueError):
        return str(v)


def g_eff_of(T: str) -> tuple[float, pd.DataFrame]:
    df = pd.read_csv(DD.ROOT / "splits" / f"{T}.csv", keep_default_na=False)
    n = df["cluster"].value_counts().to_numpy()
    return float(n.sum() ** 2 / (n ** 2).sum()), df


class UnitPreds:
    """1 単位の保存（npz）から，族に入る行のテストの判定を取り出す。"""

    def __init__(self, base: Path):
        self.p1 = np.load(f"{base}_p1.npz")
        self.p3 = np.load(f"{base}_p3.npz")
        self.sid = self.p1["sample_test"].astype(str)

    def none(self, s: str, K: int, tether: dict) -> np.ndarray:
        if K == 0:
            return self.p1[f"p1_{s}_start_pred"]                # 出発点
        cw, cb = tether[f"{s}|{K}"]
        items = [str(x) for x in self.p1[f"p1_{s}_items"]]
        return self.p1[f"p1_{s}_pred"][items.index(f"{K}|{cw}|{cb}")]

    def p3item(self, s: str, K: int, **want) -> np.ndarray:
        items = [json.loads(x) for x in self.p3[f"p3_{s}_K{K}_items"]]
        hits = [j for j, it in enumerate(items) if all(norm(it.get(k)) == norm(v) for k, v in want.items())]
        assert len(hits) == 1, (s, K, want, len(hits))
        return self.p3[f"p3_{s}_K{K}_pred"][hits[0]]

    def cotrain_single(self, s: str, K: int, rho) -> np.ndarray:
        return self.p3[f"p3c_K{K}_{s}_pred"][list(dai1.RHO_GRID).index(float(rho))]

    def coreg(self, s: str, K: int, rho) -> np.ndarray:
        return self.p3[f"p3g_K{K}_{s}_pred"][list(dai1.COREG_RHO).index(float(rho))]

    def rolesep(self, s: str, K: int, teacher: str, rho, a) -> np.ndarray:
        ja = [(tv, r) for tv in ("sel", "one") for r in dai1.RHO_GRID].index((teacher, float(rho)))
        if K == 0:
            return self.p3[f"p3r_{s}_K{K}_pred_a"][ja]
        return self.p3[f"p3r_{s}_K{K}_pred_b"][ja * len(dai1.A_GRID) + list(dai1.A_GRID).index(int(float(a)))]


def unit_preds(fz: dict, up: UnitPreds, s: str, K: int, startp) -> dict:
    """族に入る 5 つの行の判定。"""
    sel = fz["select"]
    v = fz["main"][str(K)]
    j = sel[f"{s}|{K}|joint|{v}"]
    rs = sel[f"{s}|{K}|rolesep|{v}"]
    part = fz["partner"][f"{s}|{K}"]
    pm, pv = part["method"], part["version"]
    pp = {k: x for k, x in sel[f"{s}|{K}|{pm}|{pv}"].items() if k != "pseudo_uar"}
    out = {"prop": up.p3item(s, K, method="joint", teacher=v, rho=j["rho"], a=j["a"]),
           "none": startp if K == 0 else up.none(s, K, fz["tether"]),
           "rolesep": up.rolesep(s, K, v, rs["rho"], rs["a"]),
           "default": up.p3item(s, K, method="joint", teacher="one", rho=1.0, a=1)}
    if pm == "cotrain" and s != "AVT":
        out["partner"] = up.cotrain_single(s, K, pp["rho"])
    elif pm == "coreg":
        out["partner"] = up.coreg(s, K, pp["rho"])
    else:
        out["partner"] = up.p3item(s, K, method=pm, teacher=pv, **pp)
    return out


def counts(y: np.ndarray, pred: np.ndarray, cl: np.ndarray):
    """塊ごとの数：(塊の名前の並び, 感情ごとの正解の数 (塊, 4), 感情ごとの件数 (塊, 4))。"""
    names, inv = np.unique(cl, return_inverse=True)
    cor = np.zeros((len(names), 4)); tot = np.zeros((len(names), 4))
    np.add.at(tot, (inv, y), 1)
    hit = pred == y
    np.add.at(cor, (inv[hit], y[hit]), 1)
    return names, cor, tot


def uar_rows(cor: np.ndarray, tot: np.ndarray) -> np.ndarray:
    """(…, 4) の数 → UAR（%）．テストが 30 本未満か感情が欠けるなら NaN（設計 9.3 節）。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        u = 100.0 * (cor / tot).mean(-1)
    bad = (tot.sum(-1) < MIN_TEST) | (tot == 0).any(-1)
    return np.where(bad, np.nan, u)


def uar_full(cnt) -> float:
    """テストに現れる感情だけの再現率の平均（rows.csv と同じ定義）。"""
    _, cor, tot = cnt
    c, t_ = cor.sum(0), tot.sum(0)
    ok = t_ > 0
    return 100.0 * float((c[ok] / t_[ok]).mean())


def theta(diffs: dict) -> float:
    """diffs：{f: [d_fr]} → θ（空の分割は外す）。"""
    vals = [np.mean(v) for v in diffs.values() if len(v)]
    return float(np.mean(vals))


def effect_stats(cellA: dict, cellB: dict, g_eff: float, rng) -> dict:
    """cellA，cellB：{(f, r): counts の戻り}．同じ (f, r) の組をもつ。"""
    keys = sorted(cellA)
    folds = sorted({f for f, _ in keys})
    rs_of = {f: [r for ff, r in keys if ff == f] for f in folds}
    d = {f: [uar_full(cellA[(f, r)]) - uar_full(cellB[(f, r)]) for r in rs_of[f]] for f in folds}
    est = theta(d)
    R = {f: len(d[f]) for f in folds}
    s2 = {f: (float(np.var(d[f], ddof=1)) if R[f] > 1 else 0.0) for f in folds}
    v_seed = float(sum(s2[f] / R[f] for f in folds) / len(folds) ** 2)
    res = dict(effect=est, n_folds=len(folds), n_units=sum(R.values()),
               seed_varies=bool(any(np.ptp(d[f]) > 0 for f in folds if R[f] > 1)))
    if g_eff >= 20:
        boots = np.zeros(N_BOOT); nval = np.zeros(N_BOOT)
        for f in folds:
            names = cellA[(f, rs_of[f][0])][0]
            ncl = len(names)
            Wb = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl), size=N_BOOT).astype(float)   # 分割ごとに 1 回．全部の種に同じ組（S1）
            ua = np.stack([uar_rows(Wb @ cellA[(f, r)][1], Wb @ cellA[(f, r)][2]) for r in rs_of[f]])   # (R, B)
            ub = np.stack([uar_rows(Wb @ cellB[(f, r)][1], Wb @ cellB[(f, r)][2]) for r in rs_of[f]])
            dm = np.nanmean(ua - ub, axis=0)                                # 種は引き直さず平均する
            ok = np.isfinite(dm)
            boots[ok] += dm[ok]; nval[ok] += 1
        boots = np.where(nval > 0, boots / np.maximum(nval, 1), np.nan)   # 全分割が無効の回は除く（N9）
        boots = boots[np.isfinite(boots)]
        v_cl = float(np.var(boots, ddof=1))
        se = float(np.sqrt(v_cl + v_seed))
        p, ident = pval(est, se, None)
        return res | dict(lo=est - 1.96 * se, hi=est + 1.96 * se, p=p, se=se, method="boot_cl+seed", df=np.nan,
                          v_jk=v_cl, v_seed=v_seed, v_seed_all=v_seed, identical=ident,
                          boot_lo=float(np.percentile(boots, 2.5)), boot_hi=float(np.percentile(boots, 97.5)))
    all_cl = sorted({c for k in keys for c in cellA[k][0]})
    jk, dropped = [], set()
    for g in all_cl:
        dg = {}
        for f in folds:
            vals = []
            for r in rs_of[f]:
                (na, ca, ta), (nb, cb, tb) = cellA[(f, r)], cellB[(f, r)]
                if g in na:
                    ia = int(np.flatnonzero(na == g)[0]); ib = int(np.flatnonzero(nb == g)[0])
                    a_ = uar_rows(ca.sum(0) - ca[ia], ta.sum(0) - ta[ia]); b_ = uar_rows(cb.sum(0) - cb[ib], tb.sum(0) - tb[ib])
                    if np.isfinite(a_) and np.isfinite(b_):
                        vals.append(float(a_ - b_))
                else:
                    vals.append(uar_full(cellA[(f, r)]) - uar_full(cellB[(f, r)]))
            if not vals:
                dropped.add(f)                                              # この塊を除くと分割 f ごと外れる
            dg[f] = vals
        jk.append(theta(dg))
    jk = np.array(jk)
    G = len(all_cl)
    v_jk = float((G - 1) / G * ((jk - jk.mean()) ** 2).sum())
    v_seed_kept = float(sum(s2[f] / R[f] for f in folds if f not in dropped) / len(folds) ** 2)
    se = float(np.sqrt(v_jk + v_seed_kept))
    df = max(1, int(np.floor(g_eff)) - 1)
    p, ident = pval(est, se, df)
    q = float(st.t.ppf(0.975, df))
    return res | dict(lo=est - q * se, hi=est + q * se, p=p, se=se, method="jk+seed", df=df, v_jk=v_jk, v_seed=v_seed_kept,
                      v_seed_all=v_seed, folds_dropped=",".join(map(str, sorted(dropped))), identical=ident,
                      t=(est / se if se > 0 else np.nan))


def pval(est: float, se: float, df) -> tuple[float, bool]:
    """両側の p．標準誤差が 0 なら，差も 0 のとき p = 1（同じ判定の 2 行），差が 0 でないとき p = 0 として印を付ける。"""
    if se <= 0:
        return (1.0, True) if est == 0 else (0.0, True)
    z = abs(est) / se
    return (float(2 * st.norm.sf(z)) if df is None else float(2 * st.t.sf(z, df))), False


def holm(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def crit_first(g_eff: float, m: int) -> float:
    """Holm の最初の段の基準（両側 α/m）を標準誤差の倍数で。"""
    q = 1 - ALPHA / m / 2
    return float(st.norm.ppf(q)) if g_eff >= 20 else float(st.t.ppf(q, max(1, int(np.floor(g_eff)) - 1)))


def collect(out: Path, tag: str, T: str, c: str, fz: dict, units, y_of, cl_of, check_rows=True):
    """単位の判定を集め，(生徒, K, 行) → {(f, r): 塊ごとの数}．
    照合：段 3 の格子の行（joint・kd・agree・mix・tt と AVT の cotrain）について，保存した判定から作った UAR の並びと
    rows.csv の UAR の並びが（生徒，K）ごとに一致するかを数える（統計の検査 S3，2 回目の残り）。"""
    cells, mism, total = {}, 0, 0
    cname = "ALL" if tag == "eval" else c
    for f, r in units:
        base = out / tag / "units" / f"{T}_{cname}_f{f}_r{r}"
        up = UnitPreds(base)
        y, cl = y_of(up.sid), cl_of(up.sid)
        rows3 = pd.read_csv(f"{base}_p3.csv", low_memory=False) if check_rows else None
        for s in DD.STUDENTS:
            sp = up.none(s, 0, None)
            for K in KS_MAIN:
                P = unit_preds(fz, up, s, K, sp)
                for name, pr in P.items():
                    cells.setdefault((s, K, name), {})[(f, r)] = counts(y, pr, cl)
                if check_rows:
                    pred = up.p3[f"p3_{s}_K{K}_pred"]
                    mine = np.sort([uar_full(counts(y, pr, cl)) for pr in pred])
                    meths = {json.loads(x)["method"] for x in up.p3[f"p3_{s}_K{K}_items"]}
                    rr = rows3[(rows3.stage == 3) & (rows3.student == s) & (rows3.K == K) & rows3.method.isin(meths)]
                    if s != "AVT":
                        rr = rr[rr.method != "cotrain"]                     # 単一の生徒の共訓練の行は p3 の問ではない
                    theirs = np.sort(rr.uar.to_numpy(float))
                    total += 1
                    mism += int(len(mine) != len(theirs) or not np.allclose(mine, theirs, atol=1e-6))
    return cells, mism, total


def label_maps(c: str):
    cp = dai1.Corpus.get(c)
    ymap = dict(zip(cp.df["sample_id"].astype(str), cp.y))
    cmap = dict(zip(cp.df["sample_id"].astype(str), cp.df["cluster"].astype(str)))
    return (lambda sid: np.array([ymap[x] for x in sid])), (lambda sid: np.array([cmap[x] for x in sid]))


def expected_effects(out: Path, X: str, tw: dict) -> dict:
    """評価のターゲット X の効果の見込み（統計の検査の 2 回目の 4，ML の査読者の m53 (b)）：X のテストを使わないため，
    T = X の仮のターゲットの回（c ≠ X）だけを使う．仮のターゲット c を 1 つ除いた 20 単位で設定を選び直し，除いた c の 10 単位で
    族のセルの効果を測り，3 つの c で平均する（選んだのと同じ単位で測ると選択の上乗せが入るため）。"""
    import dai1_select as SEL
    p1, p2, p3 = (SEL.load(out, X, w) for w in (1, 2, 3))
    acc = {}
    for c in DD.others(X):
        keep = [d[d.c != c] for d in (p1, p2, p3)]
        r = SEL.select_T(tw, *keep, n_units=len([u for u in tw["units"] if not u.startswith(c + "_")]))
        fz_c = {"select": r["select"], "main": r["main"], "partner": r["partner"], "tether": tw["tether"]}
        units = [(f, rr) for f in range(DD.N_FOLDS) for rr in (0, 1)]
        y_of, cl_of = label_maps(c)
        cells, _, _ = collect(out, "pseudo", X, c, fz_c, units, y_of, cl_of, check_rows=False)
        for fam, (A, B, ks) in FAMS.items():
            for s in DD.STUDENTS:
                for K in ks:
                    d = [uar_full(cells[(s, K, A)][u]) - uar_full(cells[(s, K, B)][u]) for u in units]
                    acc.setdefault(f"{fam}|{s}|{K}", []).append(float(np.mean(d)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def detect(out: Path, targets=DD.CORPORA) -> None:
    """選ぶ段の後・評価の回の前（統計の検査 S2）．評価のターゲット X ごとに：
    (1) 標準誤差の見込み：ほかの T の仮のターゲットの回で c = X の単位に 9.3 節の区間の規則を当て，塊の項と種の項を分けて平均する．
    (2) 種の数 R：K > 0 のセルの中央値で種の項が分散の 25% 以下になる最小の R を {10, 20} から（どちらでも超えるなら 20）。
    (3) 検出できる差：Holm の最初の段（族の大きさ m）と最後の段の基準 × 標準誤差の見込み．
    (4) 効果の見込み：`expected_effects`（X のテストを使わない）．検出できる差（最初の段）が効果の見込みの大きさを超える
        （検出力が 50% 未満の見込みの）セルは「区間つきの記述」に回す。
    (5) 同じ水準の δ = 2√(V/n)，n = 5R（V は c = X の比較の基準の種の間の分散の平均）と，区間の半幅の見込みが δ 以下になるか。"""
    fzp = out / "select" / "frozen.json"
    frozen = json.load(open(fzp))
    if (out / "eval" / "units").exists() and any((out / "eval" / "units").iterdir()):
        sys.exit("評価の回の出力がすでにある")
    rng = np.random.default_rng(20261104)
    acc = {}
    for T in targets:
        fz = frozen[T]
        for c in DD.others(T):
            g_eff, _ = g_eff_of(c)
            units = [(f, r) for f in range(DD.N_FOLDS) for r in (0, 1)]
            y_of, cl_of = label_maps(c)
            cells, mism, total = collect(out, "pseudo", T, c, fz, units, y_of, cl_of)
            for fam, (A, B, ks) in FAMS.items():
                for s in DD.STUDENTS:
                    for K in ks:
                        e = effect_stats(cells[(s, K, A)], cells[(s, K, B)], g_eff, rng)
                        acc.setdefault(c, []).append(dict(T=T, family=fam, student=s, K=K, **e, mismatch=mism, checked=total))
            print(f"detect {T} c={c}: 照合の食い違い {mism}/{total}", flush=True)
    for X, lst in acc.items():
        df = pd.DataFrame(lst)
        g_eff, _ = g_eff_of(X)
        crit95 = 1.959964 if g_eff >= 20 else float(st.t.ppf(0.975, max(1, int(np.floor(g_eff)) - 1)))
        # 分散を塊の項と種の項に分ける．jackknife で分割ごと外れる分割の種の揺れは v_jk に入っているので，
        # 塊の項は max(0, v_jk − (v_seed_all − v_seed))，種の項は全部の分割の v_seed_all（統計の検査の 3 回目）
        df["v_cl"] = (df.v_jk - (df.v_seed_all - df.v_seed)).clip(lower=0)
        n_seeds = 20
        for R in (10, 20):
            share = []
            for _, r in df[df.K > 0].iterrows():
                vs = r.v_seed_all * 2 / R                                  # 仮のターゲットは種 2 つ
                tot = r.v_cl + vs
                share.append(vs / tot if tot > 0 else 0.0)
            if np.median(share) <= 0.25:
                n_seeds = R
                break
        exp_eff = expected_effects(out, X, json.load(open(out / "pseudo" / f"{X}_tether_w.json"))) if X in targets else {}
        cells = {}
        for (fam, s, K), g in df.groupby(["family", "student", "K"]):
            vjk = float(g.v_cl.mean())
            vseed = float(g.v_seed_all.mean()) * 2 / n_seeds if K > 0 else 0.0
            se = float(np.sqrt(vjk + vseed))
            m = 12 if fam in (1, 3, 4) else 8
            key = f"{fam}|{s}|{K}"
            cells[key] = dict(se=se, v_cl_or_jk=vjk, v_seed=vseed, detectable_first=crit_first(g_eff, m) * se,
                              detectable_last=crit_first(g_eff, 1) * se, halfwidth=crit95 * se,
                              expected_effect=exp_eff.get(key, float("nan")))
        V = []
        for T in targets:
            if X == T:
                continue
            p1 = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted((out / "pseudo" / "units").glob(f"{T}_{X}_*_p1.csv"))])
            for s in DD.STUDENTS:
                for K in (5, 25):
                    cw, cb = frozen[T]["tether"][f"{s}|{K}"]
                    d = p1[(p1.method == "none") & (p1.kind == s) & (p1.K == K) & np.isclose(p1.Cw, cw) & np.isclose(p1.Cb, cb)]
                    V.append(d.groupby("fold")["uar"].var(ddof=1).mean())
        delta = float(2 * np.sqrt(np.mean(V) / (DD.N_FOLDS * n_seeds)))
        for key, v in cells.items():
            v["same_level_possible"] = bool(int(key.split("|")[2]) > 0 and v["halfwidth"] < delta)
            v["descriptive"] = bool(not np.isfinite(v["expected_effect"]) or v["detectable_first"] > abs(v["expected_effect"]))
        det = dict(n_seeds_eval=n_seeds, g_eff=g_eff, delta_same=delta, cells=cells,
                   delta_same_note="種による揺れの 2 標準誤差．K>0 の比較だけに使う．同じ水準は区間が ±δ に収まるときだけ",
                   descriptive_cells=[k for k, v in cells.items() if v["descriptive"]],
                   same_level_cells=[k for k, v in cells.items() if v["same_level_possible"]])
        frozen.setdefault(X, {})["n_seeds_eval"] = n_seeds
        frozen[X]["detect"] = det
        print(f"{X}: 種 {n_seeds}，δ {delta:.2f}，記述に回すセル {len(det['descriptive_cells'])}，同じ水準を出しうるセル "
              f"{len(det['same_level_cells'])}", flush=True)
    frozen["_meta"]["detect"] = "dai1_stats.py detect（統計の検査 S2 と 2 回目）"
    frozen["_meta"]["detect_commit"], frozen["_meta"]["detect_dirty"] = dai1.git_commit()      # M6
    json.dump(frozen, open(fzp, "w"), indent=1, ensure_ascii=False)
    if out.resolve().parent == (dai1.ROOT / "results").resolve():
        shutil.copy(fzp, dai1.ROOT / "frozen" / f"{out.name}_frozen.json")
    print("書いた:", fzp, hashlib.sha256(open(fzp, "rb").read()).hexdigest())


def evaluate(out: Path) -> None:
    frozen = json.load(open(out / "select" / "frozen.json"))
    rng = np.random.default_rng(20261104)
    rows = []
    for T in DD.CORPORA:
        fz = frozen[T]
        g_eff, _ = g_eff_of(T)
        R = int(fz["n_seeds_eval"])
        units = [(f, r) for f in range(DD.N_FOLDS) for r in range(R)
                 if Path(f"{out}/eval/units/{T}_ALL_f{f}_r{r}_p3.npz").exists()]
        assert len(units) == DD.N_FOLDS * R, (T, len(units))
        y_of, cl_of = label_maps(T)
        cells, mism, total = collect(out, "eval", T, T, fz, units, y_of, cl_of)
        for fam, (A, B, ks) in FAMS.items():
            res = []
            for s in DD.STUDENTS:
                for K in ks:
                    e = effect_stats(cells[(s, K, A)], cells[(s, K, B)], g_eff, rng)
                    if K == 0:
                        e["n_units"] = DD.N_FOLDS
                    res.append(dict(family=fam, T=T, student=s, K=K, A=A, B=B, g_eff=g_eff, mismatch=mism, checked=total, **e))
            for x, a in zip(res, holm(np.array([x["p"] for x in res]))):
                x["p_holm"] = float(a)
            rows += res
        print(f"{T}: 単位 {len(units)}，実効の数 {g_eff:.1f}，照合の食い違い {mism}/{total}", flush=True)
    (out / "stats").mkdir(exist_ok=True)
    cm, dt = dai1.git_commit()
    pd.DataFrame(rows).assign(commit=cm, dirty=dt).to_csv(out / "stats" / "families.csv", index=False)   # M6
    print("書いた:", out / "stats" / "families.csv")


if __name__ == "__main__":
    mode = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else dai1.ROOT / "results" / "dai1"
    tg = sys.argv[3].split(",") if len(sys.argv) > 3 else list(DD.CORPORA)     # 試走だけ
    detect(out, tg) if mode == "detect" else evaluate(out)
