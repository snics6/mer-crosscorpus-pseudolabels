"""DAI 実験 1 の選ぶ段（設計 `docs/設計/実験1_設計.md` 8 節）。仮のターゲットの出力だけを読み，固定のファイルを書く。

- 評価のターゲットの出力（`results/dai1/eval/units/`）があれば止まる（漏れの検査 L8）。
- 段 1〜3 の単位の集合が `<T>_tether_w.json` の単位の一覧と一致しなければ止まる（m60）。
- 段 1〜3 の行のコミットが 1 つでなければ止まる（比較の検査 C6）。
- `select_T` は単位の部分集合（仮のターゲットの一部）でも選べる．`dai1_stats.py detect` が仮のターゲットを 1 つ除いて選び直すのに使う。
"""

from __future__ import annotations

import glob
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dai1
import dai_data as DD

GRIDS = {
    "joint": [("rho", dai1.RHO_GRID), ("a", dai1.A_GRID)],
    "rolesep": [("rho", dai1.RHO_GRID), ("a", dai1.A_GRID)],
    "kd": [("temp", dai1.KD_T), ("rho", dai1.RHO_GRID)],
    "agree": [("tau", dai1.AG_TAU), ("f", dai1.AG_F)],
    "mix": [("alpha", dai1.MIX_ALPHA), ("f", dai1.MIX_F)],
    "cotrain": [("rho", dai1.RHO_GRID)],
    "coreg": [("rho", dai1.COREG_RHO)],
    "tgtonly": [("C", dai1.TGT_C)],
    "kbias": [("Cb", (1.0, 10.0))],
    "joint_scratch": [("C", dai1.TGT_C)],
}
EXTRA_DEFAULTS = {"kbias": {"Cb": 10.0}, "joint_scratch": {"C": 1.0}}       # 選択なしの値（設計 5 節・7 節）
FAMILY_METHODS = ("joint", "rolesep", "kd", "agree", "mix", "cotrain", "coreg")   # 族に入りうる手法（同点の数え上げを分ける）
PARTNERS = ("kd", "agree", "mix", "cotrain", "coreg")                     # 族 3 の相手（同じ情報を受け取る比較相手）．同点はこの並び
PARTNER_VERSIONS = {"cotrain": ["ct"], "coreg": ["cr"]}                    # 予測の版を持たない相手


def num(v) -> float:
    return float(v)


def load(out: Path, T: str, which: int) -> pd.DataFrame:
    fs = sorted(glob.glob(str(out / "pseudo" / "units" / f"{T}_*_p{which}.csv")))
    assert fs, (T, which)
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


def unit_names(d: pd.DataFrame) -> list[str]:
    return sorted({f"{c}_f{f}_r{r}" for c, f, r in d[["c", "fold", "seed"]].drop_duplicates().itertuples(index=False)})


def delta_sel(p1: pd.DataFrame, tether: dict, n_units: int) -> dict:
    """(生徒, K) ごとの δ_sel = 2√(V/n)．V は比較の基準の UAR の，同じ (c, 分割) の中の種の間の不偏分散の (c, 分割) にわたる平均
    （2 つの値の分散の中央値は真の値の約 0.455 倍に偏るので使わない．統計の検査 S3，定義の検査 D7）。"""
    out = {}
    for s in DD.STUDENTS:
        for K in DD.KS:
            cw, cb = tether[f"{s}|{K}"]
            d = p1[(p1.stage == 1) & (p1.method == "none") & (p1.kind == s) & (p1.K == K) &
                   np.isclose(p1.Cw, cw) & np.isclose(p1.Cb, cb)]
            v = d.groupby(["c", "fold"])["uar"].var(ddof=1).mean()
            out[(s, K)] = float(2 * np.sqrt(v / n_units))
        out[(s, 0)] = out[(s, 1)]                                        # K=0 の比較の基準は出発点で種の揺れが無い
    return out


def pick(g: pd.DataFrame, method: str, delta: float, default: dict) -> tuple[dict, bool]:
    """格子の点の平均 UAR が最大の点．最良との差が δ 以内の点が複数なら，選択なしの値に最も近い点
    （各つまみの格子の番号の差の和．それも同じなら番号の並びで先）。"""
    keys = [k for k, _ in GRIDS[method]]
    grid = {k: list(v) for k, v in GRIDS[method]}
    g = g.copy()
    for k in keys:
        g[k] = g[k].map(num)
    m = g.groupby(keys)["uar"].mean()
    best = m.max()
    cands = list(m[m >= best - delta].index)

    def idx(k, v):
        return min(range(len(grid[k])), key=lambda i: abs(grid[k][i] - v))

    def as_t(ix):
        return ix if isinstance(ix, tuple) else (ix,)
    dflt = [idx(k, num(default[k])) for k in keys]
    chosen = min(cands, key=lambda ix: (sum(abs(idx(k, v) - d) for k, v, d in zip(keys, as_t(ix), dflt)),
                                        [idx(k, v) for k, v in zip(keys, as_t(ix))]))
    am = m.idxmax()
    return ({k: v for k, v in zip(keys, as_t(chosen))} | {"pseudo_uar": float(m.loc[chosen])}, as_t(chosen) != as_t(am))


def select_T(tw: dict, p1: pd.DataFrame, p2: pd.DataFrame, p3: pd.DataFrame, n_units: int) -> dict:
    """1 つの評価のターゲット T の選択（設計 8 節の 1〜3）．p1〜p3 は使う単位の行だけ。"""
    dsel = delta_sel(p1, tw["tether"], n_units)
    select, tie_family, tie_extra = {}, 0, 0
    d3 = p3[(p3.stage == 3) & p3.method.isin([m for m in GRIDS if m not in ("tgtonly", "kbias")])]
    for (s, K, method, ver), g in d3.groupby(["student", "K", "method", "teacher"]):
        dflt = EXTRA_DEFAULTS.get(method) or dai1.DEFAULTS[method]
        ch, used = pick(g, method, dsel[(s, int(K))], dflt)
        select[f"{s}|{int(K)}|{method}|{ver}"] = ch
        if method in FAMILY_METHODS:
            tie_family += int(used)
        else:
            tie_extra += int(used)
        if method == "joint":                                             # 提案を K 本の倍率 1 に制限した版（比較の検査 C1・C2）
            ch1, used1 = pick(g[g["a"].map(num) == 1.0], "joint", dsel[(s, int(K))], dai1.DEFAULTS["joint"])
            select[f"{s}|{int(K)}|joint_a1|{ver}"] = ch1
            tie_family += int(used1)
    for meth in ("tgtonly", "kbias"):
        d1 = p1[(p1.stage == 1) & (p1.method == meth)]
        for (s, K), g in d1.groupby(["kind", "K"]):
            if s in DD.STUDENTS:
                ch, used = pick(g, meth, dsel[(s, int(K))], EXTRA_DEFAULTS.get(meth) or dai1.DEFAULTS[meth])
                select[f"{s}|{int(K)}|{meth}|-"] = ch
                tie_extra += int(used)
    main_ver, partner = {}, {}
    for K in (0,) + DD.KS:
        ms = np.mean([select[f"{s}|{K}|joint|sel"]["pseudo_uar"] for s in DD.STUDENTS])
        mo = np.mean([select[f"{s}|{K}|joint|one"]["pseudo_uar"] for s in DD.STUDENTS])
        dm = float(np.mean([dsel[(s, K)] for s in DD.STUDENTS]))
        same = tuple(tw["w"][str(K)]) == (1.0, 1.0, 1.0)
        main_ver[str(K)] = "sel" if (not same and ms - mo > dm) else "one"
        for s in DD.STUDENTS:
            cands = []
            for pm in PARTNERS:
                vers = PARTNER_VERSIONS.get(pm, ["sel", "one"])
                best_v = max(vers, key=lambda v: (select[f"{s}|{K}|{pm}|{v}"]["pseudo_uar"], v == "one"))   # 相手は自分の成績で版を選ぶ
                cands.append((select[f"{s}|{K}|{pm}|{best_v}"]["pseudo_uar"], pm, best_v))
            top = max(c[0] for c in cands)
            near = [c for c in cands if c[0] >= top - dsel[(s, K)]]
            chosen = min(near, key=lambda c: PARTNERS.index(c[1]))
            partner[f"{s}|{K}"] = {"method": chosen[1], "version": chosen[2], "pseudo_uar": chosen[0]}
    # 格子の端を選んだセルの数（手法ごと．比較の検査の 3・4 回目）
    edge = {}
    for key, ch in select.items():
        meth = key.split("|")[2]
        gm = "joint" if meth == "joint_a1" else meth
        if gm not in GRIDS:
            continue
        keys = [(k, v) for k, v in GRIDS[gm] if len(v) > 2 and k in ch and not (meth == "joint_a1" and k == "a")]   # 2 点の格子と固定のつまみは数えない（M5）
        if not keys:
            continue
        hit = any(num(ch[k]) in (float(min(v)), float(max(v))) for k, v in keys)
        edge.setdefault(meth, [0, 0])
        edge[meth][0] += int(hit); edge[meth][1] += 1
    return dict(dsel=dsel, select=select, main=main_ver, partner=partner, tie_family=tie_family, tie_extra=tie_extra,
                grid_edge=edge)


def main(out: Path, targets=DD.CORPORA) -> None:
    if list((out / "eval" / "units").glob("*")) if (out / "eval" / "units").exists() else []:
        sys.exit("評価の回の出力がすでにある．選ぶ段は評価の前にだけ流す（漏れの検査 L8）")
    sel_dir = out / "select"
    sel_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = dai1.git_commit()
    frozen = {"_meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "dirty": dirty,
                        "rule": "docs/設計/実験1_設計.md 8 節", "partners": list(PARTNERS)}}
    if list(targets) != list(DD.CORPORA):
        frozen["_meta"]["partial_targets"] = list(targets)              # 試走だけ．本番は 4 つ全部
    for T in targets:
        tw = json.load(open(out / "pseudo" / f"{T}_tether_w.json"))
        p1, p2, p3 = load(out, T, 1), load(out, T, 2), load(out, T, 3)
        for which, d in ((1, p1), (2, p2), (3, p3)):
            if unit_names(d) != sorted(tw["units"]):
                sys.exit(f"{T}: 段 {which} の単位が tether_w.json の一覧と違う（m60）")
        commits = set(p1.commit.astype(str)) | set(p2.commit.astype(str)) | set(p3.commit.astype(str))
        if len(commits) != 1 or commits != {str(tw.get("commit", ""))}:
            sys.exit(f"{T}: 段 1〜3 の行のコミットが 1 つでない {commits}（比較の検査 C6）")
        n_units = len(tw["units"])
        if n_units != 3 * DD.N_FOLDS * 2 and "partial_targets" not in frozen["_meta"]:
            sys.exit(f"{T}: 単位の数 {n_units} が設計の 30 でない（M7）")
        r = select_T(tw, p1, p2, p3, n_units)
        # 指数を分割を 1 つずつ除いて選び直したときの一致（定義の検査 D13）
        agree_w = {}
        prod = p2[(p2.stage == 2) & (p2.method == "product")]
        for K in (0,) + DD.KS:
            hits = 0
            for f in range(DD.N_FOLDS):
                w_f = dai1.select_w(prod[prod.fold != f])
                hits += int(tuple(w_f[K]) == tuple(tw["w"][str(K)]))
            agree_w[str(K)] = hits
        frozen[T] = {"tether": tw["tether"], "w": tw["w"], "w_lofo_agree_of5": agree_w, "n_units": n_units,
                     "delta_sel": {f"{s}|{K}": v for (s, K), v in r["dsel"].items()},
                     "select": r["select"], "main": r["main"], "partner": r["partner"],
                     "tie_rule_used_family": r["tie_family"], "tie_rule_used_extra": r["tie_extra"],
                     "grid_edge_cells": r["grid_edge"],
                     "units": tw["units"], "pseudo_commit": tw.get("commit", "")}
        print(f"{T}: 単位 {n_units}，同点の規則が働いたセル 族 {r['tie_family']}・ほか {r['tie_extra']}，指数の一致 {agree_w}", flush=True)
    json.dump(frozen, open(sel_dir / "frozen.json", "w"), indent=1, ensure_ascii=False)
    h = hashlib.sha256(open(sel_dir / "frozen.json", "rb").read()).hexdigest()
    print("書いた:", sel_dir / "frozen.json", "sha256", h)
    if out.resolve().parent == (dai1.ROOT / "results").resolve():          # 本番の置き場のときだけ写しを置く（I8）
        fz = dai1.ROOT / "frozen"
        fz.mkdir(exist_ok=True)
        shutil.copy(sel_dir / "frozen.json", fz / f"{out.name}_frozen.json")
        print("写し:", fz / f"{out.name}_frozen.json")
