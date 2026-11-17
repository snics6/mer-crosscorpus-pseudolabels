"""DAI 実験 2 の選ぶ段（設計 `docs/設計/実験2_設計.md` 第 3.2 版の 1 節「選び方」と 4 節）。仮のターゲットの出力だけを読み，固定のファイルを書く。

- 規則は実験 1 の 8 節の 1 と同じ（`dai1_select.pick` をそのまま使う）：(T, 生徒, K, 手法) ごとに，格子の点を T の仮のターゲットの単位の UAR の平均で
  選び，最良との差が δ_sel 以内の点が複数なら，選択なしの値に最も近い点（各つまみの格子の番号の差の和．それも同じなら格子の並びで先）．
- δ_sel は実験 1 の固定のファイル（既定 `frozen/dai1_frozen.json`．試走では引数で与えた置き場の frozen.json）の (T, 生徒, K) の値．
  δ_sel は K 本の抜き方の揺れだけを含み，勾配法の手法の初期化の揺れ（s = r）を含まない（表の注に書く）．
- 選択なしの値が無い xMUDA 型は，δ_sel の中の点のうち格子の並び（学習率，反復，κ，μ の順の辞書順）で先の点を選ぶ．
- 選択なしの値（`dai2.DEFAULTS`・`dai2_grad.DEFAULTS`）は，固定のファイルの _meta に写す．格子の点を持たない saerens は平均だけを書く．
- 止まる条件（gate-code の I12 を含む）：評価の回の出力（`<out>/eval/units/`）が 1 つでもある．仮のターゲットの単位の集合が実験 1 の固定の
  ファイルの単位の一覧と違う（部分ごと）．(生徒, K, 手法) のセルが設計どおりにそろわない（線形 9 手法 × 生徒 4 × K 5，ラベル伝播は K ≥ 1，
  map 3 手法 × 生徒 4 × K 5，xMUDA 型は生徒 A・V・T，CORAL は生徒 4）．格子の値が設計の格子から削り方（npz の extra_tag）を引いたものと違う．
  格子の点の行の数が単位の数と違う（欠けか重複）．全部の行のコミットが 1 つでない．引き戻しの鍵が実験 1 の固定のファイルの引き戻しと違う．
  仮のターゲットの回の台本の sha256（`<out>/provenance_log.jsonl` の段 pseudo の scripts_sha256）が 1 つでないか，選ぶ段の台本と違う．
  本番の置き場（`results/dai2`）では，未コミットの変更がある，行の dirty が 1 つでも真，実験 1 の固定のファイルの T に n_seeds_eval と detect
  が無い，のどれでも止まる（試走の置き場では --allow-dirty で進め，_meta に残す）．
- 順序：実験 1 の `dai1_stats.py detect` は実験 1 の固定のファイルに n_seeds_eval と detect を書き足す．その前に実験 2 の選ぶ段を流すと，
  評価の段が実験 1 の固定のファイルの sha256 の違いで止まる（選び直しとコミットのやり直しになる）．本番の置き場では detect の前には動かない．
- 書き出し：`<out>/select/frozen.json`．本番の置き場のときだけ写しを `frozen/dai2_frozen.json` に置く（コミットは人が行う）．
  _meta に台本の sha256（評価の段が照らす．ML の査読者の m72），実験 1 の固定のファイルの sha256，勾配法の削り方，選択なしの値と格子を書く．
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_select  # noqa: E402
import dai2  # noqa: E402
import dai_data as DD  # noqa: E402

ROOT = DD.ROOT
GRIDS = dict(dai2.GRIDS)
DEFAULTS = dict(dai2.DEFAULTS)
PARTS = {"lin": ("self", "entmin", "im", "t3a", "t3a_b", "lp", "own", "st", "saerens"),
         "map": ("dann", "mme", "mapctl"), "xmuda": ("xmuda",), "coral": ("coral",)}
NO_DEFAULT = ("xmuda",)


def _grad_module():
    import dai2_grad                                                       # torch の TF32 の設定を変えるので，選ぶ段でだけ読む
    return dai2_grad


def load_part(out: Path, T: str, part: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(out / "pseudo" / "units" / f"{T}_*_{part}.csv")))
    if not fs:
        sys.exit(f"{T}: 部分 {part} の仮のターゲットの出力が無い")
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


def pick_order(g: pd.DataFrame, keys: list, grid: dict, delta: float) -> tuple[dict, bool]:
    """選択なしの値が無い手法（xMUDA 型）：平均が最大の点との差が δ 以内の点のうち，格子の並び（keys の順の辞書順）で先の点。"""
    g = g.copy()
    for k in keys:
        g[k] = g[k].map(float)
    m = g.groupby(keys)["uar"].mean()
    best = m.max()
    cands = list(m[m >= best - delta].index)

    def idx(ix):
        ix = ix if isinstance(ix, tuple) else (ix,)
        return [min(range(len(grid[k])), key=lambda i: abs(grid[k][i] - v)) for k, v in zip(keys, ix)]
    chosen = min(cands, key=idx)
    ch = chosen if isinstance(chosen, tuple) else (chosen,)
    am = m.idxmax()
    am = am if isinstance(am, tuple) else (am,)
    return {k: v for k, v in zip(keys, ch)} | {"pseudo_uar": float(m.loc[chosen])}, ch != am


def grid_expected(method: str, cut: dict) -> list:
    """設計の格子から，npz の extra_tag に書いた削り方（MME の sw = 0.1，反復 200）だけを引いた格子（I12 (2)）。"""
    out = []
    for k, vals in GRIDS[method]:
        if k == "sw" and cut.get("mme_sw01"):
            vals = tuple(v for v in vals if float(v) == 0.0)
        if k == "iters" and cut.get("iter200"):
            vals = tuple(v for v in vals if int(v) != 200)
        out.append((k, tuple(vals)))
    return out


def expected_cells() -> set:
    """設計どおりの (生徒, K, 手法) のセルの集合（I12 (1)）。"""
    cells = set()
    for part, meths in PARTS.items():
        for m in meths:
            studs = ("A", "V", "T") if m == "xmuda" else DD.STUDENTS
            ks = DD.KS if m == "lp" else (0,) + DD.KS
            cells |= {(s, K, m) for s in studs for K in ks}
    return cells


def select_T(T: str, d: pd.DataFrame, fz1: dict, n_units: int, cut: dict) -> dict:
    dsel = {tuple([k.split("|")[0], int(k.split("|")[1])]): float(v) for k, v in fz1[T]["delta_sel"].items()}
    select, tie_used, mme_sw0 = {}, {}, 0
    meths = [m for p in PARTS.values() for m in p]
    dm = d[d.method.isin(meths)]
    have = {(s, int(K), m) for s, K, m in dm[["student", "K", "method"]].drop_duplicates().itertuples(index=False)}
    want = expected_cells()
    if have != want:
        sys.exit(f"{T}: (生徒, K, 手法) のセルが設計どおりでない．欠け {sorted(want - have)[:6]}（{len(want - have)} 個），"
                 f"余分 {sorted(have - want)[:6]}（{len(have - want)} 個）（I12）")
    for (s, K, method), g in dm.groupby(["student", "K", "method"]):
        K = int(K)
        grid = grid_expected(method, cut)
        keys = [k for k, _ in grid]
        for k, vals in grid:                                                 # 格子の値が設計（から削り方を引いたもの）とそろうか（I12 (2)）
            if set(g[k].map(float)) != {float(v) for v in vals}:
                sys.exit(f"{T} {s} K={K} {method}: つまみ {k} の値 {sorted(set(g[k].map(float)))} が格子 {list(vals)} と違う（I12）")
        if keys:                                                           # 欠けと重複（格子の点ごとに単位の数だけ行がある）
            cnt = g.groupby(keys).size()
            npts = int(np.prod([len(v) for _, v in grid]))
            if len(cnt) != npts or (cnt != n_units).any():
                sys.exit(f"{T} {s} K={K} {method}: 格子の点 {len(cnt)}/{npts}，点ごとの行の数 {sorted(set(cnt))}（単位 {n_units}）")
        elif len(g) != n_units:
            sys.exit(f"{T} {s} K={K} {method}: 行の数 {len(g)}（単位 {n_units}）")
        if not keys:
            select[f"{s}|{K}|{method}"] = {"pseudo_uar": float(g["uar"].mean())}
            continue
        delta = dsel[(s, K)]
        if method in NO_DEFAULT:
            ch, used = pick_order(g, keys, dict(grid), delta)
        else:
            dai1_select.GRIDS[f"dai2:{method}"] = grid                        # 同じ規則の関数をそのまま使う（格子だけ渡す）
            ch, used = dai1_select.pick(g, f"dai2:{method}", delta, DEFAULTS[method])
        select[f"{s}|{K}|{method}"] = {k: (float(v) if k not in ("iters", "k", "form", "M", "r", "a") else int(v)) for k, v in ch.items()
                                       if k != "pseudo_uar"} | {"pseudo_uar": ch["pseudo_uar"]}
        tie_used[method] = tie_used.get(method, 0) + int(used)
        if method == "mme" and float(ch.get("sw", 0.0)) == 0.0:
            mme_sw0 += 1
    st = d[(d.method == "st")]
    st_edge = {f"{s}|{srcs}": bool(e) for (s, srcs), e in st.groupby(["student", "srcs"])["C_edge"].first().items()}
    return dict(select=select, tie_used=tie_used, mme_sw0_cells=mme_sw0, st_C_edge=st_edge, st_C_edge_n=int(sum(st_edge.values())),
                delta_sel={f"{s}|{K}": v for (s, K), v in dsel.items()})


def main(out: Path, targets=DD.CORPORA, dai1_frozen: Path = ROOT / "frozen" / "dai1_frozen.json", allow_dirty: bool = False) -> None:
    grad = _grad_module()
    GRIDS.update(grad.GRIDS)
    DEFAULTS.update(grad.DEFAULTS)
    ev = out / "eval" / "units"
    if ev.exists() and any(ev.iterdir()):
        sys.exit("評価の回の出力がすでにある．選ぶ段は評価の前にだけ流す（漏れの検査 L8）")
    prod = out.resolve() == (ROOT / "results" / "dai2").resolve()
    commit, dirty = dai1.git_commit()
    if dirty and (prod or not allow_dirty):
        sys.exit("選ぶ段は未コミットの変更があると流さない（試走の置き場だけ --allow-dirty で進める）")
    fz1 = json.load(open(dai1_frozen))
    if prod:                                                               # 実験 1 の detect の後でなければ評価の段が止まる（記録の指摘）
        miss = [T for T in targets if "n_seeds_eval" not in fz1.get(T, {}) or "detect" not in fz1.get(T, {})]
        if miss:
            sys.exit(f"実験 1 の固定のファイルに n_seeds_eval・detect が無い T がある {miss}（dai1_stats.py detect の後に流す）")
    now_sha = {n: dai2.sha(HERE / n) for n in dai2.SCRIPTS}
    prov = [json.loads(x) for x in open(out / "provenance_log.jsonl")] if (out / "provenance_log.jsonl").exists() else []
    prov = [x for x in prov if x.get("stage") == "pseudo" and x.get("T") in targets]
    shas = {json.dumps(x.get("scripts_sha256"), sort_keys=True) for x in prov}
    if not prov or len(shas) != 1 or json.loads(shas.pop()) != now_sha:
        sys.exit("仮のターゲットの回の台本の sha256 が 1 つでないか，選ぶ段の台本と違う（provenance_log.jsonl の段 pseudo．I12 (4)）")
    cuts = set()
    for T in targets:
        for part in ("map", "xmuda"):
            for f in sorted(glob.glob(str(out / "pseudo" / "units" / f"{T}_*_{part}.npz"))):
                cuts.add((part, str(np.load(f)["extra_tag"])))
    mcut = {json.loads(t)["mme_sw01"] for p, t in cuts if p == "map"}
    icut = {json.loads(t)["iter200"] for p, t in cuts}
    if len(mcut) > 1 or len(icut) > 1:
        sys.exit(f"単位によって勾配法の削り方が違う {cuts}")
    cut = {"mme_sw01": bool(mcut.pop()) if mcut else False, "iter200": bool(icut.pop()) if icut else False}
    sel_dir = out / "select"
    sel_dir.mkdir(parents=True, exist_ok=True)
    frozen = {"_meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "dirty": dirty, "trial": not prod,
                        "rule": "docs/設計/実験2_設計.md 1 節・4 節（実験 1 の 8 節の 1）",
                        "dai1_frozen": str(dai1_frozen), "dai1_frozen_sha256": dai2.sha(dai1_frozen),
                        "script_sha256": {n: dai2.sha(HERE / n) for n in dai2.SCRIPTS},
                        "grids": {m: [(k, list(v)) for k, v in gr] for m, gr in GRIDS.items()}, "defaults": DEFAULTS,
                        "no_default_tie_rule": {m: "δ_sel の中で格子の並び（辞書順）で先" for m in NO_DEFAULT},
                        "note": "δ_sel は K 本の抜き方の揺れだけを含み，勾配法の手法の初期化の揺れを含まない",
                        "grad_seeds_eval": list(grad.GRAD_SEEDS_EVAL)}}
    if list(targets) != list(DD.CORPORA):
        frozen["_meta"]["partial_targets"] = list(targets)
    commits, n_dirty = set(), 0
    for T in targets:
        want = sorted(fz1[T]["units"])
        parts = {}
        for part in PARTS:
            dp = load_part(out, T, part)
            got = sorted({f"{c}_f{f}_r{r}" for c, f, r in dp[["c", "fold", "seed"]].drop_duplicates().itertuples(index=False)})
            if got != want:
                sys.exit(f"{T}: 部分 {part} の単位が実験 1 の固定のファイルの単位の一覧と違う（m60）")
            parts[part] = dp
            commits |= set(dp.commit.astype(str))
            nd = int((dp["dirty"].astype(str) == "True").sum())
            if nd and prod:
                sys.exit(f"{T}: 部分 {part} に未コミットの変更で計算した行がある（{nd} 行．I12 (3)）")
            n_dirty += nd
            keyT = dai2.tether_key(dai2.parse_tether(fz1[T]["tether"]))
            if set(dp.tether_key.astype(str)) != {keyT}:
                sys.exit(f"{T}: 部分 {part} の引き戻しが実験 1 の固定のファイルの引き戻しと違う")
        d = pd.concat(parts.values(), ignore_index=True)
        r = select_T(T, d, fz1, len(want), cut)
        frozen[T] = r | {"n_units": len(want), "units": want, "n_seeds_eval_linear": fz1[T].get("n_seeds_eval"),
                         "pseudo_commit": sorted(set(d.commit.astype(str)))}
        print(f"{T}: 単位 {len(want)}，同点の規則が働いたセル {r['tie_used']}，MME で sw = 0 を選んだセル {r['mme_sw0_cells']}，"
              f"S+T の C が格子の端 {r['st_C_edge_n']}/{len(r['st_C_edge'])}", flush=True)
    if len(commits) != 1:
        sys.exit(f"仮のターゲットの行のコミットが 1 つでない {commits}（比較の検査 C6）")
    frozen["_meta"]["grad_cut"] = cut
    frozen["_meta"]["pseudo_commit"] = commits.pop()
    frozen["_meta"]["pseudo_dirty_rows"] = n_dirty
    frozen["_meta"]["pseudo_scripts_sha256"] = now_sha
    json.dump(frozen, open(sel_dir / "frozen.json", "w"), indent=1, ensure_ascii=False)
    h = hashlib.sha256(open(sel_dir / "frozen.json", "rb").read()).hexdigest()
    print("書いた:", sel_dir / "frozen.json", "sha256", h)
    if prod:                                                               # 本番の置き場のときだけ写しを置く（I8）
        fz = ROOT / "frozen"
        fz.mkdir(exist_ok=True)
        shutil.copy(sel_dir / "frozen.json", fz / "dai2_frozen.json")
        print("写し:", fz / "dai2_frozen.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai2"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--dai1-frozen", default=str(ROOT / "frozen" / "dai1_frozen.json"))
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    main(Path(a.out_dir), a.targets, Path(a.dai1_frozen), a.allow_dirty)
