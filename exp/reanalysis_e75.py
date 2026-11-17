"""【再解析】E75（DAI 論文のステップ 1）：TAC 版の模擬査読 1 回目の指摘のうち，筋 1〜3 について保存済みの結果から
学び直しをせずに CPU で出せるものをまとめて計算する。新しい学習はしない（GPU は使わない）。

設計は `docs/exploratory-analyses.md` の「## E75．」の節（関門 1 の指摘を受けて直した版）。
項目（括弧は答える指摘．ML = `paper/reviews/round1/review-ml.md`，感 = `review-affect.md`）

  1  地図の区間を依存に合わせて作り直す：補正した resampled t と，IEMOCAP の話者でつないだテストの単位の t（ML Major-1，感 Minor-1）
  2  IEMOCAP 台本をターゲットから除いた集計（8 種類．ML Minor-2）
  3  加法モデルの調整した説明率と，値を並べ替えた基準（ML Minor-5）
  4  ターゲットのみの C の規則を変えたときの利得（ML Minor-6）
  5  合併の得を o の分と d の分に分ける順序（ML Major-3 (3)，感 Minor-5）
  6  4 つの切片で 6 対の o を同時に直せる量（ML Major-3 (1)(2)）
  7  式 (2) の閾値の損と測った閾値の損を K ごとに（ML Major-3 (6)）
  8  向きの損が負の升の数と扱い，Δ の 2 つの半分の値，縮小 LDA の縮小の強さ（ML Major-3 (5)，ML Major-6）
  9  中心化の得とクラスの割合の関係（ML Minor-15）
  10 E12 の α を選んだ合併（反復 3 回）と，C の選び方をそろえた対照（ML Minor-7）
  11 q の当てはめの診断：残差，区間，別の関数形，K を 1 つ抜いた予言（ML Major-4，感 Minor-6）
  12 λ の予言と実測の順位相関を K ごと・ターゲットの中で（ML Major-4 (5)）
  13 ソースの向きの効率 e を，ほかのコーパスを仮のターゲットにして推す（ML Optional-1，感 Optional-2）

    ./.venv/bin/python exp/reanalysis_e75.py --out results/e75                     # 全部
    ./.venv/bin/python exp/reanalysis_e75.py --out <dir> --kinds audio_wavlm --items 1 5   # 小さく流す

入力の場所はすべて引数（既定は下の DEFAULTS）で受ける。環境変数 MERDA_OUT_ROOT・MERDA_MAP_ROOT が
与えられていたら止まる（`exp/pair_table.py` の定数がそれで黙って変わるため）。出力先にファイルがあれば止まる
（既存の結果を上書きしない）。出力：`<out>/e75_<項目>_*.csv`（1 行目に `#` で設定とコミット番号），`gaps.csv`
（入力の欠けの数），`checks.csv`（保存済みの値の再現の検算），`e75_assumptions.csv`（模型と式の仮定を測る量），
`provenance.json`（設定，入力，コミット番号，部品の状態，所要時間），`code/<開始時刻>/`（コードの写しと git diff）。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU を使わない（ほかの計算が使う）．import より前に置く

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from scipy.special import softmax  # noqa: E402
from sklearn.covariance import LedoitWolf, ledoit_wolf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _v in ("MERDA_OUT_ROOT", "MERDA_MAP_ROOT"):
    if os.environ.get(_v):
        raise SystemExit(f"環境変数 {_v} が与えられている．この台本は入力の場所を引数でだけ受ける．外してから流す")
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(ROOT / "features"))
import common  # noqa: E402
import map as M  # noqa: E402   choose_c（C の同点の規則）と snapshot_code（コードの写し）だけを使う
import pair_table  # noqa: E402
from ablate import balanced_subset, fit_class_bias  # noqa: E402
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_direction_bound import q_fit  # noqa: E402
from explore_geometry import standardize  # noqa: E402
from explore_pairs import ba, measure, own_separation  # noqa: E402
from explore_synergy import O_MIN, PAIRS, d_o  # noqa: E402
from split import speaker_folds  # noqa: E402
from summarize_map import K_ORDER, kstar_table, load as load_map_dir, t_interval  # noqa: E402

KINDS_ALL = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video",
             "text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
TEXT_KINDS = ["text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text"]
MAIN4 = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
SHORT = {"audio_wavlm": "A", "audio_hubert": "A(HuBERT)", "video": "V", "fuse-audio-video": "AV",
         "text_roberta": "T", "fuse-audio-text": "AT", "fuse-video-text": "VT", "fuse-audio-video-text": "AVT"}
IEMOCAP = ("IEMOCAP-impro", "IEMOCAP-script")
SCRIPT = "IEMOCAP-script"
EMO = ["neutral", "happy", "sad", "angry"]
PAIR_IDX = [f"{a}-{b}" for a, b in PAIRS]                       # explore_synergy.PAIRS の順
PAIR_NAME_TO_IDX = {f"{EMO[a]}-{EMO[b]}": f"{a}-{b}" for a, b in PAIRS}
C_GRID = ("0.01", "0.1", "1", "10")
KS3 = ("1", "5", "10")                                          # 上流の点数が保存されている K（merged_scores・tgt_scores・transfer_scores）
ITEMS = list(range(1, 14))
DEN_SMALL = 0.5          # D11：割合の分母（点）がこれ未満で 0 より大きい単位は「小さい」と数える
LOC_MIN = 0.1            # D11：項目 6 の割合の分母（位置の損．点）がこれ以下なら割合を出さない
DELTA_MIN = 0.1          # 項目 13：Δ（own_d）がこれ以下の（ターゲット，対）を数える
Q_GRID = np.logspace(-1, 3, 400)                                # explore_direction_bound.q_fit と同じ格子
BETA_GRID = np.linspace(0.25, 3.0, 56)
SCALE_GRID = np.linspace(0.5, 2.0, 61)
LOKO_BETTER = 0.8        # 項目 11：K を 1 つ抜いた予言の誤差が F0 の 0.8 倍未満なら「良い」
E25_SRC = "0 ソースのみ（学び直し）"                          # `results/interventions_<kind>/` の E25 のソースのみの行

DEFAULTS = {
    "fixc_root": "results/fixc_map",                           # C を固定した地図（<kind>/<T>.csv）
    "cv_map_prefix": "results/map_",                            # 交差検証の地図（<prefix><kind>/<T>.csv）
    "stageb_main": "results/fixc_analysis/explore_relation",    # 段階 B（主な 4 種類）：e15b_rows.csv・e15b_pairs.csv・e15b_tests.csv・e13_pairs.csv
    "stageb_text": "results/fixc_text/explore_relation",        # 段階 B（言葉系 4 種類）
    "e29_pairs": "results/explore_relation/snapshots/20261011-1524_e29/e29_pairs.csv",
    "e29_tests": "results/explore_relation/snapshots/20261011-1524_e29/e29_tests.csv",
    "e28t9_targets": "results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_targets.csv",
    "e28t9_pairs": "results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_pairs.csv",
    "decomp_share_main": "results/fixc_analysis/explore_relation/decomp_share.csv",
    "decomp_share_text": "results/text30/explore_relation/decomp_share.csv",
    "kstar_cluster_main": "results/explore_relation/kstar_cluster.csv",
    "kstar_cluster_text": "results/text30/explore_relation/kstar_cluster.csv",
    "e12_wt": "results/fixc_analysis/explore_relation/e12_wt.csv",
    "shift_prefix": "results/shift_",                           # <prefix><kind>/logits/<S>.npz，label.csv，labeling.csv
    "tgt_scores_prefix": "results/tgt_scores_",
    "transfer_scores_prefix": "results/transfer_scores_",
    "mech_prefix": "results/mech_",
    "interventions_prefix": "results/interventions_",
}


# ---------------------------------------------------------------- 共通

class Ctx:
    """引数，入力の欠け，検算，仮定の量，項目どうしで共有する表を持つ。"""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out)
        self.gaps: list[dict] = []
        self.checks: list[dict] = []
        self.assumptions: list[dict] = []
        self.records: list[dict] = []
        self.inputs: dict[str, dict] = {}
        self._meta: dict[tuple[str, str], pd.DataFrame] = {}
        self.dims: dict[str, int] = {}
        self._e13: dict[str, pd.DataFrame] = {}
        self.cache: dict[str, object] = {}
        self.header: dict = {}

    def p(self, key: str) -> Path:
        v = Path(getattr(self.a, key))
        return v if v.is_absolute() else ROOT / v

    def note_input(self, path: Path) -> None:
        path = Path(path)
        if str(path) in self.inputs or not path.exists():
            return
        info = {"bytes": path.stat().st_size if path.is_file() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))}
        if path.is_file() and path.suffix == ".csv":
            with open(path, encoding="utf-8") as f:
                head = f.readline().strip()
            if head.startswith("#"):
                info["first_comment"] = head[:400]
        self.inputs[str(path)] = info

    def gap(self, item: int, kind: str, what: str, expected, found, detail: str = "") -> None:
        self.gaps.append({"item": item, "kind": kind, "what": what, "expected": expected,
                          "found": found, "detail": detail})

    def check(self, item: int, kind: str, what: str, value: float, tol: float, detail: str = "") -> None:
        ok = bool(np.isfinite(value) and abs(value) <= tol) if not isinstance(value, str) else False
        self.checks.append({"item": item, "kind": kind, "what": what, "value": value, "tol": tol,
                            "ok": ok, "detail": detail})

    def record(self, item: int, kind: str, what: str, max_abs_diff: float, frac_diff: float, n: int, unit: str,
               signed_mean: float = np.nan, frac_c_differs: float = np.nan) -> None:
        """別々に学んだ同じ設定の分類器どうしの差の記録（一致を求める検算ではない）．符号つきの平均と，
        2 つの学習で選ばれた C が違う行の割合も書ける（C が違う行の差は解き方の揺れではなく選択の違い）。"""
        self.records.append({"item": item, "kind": kind, "what": what, "max_abs_diff": max_abs_diff,
                             "frac_rows_diff_gt_1e-6": frac_diff, "n": n, "unit": unit,
                             "signed_mean_diff": signed_mean, "frac_rows_C_differs": frac_c_differs})

    def assume(self, item: int, kind: str, assumption: str, measure_: str, value: float, criterion: str,
               holds, detail: str = "") -> None:
        self.assumptions.append({"item": item, "kind": kind, "assumption": assumption, "measure": measure_,
                                 "value": value, "criterion": criterion, "holds": holds, "detail": detail})

    def meta(self, kind: str, group: str) -> pd.DataFrame:
        """発話一覧（sample_id・speaker_group・label_4）．特徴は `load_any` で読んで次元だけ残す（取り決め 1〜6 を通すため）。"""
        key = (kind, group)
        if key not in self._meta:
            df, X = load_any(group, kind)
            self._meta[key] = df[["sample_id", "speaker_group", "label_4"]].reset_index(drop=True)
            self.dims[kind] = int(X.shape[1])
        return self._meta[key]

    def e13(self, kind: str) -> pd.DataFrame:
        """E13 の感情の対ごとの表（d，o，own_d，損）．主な 7 種類は C 固定の解析の保存済みの表，
        HuBERT は保存が無いので `explore_pairs.measure` で作る（E28-9 の台本と同じ扱い．`results/shift_audio_hubert`
        を固定で読むので，`--shift-prefix` が既定と違うときは main で止める）。"""
        if kind not in self._e13:
            path = self.p("stageb_main") / "e13_pairs.csv"
            self.note_input(path)
            e = pd.read_csv(path)
            e = e[e.kind == kind]
            if e.empty:
                e = measure(kind).assign(kind=kind)
                self.cache.setdefault("e13_computed", []).append(kind)
            e = e.copy()
            e["pair_idx"] = e.pair.map(PAIR_NAME_TO_IDX)
            self._e13[kind] = e
        return self._e13[kind]

    def e13_note(self, kind: str) -> str:
        return "同じ計算どうし（HuBERT の E13 は台本の中で作る）" if kind in self.cache.get("e13_computed", []) else ""

    def write(self, name: str, df: pd.DataFrame) -> None:
        """1 行目に `#` で設定とコミット番号を書き，続けて CSV を書く（読むときは comment="#"）。"""
        path = self.out / name
        if path.exists():
            raise SystemExit(f"{path} がすでにある．上書きしない")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# " + json.dumps({**self.header, "file": name}, ensure_ascii=False, default=str) + "\n")
            df.to_csv(f, index=False)
        self.cache.setdefault("written", []).append({"file": name, "rows": int(len(df))})


def same_corpus(t: str, s: str) -> bool:
    return GROUP_SPEC[t][0] == GROUP_SPEC[s][0]


def corpus(g: str) -> str:
    return GROUP_SPEC[g][0]


def read_csv(ctx: Ctx, path: Path, **kw) -> pd.DataFrame:
    ctx.note_input(path)
    return pd.read_csv(path, comment="#", **kw)


def spear(x, y) -> tuple[float, float, int]:
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan, np.nan, int(ok.sum())
    r, p = stats.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def unit_interval(vals) -> dict:
    """単位（ターゲットなど）ごとの値の平均と，自由度 n − 1 の t 区間，範囲，正の数，両側の符号検定の p。"""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    n = len(v)
    m, lo, hi = t_interval(v) if n else (np.nan, np.nan, np.nan)
    nz = v[v != 0]
    p_sign = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else np.nan
    return {"n_units": n, "mean": m, "t_lo": lo, "t_hi": hi, "min": float(v.min()) if n else np.nan,
            "max": float(v.max()) if n else np.nan, "n_pos": int((v > 0).sum()), "p_sign": p_sign}


def within_target_perm(groups: list[tuple[np.ndarray, np.ndarray]], n_perm: int, rng: np.random.Generator) -> dict:
    """ターゲットの中の順位相関の中央値と，ソースをターゲットの中で（全ターゲット同時に）並べ替えた帰無の分布。
    帰無の 95% 点，片側 p（中央値が観測以上の割合），ターゲットの中の相関の平均の t 区間（自由度 n − 1）を返す。"""
    def rk(a):
        return stats.rankdata(a)
    prepared = []
    for x, y in groups:
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        rx, ry = rk(x), rk(y)
        prepared.append(((rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()))
    if not prepared:
        return {"n_targets": 0}
    obs = np.array([float(np.mean(a * b)) for a, b in prepared])
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.median([float(np.mean(a * b[rng.permutation(len(b))])) for a, b in prepared])
    med = float(np.median(obs))
    m, lo, hi = t_interval(obs)
    return {"n_targets": len(obs), "median_rho": med, "min_rho": float(obs.min()), "max_rho": float(obs.max()),
            "mean_rho": m, "mean_rho_t_lo": lo, "mean_rho_t_hi": hi,
            "perm_q95_median": float(np.quantile(null, 0.95)),
            "perm_p_median": float((1 + (null >= med - 1e-12).sum()) / (1 + n_perm))}


def kstar_from(tab: pd.DataFrame, lo_col: str) -> pd.DataFrame:
    """`summarize_map.kstar_table` を，任意の下端の列で呼ぶ（K* ＝ 下端が 0 を上回る最大の K）。"""
    x = tab[["target", "source", "K", "n_rep", lo_col]].rename(columns={lo_col: "diff_lo"}).copy()
    x["K"] = pd.Categorical(x["K"].astype(str), K_ORDER, ordered=True)
    return kstar_table(x)[["target", "source", "K_star"]]


def run_git(*cmd) -> str:
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:                                         # noqa: BLE001
        return f"error: {e}"


def git_state() -> dict:
    """dirty は追跡しているファイルの変更だけで決める（未追跡のファイルでは真にしない）。"""
    me = Path(__file__).resolve()
    return {"commit": run_git("git", "rev-parse", "HEAD"),
            "dirty": bool(run_git("git", "status", "--porcelain", "--untracked-files=no")),
            "script_status": run_git("git", "status", "--porcelain", "--", str(me.relative_to(ROOT))),
            "script_sha256": hashlib.sha256(me.read_bytes()).hexdigest()}


def module_state() -> dict:
    """読み込んだ MER-DA の部品ごとの git の状態と sha256（台本の外の変更を後から確かめるため）。"""
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if not rel.parts or rel.parts[0] not in ("exp", "measure", "features") or not p.is_file():
            continue
        out[str(rel)] = {"status": run_git("git", "status", "--porcelain", "--", str(rel)) or "clean",
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def test_sets_from_idx(z, reps) -> dict[int, frozenset]:
    """点数の npz の `test_idx_r{r}` から反復ごとのテストの集合（発話の行番号の集合）を作る。"""
    return {int(r): frozenset(np.asarray(z[f"test_idx_r{r}"]).tolist()) for r in reps}


def connected_units(sets: dict[int, frozenset]) -> dict[int, int]:
    """要素（話者や発話）を共有する集合をつないだ連結成分の番号を，反復ごとに返す（S10）。"""
    uniq = sorted(set(sets.values()), key=sorted)
    parent = list(range(len(uniq)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            if uniq[i] & uniq[j]:
                parent[find(i)] = find(j)
    comp = {c: find(i) for i, c in enumerate(uniq)}
    ids = {v: k for k, v in enumerate(sorted(set(comp.values())))}
    return {r: ids[comp[s]] for r, s in sets.items()}


# ---------------------------------------------------------------- 項目 1：区間

def test_clusters(ctx: Ctx, kind: str, target: str, reps) -> dict[int, frozenset]:
    """反復 r のテストの話者群の集合（`exp/map.py` の test_mask と同じ：speaker_folds(seed=r) の r mod 5 番）。"""
    sp = ctx.meta(kind, target).speaker_group.astype(str).reset_index(drop=True)
    out = {}
    for r in reps:
        f = speaker_folds(sp, k=5, seed=int(r))
        out[int(r)] = frozenset(sp[(f == int(r) % 5).to_numpy()].unique())
    return out


def _unit_t(x: np.ndarray, ids: np.ndarray) -> tuple[int, float, float, float]:
    cm = np.array([x[ids == c].mean() for c in np.unique(ids)])
    n_c = len(cm)
    if n_c < 2:
        return n_c, np.nan, np.nan, np.nan
    mc, lc, hc = t_interval(cm)
    se = float(np.std(cm, ddof=1) / np.sqrt(n_c))
    return n_c, lc, hc, float(2 * stats.t.sf(abs(mc) / se, n_c - 1)) if se > 0 else np.nan


def intervals(x: np.ndarray, reps: np.ndarray, rho: float, sets: dict | None) -> dict:
    """1 つの組 × K の反復ごとの差 x から，素朴な t，補正した resampled t，テストの単位の t を作る。
    テストの単位は 2 通り：A6 と同じ「異なるテストの話者の集合」（検算用）と，話者を共有する集合をつないだ連結成分（主．S10）。"""
    J = len(x)
    out = {"n_rep": J}
    m, lo, hi = t_interval(x)                                  # 反復を独立とみなす（今の地図の区間）
    out.update(mean=m, naive_lo=lo, naive_hi=hi)
    if J >= 2:
        sd = float(np.std(x, ddof=1))
        se_c = sd * np.sqrt(1.0 / J + rho)                     # Nadeau & Bengio (2003)
        tq = stats.t.ppf(0.975, J - 1)
        out.update(corr_lo=m - tq * se_c, corr_hi=m + tq * se_c,
                   corr_p=float(2 * stats.t.sf(abs(m) / se_c, J - 1)) if se_c > 0 else np.nan)
    else:
        out.update(corr_lo=np.nan, corr_hi=np.nan, corr_p=np.nan)
    if sets is not None:
        keys = {c: i for i, c in enumerate(sorted({sets[int(r)] for r in reps}, key=sorted))}
        n_c, lc, hc, pc = _unit_t(x, np.array([keys[sets[int(r)]] for r in reps]))
        out.update(n_sets=n_c, a6_lo=lc, a6_hi=hc, a6_p=pc)
        comp = connected_units({int(r): sets[int(r)] for r in reps})
        n_u, lu, hu, pu = _unit_t(x, np.array([comp[int(r)] for r in reps]))
        out.update(n_units=n_u, unit_lo=lu, unit_hi=hu, unit_p=pu)
    return out


def gain_table(ctx: Ctx, kind: str, item: int, tgt_col: str = "uar",
               merged_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """反復ごとの得（合併 − ターゲットのみ）．tgt_col でターゲットのみの UAR の列を替える（項目 4）。
    返すのは (得の表, 地図の表)。"""
    d_dir = merged_dir if merged_dir is not None else ctx.p("fixc_root") / kind
    for f in sorted(d_dir.glob("*.csv")):
        ctx.note_input(f)
    df = load_map_dir(d_dir)
    if not np.isfinite(df[df.method != "src_only"]["uar"]).all():
        raise SystemExit(f"{d_dir} の uar に NaN か Inf がある")
    if tgt_col == "uar" and merged_dir is None:
        pair_table.MAP_ROOT = ctx.p("fixc_root")              # 環境変数ではなく引数で地図の場所を決める
        g = pair_table.gains(kind)
    else:
        t = df[df.method == "tgt_only"].set_index(["target", "K", "rep"])[tgt_col]
        m = df[df.method == "merged"].copy()
        m["gain"] = m["uar"].to_numpy() - t.loc[list(zip(m.target, m.K, m.rep))].to_numpy()
        g = m[["target", "source", "K", "rep", "gain"]].reset_index(drop=True)
    if not np.isfinite(g.gain).all():
        raise SystemExit(f"{kind}：得に NaN か Inf がある（{tgt_col}）")
    return g, df


def interval_table(ctx: Ctx, kind: str, g: pd.DataFrame, df: pd.DataFrame, item: int,
                   n_rep_expected: int | None, label: str = "") -> pd.DataFrame:
    tg = df[df.method == "tgt_only"]
    allrow = tg[tg.K == "all"]
    rho_t = (allrow.n_test / allrow.n_tgt).groupby(allrow.target).mean()
    groups = sorted(g.target.unique())
    sets = {t: test_clusters(ctx, kind, t, sorted(g[g.target == t].rep.unique())) for t in groups if t in IEMOCAP}
    rows, bad = [], []
    for (t, s, K), h in g.groupby(["target", "source", "K"]):
        h = h.sort_values("rep")
        x, reps = h.gain.to_numpy(float), h.rep.to_numpy()
        if n_rep_expected is not None and len(x) != n_rep_expected:
            bad.append(len(x))
        r = intervals(x, reps, float(rho_t.get(t, np.nan)), sets.get(t))
        main = "units" if t in IEMOCAP else "corrected"
        pre = "unit" if main == "units" else "corr"
        r.update(kind=kind, target=t, source=s, K=K, same_corpus=same_corpus(t, s), rho=float(rho_t.get(t, np.nan)),
                 main_method=main, main_lo=r[f"{pre}_lo"], main_hi=r[f"{pre}_hi"], main_p=r[f"{pre}_p"])
        rows.append(r)
    if bad:
        ctx.gap(item, kind, f"反復の数が {n_rep_expected} でない組 × K の数{('（' + label + '）') if label else ''}",
                n_rep_expected, len(bad), f"見つかった反復の数：{sorted(set(bad))}")
    P = pd.DataFrame(rows)
    # Holm：種類 × K の中で，同じコーパスの組を除いた組を 1 つの族にする
    P["main_p_holm"] = np.nan
    for (_, K), h in P[~P.same_corpus].groupby(["kind", "K"]):
        ok = h.main_p.notna()
        P.loc[h.index[ok], "main_p_holm"] = pair_table.holm(h.main_p[ok].to_numpy())
    return P


def target_level(P: pd.DataFrame, col: str = "mean") -> pd.DataFrame:
    """種類 × K ごとに，ターゲットごとの平均（ソースで平均）を単位にした区間（S5）。"""
    rows = []
    for (kind, K), h in P.groupby(["kind", "K"]):
        tm = h.groupby("target")[col].mean() * 100
        u = unit_interval(tm.to_numpy())
        rows.append({"kind": kind, "K": K, **{f"tgt_{k}": v for k, v in u.items()}})
    return pd.DataFrame(rows)


def item1(ctx: Ctx) -> None:
    allP, testsets = [], []
    for kind in ctx.a.kinds:
        g, df = gain_table(ctx, kind, 1)
        groups = groups_for(kind)
        exp_pairs = len(groups) * (len(groups) - 1)
        found = g.groupby(["target", "source"]).ngroups
        if found != exp_pairs:
            ctx.gap(1, kind, "組の数", exp_pairs, found)
        P = interval_table(ctx, kind, g, df, 1, n_rep_expected=30)
        allP.append(P)
        # 検算：素朴な区間が地図のまとめ（summary/pairs.csv．summarize_map.pair_table の出力）と一致するか
        sp = ctx.p("fixc_root") / kind / "summary" / "pairs.csv"
        if sp.exists():
            ctx.note_input(sp)
            ref = pd.read_csv(sp, dtype={"K": str}).set_index(["target", "source", "K"])
            mine = P.set_index(["target", "source", "K"])
            j = mine.join(ref[["diff_lo", "diff_hi", "diff_mean"]], how="inner")
            ctx.check(1, kind, "素朴な t 区間の下端・上端・平均と summary/pairs.csv の差の最大",
                      float(np.nanmax(np.abs(np.r_[j.naive_lo - j.diff_lo, j.naive_hi - j.diff_hi, j["mean"] - j.diff_mean]))),
                      1e-9, f"{len(j)} 行")
        # テストの話者の集合の数（S6．感 Minor-1 の「異なるテストの折りの数」）と，話者でつないだ単位の数
        for t in groups:
            reps = sorted(df[(df.target == t) & (df.method == "tgt_only")].rep.unique())
            cl = test_clusters(ctx, kind, t, reps)
            cl10 = {r: cl[r] for r in reps if r < 10}
            testsets.append({"kind": kind, "target": t, "n_reps": len(reps),
                             "n_distinct_test_sets": len(set(cl.values())),
                             "n_units_connected": len(set(connected_units(cl).values())) if t in IEMOCAP else np.nan,
                             "n_distinct_test_sets_reps0_9": len(set(cl10.values())),
                             "n_speaker_groups": int(ctx.meta(kind, t).speaker_group.nunique())})
    P = pd.concat(allP, ignore_index=True)
    ctx.cache["item1_pairs"] = P
    # K*：素朴，補正（全ターゲットで同じ作り方），主（IEMOCAP は話者でつないだ単位，ほかは補正），A6 の集合
    ks = []
    for kind, h in P.groupby("kind"):
        k = None
        for name, col in (("naive", "naive_lo"), ("corrected", "corr_lo"), ("main", "main_lo"), ("a6", "a6_lo"),
                          ("units", "unit_lo")):
            if col not in h or h[col].notna().sum() == 0:
                continue
            x = kstar_from(h, col).rename(columns={"K_star": f"K_star_{name}"})
            if name in ("a6", "units"):                          # テストの単位の区間は IEMOCAP のターゲットだけで作る
                x.loc[~x.target.isin(IEMOCAP), f"K_star_{name}"] = np.nan
            k = x if k is None else k.merge(x, on=["target", "source"], how="outer")
        ks.append(k.assign(kind=kind))
    KS = pd.concat(ks, ignore_index=True)
    KS["same_corpus"] = [same_corpus(t, s) for t, s in zip(KS.target, KS.source)]
    ctx.cache["item1_kstar"] = KS
    # 検算：素朴な K* が summary/kstar.csv と，A6 の集合の K* が kstar_cluster.csv（C 固定の地図）と一致するか
    for kind, h in KS.groupby("kind"):
        kp = ctx.p("fixc_root") / kind / "summary" / "kstar.csv"
        if kp.exists():
            ctx.note_input(kp)
            ref = pd.read_csv(kp, dtype=str).set_index(["target", "source"]).K_star
            j = h.set_index(["target", "source"]).K_star_naive
            ctx.check(1, kind, "素朴な K* が summary/kstar.csv と食い違う組の数", float((j != ref.reindex(j.index)).sum()), 0)
        for key in ("kstar_cluster_main", "kstar_cluster_text"):
            cp = ctx.p(key)
            if cp.exists() and "K_star_a6" in h:
                ctx.note_input(cp)
                ref = pd.read_csv(cp, dtype={"K_star_cluster": str})
                ref = ref[(ref["map"] == "fixc") & (ref.kind == kind)].set_index(["target", "source"]).K_star_cluster
                if len(ref):
                    j = h.dropna(subset=["K_star_a6"]).set_index(["target", "source"]).K_star_a6
                    common_idx = j.index.intersection(ref.index)
                    ctx.check(1, kind, f"A6 の集合を単位にした K* が {cp.name}（{cp.parent.parent.name}）と食い違う組の数",
                              float((j.loc[common_idx] != ref.loc[common_idx]).sum()), 0, f"{len(common_idx)} 組")
    # 種類 × K の数え上げ（同じコーパスの組を除く）．IEMOCAP 以外の組だけの数と，ターゲットを単位にした区間も付ける
    cnt = []
    Q = P[~P.same_corpus]
    TL = target_level(Q).set_index(["kind", "K"])
    for (kind, K), h in Q.groupby(["kind", "K"]):
        row = {"kind": kind, "short": SHORT.get(kind, kind), "K": K, "n_pairs": len(h), "mean_gain": h["mean"].mean() * 100}
        for name, lo, hi in (("naive", "naive_lo", "naive_hi"), ("corrected", "corr_lo", "corr_hi"),
                             ("main", "main_lo", "main_hi")):
            row[f"up_{name}"], row[f"down_{name}"] = int((h[lo] > 0).sum()), int((h[hi] < 0).sum())
        sig = h.main_p_holm < 0.05
        row["up_main_holm"], row["down_main_holm"] = int((sig & (h["mean"] > 0)).sum()), int((sig & (h["mean"] < 0)).sum())
        non = h[~h.target.isin(IEMOCAP)]
        row["n_pairs_nonIEMOCAP"] = len(non)
        row["up_corrected_nonIEMOCAP"], row["down_corrected_nonIEMOCAP"] = int((non.corr_lo > 0).sum()), int((non.corr_hi < 0).sum())
        iem = h[h.target.isin(IEMOCAP)]
        row["n_pairs_IEMOCAP"] = len(iem)
        row["up_units_IEMOCAP"], row["down_units_IEMOCAP"] = int((iem.unit_lo > 0).sum()) if len(iem) else 0, \
            int((iem.unit_hi < 0).sum()) if len(iem) else 0
        row.update(TL.loc[(kind, K)].to_dict())
        cnt.append(row)
    C = pd.DataFrame(cnt)
    C["K"] = pd.Categorical(C.K, K_ORDER, ordered=True)
    # ターゲットごとの内訳
    bt = []
    for (kind, K, t), h in Q.groupby(["kind", "K", "target"]):
        bt.append({"kind": kind, "K": K, "target": t, "n_sources": len(h), "mean_gain": h["mean"].mean() * 100,
                   "main_method": h.main_method.iloc[0], "up_main": int((h.main_lo > 0).sum()),
                   "down_main": int((h.main_hi < 0).sum()), "up_corrected": int((h.corr_lo > 0).sum()),
                   "down_corrected": int((h.corr_hi < 0).sum())})
    dist = []
    for kind, h in KS[~KS.same_corpus].groupby("kind"):
        for name in ("naive", "corrected", "main"):
            col = f"K_star_{name}"
            if col in h:
                vc = h[col].value_counts()
                dist.append({"kind": kind, "short": SHORT.get(kind, kind), "interval": name, "n_pairs": len(h),
                             **{f"K*={k}": int(vc.get(k, 0)) for k in K_ORDER + ["none"]}})
    # 仮定：補正した t の幅 ÷ 話者でつないだ単位の t の幅（IEMOCAP の組 × K）
    for kind, h in Q[Q.target.isin(IEMOCAP)].groupby("kind"):
        w = ((h.corr_hi - h.corr_lo) / (h.unit_hi - h.unit_lo)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(w):
            med = float(w.median())
            ctx.assume(1, kind, "補正した resampled t の ρ = n_test/n_pool が反復の間の相関を表す",
                       "IEMOCAP の組 × K の（補正した t の幅 ÷ 話者でつないだ単位の t の幅）の中央値", med,
                       "0.8〜1.25", bool(0.8 <= med <= 1.25), f"{len(w)} 組 × K")
    ctx.write("e75_1_pairs.csv", P)
    ctx.write("e75_1_counts.csv", C.sort_values(["kind", "K"]))
    ctx.write("e75_1_by_target.csv", pd.DataFrame(bt))
    ctx.write("e75_1_kstar.csv", KS)
    ctx.write("e75_1_kstar_dist.csv", pd.DataFrame(dist))
    ctx.write("e75_1_testsets.csv", pd.DataFrame(testsets))


def need_item1(ctx: Ctx) -> None:
    if "item1_pairs" not in ctx.cache:
        item1(ctx)


# ---------------------------------------------------------------- 項目 2：IEMOCAP 台本を除いた集計（8 種類）

def item2(ctx: Ctx) -> None:
    need_item1(ctx)
    P, KS = ctx.cache["item1_pairs"], ctx.cache["item1_kstar"]
    rows, tg, kd = [], [], []
    for kind in ctx.a.kinds:
        h = P[(P.kind == kind) & ~P.same_corpus]
        for subset, x in (("all_targets", h), ("excl_script_target", h[h.target != SCRIPT])):
            for K, xk in x.groupby("K"):
                rows.append({"kind": kind, "short": SHORT[kind], "subset": subset, "K": K, "n_pairs": len(xk),
                             "n_targets": xk.target.nunique(), "mean_gain": xk["mean"].mean() * 100,
                             "median_gain": xk["mean"].median() * 100,
                             "up_main": int((xk.main_lo > 0).sum()), "down_main": int((xk.main_hi < 0).sum())})
            k = KS[(KS.kind == kind) & ~KS.same_corpus]
            k = k if subset == "all_targets" else k[k.target != SCRIPT]
            vc = k.K_star_main.value_counts()
            kd.append({"kind": kind, "short": SHORT[kind], "subset": subset, "interval": "main",
                       "n_pairs": len(k), **{f"K*={kk}": int(vc.get(kk, 0)) for kk in K_ORDER + ["none"]}})
        for (t, K), xk in h.groupby(["target", "K"]):
            tg.append({"kind": kind, "short": SHORT[kind], "target": t, "K": K, "n_sources": len(xk),
                       "mean_gain": xk["mean"].mean() * 100})
    ctx.write("e75_2_script_excluded.csv", pd.DataFrame(rows))
    ctx.write("e75_2_kstar_dist.csv", pd.DataFrame(kd))
    ctx.write("e75_2_targets.csv", pd.DataFrame(tg))


# ---------------------------------------------------------------- 項目 3：加法モデル

def additive(y: np.ndarray, d: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> dict:
    Xt, Xts = pair_table.design(d, True, False), pair_table.design(d, True, True)
    n, k_t, k_ts = len(y), Xt.shape[1] - 1, Xts.shape[1] - 1
    r2t, _ = pair_table.r2(y, Xt)
    r2ts, _ = pair_table.r2(y, Xts)
    adj = lambda r, k: 1 - (1 - r) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else np.nan  # noqa: E731
    null_ts, null_t = np.empty(n_perm), np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(y)
        null_t[b] = pair_table.r2(yp, Xt)[0]
        null_ts[b] = pair_table.r2(yp, Xts)[0]
    # 1 つのターゲットを除いたときの調整した R²_ts の範囲（S5）
    loto = []
    for t in sorted(d.target.unique()):
        m = (d.target != t).to_numpy()
        dd = d[m]
        if len(dd) < 8 or dd.target.nunique() < 2:
            continue
        X2 = pair_table.design(dd, True, True)
        r, _ = pair_table.r2(y[m], X2)
        nn, kk = len(dd), X2.shape[1] - 1
        loto.append(1 - (1 - r) * (nn - 1) / (nn - kk - 1) if nn - kk - 1 > 0 else np.nan)
    loto = np.array([v for v in loto if np.isfinite(v)])
    return {"n": n, "n_params": k_ts + 1, "df_resid": n - k_ts - 1, "r2_target": r2t, "r2_both": r2ts,
            "r2_source": r2ts - r2t, "adj_r2_target": adj(r2t, k_t), "adj_r2_both": adj(r2ts, k_ts),
            "null_expect_r2_both": k_ts / (n - 1), "perm_mean_r2_both": float(null_ts.mean()),
            "perm_q95_r2_both": float(np.quantile(null_ts, 0.95)),
            "perm_p_r2_both": float((1 + (null_ts >= r2ts - 1e-12).sum()) / (1 + n_perm)),
            "perm_mean_r2_target": float(null_t.mean()),
            "perm_p_r2_target": float((1 + (null_t >= r2t - 1e-12).sum()) / (1 + n_perm)),
            "loto_adj_r2_both_min": float(loto.min()) if len(loto) else np.nan,
            "loto_adj_r2_both_max": float(loto.max()) if len(loto) else np.nan}


def item3(ctx: Ctx) -> None:
    need_item1(ctx)
    P, KS = ctx.cache["item1_pairs"], ctx.cache["item1_kstar"]
    rank = pair_table.KSTAR_RANK
    rows = []
    for kind in ctx.a.kinds:
        h = P[P.kind == kind]
        piv = h.pivot_table(index=["target", "source"], columns="K", values="mean").reset_index()
        k = KS[KS.kind == kind].set_index(["target", "source"])
        for name in ("corrected", "main"):
            piv[f"kstar_{name}"] = [rank.get(k.loc[(t, s), f"K_star_{name}"], np.nan) for t, s in zip(piv.target, piv.source)]
        piv["kstar_naive_nonIEMOCAP"] = [np.nan if t in IEMOCAP else rank.get(k.loc[(t, s), "K_star_naive"], np.nan)
                                         for t, s in zip(piv.target, piv.source)]
        piv["same_corpus"] = [same_corpus(t, s) for t, s in zip(piv.target, piv.source)]
        for pair_set, base in (("excl_same_corpus", piv[~piv.same_corpus]), ("all_pairs", piv)):
            for var in [f"gain_K{K}" for K in K_ORDER] + ["kstar_corrected", "kstar_main", "kstar_naive_nonIEMOCAP"]:
                col = var.replace("gain_K", "") if var.startswith("gain_K") else var
                x = base[["target", "source", col]].dropna()
                if len(x) < 8:
                    ctx.gap(3, kind, f"{pair_set} {var} の組が少ない", 8, len(x))
                    continue
                rng = common.rng_for(f"e75-perm:{kind}:{pair_set}:{var}", ctx.a.seed)
                y = x[col].to_numpy(float) * (100 if var.startswith("gain_K") else 1)
                rows.append({"kind": kind, "short": SHORT[kind], "pair_set": pair_set, "variable": var,
                             "role": {"kstar_corrected": "main", "kstar_main": "secondary",
                                      "kstar_naive_nonIEMOCAP": "check"}.get(var, "main"),
                             **additive(y, x, ctx.a.n_perm, rng)})
    R = pd.DataFrame(rows)
    # 検算：全組の R² が保存済みの decomp_share.csv（C 固定の地図）と一致するか．fixc_analysis の版は言葉系が
    # 反復 5 回の時点（2026-10-09）の値なので主な種類だけと比べ，text30 の版（30 回）はすべての種類と比べる
    for key, only in (("decomp_share_main", MAIN4), ("decomp_share_text", KINDS_ALL)):
        path = ctx.p(key)
        if not path.exists():
            continue
        ctx.note_input(path)
        ref = pd.read_csv(path)
        ref = ref[ref.kind.isin(only)]
        for var, refvar in (("gain_K1", "gain_K1"), ("gain_K5", "gain_K5"), ("kstar_naive_nonIEMOCAP", "kstar")):
            a = R[(R.pair_set == "all_pairs") & (R.variable == var)].set_index("kind")
            b = ref[ref.variable == refvar].set_index("kind")
            j = a.join(b[["r2_target", "r2_both"]], rsuffix="_ref", how="inner")
            if len(j):
                ctx.check(3, ",".join(j.index), f"{var} の R²（全組）と {path.parent.parent.name}/decomp_share.csv の差の最大",
                          float(np.nanmax(np.abs(np.r_[j.r2_target - j.r2_target_ref, j.r2_both - j.r2_both_ref]))), 1e-6)
    ctx.write("e75_3_additive.csv", R)


# ---------------------------------------------------------------- 項目 4：C の規則

def item4(ctx: Ctx) -> None:
    rows, pairs_out, crs = [], [], []
    base_rules = [("map_rule", "uar")] + [(f"C{c}", f"test_uar_C{c}") for c in C_GRID] + [("oracle", "uar_oracle")]
    for kind in ctx.a.kinds:
        results = []                                               # (規則の名前, P)
        fixc_g = {}
        for rule, col in base_rules:
            g, df = gain_table(ctx, kind, 4, tgt_col=col)
            fixc_g[rule] = (g, df)
            results.append((rule, interval_table(ctx, kind, g, df, 4, n_rep_expected=30, label=rule)))
        cvdir = ROOT / f"{ctx.a.cv_map_prefix}{kind}"
        if cvdir.exists():
            g, df = gain_table(ctx, kind, 4, tgt_col="uar", merged_dir=cvdir)
            results.append(("cv_map", interval_table(ctx, kind, g, df, 4, n_rep_expected=30, label="cv_map")))
            # 交差検証の地図の反復が少ない種類（言葉系は 0〜4）では，C 固定の地図の全規則を同じ反復に絞った行を足す（R8）
            cv_reps = sorted(g.rep.unique())
            fx_reps = sorted(fixc_g["map_rule"][0].rep.unique())
            if cv_reps != fx_reps:
                tag = f"@reps{min(cv_reps)}-{max(cv_reps)}"
                for rule, _ in base_rules:
                    gg, dff = fixc_g[rule]
                    gg = gg[gg.rep.isin(cv_reps)]
                    results.append((rule + tag, interval_table(ctx, kind, gg, dff, 4, n_rep_expected=len(cv_reps),
                                                               label=rule + tag)))
            # 検算：cv_map の得の平均が交差検証の地図のまとめ（summary/pairs.csv）と一致するか
            sp = cvdir / "summary" / "pairs.csv"
            if sp.exists():
                ctx.note_input(sp)
                ref = pd.read_csv(sp, dtype={"K": str}).set_index(["target", "source", "K"]).diff_mean
                cvP = [p for r, p in results if r == "cv_map"][0].set_index(["target", "source", "K"])["mean"]
                ctx.check(4, kind, "cv_map の得の平均と交差検証の地図の summary/pairs.csv の差の最大",
                          float(np.nanmax(np.abs(cvP - ref.reindex(cvP.index)))), 1e-9, f"{len(cvP)} 行")
        else:
            ctx.gap(4, kind, "交差検証の地図が無い", 1, 0, str(cvdir))
        for rule, P in results:
            P = P[~P.same_corpus]
            TL = target_level(P).set_index(["kind", "K"])
            for K, h in P.groupby("K"):
                rows.append({"kind": kind, "short": SHORT[kind], "rule": rule, "K": K, "n_pairs": len(h),
                             "n_rep_min": int(h.n_rep.min()), "mean_gain": h["mean"].mean() * 100,
                             "n_pairs_mean_neg": int((h["mean"] < 0).sum()),
                             "up_main": int((h.main_lo > 0).sum()), "down_main": int((h.main_hi < 0).sum()),
                             **TL.loc[(kind, K)].to_dict()})
            cols = ["kind", "target", "source", "K", "n_rep", "mean", "main_lo", "main_hi", "main_p", "main_method"]
            cols += [c for c in ("n_units",) if c in P]
            pairs_out.append(P.assign(rule=rule)[["rule"] + cols])
        tg = fixc_g["map_rule"][1]
        tg = tg[tg.method == "tgt_only"]
        for K, h in tg.groupby("K"):
            vc = h.C_rule.value_counts(normalize=True)
            crs.append({"kind": kind, "short": SHORT[kind], "K": K, "n_rows": len(h),
                        **{f"share_{r}": float(vc.get(r, 0.0)) for r in ("cv", "tie", "fixed")}})
    R = pd.DataFrame(rows)
    kneg = []
    for (kind, rule), h in R.groupby(["kind", "rule"]):
        h = h.set_index("K").reindex(K_ORDER)
        neg = [K for K in K_ORDER if h.loc[K, "mean_gain"] < 0]
        kneg.append({"kind": kind, "short": SHORT[kind], "rule": rule, "n_rep_min": int(h.n_rep_min.min()),
                     "first_K_mean_gain_negative": neg[0] if neg else "none"})
    ctx.write("e75_4_c_rule.csv", R)
    ctx.write("e75_4_tgt_c_rule_share.csv", pd.DataFrame(crs))
    ctx.write("e75_4_first_negative_K.csv", pd.DataFrame(kneg))
    ctx.write("e75_4_pairs.csv", pd.concat(pairs_out, ignore_index=True))
    ctx.cache["item4_crs"] = pd.DataFrame(crs)


# ---------------------------------------------------------------- 項目 5：分解の順序

def stageb_rows(ctx: Ctx, kinds: list[str], item: int, max_rep: int | None = 10) -> pd.DataFrame:
    out = []
    for key, ks in (("stageb_main", MAIN4), ("stageb_text", TEXT_KINDS)):
        use = [k for k in kinds if k in ks]
        if not use:
            continue
        r = read_csv(ctx, ctx.p(key) / "e15b_rows.csv", dtype={"K": str, "pair": str})
        r = r[r.kind.isin(use)]
        if max_rep is not None:
            r = r[r.rep < max_rep]                                 # E29 と同じく反復 0〜9
        out.append(r)
    R = pd.concat(out, ignore_index=True)
    R = R[[not same_corpus(t, s) for t, s in zip(R.target, R.source)]].copy()
    if max_rep is not None:
        for (kind, t, s, K), h in R.groupby(["kind", "target", "source", "K"]):
            if len(h) != 6 * max_rep:
                ctx.gap(item, kind, f"段階 B の行（反復 {max_rep} × 対 6）{t}←{s} K={K}", 6 * max_rep, len(h))
        for kind in kinds:
            groups = groups_for(kind)
            exp = sum(1 for t in groups for s in groups if t != s and not same_corpus(t, s)) * len(KS3)
            found = R[R.kind == kind].groupby(["target", "source", "K"]).ngroups
            if found != exp:
                ctx.gap(item, kind, "段階 B の組 × K の数", exp, found)
    if not np.isfinite(R[["d_s", "o_s", "d_m", "o_m"]].to_numpy(float)).all():
        raise SystemExit("段階 B の d・o に NaN か Inf がある")
    return R


def shares(pos: pd.Series, tot: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """D11：分母（合計）が 0 以下の単位は割合を定めない（NaN）．0 < 分母 < DEN_SMALL は印を付ける。"""
    sh = pos / tot
    sh[tot <= 0] = np.nan
    return sh, tot <= 0, (tot > 0) & (tot < DEN_SMALL)


def item5(ctx: Ctx) -> None:
    kinds = [k for k in ctx.a.kinds if k in MAIN4 + TEXT_KINDS]
    R = stageb_rows(ctx, kinds, 5)
    R["pos_f"] = (ba(R.d_s, R.o_m) - ba(R.d_s, R.o_s)) * 100           # 今の順序（E15 の 8 節，E29）：位置が先
    R["dir_f"] = (ba(R.d_m, R.o_m) - ba(R.d_s, R.o_m)) * 100
    R["dir_r"] = (ba(R.d_m, R.o_s) - ba(R.d_s, R.o_s)) * 100           # 逆の順序：向きが先
    R["pos_r"] = (ba(R.d_m, R.o_m) - ba(R.d_m, R.o_s)) * 100
    R["neutral_pair"] = R.pair.str.startswith("0-")
    key = ["kind", "target", "source", "K"]
    P = R.groupby(key)[["pos_f", "dir_f", "pos_r", "dir_r"]].mean().reset_index()
    P["pos_sh"], P["dir_sh"] = (P.pos_f + P.pos_r) / 2, (P.dir_f + P.dir_r) / 2
    P["total"] = P.pos_f + P.dir_f
    for o in ("f", "r", "sh"):
        P[f"share_{o}"], P["den_le0"], P["den_small"] = shares(P[f"pos_{o}"], P.total)
    # UAR の単位での割合（E29 の G_crit ÷ G_merge．D2）を同じ表に付ける
    e29 = None
    path = ctx.p("e29_pairs")
    if path.exists():
        e29 = read_csv(ctx, path, dtype={"K": str})
        j = P.merge(e29[key + ["pos", "dir"]], on=key, how="inner")
        ctx.check(5, "all", "今の順序の位置の分・向きの分と e29_pairs.csv の差の最大",
                  float(np.nanmax(np.abs(np.r_[j.pos_f - j.pos * 1, j.dir_f - j.dir]))), 1e-6, f"{len(j)} 行")
        P = P.merge(e29[key + ["G_crit", "G_merge"]], on=key, how="left")
    ctx.cache["item5_pairs"] = P
    # 感情の対の種類（平静を含む対とそれ以外）ごとと，6 対ごとの平均
    E = R.groupby(["kind", "K", "neutral_pair"])[["pos_f", "dir_f", "pos_r", "dir_r"]].mean().reset_index()
    E6 = R.groupby(["kind", "K", "pair"])[["pos_f", "dir_f", "pos_r", "dir_r"]].mean().reset_index()
    rows, bt = [], []
    groups_def = [(k, P.kind == k) for k in kinds] + \
                 [(f"{k}_excl_script_target", (P.kind == k) & (P.target != SCRIPT)) for k in kinds]
    tk = [k for k in kinds if k in TEXT_KINDS]
    if tk:
        groups_def += [("text4", P.kind.isin(tk)), ("text4_excl_script_target", P.kind.isin(tk) & (P.target != SCRIPT))]
    for name, mask in groups_def:
        for K, h in P[mask].groupby("K"):
            row = {"group": name, "short": SHORT.get(name, name), "K": K, "n_pairs": len(h),
                   "n_targets": h.target.nunique(), "n_den_le0": int(h.den_le0.sum()),
                   "n_den_small": int(h.den_small.sum()), "mean_total": h.total.mean()}
            for o in ("f", "r", "sh"):
                row[f"mean_pos_{o}"], row[f"mean_dir_{o}"] = h[f"pos_{o}"].mean(), h[f"dir_{o}"].mean()
                row[f"median_share_{o}"] = h[f"share_{o}"].median()
                row[f"ratio_of_sums_{o}"] = h[f"pos_{o}"].sum() / h.total.sum() if h.total.sum() > 0 else np.nan
                row[f"n_share_gt_half_{o}"] = int((h[f"share_{o}"] > 0.5).sum())
                tmed = h.groupby("target")[f"share_{o}"].median()
                row[f"n_targets_median_gt_half_{o}"] = int((tmed > 0.5).sum())
                loto = [h[h.target != t][f"share_{o}"].median() for t in sorted(h.target.unique())]
                row[f"loto_median_min_{o}"], row[f"loto_median_max_{o}"] = float(np.nanmin(loto)), float(np.nanmax(loto))
            if "G_merge" in h:
                gm, gc = h.G_merge, h.G_crit
                row["uar_ratio_of_sums_Gcrit_Gmerge"] = gc.sum() / gm.sum() if gm.sum() > 0 else np.nan
                row["uar_median_Gcrit_over_Gmerge"] = float((gc / gm).where(gm > 0).median())
                row["uar_n_Gmerge_le0"] = int((gm <= 0).sum())
            rows.append(row)
            if name in kinds:
                for t, ht in h.groupby("target"):
                    bt.append({"kind": name, "K": K, "target": t, "n_sources": len(ht),
                               **{f"median_share_{o}": ht[f"share_{o}"].median() for o in ("f", "r", "sh")}})
    ctx.write("e75_5_pairs.csv", P)
    ctx.write("e75_5_summary.csv", pd.DataFrame(rows))
    ctx.write("e75_5_by_target.csv", pd.DataFrame(bt))
    ctx.write("e75_5_neutral_pairs.csv", E)
    ctx.write("e75_5_by_emotion_pair.csv", E6)


# ---------------------------------------------------------------- 項目 6：6 対を同時に直す切片

def pair_sd(L: np.ndarray, y: np.ndarray, base: list[tuple[float, float]]) -> np.ndarray:
    """感情の対ごとの 2 つの山の標準偏差 s．`d_o` は s を返さないので，k 列に 1 を足したときの o の変化
    （o は −(中点)/s なので，k が対の先なら −1/s，後なら +1/s 動く）から逆算する。"""
    sd = np.full(len(PAIRS), np.nan)
    for k in (1, 2, 3):
        L2 = L.copy()
        L2[:, k] += 1.0
        moved = d_o(L2, y)
        for i, (a, b) in enumerate(PAIRS):
            if a == k:
                sd[i] = 1.0 / (base[i][1] - moved[i][1])
            elif b == k:
                sd[i] = 1.0 / (moved[i][1] - base[i][1])
    return sd


def loc_loss_clipped(d: np.ndarray, o: np.ndarray) -> np.ndarray:
    """対ごとの位置の損 A(d,0) − A(d,o) を 0 で切る（D11．d ≤ 0 の対では負になり，最小化が o を広げてしまうため）。"""
    return np.maximum(ba(d, 0) - ba(d, o), 0.0)


def joint_offset(d: np.ndarray, o: np.ndarray, sd: np.ndarray):
    """2 つの山の模型で，4 つの切片のずらし c（c_0 = 0）を 1 組だけ選び，6 対の位置の損（0 で切る）の平均を最小にする。"""
    A = np.zeros((len(PAIRS), 3))
    for i, (a, b) in enumerate(PAIRS):
        if a > 0:
            A[i, a - 1] += 1.0 / sd[i]
        if b > 0:
            A[i, b - 1] -= 1.0 / sd[i]

    def f(c):
        return float(np.mean(loc_loss_clipped(d, o - A @ c)))
    starts = [np.zeros(3), np.linalg.lstsq(A, o, rcond=None)[0]]
    best = None
    for s0 in starts:
        r = optimize.minimize(f, s0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return float(best.fun), best.x, bool(best.success), int(best.nit)


def pair_acc(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """6 対の実際の正解率 ½[P(z > 0 | k) + P(z < 0 | l)]（E13 の acc_actual と同じ定義）。"""
    out = []
    for a, b in PAIRS:
        za, zb = L[y == a, a] - L[y == a, b], L[y == b, a] - L[y == b, b]
        out.append(0.5 * ((za > 0).mean() + (zb < 0).mean()))
    return np.array(out)


def pair_best_threshold(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """対ごとに境目を最良の位置に置いたときの実際の正解率（対ごとに取り戻せる量）。"""
    out = []
    for a, b in PAIRS:
        za, zb = L[y == a, a] - L[y == a, b], L[y == b, a] - L[y == b, b]
        cand = np.unique(np.r_[za, zb, -np.inf, np.inf])
        th = np.r_[(cand[:-1] + cand[1:]) / 2]
        th = th[np.isfinite(th)] if np.isfinite(th).any() else np.array([0.0])
        sa, sb = np.sort(za), np.sort(zb)
        pa = 1 - np.searchsorted(sa, th, side="right") / len(sa)     # P(z > t | a)
        pb = np.searchsorted(sb, th, side="left") / len(sb)           # P(z < t | b)
        out.append(float(np.max(0.5 * (pa + pb))))
    return np.array(out)


def uar(L: np.ndarray, y: np.ndarray) -> float:
    pred = L.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(4)]))


def ce_grad_max(L: np.ndarray, y: np.ndarray, b: np.ndarray) -> float:
    """`fit_class_bias`（l2 = 0）の解での，平均の交差エントロピーの切片についての勾配の最大絶対値（和 0 の部分空間）。
    共通の部品を書き換えずに，最小化が止まった点が停留点かを確かめる（I11）。"""
    g = (softmax(L + b, axis=1) - np.eye(4)[y]).mean(0)
    g = g - g.mean()
    return float(np.abs(g).max())


def item6(ctx: Ctx) -> None:
    rows = []
    e25_diff: dict[str, list[float]] = {}
    for kind in ctx.a.kinds:
        groups = groups_for(kind)
        lab = ROOT / f"{ctx.a.shift_prefix}{kind}" / "labeling.csv"
        labdf = read_csv(ctx, lab).set_index(["target", "source"]) if lab.exists() else None
        src_uar = labdf.uar_t if labdf is not None else None
        c_mismatch, c_compared = 0, 0
        e13 = ctx.e13(kind)
        ivdir = ROOT / f"{ctx.a.interventions_prefix}{kind}"
        tsdir = ROOT / f"{ctx.a.transfer_scores_prefix}{kind}"
        for s in groups:
            path = ROOT / f"{ctx.a.shift_prefix}{kind}" / "logits" / f"{s}.npz"
            if not path.exists():
                ctx.gap(6, kind, f"ソースのみの点数が無い {s}", 1, 0, str(path))
                continue
            ctx.note_input(path)
            z = np.load(path)
            for t in groups:
                if t == s or same_corpus(t, s):
                    continue
                if f"t_{t}" not in z.files:
                    ctx.gap(6, kind, f"点数の行が無い t_{t}（ソース {s}）", 1, 0)
                    continue
                L = z[f"t_{t}"].astype(np.float64)
                m = ctx.meta(kind, t)
                y = m.label_4.to_numpy()
                if len(L) != len(y):
                    raise SystemExit(f"{kind} {s}→{t}：点数 {len(L)} 行と発話 {len(y)} 本が合わない")
                base = d_o(L, y)
                d, o = np.array([b[0] for b in base]), np.array([b[1] for b in base])
                sd = pair_sd(L, y, base)
                loc_raw = float(np.mean(ba(d, 0) - ba(d, o)))           # E13 の定義（d ≤ 0 の対で負になりうる）
                loc = float(np.mean(loc_loss_clipped(d, o)))
                resid, c, ok, nit = joint_offset(d, o, sd)
                acc0, best = pair_acc(L, y), pair_best_threshold(L, y)
                rng = common.rng_for(f"e75-bias:{kind}:{s}:{t}", 0)
                idx = balanced_subset(y, rng)
                b_all = fit_class_bias(L[idx], y[idx], l2=0.0)
                acc1 = pair_acc(L + b_all, y)
                mod1 = d_o(L + b_all, y)
                loc_after_fit = float(np.mean([max(ba(dd, 0) - ba(dd, oo), 0.0) for dd, oo in mod1]))
                # 話者の半分で切片を学び，もう半分で測る（E13 の Δ と同じ分け方：speaker_folds(k=2, seed=0)）
                half = speaker_folds(m.speaker_group, k=2, seed=0).to_numpy()
                cf = {"uar0": [], "uar1": [], "acc0": [], "acc1": [], "grad": []}
                for h in (0, 1):
                    fi, ev = half == h, half == 1 - h
                    if any(np.bincount(y[fi], minlength=4) == 0) or any(np.bincount(y[ev], minlength=4) == 0):
                        ctx.gap(6, kind, f"半分に欠ける感情 {t}", 1, 0, f"half={h}")
                        continue
                    r2 = common.rng_for(f"e75-bias-half:{kind}:{s}:{t}", h)
                    ii = np.flatnonzero(fi)[balanced_subset(y[fi], r2)]
                    bh = fit_class_bias(L[ii], y[ii], l2=0.0)
                    cf["grad"].append(ce_grad_max(L[ii], y[ii], bh))
                    cf["uar0"].append(uar(L[ev], y[ev]))
                    cf["uar1"].append(uar(L[ev] + bh, y[ev]))
                    cf["acc0"].append(pair_acc(L[ev], y[ev]).mean())
                    cf["acc1"].append(pair_acc(L[ev] + bh, y[ev]).mean())
                ref = e13[(e13.target == t) & (e13.source == s)].set_index("pair_idx")
                if len(ref):
                    ctx.check(6, kind, f"位置の損（対ごと）と e13_pairs の差 {t}←{s}",
                              float(np.max(np.abs(ref.loss_offset.reindex(PAIR_IDX).to_numpy() - (ba(d, 0) - ba(d, o))))),
                              1e-6, ctx.e13_note(kind))
                if src_uar is not None and (t, s) in src_uar.index:
                    ctx.check(6, kind, f"ソースのみの UAR と labeling.csv の uar_t の差 {t}←{s}",
                              abs(uar(L, y) - float(src_uar.loc[(t, s)])), 1e-6)
                # I14：この点数の分類器が E25 の「ソースのみ（学び直し）」と同じかを，テストの袋 r（反復 0〜9）の UAR で確かめる
                ivf, tsf = ivdir / f"{t}.csv", tsdir / f"{t}.npz"
                if ivf.exists() and tsf.exists():
                    iv = read_csv(ctx, ivf, dtype={"K": str, "source": str})
                    iv = iv[(iv.method == E25_SRC) & (iv.source == s)]
                    # 2 つのソースのみの分類器の C が同じか（離散の値なので許容 0．compare の追記の「記録する」）
                    if labdf is not None and (t, s) in labdf.index and len(iv):
                        c_shift = float(labdf.loc[(t, s), "C"])
                        c_mismatch += int((np.abs(iv.C.astype(float).to_numpy() - c_shift) > 0).any())
                        c_compared += 1
                    iv = iv.set_index("rep").uar
                    zt = np.load(tsf)
                    for r, u in iv.items():
                        key = f"test_idx_r{int(r)}"
                        if key in zt.files:
                            ti = np.asarray(zt[key])
                            e25_diff.setdefault(kind, []).append(abs(uar(L[ti], y[ti]) - float(u)))
                cpad = np.r_[0.0, c]
                rows.append({"kind": kind, "target": t, "source": s, "n_target": len(y), "n_fit_balanced": len(idx),
                             "n_pairs_d_le0": int((d <= 0).sum()),
                             "loc_loss_pairwise_raw": loc_raw * 100, "loc_loss_pairwise": loc * 100,
                             "loc_loss_after_joint_model": resid * 100,
                             "loc_recovered_joint_model": (loc - resid) * 100,
                             "frac_recovered_joint_model": (loc - resid) / loc if loc * 100 > LOC_MIN else np.nan,
                             "joint_success": ok, "joint_nit": nit,
                             "uar_after_joint_model": uar(L + cpad, y) * 100,
                             "loc_loss_after_ce_fit": loc_after_fit * 100,
                             "acc_pair_src": acc0.mean() * 100, "acc_pair_best_per_pair": best.mean() * 100,
                             "acc_pair_after_ce_fit": acc1.mean() * 100,
                             "uar_src": uar(L, y) * 100, "uar_after_ce_fit": uar(L + b_all, y) * 100,
                             "ce_grad_max_all": ce_grad_max(L[idx], y[idx], b_all),
                             "ce_grad_max_halves": max(cf["grad"]) if cf["grad"] else np.nan,
                             "cf_uar_src": np.mean(cf["uar0"]) * 100 if cf["uar0"] else np.nan,
                             "cf_uar_after_fit": np.mean(cf["uar1"]) * 100 if cf["uar1"] else np.nan,
                             "cf_acc_pair_src": np.mean(cf["acc0"]) * 100 if cf["acc0"] else np.nan,
                             "cf_acc_pair_after_fit": np.mean(cf["acc1"]) * 100 if cf["acc1"] else np.nan,
                             **{f"c_joint_{k}": v for k, v in zip((1, 2, 3), c)},
                             **{f"b_ce_{k}": v for k, v in enumerate(b_all)}})
        groups_n = sum(1 for t in groups for s in groups if t != s and not same_corpus(t, s))
        found = sum(1 for r in rows if r["kind"] == kind)
        if found != groups_n:
            ctx.gap(6, kind, "組の数", groups_n, found)
        if c_compared:
            ctx.check(6, kind, "shift のソースのみの C（labeling.csv）と E25 のソースのみの C が違う組の数",
                      float(c_mismatch), 0, f"{c_compared} 組")
        if kind in e25_diff:
            dv = np.array(e25_diff[kind])
            # 同じデータと C で別々に解いた 2 つのソースのみの分類器なので，答えの数本の入れ替わりの差が残る．
            # 一致を確かめる検算ではなく，差の大きさの記録として `records.csv` に書く（許容の幅を後から決めない）
            ctx.record(6, kind, "保存した点数のテストの袋 r の UAR と E25 のソースのみ（学び直し）の UAR の差（I14）",
                       float(dv.max()), float(np.mean(dv > 1e-6)), len(dv), "組 × 反復")
        else:
            ctx.gap(6, kind, "E25 のソースのみの行かテストの行番号が無い（I14 の検算ができない）", 1, 0)
    P = pd.DataFrame(rows)
    ctx.cache["item6_pairs"] = P
    S = []
    for kind, h in P.groupby("kind"):
        tmed = h.groupby("target").frac_recovered_joint_model.median()
        S.append({"kind": kind, "short": SHORT[kind], "n_pairs": len(h),
                  "n_frac_undefined": int(h.frac_recovered_joint_model.isna().sum()),
                  "n_pairs_with_d_le0": int((h.n_pairs_d_le0 > 0).sum()),
                  "n_joint_not_success": int((~h.joint_success).sum()),
                  "max_ce_grad": float(np.nanmax(np.r_[h.ce_grad_max_all, h.ce_grad_max_halves])),
                  **{f"mean_{c}": h[c].mean() for c in P.columns if c.startswith(("loc_", "acc_", "uar_", "cf_"))},
                  "median_frac_recovered_joint_model": h.frac_recovered_joint_model.median(),
                  "min_frac_recovered_joint_model": h.frac_recovered_joint_model.min(),
                  "target_median_frac_min": float(tmed.min()), "target_median_frac_max": float(tmed.max())})
    ctx.write("e75_6_pairs.csv", P)
    ctx.write("e75_6_summary.csv", pd.DataFrame(S))


# ---------------------------------------------------------------- 項目 7：式 (2) と模型で評価した位置の損

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)              # 標準正規の期待値のための節点


def item7(ctx: Ctx) -> None:
    rows, tsets = [], []
    for kind in ctx.a.kinds:
        for clf, prefix in (("tgt_only", ctx.a.tgt_scores_prefix), ("transfer_h7", ctx.a.transfer_scores_prefix)):
            dpath = ROOT / f"{prefix}{kind}"
            if not dpath.exists():
                ctx.gap(7, kind, f"{clf} の点数が無い", 1, 0, str(dpath))
                continue
            for f in sorted(dpath.glob("*.npz")):
                ctx.note_input(f)
                t = f.stem
                z = np.load(f)
                reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
                if len(reps) != 10:
                    ctx.gap(7, kind, f"{clf} {t} の反復", 10, len(reps))
                tsets.append({"kind": kind, "clf": clf, "target": t,
                              "n_test_sets_reps": len(set(test_sets_from_idx(z, reps).values()))})
                if clf == "tgt_only":
                    keys = [(None, K, r, f"t_K{K}_r{r}") for r in reps for K in KS3]
                else:
                    srcs = sorted({k.split("_")[1] for k in z.files if k.startswith("h7_")})
                    keys = [(s, K, r, f"h7_{s}_K{K}_r{r}") for s in srcs if not same_corpus(t, s)
                            for r in reps for K in KS3]
                src_do: dict[tuple, list] = {}
                for s, K, r, key in keys:
                    if key not in z.files:
                        ctx.gap(7, kind, f"{clf} の点数の鍵が無い {t} {key}", 1, 0)
                        continue
                    y = z[f"y_test_r{r}"]
                    Lk = z[key].astype(np.float64)
                    act = pair_best_threshold(Lk, y) - pair_acc(Lk, y)       # 実際の正解率での位置の損（対ごと）
                    ds = os_ = [np.nan] * 6
                    if s is not None:
                        sk = f"s_{s}_r{r}"
                        if (s, r) not in src_do and sk in z.files:
                            src_do[(s, r)] = d_o(z[sk].astype(np.float64), y)
                        if (s, r) in src_do:
                            ds = [v[0] for v in src_do[(s, r)]]
                            os_ = [v[1] for v in src_do[(s, r)]]
                    for i, ((a, b), (dd, oo)) in enumerate(zip(PAIRS, d_o(Lk, y))):
                        rows.append({"kind": kind, "clf": clf, "target": t, "source": s or "", "K": K, "rep": r,
                                     "pair": f"{a}-{b}", "d": dd, "o": oo, "d_s": ds[i], "o_s": os_[i],
                                     "loss_actual": act[i] * 100})
    if not rows:
        return
    R = pd.DataFrame(rows)
    Kn = R.K.astype(int).to_numpy()
    d, o = R.d.to_numpy(), R.o.to_numpy()
    R["loss_model"] = (ba(d, 0) - ba(d, o)) * 100                  # 模型で評価した位置の損
    R["pred_2nd"] = d / 4 * stats.norm.pdf(d / 2) / (2 * Kn) * 100
    R["quad_with_measured_o"] = d / 4 * stats.norm.pdf(d / 2) * o ** 2 * 100   # 2 次の近似に測った o を入れた値
    sdv = np.sqrt(1 / (2 * Kn))
    ex = np.zeros(len(R))
    for x, w in zip(GH_X, GH_W):
        ex += w * (ba(d, 0) - ba(d, x * sdv))
    R["pred_exact"] = ex / np.sqrt(2 * np.pi) * 100
    R["o2"] = o ** 2
    R["delta_d"] = R.d - R.d_s                                       # 仮説転移の重みの動き（ソースのみからの d の変化）
    # 全反復に共通の系統的なずれ：同じ（ターゲット，ソース，対，K）の反復の平均の o（テストの話者は反復ごとに変わるので，
    # プールとテストの話者のずれの大部分は残り（揺れ）の側に入る）
    g = R.groupby(["kind", "clf", "target", "source", "K", "pair"]).o
    R["o_cell_mean"] = g.transform("mean")
    # ソースのみの o_s の反復の間の分散：ソースのみの分類器は反復の間で同じなので，テストの話者の入れ替えだけによる揺れ
    vo = R[R.clf == "transfer_h7"].drop_duplicates(["kind", "target", "source", "rep", "pair"])
    vo = vo.groupby(["kind", "target", "source", "pair"]).o_s.var(ddof=1).groupby(level="kind").mean()
    TS = pd.DataFrame(tsets)
    S = []
    for (kind, clf, K), h in R.groupby(["kind", "clf", "K"]):
        per_t = h.groupby("target").loss_model.mean() / h.groupby("target").pred_2nd.mean()
        ts = TS[(TS.kind == kind) & (TS.clf == clf)].n_test_sets_reps
        row = {"kind": kind, "short": SHORT[kind], "clf": clf, "K": K, "n_rows": len(h),
               "mean_loss_model": h.loss_model.mean(), "mean_loss_actual": h.loss_actual.mean(),
               "mean_pred_2nd": h.pred_2nd.mean(), "mean_pred_exact": h.pred_exact.mean(),
               "mean_quad_with_measured_o": h.quad_with_measured_o.mean(),
               "ratio_model_to_2nd": h.loss_model.mean() / h.pred_2nd.mean(),
               "ratio_model_to_exact": h.loss_model.mean() / h.pred_exact.mean(),
               "target_ratio_min": float(per_t.min()), "target_ratio_max": float(per_t.max()),
               "mean_o2": h.o2.mean(), "theory_o2": 1 / (2 * int(K)),
               "systematic_part_o2": float((h.o_cell_mean ** 2).mean()),
               "rep_varying_part_o2": float(((h.o - h.o_cell_mean) ** 2).mean()),
               "mean_d": h.d.mean(), "n_test_sets_min": int(ts.min()) if len(ts) else np.nan,
               "n_test_sets_max": int(ts.max()) if len(ts) else np.nan}
        if clf == "transfer_h7":
            rel = (h.delta_d.abs() / h.d_s).replace([np.inf, -np.inf], np.nan)
            row.update(mean_delta_d=h.delta_d.mean(), median_abs_delta_d_over_d_s=float(rel.median()),
                       var_o_s_over_reps=float(vo.get(kind, np.nan)),
                       var_o_s_over_theory=float(vo.get(kind, np.nan)) / (1 / (2 * int(K))))
            ctx.assume(7, kind, f"式 (2)：重みが動かない（仮説転移，K={K}）", "|d_h7 − d_s| ÷ d_s の中央値",
                       row["median_abs_delta_d_over_d_s"], "0.1 未満", bool(row["median_abs_delta_d_over_d_s"] < 0.1))
            ctx.assume(7, kind, f"式 (2)：o の揺れは K 本の抜き方だけ（K={K}）",
                       "ソースのみの o_s の反復の間の分散 ÷ 1/(2K)", row["var_o_s_over_theory"], "0.5 未満",
                       bool(row["var_o_s_over_theory"] < 0.5))
        S.append(row)
    # 検算：ターゲットのみの d の平均が E28-9 の dT_K と一致するか（gate-code の I7）
    path = ctx.p("e28t9_targets")
    if path.exists():
        ref = read_csv(ctx, path)
        tg = R[R.clf == "tgt_only"].groupby(["kind", "target", "K"]).d.mean().reset_index()
        for kind, h in tg.groupby("kind"):
            rr = ref[ref.kind == kind].set_index("target")
            diffs = [abs(v - rr.loc[t, f"dT_K{K}"]) for t, K, v in zip(h.target, h.K, h.d) if t in rr.index]
            if diffs:
                ctx.check(7, kind, "ターゲットのみの d の平均と e28t9_targets.csv の dT_K の差の最大", float(max(diffs)), 1e-9,
                          f"{len(diffs)} 行")
    ctx.write("e75_7_rows.csv", R)
    ctx.write("e75_7_summary.csv", pd.DataFrame(S))
    ctx.write("e75_7_test_sets.csv", TS)


# ---------------------------------------------------------------- 項目 8：向きの損が負の升，Δ の 2 つの半分，縮小の強さ

def own_halves(X: np.ndarray, y: np.ndarray, half: np.ndarray):
    """`explore_pairs.own_separation` と同じ式で，2 つの半分の値を別々に返す（D7）。共分散は `ledoit_wolf`
    （`LedoitWolf.fit` が内部で呼ぶ関数．精度行列を作らないので速い）を float64 で呼ぶ。縮小の強さも返す。"""
    def dirs(Xa, ya):
        m = np.stack([Xa[ya == k].mean(0) for k in range(4)])
        S, sh = ledoit_wolf((Xa - m[ya]).astype(np.float64), assume_centered=True)
        return np.linalg.solve(S, np.stack([m[k] - m[j] for k, j in PAIRS]).T).T, float(sh)

    def sep(U, Xb, yb):
        out = []
        for u, (k, j) in zip(U, PAIRS):
            pk, pj = Xb[yb == k] @ u, Xb[yb == j] @ u
            out.append((pk.mean() - pj.mean()) / np.sqrt((pk.var(ddof=1) + pj.var(ddof=1)) / 2))
        return np.array(out)
    A, B = half == 0, half == 1
    UA, shA = dirs(X[A], y[A])
    UB, shB = dirs(X[B], y[B])
    return sep(UA, X[B], y[B]), sep(UB, X[A], y[A]), shA, shB


def item8(ctx: Ctx) -> None:
    rows, prs, shr, dh = [], [], [], []
    for kind in ctx.a.kinds:
        e = ctx.e13(kind)
        e = e.assign(same_corpus=[same_corpus(t, s) for t, s in zip(e.target, e.source)])
        for subset, h in (("all_pairs", e), ("excl_same_corpus", e[~e.same_corpus])):
            neg = h.loss_direction < 0
            rows.append({"kind": kind, "short": SHORT[kind], "subset": subset, "n_cells": len(h),
                         "n_neg": int(neg.sum()), "frac_neg": float(neg.mean()),
                         "median_neg_points": float(h.loss_direction[neg].median() * 100) if neg.any() else np.nan,
                         "min_points": float(h.loss_direction.min() * 100),
                         **{f"n_neg_{p}": int((neg & (h.pair == p)).sum()) for p in PAIR_NAME_TO_IDX}})
        x = e[~e.same_corpus].copy()
        # 仮定：2 つの山の模型が対の正解率を当てる
        err = (x.acc_model - x.acc_actual).abs() * 100
        med_err, med_loc, med_dir = float(err.median()), float(x.loss_offset.median() * 100), float(x.loss_direction.median() * 100)
        ctx.assume(8, kind, "2 つの山の模型が対の正解率を当てる（位置の損）", "|acc_model − acc_actual| の中央値 ÷ 位置の損の中央値",
                   med_err / med_loc if med_loc > 0 else np.nan, "1/3 未満", bool(med_loc > 0 and med_err < med_loc / 3),
                   f"誤差の中央値 {med_err:.3g} 点，位置の損の中央値 {med_loc:.3g} 点")
        ctx.assume(8, kind, "2 つの山の模型が対の正解率を当てる（向きの損）", "|acc_model − acc_actual| の中央値 ÷ 向きの損の中央値",
                   med_err / med_dir if med_dir > 0 else np.nan, "1/3 未満", bool(med_dir > 0 and med_err < med_dir / 3),
                   f"誤差の中央値 {med_err:.3g} 点，向きの損の中央値 {med_dir:.3g} 点")
        x["dir_clip"] = x.loss_direction.clip(lower=0)
        pp = x.groupby(["target", "source"]).agg(L_crit=("loss_offset", "mean"), L_disc_raw=("loss_direction", "mean"),
                                                 L_disc_clip=("dir_clip", "mean")).reset_index()
        for c in ("L_crit", "L_disc_raw", "L_disc_clip"):
            pp[c] *= 100
        den_raw, den_clip = pp.L_crit + pp.L_disc_raw, pp.L_crit + pp.L_disc_clip
        pp["share_loss_e29"] = pp.L_crit / den_raw                       # E29 と同じ扱い（分母が 0 以下でも割る）
        pp["share_loss_d11"] = (pp.L_crit / den_raw).where(den_raw > 0)  # D11：分母が 0 以下の組を除く
        pp["share_loss_clip"] = (pp.L_crit / den_clip).where(den_clip > 0)
        pp["den_le0_raw"], pp["den_le0_clip"] = den_raw <= 0, den_clip <= 0
        prs.append(pp.assign(kind=kind))
        # Δ の 2 つの半分の値と縮小の強さ：E13 と同じ入力（standardize した特徴，話者の半分）
        groups = groups_for(kind)
        data = standardize({g: load_any(g, kind) for g in groups})
        # `own_separation` との一致は最も小さい 1 群で確かめる（`own_separation` は精度行列まで作るので遅い）
        small = sorted(data, key=lambda g: len(data[g][0]))[:1]
        for g, (df, X) in data.items():
            yv = df.label_4.to_numpy()
            half = speaker_folds(df.speaker_group, k=2, seed=0).to_numpy()
            dAB, dBA, shA, shB = own_halves(X, yv, half)
            mean2 = (dAB + dBA) / 2
            ref = e[e.target == g].groupby("pair_idx").own_d.mean().reindex(PAIR_IDX).to_numpy()
            if np.isfinite(ref).all():
                ctx.check(8, kind, f"Δ の 2 つの半分の平均と保存済みの own_d の差 {g}", float(np.max(np.abs(mean2 - ref))),
                          1e-6, ctx.e13_note(kind))
            if g in small:
                own = own_separation(X, yv, half)
                ctx.check(8, kind, f"Δ の 2 つの半分の平均と own_separation の差 {g}", float(np.max(np.abs(mean2 - own))), 1e-8)
                Xa, ya = X[half == 0], yv[half == 0]
                mu = np.stack([Xa[ya == k].mean(0) for k in range(4)])
                ref_sh = LedoitWolf(assume_centered=True).fit(Xa - mu[ya]).shrinkage_
                ctx.check(8, kind, f"縮小の強さと LedoitWolf.fit の shrinkage_ の差 {g}", abs(shA - ref_sh), 1e-12)
            for hh, sh in ((0, shA), (1, shB)):
                shr.append({"kind": kind, "group": g, "half": hh, "n": int((half == hh).sum()), "dim": X.shape[1],
                            "shrinkage": sh})
            for i, pidx in enumerate(PAIR_IDX):
                dh.append({"kind": kind, "group": g, "pair": pidx, "delta_dirA_measB": dAB[i], "delta_dirB_measA": dBA[i],
                           "delta_mean": mean2[i], "abs_diff_halves": abs(dAB[i] - dBA[i])})
        del data
    P = pd.concat(prs, ignore_index=True)
    # 同時に取り戻せる位置の損（項目 6）で作った損に占める o の割合（D3）
    if "item6_pairs" in ctx.cache:
        J = ctx.cache["item6_pairs"][["kind", "target", "source", "loc_recovered_joint_model"]]
        P = P.merge(J, on=["kind", "target", "source"], how="left")
        den = P.loc_recovered_joint_model + P.L_disc_raw
        P["share_loss_joint"] = (P.loc_recovered_joint_model / den).where(den > 0)
    # 合併の得に占める o の割合（項目 5 の 3 通りの順序，K=5）と損に占める o の割合の順位相関（P29-3 の形）
    p293 = []
    if "item5_pairs" in ctx.cache:
        Q = ctx.cache["item5_pairs"]
        Q = Q[Q.K == "5"].merge(P, on=["kind", "target", "source"])
        treat = ["e29", "d11", "clip"] + (["joint"] if "share_loss_joint" in Q else [])
        for kind, h in list(Q.groupby("kind")) + [("all", Q)]:
            for o in ("f", "r", "sh"):
                for v in treat:
                    # E29 と同じ扱いの行は，得の側も E29 と同じく分母が 0 以下の組を除かない（share_pos = pos/(pos+dir)）
                    gx = h[f"pos_{o}"] / h.total if v == "e29" else h[f"share_{o}"]
                    r, pv, n = spear(gx, h[f"share_loss_{v}"])
                    p293.append({"kind": kind, "short": SHORT.get(kind, kind), "order": o, "loss_treatment": v,
                                 "rho": r, "p": pv, "n": n, "median_share_loss": h[f"share_loss_{v}"].median(),
                                 "median_share_gain": gx.median()})
        # 検算：E29 の P29-3（all8，今の順序，E29 と同じ扱い）と V の損に占める o の割合の中央値
        tp = ctx.p("e29_tests")
        if tp.exists():
            T = read_csv(ctx, tp)
            ref = T[(T.group == "all8") & (T.test == "P29-3")]
            mine = [r for r in p293 if r["kind"] == "all" and r["order"] == "f" and r["loss_treatment"] == "e29"]
            if len(ref) and mine and len(Q.kind.unique()) == 8:
                ctx.check(8, "all", "P29-3（8 種類，今の順序，E29 と同じ扱い）と e29_tests.csv の差",
                          abs(mine[0]["rho"] - float(ref.rho.iloc[0])), 1e-9, f"n = {mine[0]['n']}")
        e29 = read_csv(ctx, ctx.p("e29_pairs"), dtype={"K": str})
        for kind in P.kind.unique():
            refm = e29[(e29.kind == kind) & (e29.K == "5")].share_loss.median()
            ctx.check(8, kind, "損に占める o の割合の中央値（E29 と同じ扱い）と e29_pairs.csv の差",
                      abs(P[P.kind == kind].share_loss_e29.median() - refm), 1e-9)
    else:
        ctx.gap(8, "-", "項目 5 を同じ回に流していないので P29-3 の行を作らない", 1, 0)
    DH = pd.DataFrame(dh)
    dsum = DH.groupby("kind").agg(median_abs_diff_halves=("abs_diff_halves", "median"),
                                  max_abs_diff_halves=("abs_diff_halves", "max"),
                                  n_delta_le_min=("delta_mean", lambda v: int((v <= DELTA_MIN).sum()))).reset_index()
    ctx.write("e75_8_negative_direction.csv", pd.DataFrame(rows))
    if p293:
        ctx.write("e75_8_p29_3.csv", pd.DataFrame(p293))
    ctx.write("e75_8_pairs.csv", P)
    ctx.write("e75_8_shrinkage.csv", pd.DataFrame(shr))
    ctx.write("e75_8_delta_halves.csv", DH)
    ctx.write("e75_8_delta_halves_summary.csv", dsum)


# ---------------------------------------------------------------- 項目 9：中心化の得とクラスの割合

def item9(ctx: Ctx) -> None:
    path = ctx.p("e29_pairs")
    e = read_csv(ctx, path, dtype={"K": str})
    e = e[e.K == "5"]
    rows, prs = [], []
    for kind in ctx.a.kinds:
        lp = ROOT / f"{ctx.a.shift_prefix}{kind}" / "label.csv"
        if not lp.exists():
            ctx.gap(9, kind, "label.csv が無い", 1, 0)
            continue
        lab = read_csv(ctx, lp)
        lab["tv_target_uniform"] = 0.5 * sum((lab[f"p_t_{k}"] - 0.25).abs() for k in range(4))
        lab["neutral_share_target"] = lab.p_t_0
        h = e[e.kind == kind][["kind", "target", "source", "G_center"]].merge(
            lab[["target", "source", "l1_true", "tv_target_uniform", "neutral_share_target"]], on=["target", "source"])
        h = h[[not same_corpus(t, s) for t, s in zip(h.target, h.source)]]
        if h.empty:
            ctx.gap(9, kind, "中心化の得の行が無い", 1, 0)
            continue
        prs.append(h)
        r, p, n = spear(h.G_center, h.l1_true)
        pr = pair_table.partial_rank(h, "G_center", "l1_true")
        dfree = (pr or {}).get("df", 0)
        # 1 つのコーパスを除いた順位相関の範囲（ターゲットとソースの両方でそのコーパスを含む組を除く）
        loco = []
        for c in sorted({corpus(g) for g in pd.concat([h.target, h.source])}):
            hh = h[[corpus(t) != c and corpus(s) != c for t, s in zip(h.target, h.source)]]
            rc, _, nc = spear(hh.G_center, hh.l1_true)
            if np.isfinite(rc):
                loco.append(rc)
        tgt = h.groupby("target").agg(G=("G_center", "mean"), tv=("tv_target_uniform", "first"),
                                     neu=("neutral_share_target", "first"))
        rt, _, nt = spear(tgt.G, tgt.tv)
        rn, _, _ = spear(tgt.G, tgt.neu)
        rows.append({"kind": kind, "short": SHORT[kind], "n_pairs": n, "n_corpora": len({corpus(g) for g in h.target}),
                     "rho_Gcenter_l1": r, "p_Gcenter_l1": p,
                     "r_partial_Gcenter_l1": pr["r_partial_rank"] if pr and dfree >= 3 else np.nan,
                     "p_partial_Gcenter_l1": pr["p"] if pr and dfree >= 3 else np.nan, "df_partial": dfree,
                     "loco_rho_min": float(min(loco)) if loco else np.nan, "loco_rho_max": float(max(loco)) if loco else np.nan,
                     "n_loco": len(loco),
                     "n_targets": nt, "rho_target_Gcenter_tv": rt, "rho_target_Gcenter_neutral": rn,
                     "n_Gcenter_neg": int((h.G_center < 0).sum()),
                     "median_l1_Gcenter_neg": float(h.l1_true[h.G_center < 0].median()) if (h.G_center < 0).any() else np.nan,
                     "median_l1_Gcenter_pos": float(h.l1_true[h.G_center >= 0].median())})
    ctx.write("e75_9_centering_prior.csv", pd.DataFrame(rows))
    if prs:
        ctx.write("e75_9_pairs.csv", pd.concat(prs, ignore_index=True))


# ---------------------------------------------------------------- 項目 10：E12 の α を選んだ合併と対照

def item10(ctx: Ctx) -> None:
    rows, prs = [], []
    ref_path = ctx.p("e12_wt")
    ref = read_csv(ctx, ref_path, dtype={"K": str}) if ref_path.exists() else None
    st_path = ref_path.parent / "e12_state.json"                  # e12_wt.csv を作ったときの反復（ターゲットごと）
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    if st_path.exists():
        ctx.note_input(st_path)
    cv_cols = {c: f"cv_anat_C{c:g}" for c in M.C_GRID}
    te_cols = {c: f"test_anat_C{c:g}" for c in M.C_GRID}
    for kind in ctx.a.kinds:
        mdir = ROOT / f"{ctx.a.mech_prefix}{kind}"
        if not mdir.exists():
            continue
        files = sorted(mdir.glob("*.csv"))
        for f in files:
            ctx.note_input(f)
        usec = ["op", "target", "source", "method", "K", "rep", "uar"] + list(cv_cols.values()) + list(te_cols.values())
        mech = pd.concat([pd.read_csv(f, comment="#", dtype={"K": str, "source": str}, usecols=usec) for f in files])
        w = mech[(mech.op == "wt") & (mech.method == "merged")].copy()
        if w[list(cv_cols.values()) + list(te_cols.values())].isna().any().any():
            ctx.gap(10, kind, "α = nat の交差検証かテストの列に NaN がある行", 0,
                    int(w[list(cv_cols.values())].isna().any(axis=1).sum()))
        # 対照：α を nat に固定し，C だけを同じ交差検証で選んだ合併（`map.choose_c` の同点の規則）
        ctrl, cchosen = [], []
        for _, rr in w.iterrows():
            cbest, _ = M.choose_c({c: float(rr[cv_cols[c]]) for c in M.C_GRID})
            ctrl.append(float(rr[te_cols[cbest]]))
            cchosen.append(float(cbest))
        w["ctrl"], w["C_ctrl"] = ctrl, cchosen
        w = w[["target", "source", "K", "rep", "uar", "ctrl", "C_ctrl"]]
        mp = load_map_dir(ctx.p("fixc_root") / kind)
        m = mp[mp.method == "merged"][["target", "source", "K", "rep", "uar"]].rename(columns={"uar": "m"})
        t = mp[mp.method == "tgt_only"][["target", "K", "rep", "uar"]].rename(columns={"uar": "t"})
        x = w.merge(m, on=["target", "source", "K", "rep"]).merge(t, on=["target", "K", "rep"])
        cvdir = ROOT / f"{ctx.a.cv_map_prefix}{kind}"
        if cvdir.exists():
            cvm = load_map_dir(cvdir)
            cvm = cvm[cvm.method == "merged"][["target", "source", "K", "rep", "uar", "C"]].rename(
                columns={"uar": "cvm", "C": "C_cvm"})
            x = x.merge(cvm, on=["target", "source", "K", "rep"], how="left")
            sd_ = x.ctrl - x.cvm
            # E12 の学び直しと交差検証の地図は温め直しの出発点が違うので，テストの答えが数本入れ替わる差が残る（関門 1 の
            # compare の指摘のとおり）．一致の検算ではなく差の大きさの記録として `records.csv` に書く．符号つきの平均と，
            # 2 つの学習で選ばれた C が違う行の割合も書く
            ok = sd_.notna()
            ctx.record(10, kind, "対照（α = nat，C を交差検証）と交差検証の地図の合併の UAR の差（I14）",
                       float(sd_[ok].abs().max()), float((sd_[ok].abs() > 1e-6).mean()), int(ok.sum()),
                       "行（組 × K × 反復）", signed_mean=float(sd_[ok].mean()),
                       frac_c_differs=float((x.C_ctrl[ok].astype(float) != x.C_cvm[ok].astype(float)).mean()))
        else:
            x["cvm"], x["C_cvm"] = np.nan, np.nan
        if len(x) != len(w):
            ctx.gap(10, kind, "E12 の行のうち地図と突き合わせられた数", len(w), len(x))
        for (tt, s, K), h in x.groupby(["target", "source", "K"]):
            if len(h) != 3:
                ctx.gap(10, kind, f"E12 の反復 {tt}←{s} K={K}", 3, len(h))
        pp = x.groupby(["target", "source", "K"]).agg(wt=("uar", "mean"), ctrl=("ctrl", "mean"), m=("m", "mean"),
                                                      cvm=("cvm", "mean"), t=("t", "mean"),
                                                      n_rep=("rep", "nunique")).reset_index()
        pp["gain_wt"], pp["gain_nat"] = (pp.wt - pp.t) * 100, (pp.m - pp.t) * 100
        pp["gain_ctrl"], pp["gain_cvmap"] = (pp.ctrl - pp.t) * 100, (pp.cvm - pp.t) * 100
        # 差は同じ学習の中だけで取る 3 つの項に分ける（compare の追記）．3 つの和は wt − m に等しい
        pp["alpha_part"] = (pp.wt - pp.ctrl) * 100                     # α を選ぶ分（E12 の学び直しの中）
        pp["c_rule_part"] = (pp.cvm - pp.m) * 100                      # C の選び方の分（交差検証の地図の中．同じ学習）
        pp["relearn_part"] = (pp.ctrl - pp.cvm) * 100                  # 学び直しの違いの分（E12 の学び直し − 交差検証の地図）
        pp["wt_minus_nat"] = (pp.wt - pp.m) * 100
        ctx.check(10, kind, "α の分 + C の選び方の分 + 学び直しの違いの分 と wt − m の差の最大",
                  float((pp.alpha_part + pp.c_rule_part + pp.relearn_part - pp.wt_minus_nat).abs().max()), 1e-9)
        pp["wt_minus_cvmap"] = (pp.wt - pp.cvm) * 100
        pp["same_corpus"] = [same_corpus(a, b) for a, b in zip(pp.target, pp.source)]
        prs.append(pp.assign(kind=kind))
        cols = ("gain_wt", "gain_nat", "gain_ctrl", "gain_cvmap", "alpha_part", "c_rule_part", "relearn_part",
                "wt_minus_nat", "wt_minus_cvmap")
        for pair_set, base in (("excl_same_corpus", pp[~pp.same_corpus]), ("all_pairs", pp)):
            targets = sorted(base.target.unique())
            for K, h in base.groupby("K"):
                row = {"kind": kind, "short": SHORT[kind], "pair_set": pair_set, "K": K, "n_pairs": len(h),
                       "n_targets": len(targets), "n_rep_min": int(h.n_rep.min())}
                rng = common.rng_for(f"e75-e12boot:{kind}:{pair_set}:{K}", ctx.a.seed)
                by_t = {tt: h[h.target == tt] for tt in targets}
                boots = {c: [] for c in cols}
                for _ in range(ctx.a.n_boot_targets):
                    pick = rng.choice(targets, size=len(targets), replace=True)
                    hb = pd.concat([by_t[tt] for tt in pick])
                    for c in cols:
                        boots[c].append(hb[c].mean())
                for c in cols:
                    tm = h.groupby("target")[c].mean()
                    u = unit_interval(tm.to_numpy())
                    row[f"mean_{c}"] = h[c].mean()
                    row[f"tgt_t_lo_{c}"], row[f"tgt_t_hi_{c}"] = u["t_lo"], u["t_hi"]   # 主：ターゲットの平均の t 区間
                    row[f"n_tgt_pos_{c}"], row[f"p_sign_{c}"] = u["n_pos"], u["p_sign"]
                    row[f"boot_lo_{c}"], row[f"boot_hi_{c}"] = np.quantile(boots[c], [0.025, 0.975])  # 参考
                rows.append(row)
                if ref is not None and kind in state and pair_set == "all_pairs":
                    rr = ref[(ref.setting == kind) & (ref.K == K)]
                    # 検算は e12_wt.csv を作ったときの反復だけで作り直して比べる（HuBERT はそのとき反復 0〜1 の群があった）
                    in_state = np.array([int(r) in state[kind].get(tt, []) for tt, r in zip(x.target, x.rep)], dtype=bool)
                    xs = x[(x.K == K).to_numpy() & in_state]
                    ps = xs.groupby(["target", "source"]).agg(wt=("uar", "mean"), m=("m", "mean"), t=("t", "mean"))
                    if len(rr) and len(ps):
                        ctx.check(10, kind, f"E12 の gain_wt・gain_nat（K={K}．e12_state.json の反復）と e12_wt.csv の差の最大",
                                  float(max(abs((ps.wt - ps.t).mean() - rr.gain_wt.iloc[0]),
                                            abs((ps.m - ps.t).mean() - rr.gain_nat.iloc[0]))), 1e-9,
                                  f"{len(ps)} 組，{len(xs)} 行")
    if rows:
        ctx.write("e75_10_e12_wt.csv", pd.DataFrame(rows))
        ctx.write("e75_10_pairs.csv", pd.concat(prs, ignore_index=True))


# ---------------------------------------------------------------- 項目 11：q の当てはめの診断

def fit_grid(M_: np.ndarray, D: np.ndarray, Ks: np.ndarray, full: bool = True) -> dict:
    """M_[..., 対, K] に 3 つの形を当てる（最小二乗．格子で探す）．先頭の次元はブートストラップの回。"""
    # F0：d = D/√(1 + q/K)（`explore_direction_bound.q_fit` と同じ形・同じ格子）
    f0 = D[None, :, None] / np.sqrt(1 + Q_GRID[:, None, None] / Ks[None, None, :])           # (q, 対, K)
    Mb = M_[None] if M_.ndim == 2 else M_                                                     # (B, 対, K)
    sse0 = ((Mb[:, None, :, :] - f0[None]) ** 2).sum(axis=(-1, -2))                          # (B, q)
    i0 = sse0.argmin(-1)
    q, s0 = Q_GRID[i0], sse0[np.arange(len(i0)), i0]
    out = {"q": q[0] if M_.ndim == 2 else q, "sse0": s0[0] if M_.ndim == 2 else s0}
    if M_.ndim == 2 and full:
        i0 = i0[0]
        # F1：d = D/√(1 + (q/K)^β)，F2：d = s·D/√(1 + q/K)
        f1 = D[None, None, :, None] / np.sqrt(1 + (Q_GRID[:, None, None, None] / Ks[None, None, None, :])
                                              ** BETA_GRID[None, :, None, None])
        sse1 = ((M_[None, None] - f1) ** 2).sum(axis=(-1, -2))
        j1 = np.unravel_index(sse1.argmin(), sse1.shape)
        f2 = SCALE_GRID[None, :, None, None] * f0[:, None]
        sse2 = ((M_[None, None] - f2) ** 2).sum(axis=(-1, -2))
        j2 = np.unravel_index(sse2.argmin(), sse2.shape)
        out.update(q1=Q_GRID[j1[0]], beta1=BETA_GRID[j1[1]], sse1=float(sse1[j1]),
                   q2=Q_GRID[j2[0]], scale2=SCALE_GRID[j2[1]], sse2=float(sse2[j2]),
                   fit0=f0[i0], fit1=f1[j1], fit2=f2[j2])
    return out


def loko(M_: np.ndarray, D: np.ndarray, Ks: np.ndarray) -> dict:
    """K を 1 つ抜いて残り 2 つの K で当て，抜いた K の 6 対の平均の d を予言したときの誤差の絶対値の平均（3 通り）。
    同じ K の 6 対は同じ分類器を共有するので，18 点を独立とみなす AIC の代わりに使う（D4）。"""
    err = {"F0": [], "F1": [], "F2": []}
    for k in range(len(Ks)):
        keep = [i for i in range(len(Ks)) if i != k]
        fg = fit_grid(M_[:, keep], D, Ks[keep])
        true = M_[:, k].mean()
        p0 = (D / np.sqrt(1 + fg["q"] / Ks[k])).mean()
        p1 = (D / np.sqrt(1 + (fg["q1"] / Ks[k]) ** fg["beta1"])).mean()
        p2 = (fg["scale2"] * D / np.sqrt(1 + fg["q2"] / Ks[k])).mean()
        for name, p in (("F0", p0), ("F1", p1), ("F2", p2)):
            err[name].append(abs(p - true))
    return {f"loko_err_{k}": float(np.mean(v)) for k, v in err.items()}


def item11(ctx: Ctx) -> None:
    Ks = np.array([1.0, 5.0, 10.0])
    Kstr = list(KS3)
    tg_rows, res_rows, boot_rows = [], [], []
    ref_path = ctx.p("e28t9_targets")
    ref = read_csv(ctx, ref_path) if ref_path.exists() else None
    for kind in ctx.a.kinds:
        dpath = ROOT / f"{ctx.a.tgt_scores_prefix}{kind}"
        if not dpath.exists():
            continue
        own_all = ctx.e13(kind)
        for f in sorted(dpath.glob("*.npz")):
            ctx.note_input(f)
            t = f.stem
            z = np.load(f)
            reps = sorted(int(k.split("_r")[1]) for k in z.files if k.startswith("test_idx_r"))
            rec, cube = [], np.full((len(reps), 6, 3), np.nan)
            for ir, r in enumerate(reps):
                y = z[f"y_test_r{r}"]
                for ik, K in enumerate(Kstr):
                    for ip, ((a, b), (dd, oo)) in enumerate(zip(PAIRS, d_o(z[f"t_K{K}_r{r}"].astype(np.float64), y))):
                        rec.append({"rep": r, "K": K, "pair": f"{a}-{b}", "d": dd, "o": oo})
                        cube[ir, ip, ik] = dd
            if not np.isfinite(cube).all():
                ctx.gap(11, kind, f"d が NaN の升 {t}", 0, int((~np.isfinite(cube)).sum()))
                continue
            dT = pd.DataFrame(rec)
            own = own_all[own_all.target == t].groupby("pair_idx").own_d.mean()
            D = own.reindex(PAIR_IDX).to_numpy()
            q_ref = q_fit(dT, own)                                  # 今の台本の関数そのもの
            M_ = cube.mean(0)
            fg = fit_grid(M_, D, Ks)
            ctx.check(11, kind, f"格子の当てはめ（まとめて解く版）と q_fit の差 {t}", abs(float(fg["q"]) - q_ref), 0.0)
            if ref is not None:
                rr = ref[(ref.kind == kind) & (ref.target == t)]
                if len(rr):
                    ctx.check(11, kind, f"q_dir と e28t9_targets.csv の差 {t}", abs(q_ref - float(rr.q_dir.iloc[0])), 1e-9,
                              ctx.e13_note(kind))
            n = M_.size
            aic = lambda sse, k: n * np.log(sse / n) + 2 * k  # noqa: E731
            # テストの集合を単位にしたブロックのブートストラップ（同じテストの集合の反復を 1 塊で復元抽出．S1・S6）
            sets = test_sets_from_idx(z, reps)
            cid = {s: i for i, s in enumerate(sorted(set(sets.values()), key=sorted))}
            blk = np.array([cid[sets[r]] for r in reps])
            n_b = len(cid)
            cl_sum = np.stack([cube[blk == b].sum(0) for b in range(n_b)])
            cl_cnt = np.array([(blk == b).sum() for b in range(n_b)], float)
            rng = common.rng_for(f"e75-qboot:{kind}:{t}", ctx.a.seed)
            idx = rng.integers(0, n_b, size=(ctx.a.n_boot, n_b))
            Mb = cl_sum[idx].sum(1) / cl_cnt[idx].sum(1)[:, None, None]
            qb = fit_grid(Mb, D, Ks, full=False)["q"]
            boot_rows.append(pd.DataFrame({"kind": kind, "target": t, "b": np.arange(ctx.a.n_boot), "q": qb}))
            lk = loko(M_, D, Ks)
            p_dim = ctx.dims.get(kind) or int(load_any(t, kind)[1].shape[1])
            tg_rows.append({"kind": kind, "short": SHORT[kind], "target": t, "n_reps": len(reps), "n_test_sets_reps": n_b,
                            "q_dir": q_ref,
                            "q_boot_lo": float(np.quantile(qb, 0.025)), "q_boot_hi": float(np.quantile(qb, 0.975)),
                            "n_boot_at_grid_edge": int(((qb <= Q_GRID[0]) | (qb >= Q_GRID[-1])).sum()),
                            "rms_resid_F0": float(np.sqrt(fg["sse0"] / n)), "aic_F0_indep": aic(fg["sse0"], 1),
                            "q_F1": fg["q1"], "beta_F1": fg["beta1"], "rms_resid_F1": float(np.sqrt(fg["sse1"] / n)),
                            "aic_F1_indep": aic(fg["sse1"], 2), "q_F2": fg["q2"], "scale_F2": fg["scale2"],
                            "rms_resid_F2": float(np.sqrt(fg["sse2"] / n)), "aic_F2_indep": aic(fg["sse2"], 2), **lk,
                            "F1_better_loko": bool(lk["loko_err_F1"] < LOKO_BETTER * lk["loko_err_F0"]),
                            "F2_better_loko": bool(lk["loko_err_F2"] < LOKO_BETTER * lk["loko_err_F0"]),
                            "beta_at_grid_edge": bool(fg["beta1"] in (BETA_GRID[0], BETA_GRID[-1])),
                            "scale_at_grid_edge": bool(fg["scale2"] in (SCALE_GRID[0], SCALE_GRID[-1])),
                            "own_d_mean": float(D.mean()), "dim": p_dim,
                            "q_classical": float(2 * (p_dim - 1) / np.mean(D ** 2))})
            for ip, pidx in enumerate(PAIR_IDX):
                for ik, K in enumerate(Kstr):
                    res_rows.append({"kind": kind, "target": t, "pair": pidx, "K": K, "d_meas": M_[ip, ik],
                                     "d_fit_F0": float(fg["fit0"][ip, ik]), "resid_F0": M_[ip, ik] - float(fg["fit0"][ip, ik]),
                                     "d_fit_F1": float(fg["fit1"][ip, ik]), "d_fit_F2": float(fg["fit2"][ip, ik]),
                                     "own_d": D[ip], "d_meas_sd_over_reps": float(cube[:, ip, ik].std(ddof=1))})
    if not tg_rows:
        return
    T = pd.DataFrame(tg_rows)
    B = pd.concat(boot_rows, ignore_index=True)
    Rk_all = pd.DataFrame(res_rows)
    crs = ctx.cache.get("item4_crs")
    S = []
    for kind, h in T.groupby("kind"):
        nT = len(h)
        rng = common.rng_for(f"e75-qtgtboot:{kind}", ctx.a.seed)
        tb = [np.median(rng.choice(h.q_dir.to_numpy(), size=nT, replace=True)) for _ in range(ctx.a.n_boot)]
        rb = B[B.kind == kind].groupby("b").q.median()
        Rk = Rk_all[Rk_all.kind == kind]
        n_f1, n_f2 = int(h.F1_better_loko.sum()), int(h.F2_better_loko.sum())
        S.append({"kind": kind, "short": SHORT[kind], "n_targets": nT, "median_q_dir": h.q_dir.median(),
                  # 主：順序統計量の区間 [最小, 最大]（中央値の被覆は 1 − 2·0.5^n．S9）
                  "median_q_os_lo": h.q_dir.min(), "median_q_os_hi": h.q_dir.max(),
                  "median_q_os_coverage": 1 - 2 * 0.5 ** nT,
                  "median_q_blockboot_lo": float(rb.quantile(0.025)), "median_q_blockboot_hi": float(rb.quantile(0.975)),
                  "median_q_target_boot_lo": float(np.quantile(tb, 0.025)),
                  "median_q_target_boot_hi": float(np.quantile(tb, 0.975)),
                  "median_q_classical": float(h.q_classical.median()),
                  "n_targets_F1_better_loko": n_f1, "n_targets_F2_better_loko": n_f2,
                  "n_targets_F1_better_aic2_indep": int((h.aic_F1_indep < h.aic_F0_indep - 2).sum()),
                  "n_targets_F2_better_aic2_indep": int((h.aic_F2_indep < h.aic_F0_indep - 2).sum()),
                  "median_beta_F1": h.beta_F1.median(), "median_scale_F2": h.scale_F2.median(),
                  "median_rms_resid_F0": h.rms_resid_F0.median(),
                  "median_loko_err_F0": h.loko_err_F0.median(), "median_loko_err_F1": h.loko_err_F1.median(),
                  "median_loko_err_F2": h.loko_err_F2.median(),
                  "min_test_sets_reps": int(h.n_test_sets_reps.min()),
                  **{f"mean_resid_F0_K{K}": float(Rk[Rk.K == K].resid_F0.mean()) for K in Kstr}})
        ctx.assume(11, kind, "F0：d(K) は 1 母数の K の関数で表せる", "K を 1 つ抜いた予言で F1 か F2 が F0 より良い（誤差 0.8 倍未満）ターゲットの数",
                   max(n_f1, n_f2), f"{nT} ターゲットの過半に届かない", bool(max(n_f1, n_f2) <= nT // 2))
        if crs is not None:
            c = crs[crs.kind == kind].set_index("K")
            ctx.assume(11, kind, "F0：K の間でターゲットのみの C が同じ（記述）",
                       "ターゲットのみの C が交差検証で決まった割合（K=1・5・10）",
                       float(c.reindex(Kstr).share_cv.mean()) if len(c) else np.nan, "記述（C が K で変わる種類は q に C の変化を含む）",
                       np.nan, ", ".join(f"K={K}: cv {c.loc[K, 'share_cv']:.2f}, tie {c.loc[K, 'share_tie']:.2f}, fixed {c.loc[K, 'share_fixed']:.2f}"
                                         for K in Kstr if K in c.index))
    ctx.write("e75_11_targets.csv", T)
    ctx.write("e75_11_residuals.csv", Rk_all)
    ctx.write("e75_11_summary.csv", pd.DataFrame(S))
    ctx.write("e75_11_boot.csv", B)


# ---------------------------------------------------------------- 項目 12：λ の順位相関

def lam_obs_from_rows(R: pd.DataFrame) -> pd.DataFrame:
    """`exp/explore_synergy.py` の段階 B と同じ式：|o_s| ≥ O_MIN の（反復，対）で λ_obs = 1 − Σ sign(o_s)·o_m / Σ |o_s|。"""
    out = []
    for (kind, t, s, K), g in R.groupby(["kind", "target", "source", "K"]):
        v = g[g.o_s.abs() >= O_MIN]
        lo = 1 - float((np.sign(v.o_s) * v.o_m).sum() / v.o_s.abs().sum()) if len(v) else np.nan
        out.append({"kind": kind, "target": t, "source": s, "K": K, "n_valid": len(v), "lam_obs_calc": lo})
    return pd.DataFrame(out)


def item12(ctx: Ctx) -> None:
    out, win = [], []
    kinds = [k for k in ctx.a.kinds if k in MAIN4 + TEXT_KINDS]
    if not kinds:
        return
    # λ_obs を反復 0〜9 の段階 B の行から作り直す（言葉系の保存済みの lam_obs は反復 0〜29 から作られているため．D1・R8）
    Rall = stageb_rows(ctx, kinds, 12, max_rep=None)
    calc_all = lam_obs_from_rows(Rall)
    calc10 = lam_obs_from_rows(Rall[Rall.rep < 10])
    for key, ks in (("stageb_main", MAIN4), ("stageb_text", TEXT_KINDS)):
        use = [k for k in kinds if k in ks]
        if not use:
            continue
        P = read_csv(ctx, ctx.p(key) / "e15b_pairs.csv", dtype={"K": str})
        P_all = P[P.kind.isin(use)]
        # 検算：全反復から作り直した λ_obs が保存済みの lam_obs と一致するか（同じコーパスの組を除く）
        j = P_all.merge(calc_all, on=["kind", "target", "source", "K"], how="inner")
        for kind, h in j.groupby("kind"):
            ctx.check(12, kind, "全反復から作り直した λ_obs と e15b_pairs.csv の lam_obs の差の最大",
                      float((h.lam_obs - h.lam_obs_calc).abs().max()), 1e-5, f"{len(h)} 行")
        tests = ctx.p(key) / "e15b_tests.csv"
        if tests.exists():
            T = read_csv(ctx, tests)
            name = {"audio_wavlm": "W7", "audio_hubert": "Hu7", "video": "V", "fuse-audio-video": "AV"}
            for kind in use:
                if kind in name and (T.setting == name[kind]).any():
                    r, _, _ = spear(P_all[P_all.kind == kind].lam_obs, P_all[P_all.kind == kind].lam_pred_rep09)
                    ctx.check(12, kind, "K をまとめた順位相関と e15b_tests.csv の差（同じコーパスの組を含む）",
                              abs(r - float(T[T.setting == name[kind]].spearman.iloc[0])), 1e-9)
        P = P_all.merge(calc10, on=["kind", "target", "source", "K"], how="inner")
        P = P[[not same_corpus(t, s) for t, s in zip(P.target, P.source)]]
        for kind in use:
            h = P[P.kind == kind]
            r, p, n = spear(h.lam_obs_calc, h.lam_pred_rep09)
            out.append({"kind": kind, "short": SHORT[kind], "K": "pooled", "n": n, "rho": r, "p": p,
                        "mean_obs": h.lam_obs_calc.mean(), "mean_pred": h.lam_pred_rep09.mean()})
            tau = h.tau_K5.dropna()
            ctx.assume(12, kind, "λ の式の近似 (c)（ターゲットとソースのスイッチの向きの散らばりの比 τ）", "τ（K=5）の中央値",
                       float(tau.median()) if len(tau) else np.nan,
                       "記述（E15 の 8 節で τ は 20〜70 で，近似 (c) は成り立っていないと記録済み）", np.nan)
            for K, hk in h.groupby("K"):
                r, p, n = spear(hk.lam_obs_calc, hk.lam_pred_rep09)
                grp = []
                for t, ht in hk.groupby("target"):
                    if len(ht) >= 4:
                        rt, _, nt = spear(ht.lam_obs_calc, ht.lam_pred_rep09)
                        win.append({"kind": kind, "K": K, "target": t, "n_sources": nt, "rho": rt})
                        grp.append((ht.lam_pred_rep09.to_numpy(float), ht.lam_obs_calc.to_numpy(float)))
                rng = common.rng_for(f"e75-lamperm:{kind}:{K}", ctx.a.seed)
                wp = within_target_perm(grp, ctx.a.n_perm, rng) if grp else {"n_targets": 0}
                out.append({"kind": kind, "short": SHORT[kind], "K": K, "n": n, "rho": r, "p": p,
                            "mean_obs": hk.lam_obs_calc.mean(), "mean_pred": hk.lam_pred_rep09.mean(),
                            **{f"within_{k}": v for k, v in wp.items()}})
    ctx.write("e75_12_lambda.csv", pd.DataFrame(out))
    ctx.write("e75_12_within_target.csv", pd.DataFrame(win))


# ---------------------------------------------------------------- 項目 13：e をほかのコーパスで推す

def source_effect_estimate(V: pd.Series, t: str) -> pd.Series:
    """ターゲット t のラベルを使わずにソースの値を推す：t と同じコーパスの群をターゲットからもソースからも除いた組に
    v(T′, S) = a_T′ + b_S を最小二乗で当てたソースの効果 b̂_S（`pair_table.fit`．D10：ソースごとに違う集合で平均しない）。"""
    d = V.reset_index()
    d.columns = ["target", "source", "v"]
    d = d[[corpus(a) != corpus(t) and corpus(b) != corpus(t) for a, b in zip(d.target, d.source)]]
    if d.target.nunique() < 2 or d.source.nunique() < 2:
        return pd.Series(dtype=float)
    return pair_table.fit(d.reset_index(drop=True), "v")["b"]


def item13(ctx: Ctx) -> None:
    rows, win, S = [], [], []
    ref = read_csv(ctx, ctx.p("e28t9_pairs"), dtype={"K": str}) if ctx.p("e28t9_pairs").exists() else None
    for kind in ctx.a.kinds:
        e = ctx.e13(kind)
        n_small = int((e.drop_duplicates(["target", "pair"]).own_d <= DELTA_MIN).sum())
        e = e[[not same_corpus(t, s) for t, s in zip(e.target, e.source)]].copy()
        e["eff"] = (e.d / e.own_d).clip(0.0, 0.999)                   # E28-9 と同じ e = d_s/D（切る範囲も同じ）
        E = e.groupby(["target", "source"]).eff.mean()
        lab = ROOT / f"{ctx.a.shift_prefix}{kind}" / "labeling.csv"
        U = read_csv(ctx, lab).set_index(["target", "source"]).uar_t if lab.exists() else None
        U = U[[not same_corpus(t, s) for t, s in U.index]] if U is not None else None
        if ref is not None:
            rr = ref[(ref.kind == kind) & (ref.K == "5")].set_index(["target", "source"]).e
            common_idx = E.index.intersection(rr.index)
            if len(common_idx):
                ctx.check(13, kind, "e と e28t9_pairs.csv の e の差の最大", float((E.loc[common_idx] - rr.loc[common_idx]).abs().max()),
                          1e-9, f"{len(common_idx)} 組．{ctx.e13_note(kind)}")
        # 加法性：e(T,S) = a_T + b_S の R²（仮定の表）
        dE = E.reset_index()
        r2ts, _ = pair_table.r2(dE.eff.to_numpy(float), pair_table.design(dE, True, True))
        ctx.assume(13, kind, "e の加法性：ソースの効果がターゲットをまたいで同じ", "e(T,S) = a_T + b_S の R²", float(r2ts),
                   "記述（低ければ ê の解釈を限る）", np.nan)
        grp_main, grp_mean, grp_uar = [], [], []
        for t in sorted(E.index.get_level_values(0).unique()):
            b = source_effect_estimate(E, t)
            bu = source_effect_estimate(U, t) if U is not None else pd.Series(dtype=float)
            ht = E.loc[t]
            used = sorted({tp for (tp, sp) in E.index if corpus(tp) != corpus(t)})
            for s, v in ht.items():
                others = [tp for (tp, sp) in E.index if sp == s and tp != t and not same_corpus(tp, t)]
                rows.append({"kind": kind, "target": t, "source": s, "e_measured": v,
                             "e_pseudo_source_effect": float(b.get(s, np.nan)),
                             "e_pseudo_mean": float(np.mean([E[(tp, s)] for tp in others])) if others else np.nan,
                             "uar_src_pseudo_source_effect": float(bu.get(s, np.nan)),
                             "n_pseudo_targets": len(others), "pseudo_targets_used": ";".join(used)})
            if len(ht) >= 4:
                x_main = np.array([b.get(s, np.nan) for s in ht.index], float)
                x_mean = np.array([r["e_pseudo_mean"] for r in rows if r["kind"] == kind and r["target"] == t], float)
                x_uar = np.array([bu.get(s, np.nan) for s in ht.index], float)
                yv = ht.to_numpy(float)
                for name, xv, grp in (("source_effect", x_main, grp_main), ("mean", x_mean, grp_mean),
                                      ("uar_src_source_effect", x_uar, grp_uar)):
                    r, _, n = spear(xv, yv)
                    win.append({"kind": kind, "short": SHORT[kind], "estimator": name, "target": t, "n_sources": n,
                                "rho": r, "pseudo_targets_used": ";".join(used)})
                    grp.append((xv, yv))
        for name, grp in (("source_effect", grp_main), ("mean", grp_mean), ("uar_src_source_effect", grp_uar)):
            rng = common.rng_for(f"e75-eperm:{kind}:{name}", ctx.a.seed)
            wp = within_target_perm(grp, ctx.a.n_perm, rng) if grp else {"n_targets": 0}
            S.append({"kind": kind, "short": SHORT[kind], "estimator": name,
                      "role": {"source_effect": "main", "mean": "secondary", "uar_src_source_effect": "comparator"}[name],
                      "n_target_pair_delta_le_min": n_small, **wp})
    ctx.write("e75_13_pairs.csv", pd.DataFrame(rows))
    ctx.write("e75_13_within_target.csv", pd.DataFrame(win))
    ctx.write("e75_13_summary.csv", pd.DataFrame(S))


# ---------------------------------------------------------------- 本体

FUNCS = {1: item1, 2: item2, 3: item3, 4: item4, 5: item5, 6: item6, 7: item7, 8: item8, 9: item9,
         10: item10, 11: item11, 12: item12, 13: item13}


def main() -> int:
    ap = argparse.ArgumentParser(description="E75：筋 1〜3 の保存済みの結果からの再解析（CPU）")
    ap.add_argument("--out", default="results/e75", help="出力先（既定 results/e75）．ファイルがあれば止まる")
    ap.add_argument("--kinds", nargs="+", choices=KINDS_ALL, default=KINDS_ALL)
    ap.add_argument("--items", nargs="+", type=int, choices=ITEMS, default=ITEMS)
    ap.add_argument("--n-perm", type=int, default=2000, help="項目 3・12・13 の並べ替えの回数")
    ap.add_argument("--n-boot", type=int, default=1000, help="項目 11 のブートストラップの回数")
    ap.add_argument("--n-boot-targets", type=int, default=2000, help="項目 10 のターゲットを単位にしたブートストラップの回数（参考）")
    ap.add_argument("--seed", type=int, default=0)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    a = ap.parse_args()
    a.kinds = [k for k in KINDS_ALL if k in a.kinds]
    a.items = sorted(set(a.items))
    if "audio_hubert" in a.kinds and a.shift_prefix != DEFAULTS["shift_prefix"]:
        raise SystemExit("HuBERT の E13 は explore_pairs.measure が results/shift_audio_hubert を固定で読むので，"
                         "--shift-prefix を変えるときは audio_hubert を --kinds から外す")
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    a.out = str(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} にすでにファイルがある．既存の結果を上書きしないので，空のフォルダか新しい場所を指定する")
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(a)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    gs = git_state()
    code_dir = M.snapshot_code(out, started)                      # 台本と部品の写しと git diff HEAD（docs/methods.md 6.3 節）
    ctx.header = {"script": "exp/reanalysis_e75.py", "started": started, "commit": gs["commit"], "dirty": gs["dirty"],
                  "script_status": gs["script_status"], "script_sha256": gs["script_sha256"], "argv": sys.argv[1:],
                  "seed": a.seed}
    prov = {"script": "exp/reanalysis_e75.py", "argv": sys.argv[1:], "started": started, **gs,
            "code_snapshot": str(code_dir.relative_to(out)),
            "settings": {**{k: getattr(a, k) for k in ("kinds", "items", "n_perm", "n_boot", "n_boot_targets", "seed")},
                         **{k: getattr(a, k) for k in DEFAULTS},
                         "den_small_points": DEN_SMALL, "loc_min_points": LOC_MIN, "delta_min": DELTA_MIN,
                         "q_grid": "logspace(-1,3,400)", "beta_grid": "linspace(0.25,3,56)",
                         "scale_grid": "linspace(0.5,2,61)", "loko_better": LOKO_BETTER,
                         "interval_main": "IEMOCAP がターゲットの組は話者を共有するテストの集合をつないだ単位の t，ほかは補正した resampled t（rho = n_test/n_pool）",
                         "holm_family": "種類 × K の中の，同じコーパスの組を除いた組",
                         "reps_stage_b": "0-9", "gh_nodes": len(GH_X), "o_min_lambda": O_MIN,
                         "hubert_e13": "explore_pairs.measure('audio_hubert') が results/shift_audio_hubert から作る"},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                         "torch": sys.modules["torch"].__version__ if "torch" in sys.modules else None},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "elapsed_sec": {}}
    t_all = time.time()
    for it in a.items:
        t0 = time.time()
        print(f"項目 {it} を計算する", flush=True)
        FUNCS[it](ctx)
        prov["elapsed_sec"][str(it)] = round(time.time() - t0, 1)
        print(f"  項目 {it}：{prov['elapsed_sec'][str(it)]} 秒", flush=True)
    prov["elapsed_sec"]["total"] = round(time.time() - t_all, 1)
    prov["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["inputs"] = ctx.inputs
    prov["modules"] = module_state()
    prov["e13_computed_on_the_fly"] = ctx.cache.get("e13_computed", [])
    G = pd.DataFrame(ctx.gaps, columns=["item", "kind", "what", "expected", "found", "detail"])
    Ck = pd.DataFrame(ctx.checks, columns=["item", "kind", "what", "value", "tol", "ok", "detail"])
    As = pd.DataFrame(ctx.assumptions, columns=["item", "kind", "assumption", "measure", "value", "criterion", "holds", "detail"])
    ctx.write("gaps.csv", G)
    ctx.write("checks.csv", Ck)
    ctx.write("e75_assumptions.csv", As)
    ctx.write("records.csv", pd.DataFrame(ctx.records, columns=["item", "kind", "what", "max_abs_diff",
                                                                "frac_rows_diff_gt_1e-6", "n", "unit",
                                                                "signed_mean_diff", "frac_rows_C_differs"]))
    prov["written"] = ctx.cache.get("written", [])
    prov["n_gaps"], prov["n_checks"], prov["n_checks_failed"] = len(G), len(Ck), int((~Ck.ok).sum()) if len(Ck) else 0
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))
    print(f"欠け {len(G)} 件，検算 {len(Ck)} 件（外れ {prov['n_checks_failed']} 件），計 {prov['elapsed_sec']['total']} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
