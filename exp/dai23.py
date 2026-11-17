"""実験 23（DAI-23）：言葉を聞き取れなくした声の特徴は，言葉の分類器の判断をどれだけ再現するか（設計 `docs/設計/実験23_設計.md`）．

実験 14 の保存（results/dai14/units/）の目標（J_A・J_V・J_T・J_prop の真と #shuf・#shufcl）と，生徒 A（元の声）・V（顔）の
R1 の交差の予測をそのまま使い，言葉を聞き取れなくした音声の WavLM の特徴の生徒 2 つ（L：低域通過，B：逆再生）を
R1（実験 14 と同じ作り）で足す．
K = 5 だけ．コーパスは IEMOCAP（即興の部），MOSEI，MELD．AFEW は外す（言葉が音声の書き起こし）．
加えて，本当の感情 y を目標にした R(L ← y) と R(A ← y)（記述．仮定の表の「感情の情報が残る」）を測る．

主の量（操作 X ∈ {L, B}．D は D_shufcl．κ_T = D(A ← T) ÷ D(A ← V) は実験 14 の点推定で固定）：
  M1_X = D(A ← T) − D(X ← T)
  M2_X = M1_X − [D(A ← V) − D(X ← V)]          （同じ点数で減る帰無に対する余分な減り）
  M3_X = M1_X − κ_T [D(A ← V) − D(X ← V)]      （同じ割合で減る帰無に対する余分な減り）
  L3_X = D(X ← T) − D(V ← T)                    （記述）

段：
  selftest  合成の数で，同じ予測どうしの差が 0 になること，係数の符号，数えの規則を確かめる
  units     単位ごとの計算．results/dai23/units/<T>_f<f>_r<r>.npz と .csv（同じコミットの既存の単位は飛ばす）
  summary   区間と予言の判定．results/dai23/summary/*.csv

`--verify` を付けると，各単位で (1) 実験 14 の入れ替えた目標を同じ規則で作り直して一致を確かめ（いつも行う），
(2) 元の声の特徴を同じ流れで読み直して J_T・J_V を R1 で学び，実験 14 の保存の予測との一致の割合を確かめる（試しで使う）．
`--l-kind audio_wavlm --b-kind audio_wavlm` は，L と B に元の声の特徴を入れる退化の試し（M1・M2・M3 が 0 に近くなるはず）．
本番の置き場では使えない．
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai14 as D14  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
OUT1 = ROOT / "results" / "dai1"
OUT14 = ROOT / "results" / "dai14"
OUT_PROD = ROOT / "results" / "dai23"
FC_JSON = OUT_PROD / "fc.json"
TARGETS = ("IEMOCAP", "MOSEI", "MELD")
K = 5
JUDG = ("single_A", "single_V", "single_T")
VERS_L = ("true", "shuf", "shufcl")
RNG_SEED = 20261115
MEM_FRAC = 0.6
VERIFY_AGREE = 0.995          # 元の声で作り直した R1 の予測が実験 14 の保存と一致する割合の下限（--verify）
RATIO_MIN = 0.5               # 仮定：R(X ← y) − 25 が R(A ← y) − 25 のこの割合以上（設計 6 節）
KAPPA_DESIGN = {("IEMOCAP", "all"): 3.215, ("IEMOCAP", "nomarker"): 3.328, ("MOSEI", "all"): 6.165, ("MOSEI", "nomarker"): 5.779,
                ("MELD", "all"): 1.949, ("MELD", "nonshared"): 1.784, ("MELD", "nomarker"): 1.958}   # 設計 3 節．部分集合ごと．実験 14 の predictions.csv から計算し直して小数 3 桁で照らす
OPS = ("L", "B")
SCRIPTS = ("dai23.py", "dai14.py", "dai3.py", "dai1.py", "dai1_stats.py", "dai_solver.py", "dai_data.py")
FEAT_SCRIPTS = ("build_audio.py", "build_audio_lowpass.py", "check_lowpass_asr.py")
DESIGN = ROOT / "docs" / "設計" / "実験23_設計.md"
RUN = {"commit": "", "dirty": True, "partial": False}

def quant(X: str, kappa: float) -> dict:
    """操作 X の量：[(生徒, 目標の名前, 版, 係数)]．"""
    dAT = [("A", "single_T", "true", 1), ("A", "single_T", "shufcl", -1)]
    dXT = [("%s" % X, "single_T", "true", -1), (X, "single_T", "shufcl", 1)]
    dAV = [("A", "single_V", "true", -1), ("A", "single_V", "shufcl", 1)]
    dXV = [(X, "single_V", "true", 1), (X, "single_V", "shufcl", -1)]
    sc = lambda terms, k: [(a, b, c, k * d) for a, b, c, d in terms]
    return {
        f"M1_{X}": dAT + dXT,
        f"M2_{X}": dAT + dXT + dAV + dXV,
        f"M3_{X}": dAT + dXT + sc(dAV, kappa) + sc(dXV, kappa),
        f"L3_{X}": [(X, "single_T", "true", 1), (X, "single_T", "shufcl", -1), ("V", "single_T", "true", -1), ("V", "single_T", "shufcl", 1)],
        # 記述：R の版と #shuf を対照にした版，残った D
        f"M1R_{X}": [("A", "single_T", "true", 1), (X, "single_T", "true", -1)],
        f"M2R_{X}": [("A", "single_T", "true", 1), (X, "single_T", "true", -1), ("A", "single_V", "true", -1), (X, "single_V", "true", 1)],
        f"L3R_{X}": [(X, "single_T", "true", 1), ("V", "single_T", "true", -1)],
        f"M1e_{X}": [("A", "single_T", "true", 1), ("A", "single_T", "shuf", -1), (X, "single_T", "true", -1), (X, "single_T", "shuf", 1)],
        f"L3e_{X}": [(X, "single_T", "true", 1), (X, "single_T", "shuf", -1), ("V", "single_T", "true", -1), ("V", "single_T", "shuf", 1)],
        f"D{X}T": [(X, "single_T", "true", 1), (X, "single_T", "shufcl", -1)],
        f"D{X}V": [(X, "single_V", "true", 1), (X, "single_V", "shufcl", -1)],
    }


QUANT_A = {"DAT": [("A", "single_T", "true", 1), ("A", "single_T", "shufcl", -1)],
           "DAV": [("A", "single_V", "true", 1), ("A", "single_V", "shufcl", -1)]}
PRIMARY = tuple(f"{q}_{X}" for X in OPS for q in ("M1", "M2", "M3", "M3d"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def script_shas() -> dict:
    return {**{n: sha(HERE / n) for n in SCRIPTS}, **{f"features/{n}": sha(ROOT / "features" / n) for n in FEAT_SCRIPTS}}


# ------------------------------------------------------------------ 特徴（データの取り決めどおり load_group を通す）

_CORP: dict = {}


def corpus_audio(T: str, kind: str) -> tuple[pd.DataFrame, torch.Tensor]:
    """声の特徴 kind で 3 モダリティがそろう発話の表と，その声の特徴（GPU）．`dai1.Corpus` と同じ `DD.load_corpus` を通す．
    低域の特徴の名前は `measure/data.py` の FEATURE_KEYS に無いので，ここで鍵 X を登録する（台本の中だけ．部品は変えない）．"""
    if (T, kind) not in _CORP:
        DD.D.FEATURE_KEYS.setdefault(kind, ("X",))
        enc = {"A": kind, "V": DD.ENC1["V"], "T": DD.ENC1["T"]}
        df, feats = DD.load_corpus(T, enc)
        _CORP[(T, kind)] = (df, S.t(feats["A"]))
    return _CORP[(T, kind)]


MIN_WORDS_W3 = 3
KFIX_DEN_MIN = 1.0      # C′ を固定した版の D(A ← V) がこれ未満なら κ_fixC と M3_fixC を出さない（記述だけの量．ML の査読 v2）


def words_of(T: str) -> pd.Series:
    """sample_id → 人の書き起こしの単語の数（操作の確かめと同じ単語の分け方 `check_lowpass_asr.norm_words`）．"""
    from check_lowpass_asr import norm_words
    m = DD.common.load_manifest(T)
    return pd.Series([len(norm_words(t)) for t in m["transcript"]], index=m["sample_id"].astype(str))


def std_pool(X: torch.Tensor, pool: np.ndarray) -> torch.Tensor:
    """`dai1.Unit` と同じ標準化：候補（K 本を含み，テストを含まない）の平均と標準偏差．"""
    mu, sd = X[pool].mean(0), X[pool].std(0).clamp(min=1e-6)
    return (X[pool] - mu) / sd


# ------------------------------------------------------------------ 単位

def run_unit(T: str, f: int, r: int, fzT: dict, out: Path, kinds: dict, verify: bool) -> pd.DataFrame | None:
    """kinds：{"L": 低域通過の特徴の名前, "B": 逆再生の特徴の名前}．"""
    base = out / "units" / f"{T}_f{f}_r{r}"
    if Path(f"{base}.csv").exists() and Path(f"{base}.npz").exists():
        old = pd.read_csv(f"{base}.csv")
        if (old.commit == RUN["commit"]).all():
            return None
        raise SystemExit(f"{base.name} が別のコミットで既にある（消してから流し直す）")
    t0 = time.time()
    a14 = dict(np.load(OUT14 / "units" / f"{T}_f{f}_r{r}.npz", allow_pickle=False))
    c14 = pd.read_csv(OUT14 / "units" / f"{T}_f{f}_r{r}.csv")
    sid, y, cl = a14["sid"].astype(str), a14["y"].astype(int), a14["cl"].astype(str)
    h = a14["half"].astype(int)
    if not (dai3.pool_halves(cl) == h).all():
        raise SystemExit(f"{base.name}：2 組の分け方を作り直すと実験 14 の保存と違う")
    halves = [np.flatnonzero(h == 0), np.flatnonzero(h == 1)]
    # 特徴：元の声と L・B を同じ流れで読む．行の集合と候補の並びとラベルが実験 14（実験 1）と同じことを確かめる
    dfA, XA = corpus_audio(T, DD.ENC1["A"])
    pool = np.flatnonzero(DD.pool_mask(dfA, f))
    if not (dfA["sample_id"].to_numpy()[pool] == sid).all() or not (dfA["label_4"].to_numpy()[pool] == y).all():
        raise SystemExit(f"{base.name}：元の声の候補の並びかラベルが実験 14 の保存と違う")
    Z, PCA = {}, {}
    Z["A"], PCA["A"] = D14.aligned_features(std_pool(XA, pool))
    for X, kind in kinds.items():
        dfX, XX = corpus_audio(T, kind)
        if not (dfX["sample_id"].to_numpy() == dfA["sample_id"].to_numpy()).all():
            raise SystemExit(f"{base.name}：{kind} で 3 モダリティがそろう発話が元の声と違う（ok が偽か NaN の発話がある）")
        Z[X], PCA[X] = D14.aligned_features(std_pool(XX, pool))
    main = "prop_" + fzT["main"][str(K)]
    names = list(JUDG) + [main]
    TG = {(nm, ver): a14[f"tgt__K{K}__{nm}__{ver}"].astype(int) for nm in names for ver in VERS_L}
    # 入れ替えた目標を実験 14 と同じ規則で作り直して一致を確かめる（設計 6 節）
    meta = D14.corpus_meta(T).reindex(sid)
    sex = meta["sex"].to_numpy().astype(str)
    st_e = y.astype(str)
    st_cl = np.char.add(np.char.add(np.char.add(np.char.add(st_e, "|"), cl), "|"), sex)
    for nm in names:
        for ver, strata in (("shuf", st_e), ("shufcl", st_cl)):
            tg, _ = D14.shuffle_within(TG[(nm, "true")], strata, [D14.RNG_SEED, DD.CORPUS_ID[T], f, r, K, D14.code(nm), D14.code(ver)])
            if not (tg == TG[(nm, ver)]).all():
                raise SystemExit(f"{base.name}：入れ替えた目標 {nm}/{ver} を作り直すと実験 14 の保存と違う")
    frac = {(row.target, row.version): float(row.frac_changed) for row in
            c14[(c14.K == K) & (c14.split == "main") & (c14.readout == "R1")].itertuples()}
    arr = dict(sid=sid, y=y.astype(np.int8), cl=cl, half=h.astype(np.int8), tgt__K5__label__true=y.astype(np.int8))
    rows = []

    def add_rows(s, res, split, curves=None, pca=None):
        for (key, ver), (full, ag, info) in res.items():
            nm = key[1]
            arr[f"K{K}__{split}__R1__{s}__{nm}__{ver}"] = full
            if split == "main" and info.get("prob") is not None:
                arr[f"prob__K{K}__{split}__R1__{s}__{nm}__{ver}"] = info["prob"]
            tg = y if nm == "label" else TG[(nm, ver)]
            cnt = np.bincount(tg, minlength=4)
            info = {k_: v_ for k_, v_ in info.items() if k_ != "prob"}
            rows.append(dict(T=T, fold=f, seed=r, K=K, split=split, readout="R1", student=s, target=nm, version=ver,
                             agree_halfmean=ag, agree_pooled=float(D14.uar_present(*[x.sum(0) for x in D14.counts(tg, full, cl)[1:]])),
                             frac_changed=frac.get((nm, ver), 0.0 if ver == "true" else np.nan), min_class=int(cnt.min()),
                             **(pca or {}), **info))
        for (isp, hh, key), cur in (curves or {}).items():
            arr.setdefault(f"inner__{split}__{s}__K{K}__{key[1]}", np.zeros((2, 2, len(D14.GRID_C))))[hh] = cur

    # L・B：J_A・J_V・J_T・J_prop の真と #shuf・#shufcl，本当の感情 y（真だけ）
    tg_true = {(K, nm): TG[(nm, "true")] for nm in names}
    tg_true[(K, "label")] = y
    tg_all = {((K, nm), ver): TG[(nm, ver)] for nm in names for ver in VERS_L}
    tg_all[((K, "label"), "true")] = y
    gk = [((K, nm), v) for nm in JUDG for v in ("true", "shufcl")]      # C′ を固定した感度のための格子の予測（実験 14 の R1fixC と同じ範囲）
    for X in kinds:
        res, curves, grid = D14.fit_r1(Z[X], cl, halves, tg_true, tg_all, gk)
        add_rows(X, res, "main", curves, PCA[X])
        for ((_, nm), ver), ic in grid:
            arr[f"K{K}__grid{ic}__R1__{X}__{nm}__{ver}"] = grid[(((K, nm), ver), ic)]
    # A：本当の感情 y だけ（J_m の予測は実験 14 の保存を使う）
    resA, curvesA, _ = D14.fit_r1(Z["A"], cl, halves, {(K, "label"): y}, {((K, "label"), "true"): y})
    add_rows("A", resA, "main", curvesA, PCA["A"])
    verify_agree = {}
    if verify:
        tv = {(K, nm): TG[(nm, "true")] for nm in ("single_T", "single_V")}
        resV, _, _ = D14.fit_r1(Z["A"], cl, halves, tv, {(k, "true"): v for k, v in tv.items()})
        for (key, ver), (full, _, _) in resV.items():
            saved = a14[f"K{K}__main__R1__A__{key[1]}__{ver}"]
            verify_agree[key[1]] = float((full == saved).mean())
        if min(verify_agree.values()) < VERIFY_AGREE:
            raise SystemExit(f"{base.name}：元の声で作り直した R1 の予測が実験 14 の保存と一致しない（{verify_agree}）")
    # r = 0：実験 14 と同じ別の分け方で，L・B の J_A・J_V・J_T の真と #shuf・#shufcl（分け方の揺れ）
    if r == 0:
        n_alt = int(a14["n_alt"])
        arr["n_alt"] = np.array(n_alt)
        hvs = []
        for ia in range(n_alt):
            alt = a14[f"althalf__{ia}"].astype(int)
            arr[f"althalf__{ia}"] = alt.astype(np.int8)
            hvs.append([np.flatnonzero(alt == 0), np.flatnonzero(alt == 1)])
        if hvs:
            for X in kinds:
                outs, curves_alt, _ = D14.fit_r1_multi(Z[X], cl, hvs, {(K, nm): TG[(nm, "true")] for nm in JUDG},
                                                       {((K, nm), v): TG[(nm, v)] for nm in JUDG for v in VERS_L})
                for ia, o in enumerate(outs):
                    add_rows(X, o, f"alt{ia}", {k: v for k, v in curves_alt.items() if k[0] == ia}, PCA[X])
    df = pd.DataFrame(rows)
    df["commit"], df["dirty"], df["sec"] = RUN["commit"], RUN["dirty"], time.time() - t0
    df["dai14_commit"] = ",".join(sorted(c14.commit.astype(str).unique()))
    df["kinds"] = json.dumps(kinds)
    for k_, v_ in verify_agree.items():
        df[f"verify_agree_{k_}"] = v_
    np.savez_compressed(f"{base}.npz", **arr)
    df.to_csv(f"{base}.csv", index=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


# ------------------------------------------------------------------ 集計

def load_merged(out: Path, T: str) -> tuple[dict, pd.DataFrame]:
    U14 = D14.load_units(OUT14, T)
    U23 = D14.load_units(out, T)
    if set(U23) - set(U14):
        raise SystemExit(f"{T}：実験 14 に無い単位がある")
    U = {}
    for k, a in U23.items():
        b = U14[k]
        for col in ("sid", "y", "cl", "half"):
            if not (a[col] == b[col]).all():
                raise SystemExit(f"{T}{k}：{col} が実験 14 の保存と違う")
        clash = [kk for kk in a if kk in b and kk not in ("sid", "y", "cl", "half", "n_alt") and not kk.startswith("althalf__")]
        if clash:
            raise SystemExit(f"{T}{k}：実験 14 の鍵と重なる鍵がある（{clash[:3]}）")
        U[k] = {**b, **a}
    csvs = pd.concat([pd.read_csv(out / "units" / f"{T}_f{f}_r{r}.csv") for f, r in sorted(U23)])
    return U, csvs


def kappa_from_dai14() -> dict:
    """κ_T = D(A ← T) ÷ D(A ← V)：実験 14 の predictions.csv の P3 ÷ P3b（K = 5，R1，#shufcl．部分集合ごと）．設計 3 節の値と小数 3 桁で照らす．"""
    p = pd.read_csv(OUT14 / "summary" / "predictions.csv")
    x = p[(p.K == K) & (p.readout == "R1") & (p.control == "cl")]
    out = {}
    for (T, sb), kd in KAPPA_DESIGN.items():
        y_ = x[(x["T"] == T) & (x.subset == sb)]
        num = float(y_[y_.prediction == "P3"].effect.iloc[0])
        den = float(y_[y_.prediction == "P3b"].effect.iloc[0])
        out[(T, sb)] = num / den
        if abs(round(out[(T, sb)], 3) - kd) > 1e-9:
            raise SystemExit(f"{T}/{sb}：κ を作り直すと {out[(T, sb)]:.4f} で，設計の {kd} と違う")
    return out


def corpus_verdict(main: pd.DataFrame, T: str, q: str, subsets: tuple = ("all",)) -> bool:
    """コーパス T で量 q が「上回る」か．MELD は共有の対話を除いた版でも「上回る」のときだけ．"""
    x = main[(main["T"] == T) & (main.quantity == q)]
    need = list(subsets) + (["nonshared"] if T == "MELD" else [])
    return all(((x.subset == sb) & (x.verdict == "上回る")).any() for sb in need)


def criteria(main: pd.DataFrame, targets: list, asm: pd.DataFrame | None = None, fcj: dict | None = None) -> dict:
    """予言の数え（設計 5 節）．E1_X：M1_X が 3 コーパスのうち 2 以上で上回る．E2_X：M1_X・M2_X・M3_X・M3d_X（κ の揺れを 1 次で入れた区間）が
    すべて上回るコーパスを名指し．言葉の中身に帰す書き方は，E2_L と E2_B の両方が成り立ち，仮定の表の条件（操作の確かめ，感情の情報）を
    満たすコーパスだけ（lexical_both_ops_ok）．顔の判断の再現が操作で増えた（dV_X が下回る）コーパスは face_increase に名指しする．"""
    out = {}
    for X in OPS:
        e1 = [T for T in targets if corpus_verdict(main, T, f"M1_{X}")]
        e2 = [T for T in targets if all(corpus_verdict(main, T, f"{q}_{X}") for q in ("M1", "M2", "M3", "M3d"))]
        l3 = [T for T in targets if corpus_verdict(main, T, f"L3_{X}")]
        fi = [T for T in targets if ((main["T"] == T) & (main.quantity == f"dV_{X}") & main.subset.isin(["all", "nonshared"])
                                      & (main.verdict == "下回る")).any()]       # MELD は共有の対話を除いた版も見る（ML の査読の m4）
        nm = {sb: {T: {q: main[(main["T"] == T) & (main.quantity == f"{q}_{X}") & (main.subset == sb)].verdict.tolist()
                       for q in ("M1", "M2", "M3", "M3d", "L3", "dV")} for T in targets} for sb in ("nomarker", "words3")}
        out[X] = dict(E1_count=len(e1), E1_corpora=e1, E1_hit=bool(len(e1) >= 2), E2_corpora=e2, L3_above=l3,
                      face_increase=fi, side_subset_verdicts=nm)
    out["lexical_both_ops"] = [T for T in targets if T in out["L"]["E2_corpora"] and T in out["B"]["E2_corpora"]]
    ok_manip = bool((fcj or {}).get("rule_met", False)) and bool((fcj or {}).get("rev_ok", False))
    bad_info = set()
    if asm is not None and len(asm):
        li = asm[asm.kind == "label_info"]
        bad_info = set(li[li.flag.astype(bool)]["T"])
    out["assumptions"] = dict(fc_rule_met=bool((fcj or {}).get("rule_met", False)), rev_ok=bool((fcj or {}).get("rev_ok", False)),
                              label_info_flagged=sorted(bad_info))
    out["lexical_both_ops_ok"] = [T for T in out["lexical_both_ops"] if ok_manip and T not in bad_info
                                  and T not in out["L"]["face_increase"] and T not in out["B"]["face_increase"]]
    out["below"] = main[main.primary & (main.verdict == "下回る")][["T", "subset", "quantity"]].to_dict("records")
    return out


def stage_summary(out: Path, targets: list) -> None:
    sdir = out / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    fz1, _ = dai3.load_fz1(OUT1)
    kap = kappa_from_dai14()
    mains, desc, asm, conv = [], [], [], []
    for T in targets:
        U, csvs = load_merged(out, T)
        need = {(f, r) for f in range(DD.N_FOLDS) for r in range(int(fz1[T]["n_seeds_eval"]))}
        if RUN["partial"]:
            need = set(U)
        if set(U) != need:
            raise SystemExit(f"{T}：単位の格子がそろっていない（{len(U)} / {len(need)}）")
        if not RUN["partial"] and (csvs.commit.nunique() != 1 or csvs.dirty.astype(str).ne("False").any()):
            raise SystemExit(f"{T}：単位のコミットが 1 つでないか，汚れた作業木の単位がある")
        conv.append(csvs)
        for (f, r), a in U.items():                                   # 単位ごとの行の数（L・B は 4 目標 × 3 版 + y で各 13 行，A は y の 1 行，r = 0 は別の分け方 × 2 × 9 行）
            n_alt = int(a["n_alt"]) if r == 0 else 0
            exp_rows = len(OPS) * 13 + 1 + n_alt * len(OPS) * 9
            got_rows = int(((csvs.fold == f) & (csvs.seed == r)).sum())
            if got_rows != exp_rows:
                raise SystemExit(f"{T}_f{f}_r{r}：行の数 {got_rows}（見込み {exp_rows}）")
        c14 = pd.concat([pd.read_csv(OUT14 / "units" / f"{T}_f{f}_r{r}.csv") for f, r in sorted(U)])
        ic_fix = D14.fixed_c_index(c14)                              # 実験 14 の R1fixC と同じ格子の点（A と X に同じ点を当てる）
        main_nm = "prop_" + fz1[T]["main"][str(K)]
        g_eff, _ = ST.g_eff_of(T)
        rng = np.random.default_rng([RNG_SEED, DD.CORPUS_ID[T]])
        subsets = (["all", "nonshared"] if T == "MELD" else ["all"]) + ["nomarker", "words3"]
        # 記述の部分集合 words3：人の書き起こしが 3 語以上の発話（操作の確かめの標本と同じ範囲．感情認識の査読の A2）．
        # dai14 の mask_of は nomarker で ~marker を使うので，marker を「3 語未満」に置き換えた単位の写しを nomarker として渡す
        nw_of = words_of(T)
        U_w3 = {k: {**a, "marker": nw_of.reindex(a["sid"]).to_numpy() < MIN_WORDS_W3} for k, a in U.items()}
        for subset in subsets:
            Uu, msub = (U_w3, "nomarker") if subset == "words3" else (U, subset)
            if (T, subset) in kap:
                kp = kap[(T, subset)]
            else:                                                     # words3 の κ は A の項（実験 14 と同じ予測）だけから作る
                ea = D14.combo_stats([D14.Rcell(Uu, K, "R1", "A", "single_T", "true", msub), D14.Rcell(Uu, K, "R1", "A", "single_T", "shufcl", msub)], [1, -1], g_eff, rng)
                eb = D14.combo_stats([D14.Rcell(Uu, K, "R1", "A", "single_V", "true", msub), D14.Rcell(Uu, K, "R1", "A", "single_V", "shufcl", msub)], [1, -1], g_eff, rng)
                kp = ea["effect"] / eb["effect"]
            Q = dict(QUANT_A)
            for X in OPS:
                Q.update(quant(X, kp))
            got = {}

            def run_q(q, terms, primary):
                tt = [("R1", s_, nm, ver, co) for s_, nm, ver, co in terms]
                cells = [D14.Rcell(Uu, K, "R1", s_, nm, ver, msub) for _, s_, nm, ver, _ in tt]
                spv = D14.split_values(Uu, K, tt, msub)
                e = D14.combo_stats(cells, [t[-1] for t in tt], g_eff, rng, split_vals=spv)
                got[q] = e["effect"]
                mains.append(dict(T=T, K=K, subset=subset, quantity=q, kappa=kp, primary=primary) | e |
                             dict(verdict=D14.verdict(e["lo"], e["hi"]), verdict_ns=D14.verdict(e["lo_ns"], e["hi_ns"])))

            side = subset in ("nomarker", "words3")                   # 記述の部分集合
            for q, terms in Q.items():
                if side and not q.startswith(("M1_", "M2_", "M3_", "L3_", "D")):
                    continue
                run_q(q, terms, bool(q in PRIMARY and not side))
            # M3 の κ の推定の揺れを 1 次で入れた版（設計 4 節）：M3 = D_AT − D_XT − κ̂ (D_AV − D_XV)，κ̂ = D_AT / D_AV の 1 次の係数
            if not RUN["partial"] and subset != "words3" and abs(got["DAT"] / got["DAV"] - kp) > 1e-6:   # 単位を絞った試しでは比が変わるので確かめない                 # 固定の κ は同じ A の予測から作った値なので，ここで作り直した比と一致するはず
                raise SystemExit(f"{T}/{subset}：D(A ← T) ÷ D(A ← V) = {got['DAT'] / got['DAV']:.6f} が固定の κ {kp:.6f} と違う")
            for X in OPS:
                dat, dav, dxv = got["DAT"], got["DAV"], got[f"D{X}V"]
                kh = dat / dav
                co = {"AT": dxv / dav, "XT": -1.0, "AV": -kh * dxv / dav, "XV": kh}
                terms = ([("A", "single_T", "true", co["AT"]), ("A", "single_T", "shufcl", -co["AT"])] +
                         [(X, "single_T", "true", co["XT"]), (X, "single_T", "shufcl", -co["XT"])] +
                         [("A", "single_V", "true", co["AV"]), ("A", "single_V", "shufcl", -co["AV"])] +
                         [(X, "single_V", "true", co["XV"]), (X, "single_V", "shufcl", -co["XV"])])
                run_q(f"M3d_{X}", terms, bool(not side))
                run_q(f"dV_{X}", [("A", "single_V", "true", 1), ("A", "single_V", "shufcl", -1),
                                  (X, "single_V", "true", -1), (X, "single_V", "shufcl", 1)], False)
            # 記述：2 つの操作の M3 の差（M3_L − M3_B = κ [D(L ← V) − D(B ← V)] − [D(L ← T) − D(B ← T)]）
            run_q("M3_L-B", [("L", "single_T", "true", -1), ("L", "single_T", "shufcl", 1), ("B", "single_T", "true", 1), ("B", "single_T", "shufcl", -1),
                             ("L", "single_V", "true", kp), ("L", "single_V", "shufcl", -kp), ("B", "single_V", "true", -kp), ("B", "single_V", "shufcl", kp)], False)
            # 記述：C′ を固定した版（実験 14 の R1fixC と同じ点を A と X に当てる．分け方の揺れは足さない）
            if not side:
                # κ_fixC：同じ固定の点で作った A の D の比（R1 の κ を当てると帰無の下でも M3 が 0 にならないため．台本の検査 code_v2 の 1）
                fa = D14.combo_stats([D14.fixc_cell(U, ic_fix, "A", "single_T", "true", subset), D14.fixc_cell(U, ic_fix, "A", "single_T", "shufcl", subset)], [1, -1], g_eff, rng)
                fb = D14.combo_stats([D14.fixc_cell(U, ic_fix, "A", "single_V", "true", subset), D14.fixc_cell(U, ic_fix, "A", "single_V", "shufcl", subset)], [1, -1], g_eff, rng)
                kfix = fa["effect"] / fb["effect"] if abs(fb["effect"]) >= KFIX_DEN_MIN else np.nan   # 分母が 0 に近いと比が定まらないので M3 を出さない
                for X in OPS:
                    for q in ("M1", "M2", "M3"):
                        if q == "M3" and not np.isfinite(kfix):
                            continue
                        terms = quant(X, kfix)[f"{q}_{X}"]
                        cells = [D14.fixc_cell(U, ic_fix, s_, nm, ver, subset) for s_, nm, ver, _ in terms]
                        e = D14.combo_stats(cells, [t[-1] for t in terms], g_eff, rng)
                        mains.append(dict(T=T, K=K, subset=subset, quantity=f"{q}_{X}_fixC", kappa=kfix, primary=False,
                                          fixC0=D14.GRID_C[ic_fix[0]], fixC1=D14.GRID_C[ic_fix[1]]) | e |
                                     dict(verdict=D14.verdict(e["lo"], e["hi"]), verdict_ns=D14.verdict(e["lo_ns"], e["hi_ns"])))
            if side:
                continue
            # 記述：L・B の R と D（全部の目標），R(X ← y)，R(A ← y)
            for s_, nm in [(X, m) for X in OPS for m in list(JUDG) + [main_nm, "label"]] + [("A", "label")]:
                cR = D14.Rcell(U, K, "R1", s_, nm, "true", subset)
                q = [("R", [cR], [1])]
                if nm != "label":
                    for ver in ("shuf", "shufcl"):
                        cS = D14.Rcell(U, K, "R1", s_, nm, ver, subset)
                        q += [(f"R_{ver}", [cS], [1]), (f"D_{ver}", [cR, cS], [1, -1])]
                for qn, cells_, co in q:
                    e = D14.combo_stats(cells_, co, g_eff, rng)
                    desc.append(dict(T=T, K=K, student=s_, target=nm, subset=subset, quantity=qn) | e |
                                dict(verdict=D14.verdict(e["lo"], e["hi"])))
        # 仮定の表（設計 6 節）
        d_ = pd.DataFrame([x for x in desc if x["T"] == T and x["subset"] == "all"])
        m_ = pd.DataFrame([x for x in mains if x["T"] == T and x["subset"] == "all"])
        ra = float(d_[(d_.student == "A") & (d_.target == "label") & (d_.quantity == "R")].effect.iloc[0])
        dat = float(m_[m_.quantity == "DAT"].effect.iloc[0])
        dav = float(m_[m_.quantity == "DAV"].effect.iloc[0])
        for X in OPS:
            rx = float(d_[(d_.student == X) & (d_.target == "label") & (d_.quantity == "R")].effect.iloc[0])
            ratio = (rx - 25.0) / (ra - 25.0) if ra > 25.0 else np.nan
            asm.append(dict(kind="label_info", T=T, student=X, R_X_label=rx, R_A_label=ra, ratio=ratio,
                            flag=bool(not np.isfinite(ratio) or ratio < RATIO_MIN)))
            dxt = float(m_[m_.quantity == f"D{X}T"].effect.iloc[0])
            dxv = float(m_[m_.quantity == f"D{X}V"].effect.iloc[0])
            asm.append(dict(kind="retained_share", T=T, student=X, DAT=dat, DAV=dav, DXT=dxt, DXV=dxv,
                            retained_T=dxt / dat if dat else np.nan, retained_V=dxv / dav if dav else np.nan, flag=False))
            # C′ の端（実験 14 の 8 節と dai14.py の数えと同じ：単位 × 組ごとに選んだ点を数え，目標ごとに flag = max(上端, 平らでない下端) > 0.25）
            plateau = []
            for (f, r), a in U.items():
                for kk in [k for k in a if k.startswith(f"inner__main__{X}__")]:
                    cur = a[kk].mean(1)
                    for hh in (0, 1):
                        j_ = int(np.argmax(cur[hh] > cur[hh].max() - 1e-12))
                        plateau.append(dict(target=kk.split("__")[-1], chosen=D14.GRID_C[j_],
                                            low_plateau=bool(j_ == 0 and abs(cur[hh][0] - cur[hh][1]) < 1e-9)))
            for tg, x in pd.DataFrame(plateau).groupby("target"):
                lo = float((x.chosen == D14.GRID_C[0]).mean()); hi = float((x.chosen == D14.GRID_C[-1]).mean())
                lo_np = float(((x.chosen == D14.GRID_C[0]) & ~x.low_plateau).mean())
                asm.append(dict(kind="C_edge", T=T, student=X, target=tg, share_low=lo, share_high=hi, share_low_nonplateau=lo_np,
                                share_low_plateau=lo - lo_np, flag=bool(max(hi, lo_np) > D14.EDGE_SHARE)))
        fr = csvs[(csvs.split == "main") & (csvs.student == OPS[0]) & (csvs.version == "shufcl")]
        for tg, g in fr.groupby("target"):
            asm.append(dict(kind="frac_changed", T=T, target=tg, frac_mean=float(g.frac_changed.mean()), flag=bool(g.frac_changed.mean() < D14.FRAC_LOW)))
        for col in ("pca_kept", "eff_rank"):
            for s_ in OPS + ("A",):
                v = csvs[(csvs.student == s_) & (csvs.split == "main")][col]
                asm.append(dict(kind=col, T=T, student=s_, mean=float(v.mean()), min=float(v.min()), max=float(v.max()), flag=False))
    M = pd.DataFrame(mains)
    tag = dict(commit=RUN["commit"], dirty=RUN["dirty"])
    M.assign(**tag).to_csv(sdir / "main.csv", index=False)
    pd.DataFrame(desc).assign(**tag).to_csv(sdir / "descriptive.csv", index=False)
    pd.DataFrame(asm).assign(**tag).to_csv(sdir / "assumptions.csv", index=False)
    C = pd.concat(conv)
    json.dump(dict(grad_max=float(C.grad.max()), inner_grad_max=float(C.inner_grad.max()), stalled=int(C.stalled.sum()),
                   inner_stalled=int(C.inner_stalled.max())), open(sdir / "convergence.json", "w"), indent=1)
    A_ = pd.DataFrame(asm)
    fcj = json.load(open(FC_JSON)) if FC_JSON.exists() else {}
    json.dump(criteria(M, targets, A_, fcj) | dict(kappa={f"{k[0]}/{k[1]}": v for k, v in kap.items()}),
              open(sdir / "criteria.json", "w"), ensure_ascii=False, indent=1)
    print(f"summary：主の量 {len(M)} 行，記述 {len(desc)} 行，仮定 {len(asm)} 行", flush=True)


# ------------------------------------------------------------------ 自己検査

def selftest() -> None:
    rng = np.random.default_rng(0)
    U = {}
    for f in range(5):
        for r in range(2):
            n = 200
            cl = np.array([f"c{i % 8}" for i in range(n)])
            y = rng.integers(0, 4, n)
            a = dict(y=y.astype(np.int8), cl=cl, nonshared=np.ones(n, bool), marker=np.zeros(n, bool))
            for nm in ("single_T", "single_V"):
                a[f"tgt__K5__{nm}__true"] = y.astype(np.int8)
                a[f"tgt__K5__{nm}__shufcl"] = rng.permutation(y).astype(np.int8)
                for s_ in ("A", "L"):
                    p = np.where(rng.random(n) < 0.7, y, rng.integers(0, 4, n)).astype(np.int8)
                    a[f"K5__main__R1__{s_}__{nm}__true"] = p
                    a[f"K5__main__R1__{s_}__{nm}__shufcl"] = rng.integers(0, 4, n).astype(np.int8)
            U[(f, r)] = a
    # (1) 同じ予測どうしの差は 0（L に A の予測をそのまま入れた場合）
    U0 = {k: {**a, **{kk.replace("__A__", "__L__"): v for kk, v in a.items() if "__A__" in kk}} for k, a in U.items()}
    for q in ("M1_L", "M2_L", "M3_L"):
        terms = quant("L", 3.0)[q]
        tt = [("R1", s_, nm, ver, co) for s_, nm, ver, co in terms]
        e = D14.combo_stats([D14.Rcell(U0, K, "R1", s_, nm, ver) for _, s_, nm, ver, _ in tt], [t[-1] for t in tt], 5.0, np.random.default_rng(1))
        assert abs(e["effect"]) < 1e-9 and e["se"] < 1e-6, (q, e)
    # (2) 1 項の R は 60〜80（7 割の一致）
    e2 = D14.combo_stats([D14.Rcell(U, K, "R1", "A", "single_T", "true")], [1], 5.0, np.random.default_rng(1))
    assert 60 < e2["effect"] < 80, e2
    # (3) 同じ割合の帰無：D(L ← m) = c · D(A ← m) なら M3 = 0，M2 = (1 − c)(D_AT − D_AV)
    dAT, dAV, c = 19.3, 6.0, 0.4
    kap = dAT / dAV
    m1 = dAT - c * dAT
    m2 = m1 - (dAV - c * dAV)
    m3 = m1 - kap * (dAV - c * dAV)
    assert abs(m3) < 1e-12 and abs(m2 - (1 - c) * (dAT - dAV)) < 1e-12
    # (4) 係数の符号：M1・M2 の係数の和は 0，M3 の係数の和は (1 − κ)·0 = 0
    for X in OPS:
        for q, terms in quant(X, 2.5).items():
            if q.startswith(("M1_", "M2_", "M3_", "L3_")):
                assert abs(sum(t[-1] for t in terms)) < 1e-12, q
    # (5) 数えの規則
    M = pd.DataFrame([dict(T=T, subset=sb, quantity=q, verdict=v, primary=True) for T, sb, q, v in [
        ("IEMOCAP", "all", "M1_L", "上回る"), ("MOSEI", "all", "M1_L", "上回る"), ("MELD", "all", "M1_L", "上回る"), ("MELD", "nonshared", "M1_L", "判定できない"),
        ("MOSEI", "all", "M2_L", "上回る"), ("MOSEI", "all", "M3_L", "上回る"), ("MOSEI", "all", "M3d_L", "上回る"),
        ("MOSEI", "all", "M1_B", "上回る"), ("MOSEI", "all", "M2_B", "上回る"), ("MOSEI", "all", "M3_B", "判定できない")]])
    cr = criteria(M, list(TARGETS))
    assert cr["L"]["E1_corpora"] == ["IEMOCAP", "MOSEI"] and cr["L"]["E1_hit"] and cr["L"]["E2_corpora"] == ["MOSEI"]
    assert cr["B"]["E2_corpora"] == [] and cr["lexical_both_ops"] == []
    # (6) 部分集合ごとの κ を実験 14 の保存から作り直せ，設計の値と一致する．部分集合の κ を使えば帰無で M3 = 0
    kp = kappa_from_dai14()
    assert set(kp) == set(KAPPA_DESIGN)
    p14 = pd.read_csv(OUT14 / "summary" / "predictions.csv")
    for (T, sb), kv in kp.items():
        y_ = p14[(p14.K == K) & (p14.readout == "R1") & (p14.control == "cl") & (p14["T"] == T) & (p14.subset == sb)]
        dAT_, dAV_ = float(y_[y_.prediction == "P3"].effect.iloc[0]), float(y_[y_.prediction == "P3b"].effect.iloc[0])
        for c in (0.2, 0.7):
            assert abs((dAT_ - c * dAT_) - kv * (dAV_ - c * dAV_)) < 1e-9, (T, sb)
    print("selftest：6 項目とも通った")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("selftest", "units", "summary"))
    ap.add_argument("--out-dir", default=str(OUT_PROD))
    ap.add_argument("--targets", nargs="*", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--l-kind", default=None, help="L の特徴の名前（既定は results/dai23/fc.json の fc から audio_wavlm-lp<fc>）")
    ap.add_argument("--b-kind", default=None, help="B の特徴の名前（既定は audio_wavlm-rev）")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
    if args.stage == "selftest":
        selftest()
        return
    out = Path(args.out_dir).resolve()
    prod = out == OUT_PROD.resolve()
    commit, dirty = dai1.git_commit()
    RUN["commit"], RUN["dirty"], RUN["partial"] = commit, dirty, bool(args.allow_partial)
    if dirty and (prod or not args.allow_dirty):
        sys.exit("exp・measure・features に未コミットの変更がある（試しは一時の置き場で --allow-dirty）")
    if prod and not dai1.committed_clean(DESIGN):
        sys.exit("設計の文書がコミットされていない")
    if prod and (args.folds is not None or args.seeds is not None or args.allow_partial or args.l_kind is not None or args.b_kind is not None):
        sys.exit("本番の置き場では分割・種・L の特徴を変えない")
    if not FC_JSON.exists() and (prod or args.l_kind is None):
        sys.exit(f"{FC_JSON} が無い（先に features/check_lowpass_asr.py で fc を決める．退化の試しは --l-kind を付ける）")
    fcj = json.load(open(FC_JSON)) if FC_JSON.exists() else {}
    kinds = {"L": args.l_kind or f"audio_wavlm-lp{int(fcj['fc'])}", "B": args.b_kind or "audio_wavlm-rev"}
    if any(k is not None and k != DD.ENC1["A"] for k in (args.l_kind, args.b_kind)):
        sys.exit("--l-kind と --b-kind に渡せるのは元の声の特徴（退化の試し）だけ（設計 8 節）")
    if not prod and (args.l_kind is None or args.b_kind is None):
        sys.exit("本番でない置き場では --l-kind と --b-kind の両方に元の声の特徴を渡す（言葉を消した特徴の量は本番の置き場でだけ作る．設計 8 節）")
    dai1.ENC["enc"] = DD.ENC1
    fz1, fz1_sha = dai3.load_fz1(OUT1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "units").mkdir(exist_ok=True)
    cache = {f"{X}_{T}": ROOT / "data" / "cache" / f"{kd}_{T}.npz" for X, kd in kinds.items() for T in args.targets}
    cache.update({f"{m}_{T}": ROOT / "data" / "cache" / f"{kd}_{T}.npz" for m, kd in DD.ENC1.items() for T in args.targets})
    prov = dict(stage=args.stage, argv=sys.argv, commit=commit, dirty=dirty, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                host=platform.node(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""),
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                scipy=__import__("scipy").__version__, transformers=__import__("transformers").__version__,
                ffmpeg=subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
                scripts=script_shas(), design_sha256=sha(DESIGN), fc_json=fcj, fc_json_sha256=(sha(FC_JSON) if FC_JSON.exists() else ""), kinds=kinds, kappa_design={f"{k[0]}/{k[1]}": v for k, v in KAPPA_DESIGN.items()}, wavlm_model="microsoft/wavlm-large",
                cache_sha256={k: (sha(p) if p.exists() else "") for k, p in cache.items()},
                dai1_frozen_sha256=fz1_sha, K=K, grid_c=D14.GRID_C, cb_r1=D14.CB_R1, dim_r1=D14.DIM_R1, rng_seed=RNG_SEED,
                dai14_rng_seed=D14.RNG_SEED, n_boot=D14.N_BOOT, mem_frac=MEM_FRAC, verify_agree=VERIFY_AGREE, ratio_min=RATIO_MIN)
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if args.stage == "units":
        for T in args.targets:
            fzT = fz1[T]
            seeds = list(range(int(fzT["n_seeds_eval"]))) if args.seeds is None else args.seeds
            folds = list(range(DD.N_FOLDS)) if args.folds is None else args.folds
            for f in folds:
                for r in seeds:
                    df = run_unit(T, f, r, fzT, out, kinds, args.verify or r == 0)   # r = 0 の単位は本番でも照合する（設計 6 節）
                    if df is None:
                        print(f"{T}_f{f}_r{r}：済み（飛ばす）", flush=True)
                        continue
                    va = {c: float(df[c].iloc[0]) for c in df.columns if c.startswith("verify_agree_")}
                    print(f"{T}_f{f}_r{r}：{df.sec.iloc[0]:.0f} 秒，{len(df)} 行，勾配の最大 {df.grad.max():.1e}" + (f"，照合 {va}" if va else ""), flush=True)
            _CORP.clear()
            dai1.Corpus.cache.pop(T, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if prod and set(args.targets) == set(TARGETS):
            (out / "units.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    else:
        stage_summary(out, args.targets)
        json.dump(prov, open(out / "provenance.json", "w"), ensure_ascii=False, indent=1)
        if prod and set(args.targets) == set(TARGETS):
            (out / "summary.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
