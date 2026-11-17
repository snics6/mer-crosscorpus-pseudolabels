"""DAI 実験 2：残りの比較手法のうち線形の手法（設計は `docs/設計/実験2_設計.md` 第 3.2 版の 1 節・2.1 節・4 節）。

段（--stage）：
  pseudo：評価のターゲット T ごとに，ソース 3 つの 1 つずつを仮のターゲット c にし（残り 2 つがソース），2.1 節の手法を格子の全部の点で解く．
          単位は（T, c, 分割 f, K 本の抜き方の種 r）で，r ∈ {0, 1}（実験 1 と同じ 120 単位）．
  select：`dai2_select.py` を呼ぶ（`dai2_grad.py` の出力と合わせて選ぶ）．
  eval  ：評価のターゲット T をソース 3 つで回す．r ∈ {0, …, R−1}（R は実験 1 の固定のファイルの n_seeds_eval）．
          `frozen/dai2_frozen.json` がコミット済みで `<out>/select/frozen.json` と sha256 が一致し，`dai1.check_eval_gate` が通り
          （設計 4 節の m85），実験 2 の台本 3 本が選ぶ段のときの sha256 と一致し（m72），実験 1 の固定のファイルがコミット済みで
          選ぶ段で使ったものと同じときだけ動く（`check_eval_frozen`）．

共通（設計 1 節）：単位・標準化（候補の統計）・出発点（ソースのコーパスごとの識別器の平均）は `dai1.Unit` をそのまま使う．
実験 2 は第 1 のエンコーダ（WavLM・ResNet-50・RoBERTa）だけで回す．`--dai1-dir` の provenance_log.jsonl（と source/ の実体の隣のもの）に
記録されたエンコーダが 1 つも無いか，第 1 のエンコーダと違えば止まる（`check_encoder`．gate-code の R7．次元が同じなので確かめないと落ちない）．
`--students` は試走だけ（本番の置き場 `results/dai2` では受け付けない．絞った単位は extra_tag で区別する．I10）．
引き戻し (C′, C_b) は，仮のターゲットの回は `<dai1-dir>/pseudo/<T>_tether_w.json`，評価の回は `--dai1-frozen`（既定
`frozen/dai1_frozen.json`）の値．K=0 の手法は K=1 の値（実験 1 と同じ）．ρ は候補 1 本あたりの重み（重みの合計 ÷ 候補の本数 n）．

手法（行の method）．目的関数は実験 1 の設計 4 節の形 [損失の和 ＋ ‖W − W0‖²/(2C′) ＋ ‖b − b0‖²/(2C_b)] / S（S は重みの和）：
  none   ：比較の基準 none[K]．実験 1 の段 1 と同じ問を解き直す（他の手法の出発点．選ばない）．dai1 の段 1 の保存があり，テストと K 本の
           sample_id が同じときだけ，判定の一致を行に残す（none_agree_dai1．dai1_same_units が偽なら NaN）
  saerens：Saerens ほか 2002 の EM．none[K] の候補の確率で感情の割合 π を推定し（反復 200 回まで，差 1e-9 で止める．E77 の値），テストの確率に π を掛ける
  self   ：自己学習．none[K] から始め，3 巡のあいだ，生徒自身の今の確率の最大が τ 以上の候補（K 本も含む）に硬いラベル（1 本あたり重み ρ．
           感情の釣り合いは取らない）を置き，K 本（倍率 1）と合わせて出発点から学び直す．残る候補が 0 本の巡は K 本だけで学ぶ．S = 4K ＋ ρ × 残った本数．
           巡ごとの残った本数（感情ごと）を `_self_round` の行に残す．npz には，巡 t ごとに置いた硬いラベル `self_<生徒>_K<K>_r<t>_pl`
           （候補の番号ごとの感情．−1 は無し．int8）と，1・2 巡の解のテストの判定 `self_<生徒>_K<K>_r<t>_pred` を残す（3 巡の解は
           `self_<生徒>_K<K>_pred`，`_pl` は 3 巡のラベルの写し．gate-code の R9）．
           仮のターゲットの回だけ，残った候補のラベルの正しさの割合を測る（測るだけ．m20）
  entmin ：エントロピー最小化（Grandvalet & Bengio 2005）．K 本の交差エントロピー ＋ ρn × 候補の予測のエントロピーの平均 ＋ 引き戻し．S = 4K ＋ ρn．
           凸でないので none[K] の解から始め，`dai_solver.lbfgs_batch`（実験 1 と同じ L-BFGS．問ごとの履歴 10 と Armijo の後退，float64，
           勾配の最大 1e-6 まで，最大 20,000 反復）で解く（`fit_nc`）．none[K] から解く局所解である（設計 1 節）
  im     ：情報最大化（SHOT の IM，TIM）．K 本の交差エントロピー ＋ ρn × (α × 条件のエントロピーの平均 − 候補の予測の平均の感情ごとのエントロピー)
           ＋ 引き戻し．S = 4K ＋ ρn．none[K] から `fit_nc` で解く
  t3a    ：T3A（Iwasawa & Matsuo 2021．E77 の `t3a_probs` と同じ）．候補の標準化した特徴を L2 で正規化し，支えの初期値は none[K] の W の行を正規化
           したもの．候補を none[K]（切片を含む）の判定の感情に入れ，エントロピーの小さい M 本（感情ごと）だけを残す．K 本は真のラベルで支えに
           足し，落とさない．支えの和を正規化した原型と，正規化しないテストの特徴の内積が点数．主の版（t3a）は切片を捨て，t3a_b は none[K] の
           切片を足す．M = −1 は全部
  lp     ：ラベル伝播（Zhou ほか 2004，Iscen ほか 2019．K ≥ 1）．候補の標準化した特徴の余弦の類似度の k 近傍（γ = 3 乗，負は 0，対称にする）
           のグラフで，K 本のラベルを (I − α_LP·D^−½ W D^−½) F = Y の共役勾配法（50 回まで．E77 の値）で伝播する．F を 0 以上に切り，K 本の
           行と F が全部 0 の行を除き，形が硬い（form = 0）なら最大の感情の one-hot，そのまま（form = 1）なら行で正規化した F を疑似ラベルにして，
           提案の同時に解く版の形（感情ごとの総量をそろえ，重みの合計 n，K 本の倍率 1）で出発点から学ぶ（`dai1.make_G` の joint と同じ式）．
           近傍は float64 の類似度を安定な並べ替えで選ぶ（同点は番号の小さい候補が先．E77 の計算機による違いの対策）
  own    ：提案の同時に解く版の予測を生徒自身の none[K] の候補の確率に替えた版（`dai1.make_G` の joint．ρ × a の 12 点）
  st     ：S+T．ソース（仮のターゲットの回は 2 つ，評価の回は 3 つ）の全ラベル付き発話（各ソースの全発話の平均と標準偏差で標準化）と K 本
           （候補の統計で標準化）を合わせ，W0 = 0・切片に罰則なしで学ぶ．ソースは合わせた全体で感情の釣り合いを取り（1 本あたり平均 1），
           K 本は釣り合いの重み × β．C は合わせたソースの段 source と同じ 5 分割の交差検証（各ソースの学習側の統計で標準化，合わせた評価側の
           UAR の平均が最大，同点は小さい C．K 本を使わない）で 12 点の格子から選び，`<out>/st_cv/` に残して使い回す．K = 0 は β の項が無いので
           1 問を解き，β の 3 行に同じ値を書く（beta_dummy = True）

保存（設計 1 節．実験 1 の 11 節と同じ形）：単位ごとに `<out>/<pseudo|eval>/units/<T>_<c>_f<f>_r<r>_lin.npz` と `.csv`（評価の回の c は "ALL"）．
  npz：テストの判定（int8），W・b（float32．候補の統計で標準化した座標．設計 1 節「目的関数の形」），問の名前（JSON），候補・テスト・K 本の
  sample_id，生徒の標準化の平均と標準偏差（生の特徴の上の係数は W / sd，b − (W / sd)·mean で作れる），引き戻し，コミット，dirty，
  固定のファイルと eval_gate.json の sha256，引き戻しの鍵．
  csv：全部の問の UAR（テスト），S，勾配の最大，行き詰まり，有限，反復の回数，引き戻し，コミット，dirty，計算機，所要．
  実行ごとに `<out>/provenance_log.jsonl`．解のどれかに NaN・Inf があれば保存せずに止まる．保存済みの単位は，コミット・引き戻し・固定のファイルが
  今の実行と一致し，有限のときだけ読む（再開）．

E77（`exp/compare_dai.py`）から変えた点：出発点と引き戻しを E39 から実験 1 の値に，総重み m_eff ∈ {100, 500, 2000, n} を ρ ∈ {1/16, 1/4, 1, 4}
（候補 1 本あたり）に，自己学習を総重みの等分から 1 本あたりの重み ρ に，ラベル伝播を総重み n・感情ごとの総量をそろえる形に替えた．
entmin・IM の揺らした出発点からの解き直し（E77 の種 1・2）は置かない（設計 2.1 節の表に無い）．massbal・kd・dist は実験 1 が受け持つ．
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

ROOT = DD.ROOT
KS_ALL = dai1.KS_ALL
RHO_GRID = dai1.RHO_GRID                                             # 候補 1 本あたりの重み（実験 1 と同じ）
A_GRID = dai1.A_GRID
SELF_TAU = (0.5, 0.7, 0.9)                                           # 自己学習の確信度のしきい値（E77 と同じ）
SELF_ROUNDS = 3                                                      # 自己学習の巡（E77 の ROUNDS_SELF）
IM_ALPHA = (0.1, 0.3, 1.0)                                           # IM の条件のエントロピーの重み（E77 と同じ）
T3A_M = (1, 5, 20, 50, 100, -1)                                      # T3A の感情ごとの支えの本数．−1 は全部（E77 と同じ）
LP_K = (10, 20, 50)                                                  # ラベル伝播の近傍の数（E77 と同じ）
LP_ALPHA = (0.9, 0.99)                                               # 伝播の係数（E77 と同じ）
LP_FORM = (0, 1)                                                     # 0：硬い，1：そのまま（E77 の hard・soft）
LP_GAMMA, LP_CG_ITERS = 3, 50                                        # 類似度の乗数（Iscen ほか）と共役勾配法の回数（E77 と同じ）
ST_BETA = (1.0, 10.0, 100.0)                                         # S+T の K 本の 1 本あたりの重みの倍率
ST_C = dai1.C_GRID                                                   # S+T の C の格子（段 source と同じ 12 点）
SAERENS_ITERS = 200
GRIDS = {
    "self": [("tau", SELF_TAU), ("rho", RHO_GRID)],
    "entmin": [("rho", RHO_GRID)],
    "im": [("alpha", IM_ALPHA), ("rho", RHO_GRID)],
    "t3a": [("M", T3A_M)],
    "t3a_b": [("M", T3A_M)],
    "lp": [("k", LP_K), ("alpha_lp", LP_ALPHA), ("form", LP_FORM)],
    "own": [("rho", RHO_GRID), ("a", A_GRID)],
    "st": [("beta", ST_BETA)],
    "saerens": [],
}
DEFAULTS = {                                                         # 選択なしの値（設計 2.1 節の表）
    "self": {"tau": 0.9, "rho": 1.0}, "entmin": {"rho": 1.0}, "im": {"alpha": 0.1, "rho": 1.0},
    "t3a": {"M": -1}, "t3a_b": {"M": -1}, "lp": {"k": 50, "alpha_lp": 0.99, "form": 0},
    "own": {"rho": 1.0, "a": 1}, "st": {"beta": 1.0}, "saerens": {},
}
SCRIPTS = ("dai2.py", "dai2_grad.py", "dai2_select.py", "dai1.py", "dai_solver.py", "dai_data.py")   # provenance と固定のファイルに sha256 を残す台本
ENCODER = DD.ENC1                                                    # 実験 2 は第 1 のエンコーダだけで回す（R7）
PROD_DIR = ROOT / "results" / "dai2"                                 # 本番の置き場
OWN_SCRIPTS = ("dai2.py", "dai2_grad.py", "dai2_select.py")          # 評価の段が，選ぶ段のときの sha256 と照らす台本（共通の台本は eval_gate が固める）
RUN = {"commit": "", "dirty": False, "frozen_sha256": "", "gate_sha256": "", "gpu": "", "tether_key": "", "dai1_dir": "", "out": ""}   # main が埋める
f32, pred8, probs = dai1.f32, dai1.pred8, dai1.probs


def sha(path: Path) -> str:
    return dai1.sha(Path(path))


def tether_key(tether: dict) -> str:
    """引き戻しの表の中身の sha256（先頭 16 文字）．選ぶ段が，全部の単位が実験 1 の固定のファイルと同じ引き戻しで解かれたかを照らす。"""
    d = {f"{k[0]}|{k[1]}": [float(v[0]), float(v[1])] for k, v in sorted(tether.items())}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def parse_tether(d: dict) -> dict:
    return {(k.split("|")[0], int(k.split("|")[1])): (float(v[0]), float(v[1])) for k, v in d.items()}


def tether_pseudo(dai1_dir: Path, T: str, allow_partial: bool) -> tuple[dict, Path]:
    """仮のターゲットの回の引き戻し：実験 1 の `<T>_tether_w.json`．単位の一覧が全部の分割と種でなければ止まる（試走は --allow-partial-tether）。"""
    p = dai1_dir / "pseudo" / f"{T}_tether_w.json"
    if not p.exists():
        sys.exit(f"{p} が無い（実験 1 の仮のターゲットの回の段 2 の後に流す）")
    tw = json.load(open(p))
    full = sorted(f"{c}_f{f}_r{r}" for c, _, f, r in dai1.units_for(T, "pseudo", range(DD.N_FOLDS), (0, 1)))
    if sorted(tw["units"]) != full and not allow_partial:
        sys.exit(f"{p} の単位が全部の分割と種でない（試走は --allow-partial-tether）")
    return parse_tether(tw["tether"]), p


def provenance(out: Path, args, script: str, extra: dict | None = None) -> dict:
    commit, dirty = dai1.git_commit()
    prov = dict(script=script, argv=sys.argv, commit=commit, dirty=dirty, host=platform.node(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                cuda=torch.version.cuda, numpy=np.__version__, pandas=pd.__version__, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                scripts_sha256={n: sha(HERE / n) for n in SCRIPTS if (HERE / n).exists()},
                features_sha256={p.name: sha(p) for p in sorted((ROOT / "data" / "cache").glob("*.npz"))
                                 if p.stem.split("_")[-1] in DD.CORPORA and any(p.stem.startswith(k + "_") for k in DD.ENC1.values())},
                encoder=ENCODER, afew_transcript_sha256=sha(ROOT / "data" / "metadata" / "afew_transcript_whisper.csv"),
                metadata_sha256={p.name: sha(p) for p in sorted((ROOT / "data" / "metadata").glob("*.csv"))},
                splits_sha256={p.name: sha(p) for p in sorted((ROOT / "splits").glob("*.csv"))},
                gpu_fraction=os.environ.get("MERDA_GPU_FRACTION", "0.6"), **(extra or {}))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    return prov



def check_eval_frozen(out: Path, dai1_frozen: Path) -> tuple[dict, dict]:
    """実験 2 の評価の段を流す条件（設計 4 節）：(1) `<out>/select/frozen.json` と写し `frozen/<out の名前>_frozen.json` が一致し，写しが
    コミット済み（実験 1 の eval と同じ確かめ），(2) 実験 1 の `dai1.check_eval_gate` が通る（m85．必須の一覧と台本の sha256，査読と関門の OK），
    (3) 実験 2 の台本 3 本が選ぶ段のときから変わっていない（m72．評価の値を見た後に台本を変えない），(4) 実験 1 の固定のファイルが選ぶ段で
    使ったものと同じでコミット済み．戻り：(実験 2 の固定, 実験 1 の固定)。"""
    fz = out / "select" / "frozen.json"
    cp = ROOT / "frozen" / f"{out.name}_frozen.json"
    if not fz.exists() or not cp.exists() or sha(fz) != sha(cp):
        sys.exit(f"固定のファイルが無いか，{cp.relative_to(ROOT)} と一致しない")
    if not dai1.committed_clean(cp):
        sys.exit("実験 2 の固定のファイルの写しがコミットされていない（漏れの検査 L8）")
    d2 = json.load(open(fz))
    for name in OWN_SCRIPTS:
        if sha(HERE / name) != d2["_meta"]["script_sha256"][name]:
            sys.exit(f"台本 {name} が選ぶ段のときから変わっている（固定の後に台本を変えない．ML の査読者の m72）")
    if not dai1_frozen.exists() or sha(dai1_frozen) != d2["_meta"]["dai1_frozen_sha256"]:
        sys.exit(f"実験 1 の固定のファイル {dai1_frozen} が無いか，実験 2 の選ぶ段で使ったものと違う")
    if not dai1.committed_clean(dai1_frozen):
        sys.exit("実験 1 の固定のファイルがコミットされていない")
    RUN["gate_sha256"] = dai1.check_eval_gate(str(dai1_frozen.resolve().relative_to(ROOT)))
    RUN["frozen_sha256"] = sha(fz)
    return d2, json.load(open(dai1_frozen))


def unit_base(out: Path, tag: str, T: str, c: str, f: int, r: int, part: str) -> Path:
    cname = "ALL" if tag == "eval" else c
    return out / tag / "units" / f"{T}_{cname}_f{f}_r{r}_{part}"


def run_unit(out: Path, tag: str, T: str, c: str, srcs: list[str], f: int, r: int, part: str, compute, tether: dict,
             extra_tag: str = "") -> pd.DataFrame:
    """1 単位の 1 つの部分（lin，map，xmuda，coral）を回して保存する．保存があれば照らして読むだけ（再開）。
    compute(u, store, rows) が問を解き，store（npz の中身）と rows（csv の行）を埋める．extra_tag（格子の削り方など）も再開のときに照らす。"""
    base = unit_base(out, tag, T, c, f, r, part)
    path, rpath = Path(f"{base}.npz"), Path(f"{base}.csv")
    tag_t = np.array([f"{k[0]}|{k[1]}|{v[0]}|{v[1]}" for k, v in sorted(tether.items())])
    if path.exists() and rpath.exists():
        old = pd.read_csv(rpath, low_memory=False)
        z = np.load(path)
        bad = str(old["commit"].iloc[0]) != RUN["commit"] or (tag == "eval" and str(old["frozen_sha256"].iloc[0]) != RUN["frozen_sha256"])
        bad |= "tether" not in z or list(z["tether"]) != list(tag_t)
        bad |= "finite" in old and (old["finite"].astype(str) == "False").any()
        bad |= str(z["extra_tag"]) != extra_tag if "extra_tag" in z else extra_tag != ""
        if bad:
            sys.exit(f"保存済みの単位 {rpath.name} のコミット・固定のファイル・引き戻し・有限のどれかが今の実行と違う（消してから流し直す）")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    u = dai1.Unit(Path(RUN["dai1_dir"]), T, c, srcs, f, r)
    store, rows = {}, []
    t0 = time.time()
    compute(u, store, rows)
    store.update(sample_pool=u.cp.df["sample_id"].to_numpy()[u.pool].astype(str),
                 sample_test=u.cp.df["sample_id"].to_numpy()[u.test].astype(str),
                 **{f"sample_K{K}": u.cp.df["sample_id"].to_numpy()[u.pool[u.kpos[K]]].astype(str) for K in DD.KS},
                 **{f"stat_{k}_mu": u.stats[k][0] for k in DD.STUDENTS}, **{f"stat_{k}_sd": u.stats[k][1] for k in DD.STUDENTS},
                 tether=tag_t, tether_key=np.array(RUN["tether_key"]), extra_tag=np.array(extra_tag), commit=np.array(RUN["commit"]), dirty=np.array(RUN["dirty"]),
                 frozen_sha256=np.array(RUN["frozen_sha256"]), gate_sha256=np.array(RUN["gate_sha256"]), gpu=np.array(RUN["gpu"]))
    if any(("finite" in x and x["finite"] is False) for x in rows):
        sys.exit(f"{rpath.name}: 勾配か解に NaN・Inf の問がある（計算の失敗として止める．保存しない．I11）")
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in store.items()})
    df = pd.DataFrame(rows).assign(T=T, c="ALL" if tag == "eval" else c, fold=f, seed=r, sec=time.time() - t0, commit=RUN["commit"],
                                   dirty=RUN["dirty"], frozen_sha256=RUN["frozen_sha256"], gate_sha256=RUN["gate_sha256"],
                                   tether_key=RUN["tether_key"], gpu=RUN["gpu"])
    df.to_csv(rpath, index=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{tag} {base.name}: {time.time()-t0:.0f} 秒", flush=True)
    return df


# ------------------------------------------------------------------ 凸でない問（目的関数に追加の項を足し，出発点を替えて dai_solver.lbfgs_batch で解く）

def fit_nc(X, G, W0, b0, Cw, Cb, Wi, bi, mass, alpha, beta, max_iter: int = 20000, tol: float = 1e-6) -> dict:
    """問 p の目的関数：
        [ Σ_i Σ_k G[p,i,k]·(−log s_ik) + mass_p·(alpha_p·(1/n) Σ_i H(s_i) − beta_p·H((1/n) Σ_i s_i)) + 引き戻し ] / (ΣG_p + mass_p)
    entmin は alpha = 1，beta = 0．IM は alpha = α，beta = 1．(Wi, bi) から始め，`dai_solver.lbfgs_batch`（dai_solver.fit と同じ反復）で，
    問ごとの勾配の最大絶対値が tol 以下になるまで解く．戻りは dai_solver.fit と同じ形（f_p は最後の目的関数の値）。"""
    X = S.t(X)
    G = S.t(G)
    P, n, k = G.shape
    d = X.shape[1]
    W0 = S.t(W0).expand(P, k, d).contiguous() if S.t(W0).dim() == 2 else S.t(W0)
    b0 = S.t(b0).expand(P, k).contiguous() if S.t(b0).dim() == 1 else S.t(b0)
    Wi = S.t(Wi).expand(P, k, d).contiguous() if S.t(Wi).dim() == 2 else S.t(Wi)
    bi = S.t(bi).expand(P, k).contiguous() if S.t(bi).dim() == 1 else S.t(bi)
    vec = lambda v: S.t(np.broadcast_to(np.asarray(v, float), (P,)).copy())  # noqa: E731
    Cw, Cb, mass, alpha, beta = vec(Cw), vec(Cb), vec(mass), vec(alpha), vec(beta)
    Ssum = (G.sum((1, 2)) + mass).clamp(min=1e-12)
    lw = torch.where(torch.isinf(Cw), torch.zeros_like(Cw), 1.0 / (2 * Cw))
    lb = torch.where(torch.isinf(Cb), torch.zeros_like(Cb), 1.0 / (2 * Cb))
    kd = k * d

    def f_and_g(th, ix=None):
        sel = (lambda v: v) if ix is None else (lambda v: v[ix])
        th = th.detach().requires_grad_(True)
        Bx = th.shape[0]
        W = th[:, :kd].view(Bx, k, d)
        b = th[:, kd:]
        Z = torch.einsum("nd,pkd->pnk", X, W) + b[:, None, :]
        lp = torch.log_softmax(Z, 2)
        p = lp.exp()
        nll = -(sel(G) * lp).sum((1, 2))
        cond = -(p * lp).sum(2).mean(1)
        pbar = p.mean(1)
        marg = -(pbar * pbar.clamp(min=1e-300).log()).sum(1)
        reg = sel(lw) * ((W - sel(W0)) ** 2).sum((1, 2)) + sel(lb) * ((b - sel(b0)) ** 2).sum(1)
        f = (nll + sel(mass) * (sel(alpha) * cond - sel(beta) * marg) + reg) / sel(Ssum)
        (g,) = torch.autograd.grad(f.sum(), th)
        return f.detach(), g

    theta = torch.cat([Wi.reshape(P, kd), bi], 1).clone()
    theta, f, grad, n_iter, stalled = S.lbfgs_batch(theta, f_and_g, tol, max_iter)     # 実験 1 の学習器と同じ反復（問ごとの履歴と Armijo の後退）
    gp = grad.abs().amax(1)
    finite = torch.isfinite(gp) & torch.isfinite(theta).all(1) & torch.isfinite(f)
    return {"W": theta[:, :kd].view(P, k, d).clone(), "b": theta[:, kd:].clone(), "n_iter": int(n_iter.max()),
            "grad_max": float(gp.max()), "grad_p": gp.cpu().numpy(), "stalled": int(stalled.sum()), "stalled_p": stalled.cpu().numpy(),
            "n_iter_p": n_iter.cpu().numpy(), "finite_p": finite.cpu().numpy(), "f_p": f.cpu().numpy()}


# ------------------------------------------------------------------ 学習しない補正と T3A とラベル伝播

def saerens(Pp: np.ndarray, Pt: np.ndarray, iters: int = SAERENS_ITERS) -> tuple[np.ndarray, np.ndarray]:
    """Saerens ほか 2002 の EM（E77 の saerens_p と同じ）。戻り：(テストの補正した確率, 推定した割合 π)。"""
    pi = np.full(4, 0.25)
    for _ in range(iters):
        A = Pp * pi
        A /= A.sum(1, keepdims=True)
        pi_new = A.mean(0)
        done = np.abs(pi_new - pi).max() < 1e-9
        pi = pi_new
        if done:
            break
    Q = Pt * pi
    return Q / Q.sum(1, keepdims=True), pi


def t3a(X, Xt, Wn, bn, kpos, ykpos, Ms=T3A_M):
    """E77 の t3a_probs と同じ（公式の実装 matsuolab/T3A：正規化した支えの和を正規化し，正規化しないテストの特徴との内積）。
    戻り：{M: (主の点数, 切片を足した点数, 原型 (4, d))}。"""
    zc = F.normalize(X, dim=1)
    L = X @ Wn.T + bn
    p = torch.softmax(L, 1)
    ent = -(p * p.clamp(min=1e-12).log()).sum(1)
    lab = p.argmax(1)
    unl = torch.ones(len(X), dtype=torch.bool, device=X.device)
    if len(kpos):
        unl[torch.as_tensor(kpos, device=X.device)] = False
    init = F.normalize(Wn, dim=1)
    out = {}
    for Mv in Ms:
        protos = []
        for k in range(4):
            cand = torch.nonzero(unl & (lab == k)).squeeze(1)
            if Mv != -1 and len(cand) > int(Mv):
                cand = cand[torch.argsort(ent[cand], stable=True)[:int(Mv)]]
            parts = [init[k:k + 1], zc[cand]]
            lk = [int(i) for i, y_ in zip(kpos, ykpos) if int(y_) == k]
            if lk:
                parts.append(zc[torch.as_tensor(lk, device=X.device)])
            protos.append(F.normalize(torch.cat(parts).sum(0), dim=0))
        Pm = torch.stack(protos)
        Lt = Xt @ Pm.T
        out[Mv] = (Lt, Lt + bn, Pm)
    return out


def knn_graph(Z, k: int, gamma: int = LP_GAMMA):
    """候補の k 近傍のグラフ（余弦の類似度の γ 乗．負は 0．i→j と j→i を足して対称にする．E77 の lp_scores と同じ形）と D^−½。
    float64 の類似度を安定な並べ替えで並べ，同点は番号の小さい候補を先にする。"""
    n = len(Z)
    zn = F.normalize(Z, dim=1)
    kk = min(k, n - 1)
    ri, ci, vs = [], [], []
    for a in range(0, n, 2048):
        Sm = zn[a:a + 2048] @ zn.T
        ar = torch.arange(Sm.shape[0], device=Z.device)
        Sm[ar, a + ar] = -2.0
        v, i = torch.sort(Sm, dim=1, descending=True, stable=True)
        v, i = v[:, :kk], i[:, :kk]
        ri.append((a + ar).repeat_interleave(kk))
        ci.append(i.reshape(-1))
        vs.append(v.clamp(min=0).reshape(-1) ** gamma)
        del Sm
    ri, ci, vs = torch.cat(ri), torch.cat(ci), torch.cat(vs)
    Wm = torch.sparse_coo_tensor(torch.stack([torch.cat([ri, ci]), torch.cat([ci, ri])]), torch.cat([vs, vs]), (n, n),
                                 check_invariants=False).coalesce()
    dg = torch.sparse.sum(Wm, 1).to_dense()
    dm = torch.where(dg > 0, dg.rsqrt(), torch.zeros_like(dg))
    return Wm, dm


def lp_solve(Wm, dm, lab_pos, lab_y, alpha: float, n: int, iters: int = LP_CG_ITERS):
    """(I − α D^−½ W D^−½) F = Y を共役勾配法で解く（E77 と同じ）。戻り：(F を 0 以上に切ったもの, 回数, 最後の残差)。"""
    Y = torch.zeros(n, 4, dtype=S.DT, device=S.DEV)
    Y[torch.as_tensor(lab_pos, device=S.DEV), torch.as_tensor(lab_y, device=S.DEV)] = 1.0

    def A(Xm):
        return Xm - alpha * dm[:, None] * torch.sparse.mm(Wm, dm[:, None] * Xm)
    Xs = torch.zeros_like(Y)
    R = Y - A(Xs)
    Pm = R.clone()
    rs = (R * R).sum(0)
    it = 0
    for it in range(1, iters + 1):
        AP = A(Pm)
        a_ = rs / (Pm * AP).sum(0).clamp(min=1e-30)
        Xs = Xs + a_ * Pm
        R = R - a_ * AP
        rs_new = (R * R).sum(0)
        if float(rs_new.max()) < 1e-20:
            break
        Pm = R + (rs_new / rs.clamp(min=1e-30)) * Pm
        rs = rs_new
    resid = float(((Y - A(Xs)) ** 2).sum(0).max().sqrt())
    return Xs.clamp(min=0), it, resid


def lp_pseudo(Fs, lab_pos, form: int):
    """伝播の結果から疑似ラベルの行（K 本の行と F が全部 0 の行は 0）。form 0：最大の感情の one-hot，1：行で正規化した F。"""
    Fs = Fs.clone()
    if len(lab_pos):
        Fs[torch.as_tensor(lab_pos, device=Fs.device)] = 0.0
    tot = Fs.sum(1)
    ok = tot > 0
    if form == 0:
        R = torch.zeros_like(Fs)
        R[torch.nonzero(ok).squeeze(1), Fs[ok].argmax(1)] = 1.0
    else:
        R = torch.where(ok[:, None], Fs / tot.clamp(min=1e-30)[:, None], torch.zeros_like(Fs))
    return R, ok


# ------------------------------------------------------------------ S+T

def st_sources(u, kind: str, dai1_dir: Path):
    """ソースの全発話（各ソースの全発話の平均と標準偏差で標準化．段 source と同じ座標）とラベル。"""
    Xs, ys = [], []
    for s_ in u.srcs:
        cp = dai1.Corpus.get(s_)
        mu, sd = dai1.load_source_stats(dai1_dir, s_, kind)
        Xs.append((cp.X[kind] - mu) / sd)
        ys.append(cp.y)
    return torch.cat(Xs), np.concatenate(ys)


def st_cv(out: Path, srcs: list[str], kind: str) -> dict:
    """S+T の C：合わせたソースで段 source と同じ塊の 5 分割の交差検証（各ソースの学習側を，そのソースの学習側の統計で標準化．
    MELD はテストの対話を学習側から除く．評価側は合わせた全体の UAR）．平均が最大の C，同点は小さい C．`<out>/st_cv/` に残して使い回す。"""
    key = "+".join(sorted(srcs)) + f"_{kind}"
    p = out / "st_cv" / f"{key}.json"
    if p.exists():
        d = json.load(open(p))
        if d["commit"] != RUN["commit"]:
            sys.exit(f"{p} のコミットが今の実行と違う（消してから流し直す）")
        return d
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cv = np.zeros((DD.N_FOLDS, len(ST_C)))
    gmax = 0.0
    for f in range(DD.N_FOLDS):
        Xtr, ytr, Xte, yte = [], [], [], []
        for s_ in sorted(srcs):
            cp = dai1.Corpus.get(s_)
            X = cp.X[kind]
            tr = np.flatnonzero(DD.pool_mask(cp.df, f)); te = np.flatnonzero(cp.df["fold"].to_numpy() == f)
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
            Xtr.append((X[tr] - mu) / sd); Xte.append((X[te] - mu) / sd)
            ytr.append(cp.y[tr]); yte.append(cp.y[te])
        Xtr, Xte, ytr, yte = torch.cat(Xtr), torch.cat(Xte), np.concatenate(ytr), np.concatenate(yte)
        G = S.balanced_onehot(ytr, np.arange(len(ytr)), len(ytr))[None].expand(len(ST_C), -1, -1)
        r = S.fit(Xtr, G, torch.zeros(4, Xtr.shape[1]), torch.zeros(4), Cw=np.array(ST_C), Cb=math.inf)
        cv[f] = S.uar_batch(yte, S.logits(Xte, r["W"], r["b"]))
        gmax = max(gmax, r["grad_max"])
        del Xtr, Xte, G, r
    mean = cv.mean(0)
    best = int(np.flatnonzero(mean == mean.max())[0])
    d = {"srcs": sorted(srcs), "kind": kind, "C": ST_C[best], "C_index": best, "edge": best in (0, len(ST_C) - 1),
         "cv": cv.tolist(), "C_grid": list(ST_C), "grad_max": gmax, "commit": RUN["commit"], "sec": time.time() - t0}
    json.dump(d, open(p, "w"), indent=1)
    print(f"S+T の交差検証 {key}: C={d['C']}（{time.time()-t0:.0f} 秒）", flush=True)
    return d


# ------------------------------------------------------------------ 1 単位

def solve_batch(u, s, K, specs, W0, b0, Cw, Cb, rows, store, key, X=None, Xt=None, extra_row=None) -> dict:
    """specs：[(行の dict, G)] をまとめて dai_solver.fit で解き，行と判定と W・b を残す。"""
    X = u.Xp[s] if X is None else X
    Xt = u.Xt[s] if Xt is None else Xt
    G = torch.stack([g for _, g in specs])
    res = S.fit(X, G, W0, b0, Cw=Cw, Cb=Cb)
    Z = S.logits(Xt, res["W"], res["b"])
    ua = S.uar_batch(u.ytest, Z)
    Ssum = G.sum((1, 2)).cpu().numpy()
    cw_v = np.broadcast_to(np.asarray(Cw, float), (len(specs),))
    cb_v = np.broadcast_to(np.asarray(Cb, float), (len(specs),))
    for j, ((it, _), v) in enumerate(zip(specs, ua)):
        rows.append(dict(student=s, K=K, uar=v, S=float(Ssum[j]), grad=float(res["grad_p"][j]), stalled=bool(res["stalled_p"][j]),
                         finite=bool(res["finite_p"][j]), n_iter=int(res["n_iter_p"][j]), Cw=float(cw_v[j]), Cb=float(cb_v[j]),
                         **(extra_row or {}), **it))
    store[f"{key}_pred"] = pred8(Z)
    store[f"{key}_W"], store[f"{key}_b"] = f32(res["W"]), f32(res["b"])
    store[f"{key}_items"] = np.array([json.dumps(it, sort_keys=True) for it, _ in specs])
    return res


def pass_lin(u, store: dict, rows: list, tether: dict, pseudo: bool, students=DD.STUDENTS) -> None:
    dai1_dir = Path(RUN["dai1_dir"])
    p1old = None                                                       # 実験 1 の段 1 の保存（あれば none の判定の一致を残す）
    p1path = dai1_dir / ("pseudo" if pseudo else "eval") / "units" / f"{u.T}_{u.c if pseudo else 'ALL'}_f{u.f}_r{u.r}_p1.npz"
    same_units = False
    if p1path.exists():
        p1old = np.load(p1path)
        sid = u.cp.df["sample_id"].to_numpy()
        same_units = ("sample_test" in p1old and np.array_equal(p1old["sample_test"], sid[u.test].astype(str))
                      and all(np.array_equal(p1old[f"sample_K{K}"], sid[u.pool[u.kpos[K]]].astype(str)) for K in DD.KS))
        if not same_units:                                             # 分割か K 本が違う保存（古いコミットの試走など）とは照らさない
            p1old = None
    for s in students:
        X, Xt = u.Xp[s], u.Xt[s]
        W0, b0 = u.start[s]
        n = u.n
        graphs = {k: knn_graph(X, k) for k in LP_K}                     # ラベル伝播のグラフは K に依らない
        Xsrc, ysrc = st_sources(u, s, dai1_dir)
        cvd = st_cv(Path(RUN["out"]), u.srcs, s)
        for K in KS_ALL:
            t0 = time.time()
            cw, cb = tether[(s, max(K, 1))]
            gk = u.gK(K) if K else torch.zeros(n, 4, device=S.DEV, dtype=S.DT)
            kp = u.kpos[K]
            # 比較の基準 none[K]（K=0 は出発点）
            if K:
                rn = S.fit(X, gk[None], W0, b0, Cw=cw, Cb=cb)
                Wn, bn = rn["W"][0], rn["b"][0]
                conv = dict(grad=float(rn["grad_p"][0]), stalled=bool(rn["stalled_p"][0]), finite=bool(rn["finite_p"][0]),
                            n_iter=int(rn["n_iter_p"][0]))
            else:
                Wn, bn = W0, b0
                conv = dict(grad=0.0, stalled=False, finite=True, n_iter=0)
            Zn = S.logits(Xt, Wn, bn)
            agree = float("nan")
            if p1old is not None:
                try:
                    old = (p1old[f"p1_{s}_start_pred"] if K == 0 else
                           p1old[f"p1_{s}_pred"][[str(x) for x in p1old[f"p1_{s}_items"]].index(f"{K}|{cw}|{cb}")])
                    agree = float((Zn.argmax(1).cpu().numpy() == old).mean())
                except (KeyError, ValueError):                           # 形の違う古い保存は照らさない
                    agree = float("nan")
            rows.append(dict(method="none", student=s, K=K, Cw=cw, Cb=cb, uar=100 * S.uar(u.ytest, Zn.argmax(1).cpu().numpy()),
                             S=float(gk.sum()), none_agree_dai1=agree, dai1_same_units=same_units, **conv))
            store[f"none_{s}_K{K}_W"], store[f"none_{s}_K{K}_b"] = f32(Wn), f32(bn)
            store[f"none_{s}_K{K}_pred"] = pred8(Zn)
            own = probs(X, Wn, bn)
            # 割合の補正（付録の確かめ）
            Qt, pi = saerens(own.cpu().numpy(), probs(Xt, Wn, bn).cpu().numpy())
            rows.append(dict(method="saerens", student=s, K=K, uar=100 * S.uar(u.ytest, Qt.argmax(1)), finite=bool(np.isfinite(Qt).all()),
                             **{f"pi_{k}": float(pi[k]) for k in range(4)}))
            store[f"sae_{s}_K{K}_pred"] = Qt.argmax(1).astype(np.int8)
            store[f"sae_{s}_K{K}_pi"] = pi
            # 自己学習（3 巡．毎巡，出発点から学び直す）
            items = [(tau, rho) for tau in SELF_TAU for rho in RHO_GRID]
            cur = own[None].expand(len(items), -1, -1)
            for rnd in range(SELF_ROUNDS):
                Gs, Hl = [], []
                for j, (tau, rho) in enumerate(items):
                    conf, lab = cur[j].max(1)
                    keep = conf >= tau
                    H = torch.zeros(n, 4, device=S.DEV, dtype=S.DT)
                    H[keep, lab[keep]] = 1.0
                    Gs.append(gk + rho * H)
                    Hl.append(torch.where(keep, lab, torch.full_like(lab, -1)))
                    cnt = H.sum(0).cpu().numpy()
                    acc = float((lab[keep].cpu().numpy() == u.ypool[keep.cpu().numpy()]).mean()) if pseudo and bool(keep.any()) else float("nan")
                    rows.append(dict(method="_self_round", student=s, K=K, tau=tau, rho=rho, round=rnd + 1, n_pseudo=int(keep.sum()),
                                     **{f"n_pseudo_{k}": int(cnt[k]) for k in range(4)}, acc_pseudo=acc))
                store[f"self_{s}_K{K}_r{rnd + 1}_pl"] = torch.stack(Hl).to(torch.int8).cpu().numpy()   # 巡 rnd+1 で置いた硬いラベル（候補の番号ごとの感情．−1 は無し．R9）
                if rnd < SELF_ROUNDS - 1:
                    rs = S.fit(X, torch.stack(Gs), W0, b0, Cw=cw, Cb=cb)
                    cur = torch.softmax(S.logits(X, rs["W"], rs["b"]), -1)
                    store[f"self_{s}_K{K}_r{rnd + 1}_pred"] = pred8(S.logits(Xt, rs["W"], rs["b"]))     # 巡 rnd+1 の解のテストの判定（最後の巡は self_..._pred．R9）
                    rows.append(dict(method="_conv_self", student=s, K=K, round=rnd + 1, grad_max=rs["grad_max"], n_stalled=rs["stalled"],
                                     finite=bool(rs["finite_p"].all())))
            solve_batch(u, s, K, [(dict(method="self", tau=tau, rho=rho), g) for (tau, rho), g in zip(items, Gs)], W0, b0, cw, cb,
                        rows, store, f"self_{s}_K{K}", extra_row=dict(round=SELF_ROUNDS))
            store[f"self_{s}_K{K}_pl"] = torch.stack(Hl).to(torch.int8).cpu().numpy()   # 最後の巡の硬いラベル（−1 は無し）
            # own とラベル伝播（どちらも提案の同時に解く版の形）
            specs = []
            for rho in RHO_GRID:
                for a in A_GRID:
                    it = dict(method="own", rho=rho, a=a)
                    specs.append((it, dai1.make_G(u, "joint", own, None, gk, dict(rho=rho, a=a))[0]))
            if K:
                ky = u.ypool[kp]
                for kk in LP_K:
                    Wm, dm = graphs[kk]
                    for al in LP_ALPHA:
                        Fs, cgi, cgr = lp_solve(Wm, dm, kp, ky, al, n)
                        for form in LP_FORM:
                            R, ok = lp_pseudo(Fs, kp, form)
                            okn = ok.cpu().numpy()
                            acc = float((Fs[ok].argmax(1).cpu().numpy() == u.ypool[okn]).mean()) if pseudo and okn.any() else float("nan")
                            it = dict(method="lp", k=kk, alpha_lp=al, form=form)
                            rows.append(dict(method="_lp_diag", student=s, K=K, k=kk, alpha_lp=al, form=form, cg_iters=cgi, cg_resid=cgr,
                                             n_pseudo=int(okn.sum()), acc_pseudo=acc, **{f"colsum_{c}": float(R[:, c].sum()) for c in range(4)}))
                            specs.append((it, gk + S.soft_rows(R, dai1.mval(1.0, n))))
            solve_batch(u, s, K, specs, W0, b0, cw, cb, rows, store, f"lin_{s}_K{K}")
            # エントロピー最小化と IM（凸でない．none[K] の解から）
            nci = [dict(method="entmin", rho=rho) for rho in RHO_GRID] + [dict(method="im", alpha=a_, rho=rho) for a_ in IM_ALPHA for rho in RHO_GRID]
            mass = [dai1.mval(it["rho"], n) for it in nci]
            alpha = [1.0 if it["method"] == "entmin" else it["alpha"] for it in nci]
            beta = [0.0 if it["method"] == "entmin" else 1.0 for it in nci]
            rc = fit_nc(X, gk[None].expand(len(nci), -1, -1), W0, b0, cw, cb, Wn, bn, mass, alpha, beta)
            Zc = S.logits(Xt, rc["W"], rc["b"])
            ua = S.uar_batch(u.ytest, Zc)
            for j, it in enumerate(nci):
                rows.append(dict(student=s, K=K, uar=ua[j], S=float(gk.sum()) + mass[j], grad=float(rc["grad_p"][j]), stalled=bool(rc["stalled_p"][j]),
                                 finite=bool(rc["finite_p"][j]), n_iter=int(rc["n_iter_p"][j]), Cw=cw, Cb=cb, obj=float(rc["f_p"][j]),
                                 W_dist=float((rc["W"][j] - W0).norm()), W_dist_none=float((rc["W"][j] - Wn).norm()), **it))
            store[f"nc_{s}_K{K}_W"], store[f"nc_{s}_K{K}_b"] = f32(rc["W"]), f32(rc["b"])
            store[f"nc_{s}_K{K}_pred"] = pred8(Zc)
            store[f"nc_{s}_K{K}_items"] = np.array([json.dumps(it, sort_keys=True) for it in nci])
            # T3A（主の版と切片を残す版）
            tt = t3a(X, Xt, Wn, bn, kp, u.ypool[kp])
            pm, pb, pp = [], [], []
            for Mv, (Lm, Lb, Pm) in tt.items():
                for meth, L in (("t3a", Lm), ("t3a_b", Lb)):
                    rows.append(dict(method=meth, student=s, K=K, M=Mv, uar=100 * S.uar(u.ytest, L.argmax(1).cpu().numpy()),
                                     finite=bool(torch.isfinite(L).all())))
                pm.append(Lm.argmax(1)); pb.append(Lb.argmax(1)); pp.append(Pm)
            store[f"t3a_{s}_K{K}_pred"] = torch.stack(pm).to(torch.int8).cpu().numpy()          # 並びは T3A_M
            store[f"t3a_{s}_K{K}_pred_b"] = torch.stack(pb).to(torch.int8).cpu().numpy()
            store[f"t3a_{s}_K{K}_P"] = f32(torch.stack(pp))
            rows.append(dict(method="_sec", student=s, K=K, sec_sK=time.time() - t0))
        # S+T（K=0 は 1 問，K ≥ 1 は β の 3 点．X は ソースの全発話 ＋ K=25 の入れ子の和集合）
        k25 = u.kpos[25]
        Xall = torch.cat([Xsrc, X[k25]])
        Gsrc = S.balanced_onehot(ysrc, np.arange(len(ysrc)), len(ysrc))
        specs = [(dict(method="st", beta=ST_BETA[0], K_=0), torch.cat([Gsrc, torch.zeros(len(k25), 4, device=S.DEV, dtype=S.DT)]))]
        for K in DD.KS:
            gK25 = u.gK(K)[torch.as_tensor(k25, device=S.DEV)]
            for be in ST_BETA:
                specs.append((dict(method="st", beta=be, K_=K), torch.cat([Gsrc, be * gK25])))
        G = torch.stack([g for _, g in specs])
        res = S.fit(Xall, G, torch.zeros(4, X.shape[1]), torch.zeros(4), Cw=cvd["C"], Cb=math.inf)
        Z = S.logits(Xt, res["W"], res["b"])
        ua = S.uar_batch(u.ytest, Z)
        for j, (it, g) in enumerate(specs):
            K = it["K_"]
            base = dict(method="st", student=s, K=K, uar=ua[j], S=float(g.sum()), grad=float(res["grad_p"][j]), stalled=bool(res["stalled_p"][j]),
                        finite=bool(res["finite_p"][j]), n_iter=int(res["n_iter_p"][j]), C=cvd["C"], C_edge=bool(cvd["edge"]),
                        n_src=len(ysrc), srcs="+".join(sorted(u.srcs)))
            if K == 0:
                for be in ST_BETA:
                    rows.append(dict(base, beta=be, beta_dummy=be != ST_BETA[0]))
            else:
                rows.append(dict(base, beta=it["beta"], beta_dummy=False))
        store[f"st_{s}_pred"] = pred8(Z)
        store[f"st_{s}_W"], store[f"st_{s}_b"] = f32(res["W"]), f32(res["b"])
        store[f"st_{s}_items"] = np.array([json.dumps({"beta": it["beta"], "K": it["K_"]}, sort_keys=True) for it, _ in specs])
        del Xsrc, Xall, G, graphs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ 進め方

def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai2"))
    ap.add_argument("--dai1-dir", default=str(ROOT / "results" / "dai1"), help="実験 1 の置き場（source/ と pseudo/<T>_tether_w.json）")
    ap.add_argument("--dai1-frozen", default=str(ROOT / "frozen" / "dai1_frozen.json"),
                    help="実験 1 の固定のファイル（評価の回の引き戻しと R，選ぶ段の δ_sel．試走では一時の置き場の frozen.json）")
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(DD.N_FOLDS)))
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="既定は pseudo で 0 1，eval で実験 1 の n_seeds_eval（勾配法は 0 1 2）")
    ap.add_argument("--pseudo-c", nargs="*", default=None, help="試走用：仮のターゲットを絞る")
    ap.add_argument("--allow-dirty", action="store_true", help="試走用（評価の段は受け付けない．選ぶ段は本番の置き場では受け付けない）")
    ap.add_argument("--allow-partial-tether", action="store_true", help="試走用：tether_w.json の単位が全部でなくても進める")
    ap.add_argument("--students", nargs="*", default=None, choices=DD.STUDENTS,
                    help="試走用：生徒を絞る（本番の置き場では受け付けない．xMUDA 型は A・V・T を同時に学ぶので絞らない．I10）")


def is_production(out: Path) -> bool:
    return Path(out).resolve() == PROD_DIR.resolve()


def check_encoder(dai1_dir: Path) -> dict:
    """実験 1 の置き場のエンコーダが実験 2 のエンコーダ（第 1 のエンコーダ）と同じかを確かめる（R7）．実験 1 の provenance_log.jsonl
    （置き場のものと，source/ の実体の隣のもの）の encoder を全部読み，1 つも無いか，1 つでも違えば止まる．
    音声 1,024・顔 512・言葉 768 の次元は両エンコーダで同じなので，確かめないと誤った組み合わせでも落ちずに計算する。"""
    logs = {dai1_dir / "provenance_log.jsonl", (dai1_dir / "source").resolve().parent / "provenance_log.jsonl"}
    encs = []
    for lp in sorted(logs):
        if lp.exists():
            for line in open(lp):
                d = json.loads(line)
                if "encoder" in d:
                    encs.append((lp.name, d.get("stage"), d["encoder"]))
    if not encs:
        sys.exit(f"実験 1 の置き場 {dai1_dir} の provenance にエンコーダの記録が無い（R7）")
    bad = [e for e in encs if e[2] != ENCODER]
    if bad:
        sys.exit(f"実験 1 の置き場のエンコーダ {bad[0][2]} が実験 2 のエンコーダ {ENCODER} と違う（実験 2 は第 1 のエンコーダだけで回す．R7）")
    return ENCODER


def setup_run(args, script: str) -> Path:
    out = Path(args.out_dir)
    if args.students and is_production(out):
        sys.exit("--students は試走だけ（本番の置き場では受け付けない．I10）")
    check_encoder(Path(args.dai1_dir))
    out.mkdir(parents=True, exist_ok=True)
    commit, dirty = dai1.git_commit()
    if dirty and args.stage == "eval":
        sys.exit("評価の段は未コミットの変更があると流さない（--allow-dirty も受け付けない）")
    if dirty and not args.allow_dirty:
        sys.exit("exp・measure・features に未コミットの変更がある（試しは --allow-dirty）")
    RUN.update(commit=commit, dirty=dirty, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               dai1_dir=str(Path(args.dai1_dir)), out=str(out))
    return out


def students_tag(args) -> str:
    """生徒を絞った試走の単位を，絞らない単位として読み直さないための印（run_unit の extra_tag）。"""
    return json.dumps({"students": list(args.students)}) if args.students else ""


def units_of(args, T: str, mode: str, seeds):
    for c, srcs, f, r in dai1.units_for(T, mode, args.folds, seeds):
        if mode == "pseudo" and args.pseudo_c and c not in args.pseudo_c:
            continue
        yield c, srcs, f, r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("pseudo", "select", "eval"))
    common_args(ap)
    args = ap.parse_args()
    if args.stage == "select":
        import dai2_select
        dai2_select.main(Path(args.out_dir), args.targets, Path(args.dai1_frozen), args.allow_dirty)
        return
    out = setup_run(args, "dai2.py")
    studs = tuple(args.students) if args.students else DD.STUDENTS
    stag = students_tag(args)
    if args.stage == "pseudo":
        for T in args.targets:
            teth, tpath = tether_pseudo(Path(args.dai1_dir), T, args.allow_partial_tether)
            RUN["tether_key"] = tether_key(teth)
            provenance(out, args, "exp/dai2.py", dict(stage="pseudo", T=T, tether_file=str(tpath), tether_file_sha256=sha(tpath),
                                                         tether_key=RUN["tether_key"], self_rounds=SELF_ROUNDS, lp_gamma=LP_GAMMA,
                                                         lp_cg_iters=LP_CG_ITERS, saerens_iters=SAERENS_ITERS))
            seeds = args.seeds if args.seeds is not None else [0, 1]
            for c, srcs, f, r in units_of(args, T, "pseudo", seeds):
                run_unit(out, "pseudo", T, c, srcs, f, r, "lin", lambda u, st, rw: pass_lin(u, st, rw, teth, True, studs), teth, extra_tag=stag)
        return
    fz2, fz1 = check_eval_frozen(out, Path(args.dai1_frozen))
    for T in args.targets:
        teth = parse_tether(fz1[T]["tether"])
        RUN["tether_key"] = tether_key(teth)
        R = int(fz1[T]["n_seeds_eval"])
        seeds = args.seeds if args.seeds is not None else list(range(R))
        provenance(out, args, "exp/dai2.py", dict(stage="eval", T=T, frozen_sha256=RUN["frozen_sha256"], gate_sha256=RUN["gate_sha256"],
                                                     dai1_frozen_sha256=sha(Path(args.dai1_frozen)), tether_key=RUN["tether_key"], R=R))
        for c, srcs, f, r in units_of(args, T, "eval", seeds):
            run_unit(out, "eval", T, c, srcs, f, r, "lin", lambda u, st, rw: pass_lin(u, st, rw, teth, False, studs), teth, extra_tag=stag)


if __name__ == "__main__":
    main()
