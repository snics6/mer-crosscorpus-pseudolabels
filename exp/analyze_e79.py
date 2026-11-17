"""【本番】E79（CPU）：DAI 論文の比較実験の判定の解析（ステップ 4）。

設計は `DAI_paper/steps/step4_design.md`（関門 1 の後に `docs/exploratory-analyses.md` の E79 に写す）．正本の比較の設計は
`DAI_paper/reviews/comparison-design/proposal_v8.md`．入力は E77（`exp/compare_dai.py`）と E78（`exp/compare_dai2.py`）の出力で，
この台本は学び直しをしない．GPU を使わない（起動の最初に CUDA_VISIBLE_DEVICES を空にする）。

段（--stage．この順に流す．各段の出力は `--out/<段>/` に書き，既にあれば止まる）
  delta   ：判定の基準 δ_d・δ_u を，E77 の仮のターゲットの none の出力（`<T>_ps_<c>_r<r>_none.npz`）と `<T>_pseudo.csv` だけから計算する．
            入力の場所の中でほかのファイルを開こうとしたら止まる（`sys.addaudithook` で open を見張る．段の最初に，禁じたファイルを
            開けないことを自分で確かめる）．出力の `delta.csv` の sha256 の先頭 16 文字を表示する．これを文書に書いてコミットしてから次の段に進む
  select  ：仮のターゲットの値（`<T>_pseudo.csv`）だけで，(生徒, 選ぶ族, K) ごとに設定を選ぶ．none を含む版（incl）と除く版（excl），
            MOSEI・MELD では IEMOCAP 台本を仮のターゲットから除いた版（ns_incl・ns_excl）．固定版の行と，対照の対応する行も決める．
            oursx は E80 の R4（`--e80-kept` の kept.csv で（世界, 評価のターゲット, 生徒, K）ごとに残した教師 × 重みの付け方 5 × 総重み 4．
            残した教師が無い単位は none．KD も同じ教師の集合）．縮めた格子は R6′（KD と同じ数の部分集合を種 0〜19 で 20 回引いて選ぶ）．
            評価のターゲットの値は開かない（delta と同じ仕掛け）．δ のコミットを確かめてから動く
  measure ：評価のターゲットのテストの発話ごとの対数の確率から，反復ごとに UAR・6 対の d と o・位置の損・向きの損を全行で測り，
            選んだ行について 2 段のブートストラップ（話者と反復）の複製を作る．D（候補のラベルで学んだ LDA の向きをテストで測った d）は抽出の前に固定する．
            人物の実効の数 G_eff が 20 未満のコーパスでは人物を 1 人ずつ抜く jackknife の複製，MELD（対話）と IEMOCAP（セッション）では
            塊を引くブートストラップの複製（感度の行）も作る．共正則化は評価のターゲットでテストで選んだ上限（UAR と向きの損で別々）を決める
  judge   ：設計の文書 7.2 節の表を機械的に当てて，P4・P4'・P5-1〜P5-3・除去実験・5.3 節の集計を判定する．感度の行と付録の表も出す
            （Co-teaching とその対照，P5-4 は E78 で回さないので外した．2026-11-01 夜の変更）

    ./.venv/bin/python exp/analyze_e79.py --stage delta  [--runs …] [--e77 名前=場所 …] [--e78 名前=場所 …] [--out results/e79]
    ./.venv/bin/python exp/analyze_e79.py --stage select [--e80-kept results/e80/kept.csv]
    ./.venv/bin/python exp/analyze_e79.py --stage measure [--boot 1000]
    ./.venv/bin/python exp/analyze_e79.py --stage judge

段の間の照合（関門 1 の指摘 L-2・L-3・I12-2）：select は δ のコミットと delta の台本の版を，measure は δ・`chosen.csv`・`fixed.csv` のコミットと
前の段の台本の版と δ の sha256 を，judge はさらに measure が使った δ・`chosen.csv`・`fixed.csv` の sha256 と 4 段の台本の版を確かめ，外れたら止まる。
`exp/` か `measure/` に未コミットの変更があると止まる（--allow-dirty は試し用）．BLAS の糸の数は既定で 4（環境変数で変えられる．provenance に残す）。

試走（揃っていない出力）では --allow-incomplete（欠けを止めずに記録する），--skip-commit-check（コミットの確認を飛ばし，provenance に残す），
--allow-dirty を使う。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                         # GPU を使わない（読み込むモジュールが torch を使っても CPU に置く）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):         # E77・E78 と CPU を取り合わないよう糸の数を決める（I10-1）
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse, stats  # noqa: E402
import scipy.linalg  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_pairs import ba, own_separation  # noqa: E402
from explore_synergy import PAIRS, d_o  # noqa: E402
from multimodal_distill import WORLDS, blocks  # noqa: E402
from shift import uar as shift_uar  # noqa: E402

RUNS = ("audio_wavlm", "video", "text_roberta", "fuse-audio-video-text", "fuse-audio-video")
KS = ("0", "5", "25")                                                          # DAI 論文の短縮 c（E77・E78 は --ks 0 5 25）．読んだ provenance の K と照合する
REPS = tuple(range(10))
PSEUDO_REPS = tuple(range(5))
MEFFS = ("100", "500", "2000", "n")
WGTS = ("bal", "raw", "hard0.5", "hard0.7", "hard0.9")
SEEDED = {"dann", "mme", "mapctl", "xmuda", "entmin", "im"}                   # 種のある手法（評価は種 0・1・2 の平均．選択は種 0）
NC_SEEDED = {"entmin", "im"}                                                   # E77 の非凸の手法：種 0 の param は末尾に s が無い（`-`・`a0.1`）．評価のターゲットでは全候補に種 1・2 がある（gate-code の R3-2）
SEEDS = (0, 1, 2)
GROUP_B = ("entmin", "self", "im", "t3a", "lp", "coral", "dann", "mme")       # P4 の 8 手法（T3A は主の版）
P4_KS = ("0", "5")
P5_KS = ("0", "5", "25")
DISAGREE_MIN = 0.01                                                            # 予測が変わったとみなす食い違いの割合の下限．DA の項の効き（K ≥ 1 は種の間の食い違いの最大も超えること）と，
                                                                               # P4・P4' の「働かなかった」単位（選んだ設定のテストの予測が none と食い違う割合がこれ未満．模擬査読の S4-2）
MODEL_RATIO_MAX = 1 / 3                                                        # 2 つの山の模型：E75 の決まり（模型の誤差 ÷ 損 < 1/3）．誤差は模型が正しいときの標本の揺れを引いてから測る
MODEL_UNIT_FRAC = 0.5                                                          # 単位の印：|g| と |Δe| が δ_d のこの割合を超えたら「模型に依る」
VR_RANGE = (0.5, 2.0)                                                          # 分散の比（記述だけ）
NULL_DRAWS = 100                                                               # 模型が正しいときの誤差の分布を作る抽出の回数
CORAL_FIX_R = 64
TOL_LOGP, TOL_UAR = 1e-5, 1e-9                                                 # E77 と E78 の同じはずの値の許容幅（I14-1）
SEEN = {"P4": {"massbal", "entmin", "self", "coral"}, "P4'": {"massbal", "entmin", "self", "coral", "im"}}   # 設計の文書 2 節で「UAR を見た後の予言」の手法（L-4）
REFS = ("ref_srconly", "ref_merged", "ref_tgtonly", "ref_wstar", "teacher")
GEFF_MIN = 20                                                                  # 人物の実効の数 G_eff ＝ (Σn)²/Σn² がこれ未満のターゲットを含む単位は，百分位の区間と jackknife の t 区間の広い方（E76 の規則．模擬査読の G4）
CLUSTER_COLS = {"MELD": ("meld_utterance_manifest.csv", ("split", "dialogue_id")),   # 塊を引く区間（感度の行）：MELD は対話（split ごとに番号が振り直されるので split と組にする）
                "IEMOCAP": ("iemocap_utterance_manifest.csv", ("session",))}   # IEMOCAP はセッション
COREG = {"uni": "coreg", "fused": "coreg_sum"}                                 # 共正則化（群 C の探索）．仮のターゲットでは解かない（短縮 b）．評価のターゲットでテストで選んだ上限と比べる
KEPT_FAMS = ("oursx", "kd", "abl_bal", "abl_raw", "abl_hard")                # E80 の R3′〜R3‴ で残した教師を参照にする族．KD も同じ教師の集合（C1）．残した教師が無ければ候補が無く，none を含む版は none（R3‴）
R6_DRAWS = 20                                                                  # R6′：oursx の候補から KD の族と同じ数の部分集合を種 0〜19 で一様に引き（none は常に含む），それぞれで選ぶ
CONV_MAX_ITER, CONV_TOL = 20000, 1e-6                                          # exp/logreg.py の MAX_ITER・TOL（L-BFGS の反復の上限と勾配の許容幅．付録の収束の表）
LP_CG_MAX = 50                                                                 # exp/compare_dai.py の LP_CG_ITERS（ラベル伝播の共役勾配法の上限）
R6_PREFIX = "oursx_r6s"
P4_SELS = set(GROUP_B) | {"t3a_b", "mapctl", "coralctl"}                       # --families p4 で選ぶ族（P4・P4′・その対照と記述．E80 の段 c を待たずに流す道）
NOJ_DIST = {"uni": ("pF", "pFK", "ref"), "fused": ("poe0", "ref", "ref0")}    # ふるわない版 oursx_noJ：E77 に dist の行がある全部の教師（ownK は教師でなく除去の行）
NOJ_KD = {"uni": ("pF", "pFK", "ref"), "fused": ("poe0", "ref")}               # ふるわない版 kd_noJ：前の設計の KD と同じ教師（E77 の kd・kdh の教師）
ABL_TEACHER = {"abl_refa": "pFK", "abl_refb": "ref", "abl_pF": "pF", "abl_ref0": "ref0", "abl_poe0": "poe0"}   # 教師を固定した除去の行
EXCLUDED = ("no_teacher", "identical", "other_class")
CLUSTER_JK_MAX = 10                                                            # 塊の数がこれ未満なら，塊の百分位の区間に塊を 1 つずつ抜く jackknife の t 区間（自由度 塊の数 − 1）を和集合で足す                          # 判定の数え上げから外す単位（過半は残りのコーパスの数で決め直す）                                                        # R6′ の選択の族の名前（oursx_r6s0 〜 oursx_r6s19）
CONSTS = {"KS": KS, "DISAGREE_MIN": DISAGREE_MIN, "MODEL_RATIO_MAX": MODEL_RATIO_MAX, "MODEL_UNIT_FRAC": MODEL_UNIT_FRAC,
          "VR_RANGE": VR_RANGE, "NULL_DRAWS": NULL_DRAWS, "CORAL_FIX_R": CORAL_FIX_R, "TOL_LOGP": TOL_LOGP,
          "TOL_UAR": TOL_UAR, "P4_KS": P4_KS, "P5_KS": P5_KS, "GROUP_B": GROUP_B, "SEEN": {k: sorted(v) for k, v in SEEN.items()},
          "SEEDED": sorted(SEEDED), "NC_SEEDED": sorted(NC_SEEDED), "GEFF_MIN": GEFF_MIN,
          "CLUSTER_COLS": {k: list(v[1]) for k, v in CLUSTER_COLS.items()}, "COREG": COREG, "KEPT_FAMS": KEPT_FAMS, "R6_DRAWS": R6_DRAWS,
          "CONV_MAX_ITER": CONV_MAX_ITER, "CONV_TOL": CONV_TOL, "LP_CG_MAX": LP_CG_MAX, "NOJ_DIST": NOJ_DIST, "NOJ_KD": NOJ_KD,
          "EXCLUDED": EXCLUDED, "CLUSTER_JK_MAX": CLUSTER_JK_MAX}
EMPTY = "__none__"
OWN_CHECKED: set = set()
E80C: dict = {}                                                                # 実行 → E80 の段 c の出力の場所（parse_dirs が入れる）

# 入力の場所の中で，段ごとに開いてよいファイル（名前の末尾の正規表現）．provenance_log.jsonl は起動の引数だけを持つ（K の照合に使う）
ALLOW = {
    "delta": (r"_ps_.+_r\d+_none\.npz$", r"_pseudo\.csv$", r"^provenance\.json$", r"^provenance_log\.jsonl$"),
    "select": (r"_pseudo\.csv$", r"_chosen\.csv$", r"_settings\.json$", r"_conv\.csv$", r"^provenance\.json$", r"^provenance_log\.jsonl$"),
}


# ---------------------------------------------------------------- 道具

def corp(g):
    return GROUP_SPEC[g][0]


def strip_seed(p):
    return re.sub(r"s\d+$", "", p)


def seed_of(method, p):
    """種の番号．E78 の手法は param の末尾 `s<種>`．E77 の entmin・im は種 0 の param に末尾の s が無い（`-`・`a0.1`）ので 0 と読む（R3-2）。"""
    if method not in SEEDED:
        return -1
    m = re.search(r"s(\d+)$", p)
    if m:
        return int(m.group(1))
    return 0 if method in NC_SEEDED else -1


def role_of(run, student):
    if run not in WORLDS:
        return "kind"
    return "fused" if student == run else "uni"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def majority(n):
    return n // 2 + 1


def install_guard(stage, roots):
    """入力の場所の中で ALLOW[stage] に無いファイルを開こうとしたら PermissionError で止める．入力の場所への書き込みは全段で止める。
    段 (a)・(b) では data/（ラベルの正本と特徴）も開かない（関門 1 の L-9）。"""
    roots = [os.path.realpath(str(r)) for r in roots]
    pats = [re.compile(p) for p in ALLOW.get(stage, ())]
    read_guard = stage in ALLOW
    data_root = os.path.realpath(str(ROOT / "data"))

    def hook(event, args):
        if event != "open":
            return
        p = args[0]
        if not isinstance(p, (str, bytes, os.PathLike)):
            return
        p = os.path.realpath(os.fsdecode(p))
        if read_guard and (p == data_root or p.startswith(data_root + os.sep)):
            raise PermissionError(f"--stage {stage} は data/ を開かない：{p}")
        if not any(p == r or p.startswith(r + os.sep) for r in roots):
            return
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if any(c in mode for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError(f"入力の場所に書こうとした：{p}")
        if read_guard and not any(pt.search(os.path.basename(p)) for pt in pats):
            raise PermissionError(f"--stage {stage} は {p} を開かない（評価のターゲットの出力か，この段に要らないファイル）")
    sys.addaudithook(hook)


def guard_selftest(stage, dirs):
    """禁じたファイル（評価のターゲットの `<T>.csv` などがあればそれ，無ければ存在しない名前）を開けないことを確かめる。"""
    probe = None
    for d in dirs:
        for f in sorted(Path(d).glob("*.csv")):
            if not any(re.search(p, f.name) for p in ALLOW[stage]):
                probe = f
                break
        if probe:
            break
    probe = probe or Path(dirs[0]) / "__guard_probe__.csv"
    for pr in (probe, ROOT / "data" / "README.md"):                            # data/ も開けないこと（関門 1 の L-9）
        try:
            open(pr, "rb").close()
        except PermissionError:
            continue
        except FileNotFoundError:
            raise SystemExit(f"見張りが働いていない（{pr} を止めずに開こうとした）")
        raise SystemExit(f"見張りが働いていない（{pr} を開けた）")
    return str(probe)


def prepare_out(out, stage):
    d = Path(out) / stage
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"{d} は既にある．上書きしない（別の --out を指定するか，消してから流す）")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_prov(d, stage, args, extra):
    prov = {"script": "exp/analyze_e79.py", "experiment": "E79", "stage": stage, "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "dirty_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
            "script_sha256": sha256(Path(__file__)), "started": args._t0_str, "seconds": round(time.time() - args._t0, 1),
            "consts": CONSTS, "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "allow_incomplete": args.allow_incomplete, "skip_commit_check": args.skip_commit_check, "allow_dirty": args.allow_dirty,
            "families": getattr(args, "families", "all"), "e80c_dirs": {k: str(v) for k, v in E80C.items()}, **extra}
    (d / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))


def check_dirty(args):
    """exp/ か measure/ に未コミットの変更があれば止める（コードの版を結果から特定できるように．関門 1 の C-7）。"""
    bad = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if re.search(r"\s(exp|measure)/", ln)]
    if bad and not args.allow_dirty:
        raise SystemExit(f"exp/ か measure/ に未コミットの変更がある：{bad}．コミットしてから流す（試しは --allow-dirty）")
    return bad


def read_prov(stage, args):
    f = Path(args.out) / stage / "provenance.json"
    if not f.exists():
        raise SystemExit(f"{f} が無い．前の段を先に流す")
    return json.loads(f.read_text())


def check_chain(stage, args):
    """段の間の照合（関門 1 の L-2・L-3・I12-2）．外れたら止める．--skip-commit-check はコミットの確認だけを飛ばす。"""
    me = sha256(Path(__file__))
    out = {}
    dpath, cpath, fpath = (Path(args.out) / "delta" / "delta.csv", Path(args.out) / "select" / "chosen.csv", Path(args.out) / "select" / "fixed.csv")
    out["delta"] = require_committed(dpath, "δ", args)
    prev = ["delta"] + (["select"] if stage in ("measure", "judge") else []) + (["measure"] if stage == "judge" else [])
    for st in prev:
        p = read_prov(st, args)
        if st != "delta" and p.get("families", "all") != args.families:      # --families は select・measure・judge で同じ（P4 だけの出力を全部の族の段に渡さない）
            raise SystemExit(f"{st} の段は --families {p.get('families', 'all')} で流されている（今は {args.families}）．同じ値で流し直す")
        if p.get("script_sha256") != me:
            raise SystemExit(f"{st} の段の台本の版（{p.get('script_sha256', '')[:16]}）が今の台本（{me[:16]}）と違う．同じ版で流し直す")
        if not args.skip_commit_check and (p.get("skip_commit_check") or (p.get("delta") or {}).get("skipped")):
            raise SystemExit(f"{st} の段はコミットの確認を飛ばして流されている．確認してから流し直す")
    if stage in ("measure", "judge"):
        out["chosen"] = require_committed(cpath, "選んだ設定（chosen.csv）", args)
        out["fixed"] = require_committed(fpath, "固定版の行（fixed.csv）", args)
        ps = read_prov("select", args)
        if (ps.get("delta") or {}).get("sha16") != out["delta"]["sha16"]:
            raise SystemExit("select の段が使った δ と今の δ が違う")
        if ps.get("chosen_csv_sha256") != sha256(cpath) or ps.get("fixed_csv_sha256") != sha256(fpath):
            raise SystemExit("select の段が書いた chosen.csv・fixed.csv と今のファイルが違う")
    if stage == "judge":
        pm = read_prov("measure", args)
        if (pm.get("delta") or {}).get("sha16") != out["delta"]["sha16"]:
            raise SystemExit("measure の段が使った δ と今の δ が違う")
        if pm.get("select_chosen_sha256") != sha256(cpath) or pm.get("select_fixed_sha256") != sha256(fpath):
            raise SystemExit("measure の段が使った chosen.csv・fixed.csv と今のファイルが違う")
    out["script_sha256"] = me
    return out


def require_committed(path, what, args):
    """`path` の sha256 の先頭 16 文字が，HEAD の docs/ か DAI_paper/ のどれかの追跡しているファイルに書かれているかを確かめる。"""
    if not Path(path).exists():
        raise SystemExit(f"{path} が無い．前の段を先に流す")
    h = sha256(path)[:16]
    hit = git("grep", "-l", "-F", h, "HEAD", "--", "docs", "DAI_paper")
    if not hit:
        if args.skip_commit_check:
            return {"sha16": h, "committed": False, "skipped": True}
        raise SystemExit(f"{what}（{path}，sha256 の先頭 {h}）がコミットされた文書に無い．値を docs/ に書いてコミットしてから流す")
    return {"sha16": h, "committed": True, "files": hit.splitlines()}


def parse_dirs(args):
    """名前 → (E77 の場所, E78 の場所)．指定が無ければ results/e77_<名前>・results/e78_<名前>．
    E80 の段 c の場所（世界の実行だけ．既定 results/e80c_<名前>．無ければ読まない）は大域の E80C に入れる（波の名前 e80c）。"""
    e77 = dict(x.split("=", 1) for x in (args.e77 or []))
    e78 = dict(x.split("=", 1) for x in (args.e78 or []))
    e80c = dict(x.split("=", 1) for x in (args.e80c or []))
    out = {}
    E80C.clear()
    for r in args.runs:
        a = Path(e77.get(r, ROOT / "results" / f"e77_{r}"))
        b = Path(e78.get(r, ROOT / "results" / f"e78_{r}"))
        out[r] = (a, b if b.exists() else None)
        # 既定の results/e80c_<名前> は E77 の場所も既定のときだけ読む（試走で --e77 を替えたときに本番の段 c の行を混ぜない）
        c = Path(e80c[r]) if r in e80c else (ROOT / "results" / f"e80c_{r}" if r not in e77 else None)
        if r in e80c and not c.exists():
            raise SystemExit(f"--e80c の場所 {c} が無い")
        if r in WORLDS and c is not None and c.exists():
            E80C[r] = c
    return out


def wave_dir(w, d77, d78, run):
    """波の名前 → 出力の場所（e77・e78・e80c）。"""
    return {"e77": d77, "e78": d78, "e80c": E80C.get(run)}[w]


def check_overlap(a, b, keys, what):
    """E77 の行と E80 の段 c の行で同じ鍵が重なったら止める（段 c は E77 に無い教師だけを足す）。"""
    if a is None or b is None or a.empty or b.empty:
        return
    ka = set(map(tuple, a[keys].astype(str).drop_duplicates().itertuples(index=False)))
    kb = set(map(tuple, b[keys].astype(str).drop_duplicates().itertuples(index=False)))
    both = ka & kb
    if both:
        raise SystemExit(f"{what}：E77 と E80 の段 c の行で同じ鍵が {len(both)} 個重なる（例 {sorted(both)[:3]}）．段 c の出力を確かめる")


def check_run_name(run, d, wave, issues, args=None):
    """provenance の生徒の名前と，解いた K（E77 は `KS`，E78 は `ks`）が KS と同じかを確かめる．K は provenance.json と，
    provenance_log.jsonl のうち同じコミットの全部の起動の記録で見る（同じ場所に 2 つの流れが書くと provenance.json は後の起動だけを持つ．I12-4）。"""
    f = Path(d) / "provenance.json"
    if not f.exists():
        issues.append({"check": "provenance", "run": run, "wave": wave, "note": "provenance.json が無い"})
        if args is not None:
            fail_or_note(args, issues, check="ks_unverified", run=run, wave=wave, note="provenance.json が無く K を確かめられない")
        return None
    p = json.loads(f.read_text())
    argv = p.get("argv", [])
    nm = argv[argv.index("--kind") + 1] if "--kind" in argv else (argv[argv.index("--world") + 1] if "--world" in argv else None)
    if nm != run:
        raise SystemExit(f"{d} の provenance の生徒の名前 {nm} が {run} と違う")
    kkey = "KS" if wave == "e77" else "ks"
    recs = [("provenance.json", p)]
    lg = Path(d) / "provenance_log.jsonl"
    if lg.exists():
        recs += [(f"provenance_log.jsonl:{i}", e) for i, e in enumerate(json.loads(x) for x in lg.read_text().splitlines() if x.strip())
                 if e.get("commit") == p.get("commit")]
    for where, e in recs:
        ks = [str(k) for k in (e.get(kkey) or [])]
        ok = ks == list(KS)
        issues.append({"check": "ks_of_provenance", "run": run, "wave": wave, "where": where, "ks": ks, "ok": ok})
        if not ok and args is not None:
            fail_or_note(args, issues, check="ks_mismatch", run=run, wave=wave, where=where, ks=ks, expected=list(KS))
    return p.get("commit")


def drop_other_k(df, args, issues, drop=True, **where):
    """K が KS に無い行を記録し，drop なら落とす（provenance の K の照合が外れた試走の出力のため．本番では provenance の照合で止まっている）．
    評価のターゲットの表は保存した配列と行の数を照合するので落とさない（判定は KS の鍵だけを読む）。"""
    if df.empty or "K" not in df.columns:
        return df
    bad = ~df.K.astype(str).isin(KS)
    if bad.any():
        fail_or_note(args, issues, check="rows_with_other_k", n=int(bad.sum()), ks=sorted(set(df.K[bad].astype(str))), **where)
        if drop:
            df = df[~bad]
    return df


def fail_or_note(args, issues, **row):
    issues.append(row)
    if not args.allow_incomplete:
        raise SystemExit(f"欠けか食い違い：{row}")


def read_rows(path):
    d = pd.read_csv(path, dtype={"param": str, "meff": str, "K": str, "student": str, "method": str, "pseudo": str, "target": str})
    if d.empty or "method" not in d.columns:
        return pd.DataFrame(columns=["pseudo", "target", "rep", "row_id", "student", "method", "param", "meff", "K", "uar", "commit"])
    return d


def present_uar(y, pred):
    """現れた感情だけで平均した UAR（全部の感情があれば `shift.uar` と同じ）。"""
    ks = np.unique(y)
    return float(np.mean([(pred[y == k] == k).mean() for k in ks]))


# ---------------------------------------------------------------- 段 (a)：δ

def copied_of(ps):
    """写した仮のターゲットの行（E77・E78 の --copy-pseudo．短縮 a）の写し元と写し元のコミット。"""
    if ps is None or ps.empty or "copied_from" not in ps.columns:
        return "", []
    src = sorted(set(ps.copied_from.dropna().astype(str)))
    sc = sorted(set(ps.src_commit.dropna().astype(str))) if "src_commit" in ps.columns else []
    return ";".join(src), sc


def stage_delta(args, dirs, out):
    issues, units, inputs, copied = [], [], {}, {}
    for run, (d77, _) in dirs.items():
        check_run_name(run, d77, "e77", issues, args)
        files = sorted(Path(d77).glob("*_ps_*_none.npz"))
        targets = [t for t in (args.targets or groups_for(run)) if t in groups_for(run)]
        for T in targets:
            S = [s for s in groups_for(run) if corp(s) != corp(T)]
            expect = [(c, r) for c in S if any(corp(s2) != corp(c) for s2 in S) for r in args.pseudo_reps]
            have = {}
            for f in files:
                m = re.match(r"^(?P<T>.+)_ps_(?P<c>.+)_r(?P<r>\d+)_none\.npz$", f.name)
                if m and m["T"] == T:
                    have[(m["c"], int(m["r"]))] = f
            for c, r in expect:
                if (c, r) not in have:
                    fail_or_note(args, issues, check="ps_missing", run=run, target=T, pseudo=c, rep=r)
            pfile = Path(d77) / f"{T}_pseudo.csv"
            ps = read_rows(pfile) if pfile.exists() else pd.DataFrame()
            if pfile.exists():
                inputs[str(pfile)] = sha256(pfile)
            src, src_commit = copied_of(ps)
            if src:
                copied[f"{run}|{T}"] = {"copied_from": src, "src_commit": src_commit}
            ps_none = ps[ps.method == "none"].set_index(["pseudo", "rep", "student", "K"]).uar if len(ps) else pd.Series(dtype=float)
            if ps_none.index.duplicated().any():
                raise SystemExit(f"{pfile}：none の行が重複している")
            for (c, r), f in sorted(have.items()):
                inputs[str(f)] = sha256(f)
                z = np.load(f)
                y = np.asarray(z["y"]).astype(int)
                for i, key in enumerate(z["keys"]):
                    st, K = str(key).split("|")
                    if K not in KS:                                            # 判定に使わない K（本番では provenance の照合で止まっている）
                        fail_or_note(args, issues, check="ps_npz_other_k", run=run, target=T, pseudo=c, rep=r, student=st, K=K)
                        continue
                    lp = np.asarray(z[f"lp{i}"], dtype=np.float64)
                    if not np.isfinite(lp).all():
                        raise SystemExit(f"NaN/Inf：{f} {key}")
                    pred = lp.argmax(1)
                    u = shift_uar(y, pred)
                    a = float(np.mean([ba(dd, 0.0) for dd, _ in d_o(lp, y)]))
                    ref = ps_none.get((c, r, st, K), np.nan) if len(ps_none) else np.nan
                    if np.isfinite(ref) and abs(ref - u) > 1e-9:
                        fail_or_note(args, issues, check="ps_uar_vs_pseudo_csv", run=run, target=T, pseudo=c, rep=r, student=st, K=K, absdiff=abs(ref - u))
                    units.append({"run": run, "student": st, "role": role_of(run, st), "target": T, "pseudo": c, "rep": r, "K": K,
                                  "uar": 100 * u, "a_d0": 100 * a, "uar_csv_found": bool(np.isfinite(ref)), "copied_from": src})
    U = pd.DataFrame(units)
    if U.empty:
        raise SystemExit("δ の材料（_ps_ のファイル）が 1 つも無い")
    U.to_csv(out / "delta_reps.csv", index=False)
    g = U.groupby(["run", "student", "role", "target", "pseudo", "K", "copied_from"])
    pu = g.agg(n_reps=("rep", "nunique"), sd_u=("uar", lambda x: x.std(ddof=1)), sd_a=("a_d0", lambda x: x.std(ddof=1))).reset_index()
    pu["s_u"], pu["s_d"] = pu.sd_u / np.sqrt(10), pu.sd_a / np.sqrt(10)
    pu.to_csv(out / "delta_units.csv", index=False)
    full = pu[pu.n_reps >= 2]
    D = full.groupby(["run", "student", "role", "K"]).agg(n_units=("s_u", "size"), n_targets=("target", "nunique"),
                                                          s_u=("s_u", "median"), s_d=("s_d", "median")).reset_index()
    D["delta_u"], D["delta_d"] = 2 * D.s_u, 2 * D.s_d
    # 感度の列（関門 1 の S7）：仮のターゲットのコーパスごとに中央値を取ってから，コーパスの間で中央値を取った δ．判定には使わない
    if len(full):
        full = full.assign(pcorp=full.pseudo.map(corp))
        pc = full.groupby(["run", "student", "K", "pcorp"])[["s_u", "s_d"]].median().groupby(["run", "student", "K"]).median()
        cn = full.groupby(["run", "student", "K", "pcorp"]).size().reset_index(name="n")
        cnt = pd.DataFrame([{"run": a, "student": b, "K": c, "pairs_by_corpus": ";".join(f"{x}:{y}" for x, y in zip(g.pcorp, g.n))}
                            for (a, b, c), g in cn.groupby(["run", "student", "K"])])
        D = D.merge((2 * pc).rename(columns={"s_u": "delta_u_corpus", "s_d": "delta_d_corpus"}).reset_index(), how="left").merge(cnt, how="left")
    else:
        D = D.assign(delta_u_corpus=np.nan, delta_d_corpus=np.nan, pairs_by_corpus="")
    D["K"] = D.K.astype(str)
    D = D.assign(_k=D.K.map(KS.index)).sort_values(["run", "student", "_k"]).drop(columns="_k")
    D.to_csv(out / "delta.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "checks.csv", index=False)
    h = sha256(out / "delta.csv")
    write_prov(out, "delta", args, {"inputs_sha256": inputs, "n_issues": len(issues), "delta_csv_sha256": h, "copied_pseudo": copied,
                                    "rule": "s = 反復 0〜4 の標準偏差（ddof=1）/ √10．δ = 2 × (run, student, K) ごとの (評価のターゲット, 仮のターゲット) の中央値．単位はポイント．"
                                            "delta_*_corpus は仮のターゲットのコーパスごとの中央値のコーパスの間の中央値（感度の列）．"
                                            "copied_pseudo の評価のターゲット（IEMOCAP 台本）の仮のターゲットは写し元（即興）の写しで，"
                                            "中央値はその単位の同じ値を 2 回数える（短縮の前も即興と台本の仮のターゲットの出力はビット単位で同じだった）"})
    print(f"δ：{len(D)} 行（run × 生徒 × K），材料 {len(U)} 行，欠けと食い違い {len(issues)} 件", flush=True)
    print(f"delta.csv の sha256 の先頭 16 文字：{h[:16]}（これを文書に書いてコミットしてから --stage select に進む）", flush=True)


# ---------------------------------------------------------------- 段 (b)：選択

def e80c_teachers(df, wave="e80c"):
    """E80 の段 c の行がある教師（dist の教師，kd・kdh の教師）．ふるわない版（oursx_noJ・kd_noJ）に足す。"""
    x = df[df.wave == wave] if len(df) else df
    if x.empty:
        return (), ()
    d_ = tuple(sorted(set(x[x.method == "dist"].pnorm.str.split(":").str[0])))
    k_ = tuple(sorted(set(x[x.method.isin(["kd", "kdh"])].pnorm.str.replace(r"T\d+$", "", regex=True))))
    return d_, k_


def sel_defs(role, kept=(), noj_new=((), ())):
    """選ぶ族 → [(波, 判定の関数(method, pnorm, meff) → bool)]．ref_*・teacher の行はどの族にも入らない．共正則化は選ばない（短縮 b．測る段で
    テストで選んだ上限にする）．oursx は E80 の R4：kept（その (実行, 評価のターゲット, 生徒, K) で R3′〜R3″ が残した教師）× 重みの付け方 5 × 総重み 4．
    KD（kd と kdh）の教師も kept と同じ集合（E80 の C1）．今の ours（pF・poe0 の bal）は除去実験の 1 行（R4）．R6′ の部分集合は stage_select で作る。"""
    uni, world = role == "uni", role != "kind"
    ours_ref = "pF" if uni else "poe0"
    kept = tuple(kept)

    def dist(refs, wgts=WGTS, meffs=MEFFS):
        """dist の行（E77 と，E77 に無い教師を足した E80 の段 c の行．同じ鍵の行は読み込みで止めている）。"""
        fn = (lambda m, p, f: m == "dist" and p.split(":")[0] in refs and p.split(":")[1] in wgts and f in meffs)
        return [("e77", fn), ("e80c", fn)]

    def kdf(refs):
        fn = (lambda m, p, f: m in ("kd", "kdh") and re.sub(r"T\d+$", "", p) in refs)
        return [("e77", fn), ("e80c", fn)]
    D = {
        "entmin": [("e77", lambda m, p, f: m == "entmin")],
        "self": [("e77", lambda m, p, f: m == "self")],
        "im": [("e77", lambda m, p, f: m == "im" and f in MEFFS)],
        "t3a": [("e77", lambda m, p, f: m == "t3a")],
        "t3a_b": [("e77", lambda m, p, f: m == "t3a_b")],
        "lp": [("e77", lambda m, p, f: m == "lp")],
        "coral": [("e78", lambda m, p, f: m == "coral")],                      # E78 が単位ごとに足した E39 の C′・C_b の点も候補に入れる（関門 1 の C-1）
        "coralctl": [("e78", lambda m, p, f: m == "coralctl")],
    }
    if role != "fused":                                                       # 融合の生徒では DANN・MME を回さない（E78）
        for m_ in ("dann", "mme", "mapctl"):
            D[m_] = [("e78", (lambda mm: lambda m, p, f: m == mm)(m_))]
    D["groupB"] = [x for k in GROUP_B if k in D for x in D[k]]
    if world:
        D["kd"] = kdf(kept)                                                   # E80 の C1：提案と同じ教師
        D["oursx"] = dist(kept)                                               # E80 の R4
        # ふるわない版：その（T, 生徒, K）で行がある教師の全部（E77 の教師に，E80 の段 c で計算した教師を足したもの．本体のセッションの決定）．
        # 段 c は kept で残した新しい教師しか計算しないので，残さなかった新しい教師は入らない
        noj_d = tuple(NOJ_DIST["uni" if uni else "fused"]) + tuple(noj_new[0])
        noj_k = tuple(NOJ_KD["uni" if uni else "fused"]) + tuple(noj_new[1])
        D["oursx_noJ"] = dist(noj_d)                                        # ふるわない版（J でふるう前の，E77 に dist の行がある全部の教師．比較の検査役の追記の指摘 1）
        D["kd_noJ"] = kdf(noj_k)                                              # 前の設計の KD と同じ教師
        D["ours"] = [("e77", lambda m, p, f: m == "dist" and p == f"{ours_ref}:bal")]   # 除去実験の行（R4）
        if uni:
            D["abl_refa"], D["abl_refb"], D["abl_pF"] = dist(("pFK",)), dist(("ref",)), dist(("pF",))
        else:
            D["abl_ref0"], D["abl_poe0"] = dist(("ref0",)), dist(("poe0",))
        D["abl_bal"], D["abl_raw"] = dist(kept, ("bal",)), dist(kept, ("raw",))
        D["abl_hard"] = dist(kept, ("hard0.5", "hard0.7", "hard0.9"))
        # m_eff ＝ 500 に固定する除去の行は置かない：除去実験の行の総重みは行ごとに仮のターゲットで選ぶ（R5）
        D["abl_own"] = [("e77", lambda m, p, f: m == "own")]
        D["abl_ownK"] = dist(("ownK",))
        if uni:
            D["xmuda"] = [("e78", lambda m, p, f: m == "xmuda")]
            D["cotrain"] = [("e77", lambda m, p, f: m == "cotrain")]
        else:
            D["cotrain"] = [("e77", lambda m, p, f: m == "cotrain_fused")]
            D["cotrain_prod"] = [("e77", lambda m, p, f: m == "cotrain_prod")]
    return D


def cand_mask(df, defs):
    m = np.zeros(len(df), bool)
    for wave, fn in defs:
        w = (df.wave == wave).to_numpy()
        if w.any():
            sub = df[w]
            m[w] |= np.array([bool(fn(a, b, c)) for a, b, c in zip(sub.method, sub.pnorm, sub.meff)], bool)
    return m


def choose(cs, none_score):
    """cs：候補ごとの (wave, method, pnorm, meff, score)．同点は 'method|pnorm|meff' の文字列の順で先頭（E78 の choose_seeded と同じ）。
    none を含む版は，最良の候補の値が none の値より大きいときだけ候補を選ぶ（同点は none）。"""
    if cs.empty:
        return None, None, 0
    cs = cs.assign(_k=cs.method + "|" + cs.pnorm + "|" + cs.meff).sort_values("_k")
    best = cs.score.max()
    tie = int((cs.score == best).sum())
    ex = cs[cs.score == best].iloc[0]
    inc = ex if best > none_score else None
    return ex, inc, tie


E80C_METHODS = ("dist", "kd", "kdh", "teacher")                                  # E80 の段 c から読む手法（E77 に無い教師の行．ほかの行は照合だけに使う）


def load_pseudo(run, d77, d78, args=None, issues=None):
    parts = []
    for wave, d in (("e77", d77), ("e78", d78), ("e80c", E80C.get(run))):
        if d is None:
            continue
        for f in sorted(Path(d).glob("*_pseudo.csv")):
            T = f.name[: -len("_pseudo.csv")]
            x = read_rows(f)
            if x.empty:
                continue
            if args is not None:
                x = drop_other_k(x, args, issues, run=run, target=T, wave=wave, where="pseudo")
            if "copied_from" not in x.columns:
                x = x.assign(copied_from=np.nan)
            parts.append(x.assign(wave=wave, eval_target=T))
    if not parts:
        return pd.DataFrame()
    P = pd.concat(parts, ignore_index=True)
    if (P.wave == "e80c").any():                                              # E80 の段 c の行：none は E77 の none と照合し，dist・kd・kdh・teacher だけを足す
        kk_ = ["eval_target", "pseudo", "rep", "student", "K"]
        c_ = P[(P.wave == "e80c") & (P.method == "none")].set_index(kk_).uar
        e_ = P[(P.wave == "e77") & (P.method == "none")].set_index(kk_).uar
        if len(c_):
            j_ = pd.concat([c_.rename("c"), e_.rename("e")], axis=1, join="inner")
            dmax = float((j_.c - j_.e).abs().max()) if len(j_) else np.nan
            if issues is not None:
                issues.append({"check": "none_e77_vs_e80c_pseudo", "run": run, "absdiff_max": dmax, "n": int(len(j_)), "n_e80c_none": int(len(c_))})
            if not (dmax <= TOL_UAR) and args is not None:
                fail_or_note(args, issues, check="none_e77_vs_e80c_pseudo_over_tol", run=run, absdiff_max=dmax)
            if len(j_) < len(c_) and args is not None:                        # 段 c の none に E77 の相手が無い行がある（照合できない行を残さない．追記 4）
                fail_or_note(args, issues, check="none_e77_vs_e80c_pseudo_unmatched", run=run, n=int(len(j_)), n_e80c_none=int(len(c_)))
        other = (P.wave == "e80c") & ~P.method.isin(E80C_METHODS)
        if other.any() and issues is not None:
            issues.append({"check": "e80c_rows_not_used", "run": run, "n": int(other.sum()), "methods": sorted(set(P.method[other]))})
        P = P[~other]
        check_overlap(P[P.wave == "e77"], P[P.wave == "e80c"], ["eval_target", "pseudo", "rep", "student", "method", "param", "meff", "K"],
                      f"{run} の仮のターゲット")
    P["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(P.method, P.param)]
    P["seed"] = [seed_of(m, p) for m, p in zip(P.method, P.param)]
    co = P.method.isin(list(COREG.values()))                                  # 共正則化は仮のターゲットで選ばない（短縮 b）．行があっても使わない
    if co.any() and issues is not None:
        issues.append({"check": "coreg_pseudo_rows_ignored", "run": run, "n": int(co.sum())})
    return P[~co]


def kept_provenance(f):
    """kept.csv の由来（関門 1 の C6）：kept.csv の sha256 と大きさと時刻，作った台本（exp/e80_teacher_j.py）の今の sha256 と最後のコミットと dirty，
    kept.csv の隣にある provenance（あれば中身の commit・script_sha256 など）と j_units.csv（あれば sha256 と commit の列）。"""
    f = Path(f)
    scr = ROOT / "exp" / "e80_teacher_j.py"
    pv = {"path": str(f), "sha256": sha256(f), "bytes": f.stat().st_size, "mtime": f.stat().st_mtime, "script": "exp/e80_teacher_j.py",
          "script_sha256_now": sha256(scr) if scr.exists() else None, "script_last_commit": git("log", "-1", "--format=%H", "--", "exp/e80_teacher_j.py"),
          "script_dirty": bool(git("status", "--porcelain", "--", "exp/e80_teacher_j.py")),
          "note": "script_sha256_now・script_last_commit は E79 を流したときの台本．kept.csv を作ったときの版は sidecar（E80 が書く provenance）か j_units の commit で読む"}
    side = {}
    for nm in (f.stem + "_provenance.json", f.name + ".provenance.json", "provenance.json"):
        g = f.parent / nm
        if g.exists():
            try:
                d = json.loads(g.read_text())
            except json.JSONDecodeError:
                continue
            side[nm] = d                                                      # E80 の kept_provenance.json は全部写す（script・script_sha256・commit・dirty・rule・inputs・unit_commits・kept_sha256・made）
            ks = d.get("kept_sha256")
            if ks is not None and ks != pv["sha256"]:
                raise SystemExit(f"{g} の kept_sha256（{str(ks)[:16]}）が {f} の sha256（{pv['sha256'][:16]}）と違う．kept.csv と由来の記録をそろえる")
    pv["sidecar"] = side
    if side:
        d0 = next(iter(side.values()))
        pv.update({"made_by_script_sha256": d0.get("script_sha256"), "made_by_commit": d0.get("commit"), "made_by_dirty": d0.get("dirty"),
                   "rule": d0.get("rule"), "inputs": d0.get("inputs"), "unit_commits": d0.get("unit_commits"), "made": d0.get("made")})
    j = f.parent / "j_units.csv"
    if j.exists():
        try:
            c = pd.read_csv(j, usecols=["commit"]).commit.astype(str).unique().tolist()
        except ValueError:
            c = []
        pv["j_units"] = {"path": str(j), "sha256": sha256(j), "commits": c}
    return pv


def load_kept(args, dirs, issues):
    """E80 の kept.csv（R3′〜R3‴．（世界, 評価のターゲット, 生徒, 教師, K）ごとの keep）を読み，(実行, T, 生徒, K) → 残した教師（名前の順）と，
    (実行, T, 生徒, K) → {教師: keep}（D8 の表に使う）と，由来の記録を返す．IEMOCAP 台本の行が無いときは即興の行を使う（E80 は台本を即興の単位で測る．記録する）．
    world の実行が無いか --families p4 なら読まない。"""
    worlds = [r for r in dirs if r in WORLDS]
    if not worlds or args.families == "p4":
        return {}, None, {}, None
    f = Path(args.e80_kept)
    if not f.exists():
        if not args.allow_incomplete:
            raise SystemExit(f"E80 の kept.csv（{f}）が無い．E80 の段 b（R3′〜R3‴）が終わってから流す（R7）．P4 だけなら --families p4")
        issues.append({"check": "kept_missing", "path": str(f)})
        return {}, None, {}, None
    k = pd.read_csv(f, dtype={"K": str, "world": str, "target": str, "student": str, "teacher": str})
    need = {"world", "target", "student", "teacher", "K", "keep"}
    if not need <= set(k.columns):
        raise SystemExit(f"{f} に列 {sorted(need - set(k.columns))} が無い（E80 の書き直しの後の形：評価のターゲットごと）")
    if k.duplicated(["world", "target", "student", "teacher", "K"]).any():
        raise SystemExit(f"{f} に（世界, 評価のターゲット, 生徒, 教師, K）の重複がある")
    k["keep"] = k.keep.map(lambda v: v is True or str(v).strip().lower() in ("true", "1"))
    out, allk = {}, {}
    for (w, T, st, K), g in k.groupby(["world", "target", "student", "K"]):
        out[(w, T, st, K)] = tuple(sorted(g.teacher[g.keep]))
        allk[(w, T, st, K)] = dict(zip(g.teacher, g.keep))
    for (w, T, st, K) in list(out):                                            # 台本は即興の単位（E80 の段 a の SKIP_T）
        if T == "IEMOCAP-impro" and (w, "IEMOCAP-script", st, K) not in out:
            out[(w, "IEMOCAP-script", st, K)] = out[(w, T, st, K)]
            allk[(w, "IEMOCAP-script", st, K)] = allk[(w, T, st, K)]
            issues.append({"check": "kept_script_from_impro", "run": w, "student": st, "K": K})
    pv = kept_provenance(f)
    return out, pv["sha256"], allk, pv


def check_kept(args, run, T, PT, kept, issues):
    """kept.csv が T の (生徒, K) を全部持つか．残した教師の dist と KD（kd・kdh）の行が E77 に無ければ止める（E80 の段 c が要る）。"""
    if run not in WORLDS:
        return
    have = {}
    for st, g in PT[PT.wave.isin(["e77", "e80c"])].groupby("student"):
        d_ = set(g[g.method == "dist"].param.str.split(":").str[0])
        k_ = {meth: set(g[g.method == meth].param.str.replace(r"T\d+$", "", regex=True)) for meth in ("kd", "kdh")}
        have[st] = d_ & k_["kd"] & k_["kdh"]
    for st in sorted(set(PT.student)):
        for K in KS:
            if (run, T, st, K) not in kept:
                fail_or_note(args, issues, check="kept_unit_missing", run=run, target=T, student=st, K=K)
                continue
            miss = sorted(set(kept[(run, T, st, K)]) - have.get(st, set()))
            if miss:
                raise SystemExit(f"E80 の段 c が要る：{run} {T} の生徒 {st}（K={K}）で残した教師 {miss} の dist か KD（kd・kdh）の行が E77 に無い"
                                 "（E77 の参照は，dist が単モダリティ pFK・ref・pF，融合 ref・ref0・poe0，KD が単モダリティ pF・pFK・ref，融合 poe0・ref）．"
                                 "段 c で dist と KD の行を足してから流す")


def d8_rows(run, T, st, role, K, PK, kept_all_u, kept_u):
    """E80 の D8 の表（記述．R3″ は変えない）：E77 に dist か KD の行がある教師ごとに，kept で残したかと，その教師の dist（重みの付け方 × 総重み）と
    KD（kd・kdh × 温度 × 総重み）の最良の設定の仮のターゲットの UAR（仮のターゲット × 反復の単純平均．全部の単位に値のある設定だけ）− none の UAR。
    R3″ と同じ仮のターゲットのデータで 2 回選んだ結果どうしの一致なので，一致は高めに出る（表の注に書く）。"""
    none = PK[(PK.method == "none") & (PK.wave == "e77")]
    n_units = none.groupby(["pseudo", "rep"]).ngroups
    nv = float(none.uar.mean()) if len(none) else np.nan
    x = PK[PK.wave.isin(["e77", "e80c"])]
    dist = x[x.method == "dist"].assign(t=lambda d: d.pnorm.str.split(":").str[0])
    kd = x[x.method.isin(["kd", "kdh"])].assign(t=lambda d: d.pnorm.str.replace(r"T\d+$", "", regex=True))
    out = []

    def best(d):
        if d.empty:
            return np.nan, None
        g = d.groupby(["method", "pnorm", "meff"]).uar.agg(["mean", "size"])
        g = g[g["size"] == n_units]
        return (float(g["mean"].max()), "|".join(g["mean"].idxmax())) if len(g) else (np.nan, None)
    new_t = set(dist[dist.wave == "e80c"].t) | set(kd[kd.wave == "e80c"].t)    # E80 の段 c で足した教師（残した教師だけ．残さなかった新しい教師は測らない）
    for t in sorted(set(NOJ_DIST[role]) | set(NOJ_KD[role]) | new_t):
        bd, sd = best(dist[dist.t == t])
        bk, sk = best(kd[kd.t == t])
        keep = t in kept_u
        row = {"run": run, "target": T, "student": st, "role": role, "K": K, "teacher": t, "source": "e80c" if t in new_t else "e77",
               "in_kept_csv": bool(kept_all_u is not None and t in kept_all_u), "keep": keep, "none_pseudo_uar": 100 * nv,
               "dist_best_minus_none": 100 * (bd - nv), "dist_best": sd, "kd_best_minus_none": 100 * (bk - nv), "kd_best": sk}
        row["dist_beats_none"] = bool(np.isfinite(bd) and bd > nv) if np.isfinite(bd) else np.nan
        row["kd_beats_none"] = bool(np.isfinite(bk) and bk > nv) if np.isfinite(bk) else np.nan
        row["agree_dist"] = (keep == row["dist_beats_none"]) if isinstance(row["dist_beats_none"], bool) else np.nan
        row["agree_kd"] = (keep == row["kd_beats_none"]) if isinstance(row["kd_beats_none"], bool) else np.nan
        out.append(row)
    return out


def r6_subsets(cands, n_kd, run, T, st, K):
    """R6′：oursx の候補（none を除く．'method|pnorm|meff' の順に並べる）から，KD の族と同じ数 n_kd を種 0〜19 で一様に（重複なく）引く．
    データを見ずに決まる（乱数は種と単位の名前だけで決める）．n_kd が候補の数以上なら全部。"""
    cands = sorted(cands)
    out = []
    for s in range(R6_DRAWS):
        if n_kd >= len(cands):
            out.append(frozenset(cands))
            continue
        rng = np.random.default_rng([s, zlib.crc32(f"{run}|{T}|{st}|{K}".encode())])
        out.append(frozenset(cands[i] for i in rng.choice(len(cands), size=n_kd, replace=False)))
    return out


def coral_pseudo_units(d78, T, student):
    """仮のターゲットの（仮のターゲット, 反復）のうち，CORAL の r ごとの行が作られた単位（`_conv.csv` の what の列だけを読む）。"""
    f = Path(d78) / f"{T}_conv.csv" if d78 else None
    if f is None or not f.exists():
        return None
    c = pd.read_csv(f, usecols=lambda c_: c_ in ("pseudo", "rep", "student", "what"), dtype=str)
    if "pseudo" not in c.columns:
        return {r: set() for r in (32, 64, 128)}
    c = c[c.pseudo.notna() & (c.student == student)]
    return {r: {(p_, int(x)) for p_, x in zip(c[c.what == f"coral_r{r}"].pseudo, c[c.what == f"coral_r{r}"].rep)} for r in (32, 64, 128)}


def coral_eval_ok(d78, T, student, reps):
    """評価のターゲットの全反復で coral の r ごとの行が作られたか（`_conv.csv` の what の列だけを読む．値は読まない）。"""
    f = Path(d78) / f"{T}_conv.csv" if d78 else None
    if f is None or not f.exists():
        return None
    c = pd.read_csv(f, usecols=lambda c_: c_ in ("target", "rep", "student", "what"), dtype=str)
    if "target" not in c.columns:
        return None
    c = c[c.target.notna() & (c.student == student)]
    out = {}
    for r in (32, 64, 128):
        out[r] = set(reps) <= {int(x) for x in c[c.what == f"coral_r{r}"].rep}
    return out


def expected_pseudo(run, T, pseudo_reps):
    """delta と同じ期待：T と別のコーパスのソースのうち，仮のターゲットになれる群 × 反復。"""
    S = [s_ for s_ in groups_for(run) if corp(s_) != corp(T)]
    return {(c, r) for c in S if any(corp(s2) != corp(c) for s2 in S) for r in pseudo_reps}


def check_e80c_prov(run, args, issues, kept_sha):
    """E80 の段 c（exp/e80_stage_c.py）の provenance.json と，同じコミットの provenance_log.jsonl の記録について，解いた K（`KS`）が E79 の KS と，
    読んだ kept.csv の sha256（`kept_sha256`）が E79 の読んだ kept.csv と同じか，試しの教師（`trial`）で作っていないかを確かめる．違えば止まる．戻り：記録。"""
    d = E80C.get(run)
    if d is None:
        return None
    f = Path(d) / "provenance.json"
    if not f.exists():
        fail_or_note(args, issues, check="e80c_provenance_missing", run=run, path=str(f))
        return None
    p = json.loads(f.read_text())
    if p.get("world") not in (None, run):
        raise SystemExit(f"{f} の世界 {p.get('world')} が {run} と違う")
    recs = [("provenance.json", p)]
    lg = Path(d) / "provenance_log.jsonl"
    if lg.exists():
        recs += [(f"provenance_log.jsonl:{i}", e) for i, e in enumerate(json.loads(x) for x in lg.read_text().splitlines() if x.strip())
                 if e.get("commit") == p.get("commit")]
    for where, e in recs:
        ks = [str(k) for k in (e.get("KS") or [])]
        ksha = e.get("kept_sha256")
        issues.append({"check": "e80c_provenance", "run": run, "where": where, "ks": ks, "kept_sha16": str(ksha)[:16], "trial": e.get("trial")})
        if ks != list(KS):
            fail_or_note(args, issues, check="e80c_ks_mismatch", run=run, where=where, ks=ks, expected=list(KS))
        if kept_sha is not None and ksha != kept_sha:
            fail_or_note(args, issues, check="e80c_kept_sha_mismatch", run=run, where=where, e80c=str(ksha)[:16], e79=str(kept_sha)[:16])
        if e.get("trial"):
            fail_or_note(args, issues, check="e80c_trial_teachers", run=run, where=where)
    return {"path": str(d), "commit": p.get("commit"), "dirty": p.get("dirty"), "script_sha256": p.get("script_sha256"), "KS": p.get("KS"),
            "kept_sha256": p.get("kept_sha256"), "n_records_checked": len(recs)}


def stage_select(args, dirs, out):
    chain = check_chain("select", args)
    dmeta = chain["delta"]
    issues, rows, cand_rows, fixed, inputs, copied, r6rows = [], [], [], [], {}, {}, []
    kept, kept_sha, kept_all, kept_prov = load_kept(args, dirs, issues)
    kept_commit = require_committed(args.e80_kept, "E80 の kept.csv", args) if kept_sha else None   # E80 の節に sha256 を書いてコミットしてから（δ と同じ形）
    p4only = args.families == "p4"
    d8rows = []
    e80c_prov = {r: check_e80c_prov(r, args, issues, kept_sha) for r in dirs if r in E80C}   # 段 c の K と kept.csv の sha256 の照合

    def pick_one(ctx, sel, dd, PV):
        """1 つの族の選択（仮のターゲット × 反復の全行の UAR の単純平均の最大）．行を rows に足す。"""
        run, T, st, role, K, vname, none, n_units, none_score, ev, pu, chosen78, src = ctx
        m = cand_mask(PV, dd)
        C = PV[m]
        base = {"run": run, "target": T, "student": st, "role": role, "K": K, "selector": sel, "variant": vname, "none_score": none_score,
                "copied_from": src}
        if C.empty:
            no_kept = (sel in KEPT_FAMS or sel.startswith(R6_PREFIX)) and not kept.get((run, T, st, K))
            if no_kept and vname.endswith("incl"):                            # 残した教師が無い：候補は none だけなので none を選ぶ（R3・R4）
                rows.append({**base, "wave": "e77", "method": "none", "param": "-", "meff": "0", "score": none_score, "n_cand": 0, "n_dropped": 0,
                             "n_tie": 0, "chose_none": True})
            else:
                rows.append({**base, "wave": None, "method": EMPTY, "param": None, "meff": None, "score": np.nan, "n_cand": 0, "n_dropped": 0,
                             "n_tie": 0, "chose_none": np.nan})
            if vname == "incl" and no_kept:                                  # 残した教師が無い単位の候補は none だけ（R3‴）
                cand_rows.append({"run": run, "target": T, "student": st, "role": role, "K": K, "selector": sel, "n_cand_with_none": 1, "n_dropped": 0})
            if no_kept:
                issues.append({"check": "no_kept_teacher", "run": run, "target": T, "student": st, "K": K, "selector": sel, "variant": vname})
            elif not (sel == "lp" and K == "0"):                              # 定義から無いのはラベル伝播の K = 0 だけ（I12-1）
                fail_or_note(args, issues, check="selector_without_candidates", run=run, target=T, student=st, K=K, selector=sel, variant=vname)
            return
        kk = ["wave", "method", "pnorm", "meff"]
        sc = C.groupby(kk).uar.agg(score="mean", n="size").reset_index()
        nu = C.drop_duplicates(kk + ["pseudo", "rep"]).groupby(kk).size().rename("nu").reset_index()
        sc = sc.merge(nu)
        ok = (sc.nu == n_units) & (sc.n == n_units)                           # 選択は種 0 の行だけ（種 1・2 の行は仮のターゲットに無い）
        # 仮のターゲットの行が揃わない候補（I12-1）．ただし CORAL の候補で，欠けた（仮のターゲット, 反復）が E78 の `_conv.csv` で
        # その r の行が作られなかった単位（5r の条件で外れた単位）とちょうど一致するものは，設計どおり外すだけで止めない（関門 1 の再検査の I12-3）
        units_all = set(map(tuple, none[["pseudo", "rep"]].drop_duplicates().itertuples(index=False)))
        excused = np.zeros(len(sc), bool)
        for i in np.flatnonzero(~ok.to_numpy()):
            rr = sc.iloc[i]
            m_r = re.match(r"^r(\d+)rho", rr.pnorm) if rr.method == "coral" else None
            if m_r is None or pu is None:
                continue
            have = set(map(tuple, C[(C.wave == rr.wave) & (C.method == rr.method) & (C.pnorm == rr.pnorm) & (C.meff == rr.meff)][
                ["pseudo", "rep"]].drop_duplicates().itertuples(index=False)))
            want_missing = units_all - pu.get(int(m_r.group(1)), set())
            if want_missing and (units_all - have) == want_missing and rr.n == rr.nu:
                excused[i] = True
        n_incomplete = int((~ok.to_numpy() & ~excused).sum())
        if n_incomplete:
            fail_or_note(args, issues, check="incomplete_candidates", run=run, target=T, student=st, K=K, selector=sel, variant=vname, n=n_incomplete)
        if excused.any():
            issues.append({"check": "coral_r_pseudo_excluded", "run": run, "target": T, "student": st, "K": K, "selector": sel,
                           "variant": vname, "n": int(excused.sum())})
        # CORAL の r が 5r の条件で外れた単位：全部の仮のターゲットと評価のターゲットに値のある候補だけ．CORAL を含む全部の族に当てる（I4-1）
        is_coral = (sc.method == "coral").to_numpy()
        if is_coral.any():
            for r_ in (32, 64, 128):
                in_r = is_coral & sc.pnorm.str.startswith(f"r{r_}rho").to_numpy()
                if not in_r.any():
                    continue
                if ev is not None and not ev[r_]:
                    ok &= ~in_r
                elif ev is None and not args.allow_incomplete:
                    raise SystemExit(f"{run} {T}：E78 の _conv.csv が無く，CORAL の r の有無を確かめられない")
        dropped = int((~ok).sum())
        sc = sc[ok]
        ex, inc, tie = choose(sc, none_score)
        pick = ex if vname.endswith("excl") else inc
        base.update({"n_cand": int(len(sc)), "n_dropped": dropped, "n_tie": tie})
        if pick is None and ex is None:
            rows.append({**base, "wave": None, "method": EMPTY, "param": None, "meff": None, "score": np.nan, "chose_none": np.nan})
        elif pick is None:
            rows.append({**base, "wave": "e77", "method": "none", "param": "-", "meff": "0", "score": none_score, "chose_none": True})
        else:
            rows.append({**base, "wave": pick.wave, "method": pick.method, "param": pick.pnorm, "meff": pick.meff,
                         "score": float(pick.score), "chose_none": False if vname.endswith("incl") else np.nan})
        if vname == "incl":
            cand_rows.append({"run": run, "target": T, "student": st, "role": role, "K": K, "selector": sel,
                              "n_cand_with_none": int(len(sc)) + 1, "n_dropped": dropped})
        # E78 の choose_seeded と同じ選択か（I14）
        if vname == "excl" and sel in ("dann", "mme", "mapctl", "xmuda") and chosen78 is not None and ex is not None:
            c78 = chosen78[(chosen78.student == st) & (chosen78.method == sel) & (chosen78.K == K)]
            same = len(c78) == 1 and c78.param.iloc[0] == ex.pnorm
            issues.append({"check": "choose_seeded_match", "run": run, "target": T, "student": st, "K": K, "method": sel, "ok": same})
            if not same and not args.allow_incomplete:
                raise SystemExit(f"E78 の _chosen.csv と選択が違う：{run} {T} {st} {sel} K={K}")

    for run, (d77, d78) in dirs.items():
        check_run_name(run, d77, "e77", issues, args)
        if d78 is not None:
            check_run_name(run, d78, "e78", issues, args)
        else:
            fail_or_note(args, issues, check="e78_missing", run=run)
        P = load_pseudo(run, d77, d78, args, issues)
        if len(P) and (P.seed > 0).any():                                     # 選択は種 0（仮のターゲットに種 1・2 の行は無いはず）
            fail_or_note(args, issues, check="pseudo_rows_with_seed_gt0", run=run, n=int((P.seed > 0).sum()))
            P = P[P.seed <= 0]
        targets = [t for t in (args.targets or groups_for(run)) if t in groups_for(run)]
        for T in targets:
            for d_ in (d77, d78, E80C.get(run)):
                for nm in (f"{T}_pseudo.csv", f"{T}_chosen.csv", f"{T}_settings.json"):
                    if d_ is not None and (Path(d_) / nm).exists():
                        inputs[str(Path(d_) / nm)] = sha256(Path(d_) / nm)
            PT = P[P.eval_target == T] if len(P) else P
            if PT.empty:
                fail_or_note(args, issues, check="pseudo_missing", run=run, target=T)
                continue
            if run in WORLDS and not p4only:
                check_kept(args, run, T, PT, kept, issues)
            src = ";".join(sorted(set(PT.copied_from.dropna().astype(str))))   # 写した仮のターゲット（短縮 a）
            if src:
                copied[f"{run}|{T}"] = {w: copied_of(PT[PT.wave == w])[0] for w in sorted(set(PT.wave))}
            # 仮のターゲットと反復の組が delta と同じ期待にそろうか（関門 1 の I12-1）
            want = expected_pseudo(run, T, args.pseudo_reps)
            for w in sorted(set(PT.wave)):
                have = set(map(tuple, PT[PT.wave == w][["pseudo", "rep"]].drop_duplicates().itertuples(index=False)))
                if have != want:
                    fail_or_note(args, issues, check="pseudo_units_mismatch", run=run, target=T, wave=w,
                                 missing=sorted(want - have), extra=sorted(have - want))
            # E77 と E78 の none（仮のターゲット）が同じ値か（I14．許容幅を超えたら止める）
            nn = PT[PT.method == "none"].pivot_table(index=["pseudo", "rep", "student", "K"], columns="wave", values="uar")
            if {"e77", "e78"} <= set(nn.columns):
                dmax = float((nn.e77 - nn.e78).abs().max())
                issues.append({"check": "none_e77_vs_e78_pseudo", "run": run, "target": T, "absdiff_max": dmax})
                if not dmax <= TOL_UAR:
                    fail_or_note(args, issues, check="none_e77_vs_e78_pseudo_over_tol", run=run, target=T, absdiff_max=dmax)
            sets = json.loads((Path(d77) / f"{T}_settings.json").read_text()) if (Path(d77) / f"{T}_settings.json").exists() else {}
            chosen78 = None
            if d78 is not None and (Path(d78) / f"{T}_chosen.csv").exists() and (Path(d78) / f"{T}_chosen.csv").stat().st_size > 1:
                chosen78 = pd.read_csv(Path(d78) / f"{T}_chosen.csv", dtype=str)
            variants = [("incl", None), ("excl", None)]
            if T in ("MOSEI", "MELD"):
                variants += [("ns_incl", "IEMOCAP-script"), ("ns_excl", "IEMOCAP-script")]
            for st in sorted(PT.student.unique()):
                role = role_of(run, st)
                Ps = PT[PT.student == st]
                ev = coral_eval_ok(d78, T, st, args.reps)                     # CORAL の r が評価のターゲットの全反復で作られたか
                pu = coral_pseudo_units(d78, T, st)                           # CORAL の r が仮のターゲットのどの単位で作られたか
                for K in KS:
                    kept_u = kept.get((run, T, st, K), ())
                    PK = Ps[Ps.K == K]
                    defs = sel_defs(role, kept_u, e80c_teachers(PK))
                    if p4only:                                                # --families p4：P4・P4′ の族だけ（kept.csv を使う族を選ばない）
                        defs = {k_: v_ for k_, v_ in defs.items() if k_ in P4_SELS}
                    if role in ("uni", "fused") and not p4only:              # D8 の表（E80）：E77 に行がある教師ごとに，残したかと，dist・KD の最良が仮のターゲットで none を上回ったか
                        d8rows += d8_rows(run, T, st, role, K, PK, kept_all.get((run, T, st, K)), kept_u)
                    subsets = []
                    if "oursx" in defs:                                       # R6′：KD の族と同じ数の oursx の部分集合（none は常に含む）
                        key_ = lambda d_: set((d_.method + "|" + d_.pnorm + "|" + d_.meff).unique())  # noqa: E731
                        c_ox, c_kd = key_(PK[cand_mask(PK, defs["oursx"])]), key_(PK[cand_mask(PK, defs["kd"])])
                        subsets = r6_subsets(c_ox, len(c_kd), run, T, st, K) if c_ox else []
                        r6rows += [{"run": run, "target": T, "student": st, "role": role, "K": K, "draw": i, "n_teachers": len(kept_u),
                                    "n_oursx": len(c_ox), "n_kd": len(c_kd), "n_subset": len(sub_)} for i, sub_ in enumerate(subsets)]
                    for vname, drop in variants:
                        PV = PK[PK.pseudo != drop] if drop else PK
                        none = PV[(PV.method == "none") & (PV.wave == "e77")]
                        n_units = none.groupby(["pseudo", "rep"]).ngroups
                        none_score = float(none.uar.mean()) if len(none) else np.nan
                        ctx = (run, T, st, role, K, vname, none, n_units, none_score, ev, pu, chosen78, src)
                        for sel, dd in defs.items():
                            pick_one(ctx, sel, dd, PV)
                        if vname == "incl" and "oursx" in defs:               # R6′ は P5-1・P5-2 の感度の行（none を含む版）だけに使う
                            for i in range(R6_DRAWS):
                                sub_ = subsets[i] if subsets else frozenset()
                                fn_ = (lambda m, p, f, s_=sub_: f"{m}|{p}|{f}" in s_)
                                pick_one(ctx, f"{R6_PREFIX}{i}", [("e77", fn_), ("e80c", fn_)], PV)
                    # 固定版の行（設計の文書の「固定」．選ばない）
                    fixed += fixed_rows(run, T, st, role, K, sets, rows)
    pd.DataFrame(r6rows).to_csv(out / "r6_subsets.csv", index=False)
    pd.DataFrame(d8rows).to_csv(out / "d8_table.csv", index=False)
    pd.DataFrame([{"run": a, "target": T_, "student": b, "K": c, "teachers": ";".join(t), "n_teachers": len(t)}
                  for (a, T_, b, c), t in sorted(kept.items()) if a in dirs]).to_csv(out / "kept_used.csv", index=False)
    R = pd.DataFrame(rows)
    R.to_csv(out / "chosen.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(out / "candidates.csv", index=False)
    pd.DataFrame(fixed).to_csv(out / "fixed.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "checks.csv", index=False)
    hc, hf = sha256(out / "chosen.csv"), sha256(out / "fixed.csv")
    write_prov(out, "select", args, {"delta": dmeta, "chain": chain, "n_issues": len(issues), "inputs_sha256": inputs,
                                     "e80_kept": {"path": str(args.e80_kept), "sha256": kept_sha, "commit_check": kept_commit, "provenance": kept_prov},
                                     "e80c_provenance": e80c_prov,
                                     "copied_pseudo": copied,
                                     "rule": "仮のターゲット × 反復の全行（種のある手法は種 0）の UAR の単純平均の最大．同点は 'method|pnorm|meff' の文字列の順で先頭．"
                                             "none を含む版は none と同点なら none．oursx は E80 の R4（kept.csv で（世界, T, 生徒, K）ごとに残した教師 × 重みの付け方 5 × 総重み 4．"
                                             "残した教師が無い単位は none．R3‴），KD（kd と kdh）の教師も同じ集合（C1），"
                                             "除去実験の行の総重みは行ごとに選ぶ（R5．m_eff ＝ 500 に固定する行は置かない），"
                                             f"R6′ は oursx の候補から KD の候補と同じ数を種 0〜{R6_DRAWS - 1} で一様に引いた部分集合で選ぶ（{R6_PREFIX}<種>．none を含む版だけ．r6_subsets.csv）．"
                                             "共正則化は選ばない（短縮 b．measure の段でテストで選んだ上限にする）．Co-teaching は回していない．"
                                             "copied_pseudo の評価のターゲット（IEMOCAP 台本）の仮のターゲットの行は写し元（即興）の写しである",
                                     "chosen_csv_sha256": hc, "fixed_csv_sha256": hf})
    print(f"選択：{len(R)} 行（うち候補なし {int((R.method == EMPTY).sum())}），固定版 {len(fixed)} 行，記録 {len(issues)} 件", flush=True)
    print(f"chosen.csv の sha256 の先頭 16 文字：{hc[:16]}，fixed.csv：{hf[:16]}（δ と同じく文書に書いてコミットしてから --stage measure に進む）", flush=True)


def fixed_rows(run, T, st, role, K, sets, rows):
    """固定版（文献の既定値）と，対照の対応する行．値は見ずに決まる（設定の文字列だけで作る）．Co-teaching の固定版と cotzero は E78 で回さないので置かない。"""
    out = []
    base = {"run": run, "target": T, "student": st, "role": role, "K": K}
    out.append({**base, "fixname": "massbal", "wave": "e77", "method": "massbal", "param": "-", "meff": "0"})
    out.append({**base, "fixname": "saerens", "wave": "e77", "method": "saerens", "param": "-", "meff": "0"})
    out.append({**base, "fixname": "t3a_fix", "wave": "e77", "method": "t3a", "param": "Mall", "meff": "0"})
    if K != "0":
        out.append({**base, "fixname": "im_fix", "wave": "e77", "method": "im", "param": "a0.1", "meff": "tim",
                    "note": "TIM の既定値．種 0・1・2 の平均（E77 の --nc-seeds）"})
        out.append({**base, "fixname": "lp_fix", "wave": "e77", "method": "lp", "param": "k50a0.99hard", "meff": "n"})
    if role != "fused":
        out.append({**base, "fixname": "dann_fix", "wave": "e78", "method": "dann", "param": "lr0.001it1000lam0.1mu1.0", "meff": "-", "seedmode": "s0",
                    "note": "原論文の既定値に近い格子の点（λ = 0.1．λ は候補の数 n を掛けた尺度）．種 0"})
        out.append({**base, "fixname": "mme_fix", "wave": "e78", "method": "mme", "param": "lr0.001it1000lam0.1mu1.0sw0.0", "meff": "-", "seedmode": "s0",
                    "note": "原論文の既定値に近い格子の点（λ = 0.1，ソースのラベルを使わない sw = 0）．種 0"})
        out.append({**base, "fixname": "mapctl_fix", "wave": "e78", "method": "mapctl", "param": "lr0.001it1000mu1.0", "meff": "-", "seedmode": "s0"})
        for sel in ("dann", "mme"):                                             # 選んだ DANN・MME と同じ lr・反復・μ の対照（種 0）
            ch = [r for r in rows if r["run"] == run and r["target"] == T and r["student"] == st and r["K"] == K and r["selector"] == sel
                  and r["variant"] == "excl" and r["method"] == sel]
            if ch:
                m = re.match(r"lr(?P<lr>[^i]+)it(?P<it>\d+)lam[^m]+mu(?P<mu>[0-9.]+)", ch[0]["param"])
                if m:
                    out.append({**base, "fixname": f"mapctl_match_{sel}", "wave": "e78", "method": "mapctl",
                                "param": f"lr{m['lr']}it{m['it']}mu{m['mu']}", "meff": "-", "seedmode": "s0"})
    e39 = (sets.get("E39") or {}).get(st)
    if e39:
        cbs = "free" if e39.get("cb") is None else str(float(e39["cb"]))
        p = f"r{CORAL_FIX_R}rho{float(e39['rho_m'])}cp{float(e39['cp'])}cb{cbs}"
        out.append({**base, "fixname": "coral_fix", "wave": "e78", "method": "coral", "param": p, "meff": "-",
                    "note": "r = 64 と E39 の (ρ_M, C′, C_b)．E78 の 957fa35 が E39 の C′・C_b を格子に足した"})
    return out


# ---------------------------------------------------------------- 段 (c)：測る

def pair_moments(L, y, full=True):
    """L：(R, n, 4) の対数の確率（float64）．6 対の d・o と，full なら 2 感情の実際の正解率・分散の比・z の AUC を返す。"""
    R = L.shape[0]
    d, o, acc, vr, auc = (np.full((R, 6), np.nan) for _ in range(5))
    for p, (k, j) in enumerate(PAIRS):
        mk, mj = y == k, y == j
        if mk.sum() < 2 or mj.sum() < 2:
            continue
        zk = L[:, mk, k] - L[:, mk, j]
        zj = L[:, mj, k] - L[:, mj, j]
        ak, aj = zk.mean(1), zj.mean(1)
        vk, vj = zk.var(1, ddof=1), zj.var(1, ddof=1)
        sd = np.sqrt((vk + vj) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            d[:, p] = (ak - aj) / sd
            o[:, p] = -(ak + aj) / 2 / sd
            vr[:, p] = vk / vj
        if not full:
            continue
        acc[:, p] = 0.5 * ((zk > 0).mean(1) + (zj < 0).mean(1))
        both = np.concatenate([zk, zj], 1)
        rk = stats.rankdata(both, axis=1)[:, : zk.shape[1]]
        n1, n2 = zk.shape[1], zj.shape[1]
        auc[:, p] = (rk.sum(1) - n1 * (n1 + 1) / 2) / (n1 * n2)
    return d, o, acc, vr, auc


def lda_sep(Xa, ya, Xb, yb):
    """`explore_pairs.own_separation` の中の向き（LDA．Ledoit–Wolf）を Xa・ya で作り，Xb・yb で 6 対の離れ方を測る（同じ式）。"""
    m = np.stack([Xa[ya == k].mean(0) for k in range(4)])
    S = LedoitWolf(assume_centered=True, store_precision=False).fit(Xa - m[ya]).covariance_   # 共分散は own_separation と同じ（逆行列を作らない）
    U = scipy.linalg.solve(S, np.stack([m[k] - m[j] for k, j in PAIRS]).T, assume_a="pos").T   # 正定値なので Cholesky で解く（同じ解）
    out = []
    for u, (k, j) in zip(U, PAIRS):
        pk, pj = Xb[yb == k] @ u, Xb[yb == j] @ u
        out.append((pk.mean() - pj.mean()) / np.sqrt((pk.var(ddof=1) + pj.var(ddof=1)) / 2))
    return np.array(out)


def student_feats(run, X):
    if run not in WORLDS:
        return {run: X}
    out = {m: X[:, a:b] for m, a, b in blocks(run)}
    out[run] = X
    return out


def uar_bias_fixed(L, y):
    """切片だけを直した UAR：向き（L の差）を変えずに，テストの感情を釣り合わせた交差エントロピーを最小にする 4 つの切片を足した UAR（4 クラスのまま）。"""
    n = len(y)
    w = 1.0 / np.bincount(y, minlength=4)[y]
    oh = np.eye(4)[y]

    def f(b3):
        b = np.r_[0.0, b3]
        Z = L + b
        lse = logsumexp(Z, 1)
        P = np.exp(Z - lse[:, None])
        return float(((lse - Z[np.arange(n), y]) * w).sum()), ((P - oh) * w[:, None]).sum(0)[1:]
    res = minimize(f, np.zeros(3), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return present_uar(y, (L + np.r_[0.0, res.x]).argmax(1)), bool(res.success)


def eval_table(d, T, args=None, issues=None, run=None, wave=None):
    f = Path(d) / f"{T}.csv"
    if not f.exists():
        return None
    x = read_rows(f)
    if args is not None:                                                       # 判定に使わない K の行を記録する（行の数の照合のため落とさない）
        x = drop_other_k(x, args, issues, drop=False, run=run, target=T, wave=wave, where="eval")
    x["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(x.method, x.param)]
    x["seed"] = [seed_of(m, p) for m, p in zip(x.method, x.param)]
    return x


def key_table(sel, fixed, run, T, tab=None, extra=()):
    """ブートストラップで測る鍵（生徒, K, 波, 方法, pnorm, meff, 種の扱い）．参照の行（ref_*・teacher）も鍵にする（関門 1 の S5・C-3・R10-1）．
    extra：共正則化のテストで選んだ上限の鍵（短縮 b）。"""
    ks = set(extra)
    for w_ in ("e77", "e80c"):                                                # 参照の行．E80 の段 c の教師そのものの行（teacher）も鍵にする
        t = (tab or {}).get(w_)
        if t is None:
            continue
        refm = REFS if w_ == "e77" else ("teacher",)
        for r in t[t.method.isin(refm) & t.K.isin(KS)][["student", "K", "method", "pnorm", "meff"]].drop_duplicates().itertuples(index=False):
            ks.add((r.student, r.K, w_, r.method, r.pnorm, r.meff, "-"))
    s = sel[(sel.run == run) & (sel.target == T) & sel.method.notna() & (sel.method != EMPTY)]
    for r in s.itertuples():
        modes = ("agg", "s0") if r.method in SEEDED else ("-",)
        for md in modes:
            ks.add((r.student, r.K, r.wave, r.method, r.param, r.meff, md))
    f = fixed[(fixed.run == run) & (fixed.target == T)]
    for r in f.itertuples():
        md = getattr(r, "seedmode", None)
        md = md if isinstance(md, str) else ("agg" if r.method in SEEDED else "-")
        ks.add((r.student, r.K, r.wave, r.method, r.param, r.meff, md))
    for st in s.student.unique():
        for K in KS:
            ks.add((st, K, "e77", "none", "-", "0", "-"))
    return sorted(ks)


def rows_of_key(tab, key):
    st, K, wave, meth, pn, meff, md = key
    t = tab.get(wave)
    if t is None:
        return None
    x = t[(t.student == st) & (t.K == K) & (t.method == meth) & (t.pnorm == pn) & (t.meff == meff)]
    if md == "s0":
        x = x[x.seed == 0]
    return x


def boot_draws(run, corpus, speakers, B, seed, n_reps, tag=""):
    """コーパスごとの抽出．話者（反復 0〜9 のどれかでテストに出た話者の和集合）と反復を復元抽出する．IEMOCAP の即興と台本は同じ抽出を使う．
    tag を付けると別の乱数の流れ（塊を引く感度の行）。主の抽出は tag なしで前の版と同じ。"""
    rng = np.random.default_rng([seed, zlib.crc32(run.encode()), zlib.crc32(corpus.encode())] + ([zlib.crc32(tag.encode())] if tag else []))
    nS = len(speakers)
    C = rng.multinomial(nS, np.full(nS, 1 / nS), size=B).astype(np.float64)
    A = rng.multinomial(n_reps, np.full(n_reps, 1 / n_reps), size=B).astype(np.float64)
    return C, A


def boot_rep(L, y, sidx, Cb, Drows, pred=None):
    """1 反復ぶん．L：(R, n, 4)．sidx：発話ごとの話者の番号（0..n_ls−1）．Cb：(B, n_ls) の話者の重み．Drows：(R, 6) の行ごとの D（生徒ごと）。
    戻り：uar・向きの損・位置の損・逆の順序の向きの損 A(D,o) − A(d,o)・逆の順序の位置の損 A(D,0) − A(D,o)（B × R．ポイント），
    空の抽出（B），感情が欠けた抽出（B）。"""
    R, n, _ = L.shape
    nls = Cb.shape[1]
    GT = sparse.csr_matrix((np.ones(n), (sidx, np.arange(n))), shape=(nls, n))   # 話者 × 発話の指示行列
    Yk = np.stack([(y == k).astype(np.float64) for k in range(4)])            # 4 × n
    N = Cb @ (GT @ Yk.T)                                                       # B × 4
    empty = Cb.sum(1) == 0
    present = N > 0
    pred = L.argmax(2) if pred is None else np.asarray(pred).astype(int)   # 保存した予測（_pred.npy）を渡せば UAR はそれで数える．
    # float32 の対数の確率の argmax は，ほぼ同点の発話で保存した予測と食い違いうる（2026-11-02 の段 (c) の検算で E78 の MOSEI の 1 本）
    feats = []
    for k in range(4):
        feats.append((GT @ ((pred == k) * Yk[k]).T).T)                         # R × nls
    for k, j in PAIRS:
        z = L[:, :, k] - L[:, :, j]
        for c in (k, j):
            feats.append((GT @ (z * Yk[c]).T).T)
            feats.append((GT @ (z * z * Yk[c]).T).T)
    Fm = np.stack(feats, 2)                                                    # R × nls × 28
    M = (Cb @ Fm.transpose(1, 0, 2).reshape(nls, -1)).reshape(len(Cb), R, -1)  # B × R × 28
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = M[:, :, :4] / N[:, None, :]
        u = np.where(present[:, None, :], rec, np.nan)
        uar = np.nanmean(u, 2)
        dirs, poss, dirs_r, poss_r = [], [], [], []
        for p, (k, j) in enumerate(PAIRS):
            b0 = 4 + 4 * p
            nk, nj = N[:, k][:, None], N[:, j][:, None]
            s1k, s2k, s1j, s2j = M[:, :, b0], M[:, :, b0 + 1], M[:, :, b0 + 2], M[:, :, b0 + 3]
            mk, mj = s1k / nk, s1j / nj
            vk = (s2k - nk * mk ** 2) / (nk - 1)
            vj = (s2j - nj * mj ** 2) / (nj - 1)
            sd = np.sqrt(np.clip((vk + vj) / 2, 0, None))
            ok = (nk >= 2) & (nj >= 2) & (sd > 0)
            dd = np.where(ok, (mk - mj) / sd, np.nan)
            oo = np.where(ok, -(mk + mj) / 2 / sd, np.nan)
            a0 = ba(dd, 0.0)
            DD = Drows[:, p][None, :]
            dirs.append(ba(DD, 0.0) - a0)
            poss.append(a0 - ba(dd, oo))
            aDo = ba(DD, oo)
            dirs_r.append(aDo - ba(dd, oo))
            poss_r.append(ba(DD, 0.0) - aDo)
        dirl, posl = np.nanmean(np.stack(dirs, 2), 2), np.nanmean(np.stack(poss, 2), 2)
        dirl_r, posl_r = np.nanmean(np.stack(dirs_r, 2), 2), np.nanmean(np.stack(poss_r, 2), 2)
    for a_ in (uar, dirl, posl, dirl_r, posl_r):
        a_[empty] = np.nan
    return 100 * uar, 100 * dirl, 100 * posl, 100 * dirl_r, 100 * posl_r, empty, (~present).any(1) & ~empty


def stage_measure(args, dirs, out):
    chain = check_chain("measure", args)
    dmeta = chain["delta"]
    sel = pd.read_csv(Path(args.out) / "select" / "chosen.csv", dtype={"K": str, "param": str, "meff": str})
    fixed = pd.read_csv(Path(args.out) / "select" / "fixed.csv", dtype={"K": str, "param": str, "meff": str})
    issues, meta, inputs, e80c_prov = [], [], {}, {}
    for run, (d77, d78) in dirs.items():
        c77 = check_run_name(run, d77, "e77", issues, args)
        c78 = check_run_name(run, d78, "e78", issues, args) if d78 is not None else None
        if run in E80C:                                                       # 段 c の K と，select の段が読んだ kept.csv の sha256 との照合
            e80c_prov[run] = check_e80c_prov(run, args, issues, ((read_prov("select", args).get("e80_kept") or {}).get("sha256")))
        want = [t for t in (args.targets or groups_for(run)) if t in groups_for(run)]
        targets = [t for t in want if (Path(d77) / f"{t}.csv").exists()]
        for t in sorted(set(want) - set(targets)):
            fail_or_note(args, issues, check="eval_missing", run=run, target=t)
        by_corpus = {}
        for T in targets:
            by_corpus.setdefault(corp(T), []).append(T)
        rows_all, pair_rows, D_rows, keys_all, diag_rows, null_rows, coreg_rows = [], [], [], [], [], [], []
        args._null_rows, args._coreg_rows = null_rows, coreg_rows
        for cp, Ts in by_corpus.items():
            prep = {}
            for T in Ts:
                prep[T] = measure_target(args, run, T, d77, d78, sel, fixed, issues, rows_all, pair_rows, D_rows, diag_rows, c77, c78, inputs)
            speakers = sorted(set().union(*[p["speakers"] for p in prep.values()]))
            C, A = boot_draws(run, cp, speakers, args.boot, args.seed, len(args.reps))
            rep_col = {r: i for i, r in enumerate(args.reps)}
            sp_ix = {s: i for i, s in enumerate(speakers)}
            # 複製の種類：主の 2 段のブートストラップ（話者と反復），人物を 1 人ずつ抜く jackknife（G_eff < 20 のターゲットを含むコーパス．E76 の規則），
            # 塊（MELD の対話，IEMOCAP のセッション）と反復を引くブートストラップ（感度の行）
            geff_min = min(p["geff"] for p in prep.values())
            specs = [("boot", sp_ix, None, C, A)]
            if geff_min < GEFF_MIN and len(speakers) >= 2:
                specs.append(("jk", sp_ix, None, 1.0 - np.eye(len(speakers)), None))
            clusters = None
            if cp in CLUSTER_COLS:
                if all(p["clusters"] is not None for p in prep.values()):
                    clusters = sorted(set().union(*[p["cl_units"] for p in prep.values()]))
                    Cc, Ac = boot_draws(run, cp, clusters, args.boot, args.seed, len(args.reps), tag="cluster")
                    cl_ix = {c: i for i, c in enumerate(clusters)}
                    specs.append(("cl", cl_ix, "clusters", Cc, Ac))
                    if len(clusters) < CLUSTER_JK_MAX:                         # 塊が少ない（IEMOCAP のセッション 5 つ）：塊を 1 つずつ抜く jackknife（関門 1 の追記 stats 1）
                        specs.append(("cljk", cl_ix, "clusters", 1.0 - np.eye(len(clusters)), None))
                else:
                    issues.append({"check": "cluster_labels_missing", "run": run, "corpus": cp, "cols": list(CLUSTER_COLS[cp][1])})
            for T, p in prep.items():
                t0 = time.perf_counter()
                nk = len(p["keys"])
                acc = {nm: ({q: np.zeros((len(Cm), nk)) for q in QN}, np.zeros((len(Cm), nk))) for nm, _, _, Cm, _ in specs}
                n_empty, n_miss, n_draw = 0.0, 0.0, 0.0
                for r in p["reps"]:
                    L, y, spk, Drows, krow, pt, te = p["load"](r)
                    if L is None:
                        continue
                    Pr = p["load_pred"](r)
                    for nm, ix, labcol, Cm, Am in specs:
                        labels = spk if labcol is None else p[labcol][te]
                        gi = np.array([ix[s] for s in labels])
                        ls, sidx = np.unique(gi, return_inverse=True)
                        Cb = Cm[:, ls]
                        if nm == "boot":
                            # 検算：重みを全部 1 にすると，全テストの UAR と向きの損（点の値）に戻る
                            u1, d1, *_ = boot_rep(L[:args.chunk].astype(np.float64), y, sidx, np.ones((1, len(ls))), Drows[:args.chunk], pred=Pr[:args.chunk])
                            e1 = max(float(np.nanmax(np.abs(u1[0] - pt[:args.chunk, 0]))), float(np.nanmax(np.abs(d1[0] - pt[:args.chunk, 1]))))
                            issues.append({"check": "boot_identity", "run": run, "target": T, "rep": r, "absdiff_max": e1})
                            if not e1 <= 1e-6:
                                raise SystemExit(f"ブートストラップの式が点の値に戻らない（{run} {T} 反復 {r}：{e1}）")
                        V = {q: np.full((len(Cm), L.shape[0]), np.nan) for q in QN}
                        for a in range(0, L.shape[0], args.chunk):
                            *vals, empty, miss = boot_rep(L[a:a + args.chunk].astype(np.float64), y, sidx, Cb, Drows[a:a + args.chunk], pred=Pr[a:a + args.chunk])
                            for q, v_ in zip(QN, vals):
                                V[q][:, a:a + args.chunk] = v_
                        ar = Am[:, rep_col[r]] if Am is not None else np.ones(len(Cm))   # jackknife は反復を引かない（全反復を 1 回ずつ）
                        w = ar * (~empty)
                        if nm == "boot":
                            n_empty += float((ar * empty).sum())
                            n_miss += float((ar * miss).sum())
                            n_draw += float(ar.sum())
                        num, den = acc[nm]
                        for j, rows_j in enumerate(krow):                      # 種は複製ごとに平均する（合奏ではない）
                            if not rows_j:
                                continue
                            ok = np.isfinite(V["uar"][:, rows_j].mean(1))
                            for q in QN:
                                vq = V[q][:, rows_j].mean(1)
                                num[q][:, j] += np.where(np.isfinite(vq), w * vq, 0)
                            den[:, j] += np.where(ok, w, 0)
                res = {}
                for nm, *_ in specs:
                    num, den = acc[nm]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        res[nm] = {q: (num[q] / den).astype(np.float32) for q in QN}
                np.savez_compressed(out / f"boot_{run}_{T}.npz", **res["boot"])
                if "jk" in res:
                    np.savez_compressed(out / f"jk_{run}_{T}.npz", persons=np.array(speakers), **res["jk"])
                if "cl" in res:
                    np.savez_compressed(out / f"bootcl_{run}_{T}.npz", **res["cl"])
                if "cljk" in res:
                    np.savez_compressed(out / f"jkcl_{run}_{T}.npz", clusters=np.array(clusters), **res["cljk"])
                bu = res["boot"]["uar"]
                kt = pd.DataFrame(p["keys"], columns=["student", "K", "wave", "method", "param", "meff", "seedmode"]).assign(
                    key_id=range(nk), run=run, target=T, n_seeds=p["nseeds"])
                keys_all.append(kt)
                meta.append({"run": run, "target": T, "corpus": cp, "n_speakers_corpus": len(speakers), "n_speakers_target": p["n_spk"],
                             "max_speaker_share": p["max_share"], "g_eff": p["geff"], "g_eff_min_corpus": geff_min, "jk": "jk" in res,
                             "cluster_cols": "|".join(CLUSTER_COLS[cp][1]) if "cl" in res else "", "n_clusters_corpus": len(clusters) if clusters else 0,
                             "cluster_jk": "cljk" in res,
                             "n_keys": nk, "boot": args.boot,
                             "frac_empty_rep_draws": n_empty / max(n_draw, 1), "frac_missing_class_draws": n_miss / max(n_draw, 1),
                             "n_nan_uar_reps": int(np.isnan(bu).sum()), "sec": round(time.perf_counter() - t0, 1)})
                print(f"{run} {T}：鍵 {nk}，複製 {'・'.join(nm for nm, *_ in specs)}，{time.perf_counter() - t0:.0f} 秒", flush=True)
        if rows_all:
            pd.concat(rows_all, ignore_index=True).to_csv(out / f"rows_{run}.csv.gz", index=False)
        if pair_rows:
            pd.DataFrame(pair_rows).to_csv(out / f"pairs_{run}.csv.gz", index=False)
        if D_rows:
            pd.DataFrame(D_rows).to_csv(out / f"D_{run}.csv", index=False)
        if keys_all:
            pd.concat(keys_all, ignore_index=True).to_csv(out / f"keys_{run}.csv", index=False)
        if diag_rows:
            pd.DataFrame(diag_rows).to_csv(out / f"diag_{run}.csv", index=False)
        if null_rows:
            pd.DataFrame(null_rows).to_csv(out / f"model_null_{run}.csv", index=False)
        if coreg_rows:
            pd.DataFrame(coreg_rows).to_csv(out / f"coreg_oracle_{run}.csv", index=False)
    pd.DataFrame(meta).to_csv(out / "boot_meta.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "checks.csv", index=False)
    write_prov(out, "measure", args, {"delta": dmeta, "chain": chain, "boot": args.boot, "seed": args.seed, "n_issues": len(issues),
                                      "select_chosen_sha256": sha256(Path(args.out) / "select" / "chosen.csv"),
                                      "select_fixed_sha256": sha256(Path(args.out) / "select" / "fixed.csv"), "inputs": inputs, "e80c_provenance": e80c_prov,
                                      "notes": "E77 と E78 の教師の照合（teachers.npz）は，E78 が Co-teaching を回さず教師を書かないので行わない．"
                                               "共正則化はテストで選んだ上限（coreg_oracle_<実行>.csv．UAR は反復の平均の最大，向きの損は反復の平均の最小．none を候補に含む）"})
    print(f"測定：記録 {len(issues)} 件", flush=True)


def null_model_error(y, d6, o6, rng, draws=NULL_DRAWS):
    """模型（等分散の正規）が正しいときの 6 対の平均の |A(d̂, ô) − 実際の 2 感情の正解率|（ポイント）の分布．テストの感情の本数をそのまま使う。"""
    errs = np.zeros(draws)
    used = 0
    for p, (k, j) in enumerate(PAIRS):
        nk, nj = int((y == k).sum()), int((y == j).sum())
        if nk < 2 or nj < 2 or not (np.isfinite(d6[p]) and np.isfinite(o6[p])):
            continue
        mk, mj = d6[p] / 2 - o6[p], -d6[p] / 2 - o6[p]
        zk, zj = rng.normal(mk, 1.0, (draws, nk)), rng.normal(mj, 1.0, (draws, nj))
        ak, aj = zk.mean(1), zj.mean(1)
        sd = np.sqrt((zk.var(1, ddof=1) + zj.var(1, ddof=1)) / 2)
        dh, oh = (ak - aj) / sd, -(ak + aj) / 2 / sd
        acc = 0.5 * ((zk > 0).mean(1) + (zj < 0).mean(1))
        errs += np.abs(ba(dh, oh) - acc)
        used += 1
    return 100 * errs / max(used, 1)


def cluster_labels(T, sample_ids, issues, run):
    """塊を引く区間（感度の行）の塊：MELD は対話（split と dialogue_id の組），IEMOCAP はセッション．発話一覧（data/metadata）の列から作る．
    列が無いか，発話が一覧に無ければ記録して None。"""
    cp = corp(T)
    if cp not in CLUSTER_COLS:
        return None
    fn, cols = CLUSTER_COLS[cp]
    f = ROOT / "data" / "metadata" / fn
    head = list(pd.read_csv(f, nrows=0).columns)
    miss = [c for c in cols if c not in head]
    if miss:
        issues.append({"check": "cluster_col_missing", "run": run, "target": T, "file": fn, "missing": miss})
        return None
    m = pd.read_csv(f, usecols=["sample_id", *cols], dtype=str).drop_duplicates("sample_id").set_index("sample_id")
    lab = m.reindex(sample_ids)[list(cols)]
    if lab.isna().any().any():
        issues.append({"check": "cluster_label_na", "run": run, "target": T, "n": int(lab.isna().any(axis=1).sum())})
        return None
    return lab.astype(str).agg("|".join, axis=1).to_numpy()


def coreg_oracle(args, run, T, tab, sel_rows, reps, students, issues):
    """共正則化（群 C の探索．短縮 b）のテストで選んだ上限：（生徒, K）ごとに，全反復に値のある候補（共正則化の設定と none）のうち，
    反復の平均の UAR が最大のもの（UAR の比べ）と，反復の平均の向きの損が最小のもの（向きの損の比べ）．同点は none．
    oursx の側から片側にだけ読む（上回った・減らしたかだけ）。戻り：ブートストラップの鍵。"""
    parts = [sel_rows[("e77", r)] for r in reps if ("e77", r) in sel_rows]
    if not parts:
        return []
    X = pd.concat(parts, ignore_index=True)
    keys = []
    for st in students:
        role = role_of(run, st)
        if role not in COREG:
            continue
        for K in KS:
            x = X[(X.student == st) & (X.K == K) & X.method.isin([COREG[role], "none"])]
            g = x.groupby(["method", "pnorm", "meff"]).agg(n=("rep", "nunique"), uar=("uar", "mean"), dirloss=("dirloss", "mean")).reset_index()
            g = g[g.n == len(reps)]
            nn, cg = g[g.method == "none"], g[g.method == COREG[role]]
            if cg.empty or nn.empty:
                fail_or_note(args, issues, check="coreg_eval_missing", run=run, target=T, student=st, K=K)
                continue
            for metric, col, best_fn in (("uar", "uar", "idxmax"), ("dir", "dirloss", "idxmin")):
                b = cg.loc[getattr(cg[col], best_fn)()]
                nv = float(nn[col].iloc[0])
                take_none = nv >= b[col] if metric == "uar" else nv <= b[col]
                c = nn.iloc[0] if take_none else b
                args._coreg_rows.append({"run": run, "target": T, "student": st, "K": K, "metric": metric, "wave": "e77", "method": c.method,
                                         "param": c.pnorm, "meff": c.meff, "value": float(c[col]) * (100 if col == "uar" else 1),
                                         "none_value": nv * (100 if col == "uar" else 1), "chose_none": bool(take_none), "n_cand_with_none": int(len(cg)) + 1})
                keys.append((st, K, "e77", c.method, c.pnorm, c.meff, "-"))
    return keys


def measure_target(args, run, T, d77, d78, sel, fixed, issues, rows_all, pair_rows, D_rows, diag_rows, c77, c78, inputs=None):
    """1 つの評価のターゲット：全行の反復ごとの点の値，D，選んだ行の 6 対の内訳，診断．ブートストラップに渡す読み込みの関数を返す。"""
    t0 = time.perf_counter()
    tab = {"e77": eval_table(d77, T, args, issues, run, "e77")}
    if d78 is not None:
        tab["e78"] = eval_table(d78, T, args, issues, run, "e78")
        if tab["e78"] is None:
            fail_or_note(args, issues, check="e78_eval_missing", run=run, target=T)
            tab.pop("e78")
    d80 = E80C.get(run)
    if d80 is not None and (Path(d80) / f"{T}.csv").exists():                 # E80 の段 c（E77 に無い教師の dist・kd・kdh・teacher の行）．E77 の行に足して読む
        t80 = eval_table(d80, T, args, issues, run, "e80c")
        t77 = tab["e77"]
        check_overlap(t77, t80[t80.method.isin(E80C_METHODS)], ["rep", "student", "method", "param", "meff", "K"], f"{run} {T} の評価のターゲット")
        tab["e80c"] = t80
        n_other = int((~t80.method.isin(E80C_METHODS) & (t80.method != "none")).sum())
        if n_other:
            issues.append({"check": "e80c_eval_rows_not_used", "run": run, "target": T, "n": n_other})
        for r in sorted(set(t80.rep) & set(t77.rep)):                         # 段 c の none は E77 の none と同じ解のはず（同じ鍵の none を照合する）
            a_ = t77[(t77.rep == r) & (t77.method == "none")].set_index(["student", "K"]).row_id
            b_ = t80[(t80.rep == r) & (t80.method == "none")].set_index(["student", "K"]).row_id
            common = a_.index.intersection(b_.index)
            if len(common) < len(b_):                                          # 段 c の none に E77 の相手が無い行がある（追記 4）
                fail_or_note(args, issues, check="none_e77_vs_e80c_eval_unmatched", run=run, target=T, rep=r, n=int(len(common)), n_e80c_none=int(len(b_)))
            if len(common):
                L7 = np.load(Path(d77) / f"{T}_r{r}_logp.npy", mmap_mode="r")
                L8 = np.load(Path(d80) / f"{T}_r{r}_logp.npy", mmap_mode="r")
                v = max(float(np.abs(L7[int(a_[k])] - L8[int(b_[k])]).max()) for k in common)
                issues.append({"check": "none_e77_vs_e80c_logp", "run": run, "target": T, "rep": r, "absdiff_max": v})
                if not v <= TOL_LOGP:
                    fail_or_note(args, issues, check="none_e77_vs_e80c_over_tol", run=run, target=T, rep=r, absdiff_max=v)
    if inputs is not None:                                                     # 読んだ評価の出力（関門 1 の I9-1）．対数の確率は大きさと更新の時刻だけ
        for d_ in (d77, d78, d80):
            if d_ is None:
                continue
            for nm in (f"{T}.csv", f"{T}_samples.csv", f"{T}_diag.csv"):
                if (Path(d_) / nm).exists():
                    inputs[str(Path(d_) / nm)] = sha256(Path(d_) / nm)
            for f_ in sorted(Path(d_).glob(f"{T}_r*_logp.npy")):
                st_ = f_.stat()
                inputs[str(f_)] = {"bytes": st_.st_size, "mtime": st_.st_mtime}
    for w, t in tab.items():
        cm = set(t.commit.astype(str))
        if len(cm) != 1:
            fail_or_note(args, issues, check="commit_mixed", run=run, target=T, wave=w, commits=sorted(cm))
        issues.append({"check": "commit_of_eval_csv", "run": run, "target": T, "wave": w, "commits": sorted(cm),
                       "provenance_commit": c77 if w == "e77" else (c78 if w == "e78" else None)})
        reps = set(t.rep)
        if reps != set(args.reps):
            fail_or_note(args, issues, check="reps_incomplete", run=run, target=T, wave=w, have=sorted(reps))
        cnt = t.groupby(["rep"]).size()
        if cnt.nunique() > 1:
            fail_or_note(args, issues, check="rows_per_rep_differ", run=run, target=T, wave=w)
        dup = t.duplicated(["rep", "student", "method", "param", "meff", "K"]).sum()
        if dup:
            fail_or_note(args, issues, check="duplicate_rows", run=run, target=T, wave=w, n=int(dup))
    samp = pd.read_csv(Path(d77) / f"{T}_samples.csv")
    df, X = load_any(T, run)
    if not np.array_equal(df.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
        raise SystemExit(f"{run} {T}：_samples.csv の sample_id の並びが load_any と違う")
    if d78 is not None and (Path(d78) / f"{T}_samples.csv").exists():
        s78 = pd.read_csv(Path(d78) / f"{T}_samples.csv")
        if not np.array_equal(s78.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
            raise SystemExit(f"{run} {T}：E77 と E78 の _samples.csv が違う")
    if "e80c" in tab and (Path(d80) / f"{T}_samples.csv").exists():
        s80 = pd.read_csv(Path(d80) / f"{T}_samples.csv")
        if not np.array_equal(s80.sample_id.astype(str).to_numpy(), samp.sample_id.astype(str).to_numpy()):
            raise SystemExit(f"{run} {T}：E77 と E80 の段 c の _samples.csv が違う")
    yall = df.label_4.to_numpy().astype(int)
    spk_all = samp.speaker_group.astype(str).to_numpy()
    clusters = cluster_labels(T, samp.sample_id.astype(str).to_numpy(), issues, run)
    feats = student_feats(run, X)
    students = sorted(tab["e77"].student.unique())
    reps = [r for r in args.reps if (Path(d77) / f"{T}_r{r}_logp.npy").exists()]
    Dval, te_of = {}, {}
    sel_rows = {}                                                              # (wave, rep) → 鍵ごとの行の番号
    for r in reps:
        te = np.load(Path(d77) / f"{T}_r{r}_te.npy")
        te_of[r] = te
        y = yall[te]
        pool = np.setdiff1d(np.arange(len(yall)), te)
        for w, t in tab.items():
            dw = wave_dir(w, d77, d78, run)
            if not (Path(dw) / f"{T}_r{r}_logp.npy").exists():
                fail_or_note(args, issues, check="logp_missing", run=run, target=T, wave=w, rep=r)
                continue
            if w != "e77" and not np.array_equal(np.load(Path(dw) / f"{T}_r{r}_te.npy"), te):
                raise SystemExit(f"{run} {T} 反復 {r}：E77 と {w} のテストの行が違う")
            Lm = np.load(Path(dw) / f"{T}_r{r}_logp.npy", mmap_mode="r")
            Pm = np.load(Path(dw) / f"{T}_r{r}_pred.npy", mmap_mode="r")
            tr = t[t.rep == r].sort_values("row_id")
            if len(tr) != Lm.shape[0] or Lm.shape[1] != len(te):
                raise SystemExit(f"{run} {T} {w} 反復 {r}：行の数かテストの発話の数が合わない（{len(tr)}・{Lm.shape}・{len(te)}）")
            out_r = []
            for a in range(0, Lm.shape[0], args.chunk):
                Lc = np.asarray(Lm[a:a + args.chunk], dtype=np.float64)
                Pc = np.asarray(Pm[a:a + args.chunk]).astype(int)
                if not np.isfinite(Lc).all():
                    raise SystemExit(f"NaN/Inf：{run} {T} {w} 反復 {r}")
                u = np.array([shift_uar(y, pc) for pc in Pc])
                d, o, _, _, _ = pair_moments(Lc, y, full=False)
                nmis = int((Lc.argmax(2) != Pc).sum())
                out_r.append(pd.DataFrame({"a": np.arange(a, a + len(Lc)), "uar_pred": 100 * u, "argmax_mismatch": nmis,
                                           **{f"d{p}": d[:, p] for p in range(6)}, **{f"o{p}": o[:, p] for p in range(6)}}))
                if a == 0:                                                     # 既存の d_o と同じ値か（I6）
                    for i in range(min(3, len(Lc))):
                        ref = np.array(d_o(Lc[i], y))
                        dm = float(np.nanmax(np.abs(ref - np.stack([d[i], o[i]], 1))))
                        if not dm <= 1e-9:
                            raise SystemExit(f"d・o が explore_synergy.d_o と違う（{dm}）")
            o_r = pd.concat(out_r, ignore_index=True)
            tr = tr.reset_index(drop=True).join(o_r)
            ud = float((tr.uar_pred - 100 * tr.uar).abs().max())
            if ud > 1e-7:
                fail_or_note(args, issues, check="uar_vs_csv", run=run, target=T, wave=w, rep=r, absdiff_max=ud)
            tr["wave"] = w
            if not np.array_equal(tr.row_id.to_numpy(), np.arange(len(tr))):
                raise SystemExit(f"{run} {T} {w} 反復 {r}：row_id が 0 から並んでいない")
            for c_ in ("uar_bfix", "bias_fit_ok", "a_auc", "e_model", "e_model_abs"):
                tr[c_] = np.nan
            rows_all.append(tr)
            sel_rows[(w, r)] = tr
        # D：候補（プール）のラベルで学んだ LDA の向きをテストで測る．プールの平均と標準偏差で標準化する
        for st in students:
            Xs = np.asarray(feats[st], dtype=np.float64)
            mu, sd = Xs[pool].mean(0), Xs[pool].std(0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            Xp, Xt = (Xs[pool] - mu) / sd, (Xs[te] - mu) / sd
            Dv = lda_sep(Xp, yall[pool], Xt, y)
            Dval[(st, r)] = Dv
            for p in range(6):
                D_rows.append({"run": run, "target": T, "student": st, "rep": r, "pair": p, "D": Dv[p]})
            if run not in OWN_CHECKED:                                         # explore_pairs.own_separation と同じ式か（両向きの平均．run ごとに 1 回）
                half = np.r_[np.zeros(len(pool), int), np.ones(len(te), int)]
                Xc = np.vstack([Xp, Xt])
                yc = np.r_[yall[pool], y]
                ref = own_separation(Xc, yc, half)
                mine = (Dv + lda_sep(Xt, y, Xp, yall[pool])) / 2
                if not np.allclose(ref, mine, rtol=1e-8, atol=1e-10):
                    raise SystemExit("D の LDA が explore_pairs.own_separation と違う")
                OWN_CHECKED.add(run)
                issues.append({"check": "lda_vs_own_separation", "run": run, "target": T, "student": st, "rep": r,
                               "absdiff_max": float(np.abs(ref - mine).max())})
    # 向きの損と位置の損（全行．反復ごと．共正則化のテストで選ぶ上限に使うので鍵より先に測る）
    for tr in [x for x in rows_all if x.target.iloc[0] == T]:
        r = int(tr.rep.iloc[0])
        dmat = tr[[f"d{p}" for p in range(6)]].to_numpy()
        omat = tr[[f"o{p}" for p in range(6)]].to_numpy()
        Dm = np.stack([Dval.get((st, r), np.full(6, np.nan)) for st in tr.student])
        a0 = ba(dmat, 0.0)
        aD0, aDo, ado = ba(Dm, 0.0), ba(Dm, omat), ba(dmat, omat)
        tr["dirloss"] = 100 * np.nanmean(aD0 - a0, 1)
        tr["posloss"] = 100 * np.nanmean(a0 - ado, 1)
        tr["dirloss_rev"] = 100 * np.nanmean(aDo - ado, 1)                    # 逆の順序（関門 1 の R4-1）：向きを o の下で，位置を D で測る
        tr["posloss_rev"] = 100 * np.nanmean(aD0 - aDo, 1)
        tr["n_neg_dir_pairs"] = (aD0 - a0 < 0).sum(1)
    # 共正則化（群 C の探索）：評価のターゲットでテストで選んだ上限（短縮 b）．（生徒, K）ごとに，反復の平均の UAR が最大の設定と，
    # 反復の平均の向きの損が最小の設定を別々に選ぶ（none も候補．同点は none）．全反復に値のある候補だけ
    extra = [] if args.families == "p4" else coreg_oracle(args, run, T, tab, sel_rows, reps, students, issues)
    keys = key_table(sel, fixed, run, T, tab, extra)
    # 鍵ごとの反復の値（点）と，選んだ行の 6 対の内訳・切片だけを直した UAR
    krow, nseeds, krows_rep = [], [], {}
    for key in keys:
        x = rows_of_key(tab, key)
        nseeds.append(0 if x is None or x.empty else int(x.groupby("rep").size().max()))
    for r in reps:
        y = yall[te_of[r]]
        rows_r = []
        for j, key in enumerate(keys):
            wave = key[2]
            tr = sel_rows.get((wave, r))
            x = rows_of_key(tab, key)
            if tr is None or x is None or x.empty:
                rows_r.append([])
                continue
            ids = x[x.rep == r].row_id.tolist()
            rows_r.append([(wave, i) for i in ids])
        krows_rep[r] = rows_r
        flat = sorted({wi for lst in rows_r for wi in lst})
        pos = {wi: i for i, wi in enumerate(flat)}
        if flat:
            arrs = {}
            for w in {wv for wv, _ in flat}:
                dw = wave_dir(w, d77, d78, run)
                ids = [i for wv, i in flat if wv == w]
                arrs[w] = (ids, np.asarray(np.load(Path(dw) / f"{T}_r{r}_logp.npy", mmap_mode="r")[ids], dtype=np.float64))
            L = np.zeros((len(flat), len(y), 4))
            for w, (ids, a) in arrs.items():
                for i, rid in enumerate(ids):
                    L[pos[(w, rid)]] = a[i]
            d, o, acc, vr, auc = pair_moments(L, y)
            st_of = {}
            for j, key in enumerate(keys):
                for wi in rows_r[j]:
                    st_of[wi] = key[0]
            for wi, i in pos.items():                                          # 行ごとに 1 回（鍵との対応は rows_*.csv と keys_*.csv から引ける）
                ub, okb = uar_bias_fixed(L[i], y) if args.bias_fixed else (np.nan, True)
                Dv = Dval[(st_of[wi], r)]
                dauc6 = np.sqrt(2) * stats.norm.ppf(np.clip(auc[i], 1e-12, 1 - 1e-12))
                em = ba(d[i], o[i]) - acc[i]
                trw = sel_rows[(wi[0], r)]                                    # 行の表に，切片だけを直した UAR と模型の量を書く（判定の段が鍵ごとに読む）
                trw.loc[wi[1], ["uar_bfix", "bias_fit_ok", "a_auc", "e_model", "e_model_abs"]] = [
                    100 * ub, float(okb), 100 * np.nanmean(ba(dauc6, 0.0)), 100 * np.nanmean(em), 100 * np.nanmean(np.abs(em))]
                for p in range(6):
                    dauc = np.sqrt(2) * stats.norm.ppf(np.clip(auc[i, p], 1e-12, 1 - 1e-12))
                    pair_rows.append({"run": run, "target": T, "rep": r, "wave": wi[0], "row_id": wi[1], "student": st_of[wi], "pair": p,
                                      "d": d[i, p], "o": o[i, p], "D": Dv[p], "acc_actual": 100 * acc[i, p],
                                      "acc_model": 100 * ba(d[i, p], o[i, p]), "var_ratio": vr[i, p], "d_auc": dauc,
                                      "uar_bias_fixed": 100 * ub, "bias_fit_ok": okb})
    # 模型が正しいときの誤差の分布（テストの本数と，その反復の none の K=0 の d・o で作る．関門 1 の define の D8）
    rng_null = np.random.default_rng([args.seed, zlib.crc32(run.encode()), zlib.crc32(T.encode())])
    for r in reps:
        tr = sel_rows.get(("e77", r))
        if tr is None:
            continue
        y = yall[te_of[r]]
        for st in students:
            nn = tr[(tr.student == st) & (tr.method == "none") & (tr.K == "0")]
            if nn.empty:
                continue
            e = null_model_error(y, nn[[f"d{p}" for p in range(6)]].to_numpy()[0], nn[[f"o{p}" for p in range(6)]].to_numpy()[0], rng_null)
            args._null_rows.append({"run": run, "target": T, "student": st, "rep": r, "e_null_median": float(np.median(e)),
                                    "e_null_95": float(np.quantile(e, 0.95)), "n_test": int(len(y)),
                                    "n_min_class": int(np.bincount(y, minlength=4).min())})
    # 診断：E77 と E78 の none，K=0 の mapctl と none，DA の項と対照の予測の食い違い，P4 の選んだ設定と none の予測の食い違い
    diagnostics(args, run, T, d77, d78, tab, reps, sel, issues, diag_rows)
    speakers = set(spk_all[np.concatenate([te_of[r] for r in reps])]) if reps else set()
    if reps:                                                                   # 話者群の数と，最大の話者群が占めるテストの発話の割合（関門 1 の S3）
        vc = pd.Series(spk_all[np.concatenate([te_of[r] for r in reps])]).value_counts()
        n_spk, max_share = int(len(vc)), float(vc.iloc[0] / vc.sum())
        geff = float(vc.sum() ** 2 / (vc ** 2).sum())                          # 人物の実効の数 G_eff ＝ (Σn)²/Σn²（n は反復 0〜9 のテストの発話数を人物ごとに足したもの．E76）
    else:
        n_spk, max_share, geff = 0, float("nan"), float("nan")
    cl_units = set(clusters[np.concatenate([te_of[r] for r in reps])]) if (reps and clusters is not None) else set()

    def load(r):
        rows_r = krows_rep.get(r)
        if not rows_r:
            return None, None, None, None, None, None, None
        flat = sorted({wi for lst in rows_r for wi in lst})
        pos = {wi: i for i, wi in enumerate(flat)}
        te = te_of[r]
        L = np.zeros((len(flat), len(te), 4), np.float32)
        for w in {wv for wv, _ in flat}:
            dw = wave_dir(w, d77, d78, run)
            ids = [i for wv, i in flat if wv == w]
            a = np.load(Path(dw) / f"{T}_r{r}_logp.npy", mmap_mode="r")[ids]
            for i, rid in enumerate(ids):
                L[pos[(w, rid)]] = a[i]
        Drows = np.full((len(flat), 6), np.nan)                                 # 行ごとの D（その行の生徒の D）
        for j, lst in enumerate(rows_r):
            for wi in lst:
                Drows[pos[wi]] = Dval[(keys[j][0], r)]
        pt = np.array([sel_rows[(w, r)].set_index("row_id").loc[rid, ["uar", "dirloss"]].to_numpy(float) * [100, 1] for w, rid in flat])
        return L, yall[te], spk_all[te], Drows, [[pos[wi] for wi in lst] for lst in rows_r], pt, te

    def load_pred(r):
        """load(r) と同じ行の並びで，保存した予測（`_pred.npy`）を返す（点の値の UAR と同じ予測でブートストラップを数えるため）。"""
        rows_r = krows_rep.get(r)
        if not rows_r:
            return None
        flat = sorted({wi for lst in rows_r for wi in lst})
        pos = {wi: i for i, wi in enumerate(flat)}
        P = np.zeros((len(flat), len(te_of[r])), np.uint8)
        for w in {wv for wv, _ in flat}:
            dw = wave_dir(w, d77, d78, run)
            ids = [i for wv, i in flat if wv == w]
            a = np.load(Path(dw) / f"{T}_r{r}_pred.npy", mmap_mode="r")[ids]
            for i, rid in enumerate(ids):
                P[pos[(w, rid)]] = a[i]
        return P
    print(f"{run} {T}：点の値と D（{len(reps)} 反復），{time.perf_counter() - t0:.0f} 秒", flush=True)
    return {"keys": keys, "nseeds": nseeds, "reps": reps, "speakers": speakers, "load": load, "load_pred": load_pred, "students": students,
            "n_spk": n_spk, "max_share": max_share, "geff": geff, "clusters": clusters, "cl_units": cl_units}


def diagnostics(args, run, T, d77, d78, tab, reps, sel, issues, diag_rows):
    """E77 と E78 の同じはずの値（許容幅を超えたら止める．I14-1），DA の項の効き（種の間の食い違いと比べる），
    P4 の選んだ設定のテストの予測が none と食い違う割合（「働かなかった」単位．模擬査読の S4-2），E78 の `_diag.csv` の DA の項（選んだ設定）．
    Co-teaching は E78 で回さないので，cotzero・ṽ・教師の照合（teachers.npz は E78 が書かない）は無い。"""
    t77, t78 = tab.get("e77"), tab.get("e78")

    def chosen(st, K, selname, variant="excl"):
        ch = sel[(sel.run == run) & (sel.target == T) & (sel.student == st) & (sel.K == K) & (sel.selector == selname)
                 & (sel.variant == variant) & (sel.method == selname)]
        return None if ch.empty else ch.param.iloc[0]

    def chosen_row(st, K, selname, variant):
        ch = sel[(sel.run == run) & (sel.target == T) & (sel.student == st) & (sel.K == K) & (sel.selector == selname) & (sel.variant == variant)]
        return None if ch.empty or ch.method.iloc[0] in (EMPTY, None) or pd.isna(ch.method.iloc[0]) else ch.iloc[0]

    def rows78(a78, st, meth, K, pn):
        x = a78[(a78.student == st) & (a78.method == meth) & (a78.K == K) & (a78.pnorm == pn)]
        return {int(sd): int(rid) for sd, rid in zip(x.seed, x.row_id)}

    def add(r, st, K, what, value, **kw):
        diag_rows.append({"run": run, "target": T, "rep": r, "student": st, "K": K, "what": what, "value": value, **kw})
    for r in reps:
        L77 = np.load(Path(d77) / f"{T}_r{r}_logp.npy", mmap_mode="r")
        P77 = np.load(Path(d77) / f"{T}_r{r}_pred.npy", mmap_mode="r")
        a77 = t77[t77.rep == r]
        L78 = np.load(Path(d78) / f"{T}_r{r}_logp.npy", mmap_mode="r") if t78 is not None else None
        P78 = np.load(Path(d78) / f"{T}_r{r}_pred.npy", mmap_mode="r") if t78 is not None else None
        a78 = t78[t78.rep == r] if t78 is not None else None
        for st in a77.student.unique():
            for K in KS:
                n7 = a77[(a77.student == st) & (a77.method == "none") & (a77.K == K)].row_id
                # P4 の「働かなかった」の材料（関門 1 の追記の stats 2・define 3・code I1-inactive）．選んだ設定（none を除いて選んだ版と MOSEI・MELD の
                # 選び直しの版）について：(i) テストの予測が none と食い違う割合（判定と同じ種：excl は種 0・1・2，ns_excl は E78 の手法だけ種 0），
                # (ii) 種のある手法の種 0 だけの食い違い（entmin・im は none の解から始めた種 0 で決める），(iii) CORAL は同じ (ρ_M, C′, C_b) の coralctl との食い違い，
                # (iv) DANN・MME の選び直しの版の DA の項の材料（同じ lr・反復・μ の mapctl（種 0）との食い違いと種の間の食い違いの最大）
                if len(n7):
                    pn7 = P77[int(n7.iloc[0])]
                    for meth in GROUP_B:
                        for v in ("excl", "ns_excl"):
                            ch = chosen_row(st, K, meth, v)
                            if ch is None or ch.method == "none":
                                continue
                            t_, Pw = (a77, P77) if ch.wave == "e77" else (a78, P78)
                            if t_ is None:
                                continue
                            x = t_[(t_.student == st) & (t_.method == ch.method) & (t_.K == K) & (t_.pnorm == ch.param) & (t_.meff == ch.meff)]
                            xs = x[x.seed == 0] if ch.method in SEEDED else x
                            if v == "ns_excl" and ch.method in SEEDED - NC_SEEDED:   # 判定と同じ種（entmin・im は選び直しの版でも種の平均）
                                x = xs
                            if len(x):
                                add(r, st, K, f"pred_disagree_vs_none|{meth}|{v}", float(np.mean([(Pw[int(i)] != pn7).mean() for i in x.row_id])),
                                    n_rows=int(len(x)))
                            if ch.method in SEEDED and len(xs):
                                add(r, st, K, f"pred_disagree_vs_none_s0|{meth}|{v}", float((Pw[int(xs.row_id.iloc[0])] != pn7).mean()))
                            if ch.method == "coral" and len(x) and t_ is a78:
                                cc = t_[(t_.student == st) & (t_.method == "coralctl") & (t_.K == K) & (t_.pnorm == re.sub(r"^r\d+", "", ch.param))]
                                if len(cc):
                                    add(r, st, K, f"coral_vs_coralctl_disagree|{v}", float((Pw[int(x.row_id.iloc[0])] != Pw[int(cc.row_id.iloc[0])]).mean()))
                            if ch.method in ("dann", "mme") and v == "ns_excl" and t_ is a78:
                                sr = rows78(a78, st, ch.method, K, ch.param)
                                pairs_ = [(i, j) for i in sr for j in sr if i < j]
                                if pairs_:
                                    add(r, st, K, f"{ch.method}_seed_disagree_max|ns_excl", float(max((P78[sr[i]] != P78[sr[j]]).mean() for i, j in pairs_)))
                                m_ = re.match(r"lr(?P<lr>[^i]+)it(?P<it>\d+)lam[^m]+mu(?P<mu>[0-9.]+)", ch.param)
                                cr = rows78(a78, st, "mapctl", K, f"lr{m_['lr']}it{m_['it']}mu{m_['mu']}") if m_ else {}
                                if 0 in sr and 0 in cr:
                                    add(r, st, K, f"{ch.method}_vs_mapctl_disagree|ns_excl", float((P78[sr[0]] != P78[cr[0]]).mean()))
                if a78 is None:
                    continue
                n8 = a78[(a78.student == st) & (a78.method == "none") & (a78.K == K)].row_id
                if len(n7) and len(n8):
                    v = float(np.abs(L77[int(n7.iloc[0])] - L78[int(n8.iloc[0])]).max())
                    add(r, st, K, "none_e77_vs_e78_logp_absmax", v)
                    if not v <= TOL_LOGP:
                        fail_or_note(args, issues, check="none_e77_vs_e78_over_tol", run=run, target=T, rep=r, student=st, K=K, absdiff_max=v)
                if K == "0" and len(n8):                                     # K=0 の mapctl は none と同じ解のはず
                    for rid in a78[(a78.student == st) & (a78.method == "mapctl") & (a78.K == "0")].row_id:
                        add(r, st, K, "mapctl_K0_vs_none_logp_absmax", float(np.abs(L78[int(rid)] - L78[int(n8.iloc[0])]).max()))
                nid = int(n8.iloc[0]) if len(n8) else None
                # DA の項：a ＝ 選んだ DANN・MME（種 0）と同じ lr・反復・μ の mapctl（種 0）の食い違い，b ＝ 選んだ設定の種 0・1・2 の 3 対の食い違いの最大
                for sm in ("dann", "mme", "mapctl", "xmuda"):
                    pn = chosen(st, K, sm)
                    if pn is None:
                        continue
                    sr = rows78(a78, st, sm, K, pn)
                    pairs_ = [(i, j) for i in sr for j in sr if i < j]
                    if pairs_:
                        add(r, st, K, f"{sm}_seed_disagree_max", float(max((P78[sr[i]] != P78[sr[j]]).mean() for i, j in pairs_)), n_seeds=len(sr))
                    if sm in ("dann", "mme") and 0 in sr:
                        m = re.match(r"lr(?P<lr>[^i]+)it(?P<it>\d+)lam[^m]+mu(?P<mu>[0-9.]+)", pn)
                        cr = rows78(a78, st, "mapctl", K, f"lr{m['lr']}it{m['it']}mu{m['mu']}") if m else {}
                        if 0 in cr:
                            add(r, st, K, f"{sm}_vs_mapctl_disagree", float((P78[sr[0]] != P78[cr[0]]).mean()))
                        if nid is not None:
                            add(r, st, K, f"{sm}_vs_none_disagree", float((P78[sr[0]] != P78[nid]).mean()))
    if d78 is not None and (Path(d78) / f"{T}_diag.csv").exists():
        dg = pd.read_csv(Path(d78) / f"{T}_diag.csv", dtype={"K": str, "param": str, "meff": str})
        if "target" in dg.columns and len(dg):
            dg = dg[dg.target.notna()]
            dg["pnorm"] = dg.param.map(strip_seed)
            # DA の項の記述（選んだ設定．反復と種で平均）．判別器の損失は DANN の行だけ，エントロピーは MME の行だけで読む（E78 の再検査）
            cols = {"dann": ("anorm", "dom_first", "dom_last"), "mme": ("anorm", "ent_first", "ent_last"), "xmuda": ("anorm", "kl_first", "kl_last"),
                    "mapctl": ("anorm",)}
            for (st, K, selname), g in sel[(sel.run == run) & (sel.target == T) & (sel.variant == "excl")
                                            & sel.selector.isin(list(cols))].groupby(["student", "K", "selector"]):
                pn = g.param.iloc[0]
                if g.method.iloc[0] != selname:
                    continue
                x = dg[(dg.student == st) & (dg.method == selname) & (dg.K == K) & (dg.pnorm == pn)]
                for col in cols[selname]:
                    if col in x.columns:
                        add(-1, st, K, f"{selname}_{col}", float(x[col].mean()) if len(x) else np.nan, n=int(len(x)))


# ---------------------------------------------------------------- 段 (d)・(e)：判定と感度の行

def judge_unit(x, lo, hi, dl):
    if not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi) and np.isfinite(dl)):
        return "missing"
    if lo > 0 and x >= dl:
        return "pos"
    if hi < 0 and x <= -dl:
        return "neg"
    if lo >= -dl and hi <= dl:
        return "same"
    return "undet"


QN = ("uar", "dirloss", "posloss", "dirloss_rev", "posloss_rev")


class Store:
    """measure の出力を読み，(run, target, 鍵) → 点の値（反復の平均）と複製を返す．複製は主の 2 段のブートストラップ（boot），
    人物を 1 人ずつ抜く jackknife（jk．G_eff < 20 のコーパス），塊を引くブートストラップ（cl．MELD・IEMOCAP）。"""

    FILES = {"boot": "boot_{run}_{T}.npz", "jk": "jk_{run}_{T}.npz", "cl": "bootcl_{run}_{T}.npz", "cljk": "jkcl_{run}_{T}.npz"}

    def __init__(self, mdir, runs):
        self.mdir, self.keys, self.reps, self.point, self.idx, self.keylist = Path(mdir), {}, {}, {}, {}, {}
        self.geff, self.coreg, self.ncl = {}, {}, {}
        bm = self.mdir / "boot_meta.csv"
        if bm.exists():
            m = pd.read_csv(bm)
            if "g_eff" in m.columns:
                self.geff = {(a, b): float(c) for a, b, c in zip(m.run, m.target, m.g_eff)}
            if "n_clusters_corpus" in m.columns:
                self.ncl = {(a, b): int(c) for a, b, c in zip(m.run, m.target, m.n_clusters_corpus)}
        for run in runs:
            f = self.mdir / f"keys_{run}.csv"
            if not f.exists():
                continue
            k = pd.read_csv(f, dtype={"K": str, "param": str, "meff": str})
            self.keylist[run] = k
            self.keys[run] = {tuple(r[:8]): int(r[8]) for r in k[["target", "student", "K", "wave", "method", "param", "meff", "seedmode", "key_id"]]
                              .itertuples(index=False)}
            rows = pd.read_csv(self.mdir / f"rows_{run}.csv.gz", dtype={"K": str, "param": str, "meff": str})
            rows["pnorm"] = [strip_seed(p) if m in SEEDED else p for m, p in zip(rows.method, rows.param)]
            rows["seed"] = [seed_of(m, p) for m, p in zip(rows.method, rows.param)]
            self.point[run] = rows
            self.idx[run] = {g: ix for g, ix in rows.groupby(["target", "student", "K", "wave", "method", "meff"]).groups.items()}
            cf = self.mdir / f"coreg_oracle_{run}.csv"
            if cf.exists():
                c = pd.read_csv(cf, dtype={"K": str, "param": str, "meff": str})
                self.coreg.update({(run, r.target, r.student, r.K, r.metric): (r.wave, r.method, r.param, r.meff, "-") for r in c.itertuples()})

    def rows_of(self, run, T, st, K, wave, meth, pn, meff, md):
        ix = self.idx.get(run, {}).get((T, st, K, wave, meth, meff))
        x = self.point[run].loc[ix] if ix is not None else self.point[run].iloc[:0]
        x = x[x.pnorm == pn]
        return x[x.seed == 0] if md == "s0" else x

    def replicates(self, run, T, j, kind):
        if (run, T, kind) not in self.reps:
            f = self.mdir / self.FILES[kind].format(run=run, T=T)
            z = np.load(f) if f.exists() else None
            self.reps[(run, T, kind)] = None if z is None else {q: z[q] for q in QN if q in z.files}
        z = self.reps[(run, T, kind)]
        return None if z is None else {q: z[q][:, j] for q in z}

    def get(self, run, T, st, K, wave, meth, pn, meff, md):
        """点の値（種を反復ごとに平均してから反復で平均）と，複製（種類 → 量 → 配列．鍵が無ければ None）。"""
        pt = {q: np.nan for q in QN + ("uar_bfix", "uar_bfix_sd", "a_auc", "e_model", "ofix")}
        if run not in self.point:
            return pt, None
        x = self.rows_of(run, T, st, K, wave, meth, pn, meff, md)
        if len(x):
            g = x.groupby("rep")
            for q in QN:
                pt[q] = g[q].mean().mean() * (100 if q == "uar" else 1)
            for q in ("uar_bfix", "a_auc", "e_model"):
                if q in x.columns:
                    pt[q] = g[q].mean().mean()
            if "uar_bfix" in x.columns:
                v = g.uar_bfix.mean()
                pt["uar_bfix_sd"] = v.std(ddof=1) if v.notna().sum() > 1 else np.nan
                pt["ofix"] = pt["uar_bfix"] - pt["uar"]                        # 切片で取り戻せる分（切片だけを直した UAR − UAR．負になりうる）
        j = self.keys.get(run, {}).get((T, st, K, wave, meth, pn, meff, md))
        if j is None:
            return pt, None
        return pt, {kind: self.replicates(run, T, j, kind) for kind in self.FILES}


METRIC = {  # 値の向き：+1 なら x − y，−1 なら y − x（向きの損と位置の損は「減らしたら正」にする）．δ の種類．区間の有無
    "uar": ("uar", 1, "u", True), "dir": ("dirloss", -1, "d", True), "pos": ("posloss", -1, "d", True),
    "dir_rev": ("dirloss_rev", -1, "d", True), "pos_rev": ("posloss_rev", -1, "d", True),
    "bfix": ("uar_bfix", 1, "u", False), "ofix": ("ofix", 1, "u", False),
}


def jk_interval(theta_hat, th, geff_min):
    """人物を 1 人ずつ抜いた値 th（G 個）から SE_jk ＝ √((G − 1)/G · Σ(θ_(g) − θ̄)²) と自由度 max(1, ⌊G_eff⌋ − 1) の t 区間（E76 の規則）。"""
    th = th[np.isfinite(th)]
    G = len(th)
    if G < 2 or not np.isfinite(theta_hat) or not np.isfinite(geff_min):
        return np.nan, np.nan
    se = np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum())
    t = stats.t.ppf(0.975, max(1, int(np.floor(geff_min)) - 1))
    return theta_hat - t * se, theta_hat + t * se


def stage_judge(args, dirs, out):
    chain = check_chain("judge", args)
    mdir = Path(args.out) / "measure"
    S = Store(mdir, list(dirs))
    delta = pd.read_csv(Path(args.out) / "delta" / "delta.csv", dtype={"K": str})
    sel = pd.read_csv(Path(args.out) / "select" / "chosen.csv", dtype={"K": str, "param": str, "meff": str})
    fixed = pd.read_csv(Path(args.out) / "select" / "fixed.csv", dtype={"K": str, "param": str, "meff": str})
    units = []
    SEL = {(r.run, r.target, r.student, r.K, r.selector, r.variant): r for r in sel.itertuples(index=False)}
    if "seedmode" not in fixed.columns:
        fixed["seedmode"] = np.nan
    FIX = {(r.run, r.target, r.student, r.K, r.fixname): r for r in fixed.itertuples(index=False)}
    DEL = {(r.run, r.student, r.K): r for r in delta.itertuples(index=False)}
    KEPTU = {}                                                                 # (実行, T, 生徒, K) → E80 で残した教師（select の段の kept_used.csv）
    kf = Path(args.out) / "select" / "kept_used.csv"
    if kf.exists() and kf.stat().st_size > 1:
        kk = pd.read_csv(kf, dtype={"K": str, "teachers": str}).fillna({"teachers": ""})
        KEPTU = {(a, b, c, d): tuple(t for t in e.split(";") if t) for a, b, c, d, e in zip(kk.run, kk.target, kk.student, kk.K, kk.teachers)}
    NOKEPT = {k_ for k_, v_ in KEPTU.items() if not v_}                        # 残す教師が 0 個（R3‴）
    DIAG = {}                                                                  # (実行, T, 生徒, K, what) → 反復の平均（diag の材料）
    for run in dirs:
        f = mdir / f"diag_{run}.csv"
        if f.exists():
            dg = pd.read_csv(f, dtype={"K": str})
            dg = dg[dg.rep >= 0]
            for (T, st, K, w), v in dg.groupby(["target", "student", "K", "what"]).value.mean().items():
                DIAG[(run, T, st, K, w)] = float(v)

    def inactive_of(run, Ts, st, K, meth, variant):
        """「働かなかった」（関門 1 の追記 stats 2・define 3・code I1-inactive）．DANN・MME は DA の項の基準（同じ lr・反復・μ の mapctl との食い違い a，
        種の間の食い違いの最大 b．K ≥ 1 は a > b かつ a ≥ 1%，K ＝ 0 は a ≥ 1%）を満たさない単位，CORAL は同じ (ρ_M, C′, C_b) の coralctl との食い違いが 1% 未満，
        entmin・IM は種 0（none の解から始めた解）と none の食い違いが 1% 未満，ほかの手法（self・t3a・lp）は none との食い違いが 1% 未満．
        値は群の平均．選び直しの版で MOSEI・MELD 以外の群は excl の選択なので excl の材料を使う．戻り：(働かなかったか, a, b)。"""
        a_, b_ = [], []
        for T in Ts:
            v = variant if (variant == "excl" or T in ("MOSEI", "MELD")) else "excl"
            sfx = "" if v == "excl" else "|ns_excl"
            if meth in ("dann", "mme"):
                a_.append(DIAG.get((run, T, st, K, f"{meth}_vs_mapctl_disagree{sfx}"), np.nan))
                b_.append(DIAG.get((run, T, st, K, f"{meth}_seed_disagree_max{sfx}"), np.nan))
            elif meth == "coral":
                a_.append(DIAG.get((run, T, st, K, f"coral_vs_coralctl_disagree|{v}"), np.nan))
            elif meth in NC_SEEDED:
                a_.append(DIAG.get((run, T, st, K, f"pred_disagree_vs_none_s0|{meth}|{v}"), np.nan))
            else:
                a_.append(DIAG.get((run, T, st, K, f"pred_disagree_vs_none|{meth}|{v}"), np.nan))
        a = float(np.mean(a_)) if a_ and np.isfinite(a_).all() else np.nan
        b = float(np.mean(b_)) if b_ and np.isfinite(b_).all() else np.nan
        if not np.isfinite(a):
            return False, a, b
        if meth in ("dann", "mme"):
            # K ≥ 1 で種の幅 b が欠ける単位（選び直しの版で E78 が種 1・2 を解かなかった設定）は「予測を変えた」と言えないので働かなかったとみなす
            # （筋 4 に不利な側．da_effect.csv の da_changed_predictions と同じ扱い．関門 1 の再検査の I-ns-b）
            changed = a >= DISAGREE_MIN and (K == "0" or (np.isfinite(b) and a > b))
            return (not changed), a, b
        return a < DISAGREE_MIN, a, b

    def abl_class(a, run, T, st, K):
        """除去の行 X の単位ごとの意味（関門 1 の追記 define 1・compare 2）：subset ＝ X の候補 ⊆ oursx の候補（外した比べ），identical ＝ 候補が同じ
        （X の教師だけが残った．差は定義から 0），replace ＝ X が oursx に無い教師を使う（替えた比べ．教師が kept に無い，own・ownK），no_teacher ＝ 残す教師が 0 個。"""
        kept = KEPTU.get((run, T, st, K))
        if kept is None:
            return "unknown"
        if not kept:
            return "no_teacher"
        if a in ("abl_bal", "abl_raw", "abl_hard"):
            return "subset"
        if a in ("abl_own", "abl_ownK"):
            return "replace"
        t = ("pF" if role_of(run, st) == "uni" else "poe0") if a == "ours" else ABL_TEACHER.get(a)
        if t is None or t not in kept:
            return "replace"
        if a != "ours" and set(kept) == {t}:
            return "identical"
        return "subset"

    def pick(run, T, st, K, selector, variant):
        r = SEL.get((run, T, st, K, selector, variant))
        if r is None or pd.isna(r.method) or r.method == EMPTY:
            return None
        return (r.wave, r.method, r.param, r.meff, None)

    def fixkey(run, T, st, K, name):
        r = FIX.get((run, T, st, K, name))
        if r is None:
            return None
        md = r.seedmode if isinstance(r.seedmode, str) else ("agg" if r.method in SEEDED else "-")
        return (r.wave, r.method, r.param, r.meff, md)

    def refkey(meth, pn, wave="e77", meff="0"):
        return (wave, meth, pn, meff, "-")

    def val1(run, T, st, K, k, s0=False, nc_s0=False):
        if k is None:
            return {q: np.nan for q in QN + ("uar_bfix", "a_auc", "e_model", "ofix")}, None
        wave, meth, pn, meff = k[:4]
        # s0（MOSEI・MELD の選び直しの版）：選び直した設定に種 1・2 の無い E78 の手法だけ種 0 で読む．entmin・im は全候補に種 1・2 があるので，
        # 主の版とそろえて種 0・1・2 の平均で読む（本体のセッションの決定．選択は種 0）．nc_s0（種の感度の行．関門 1 の追記 define 4）：entmin・im を種 0 だけで読む
        md = k[4] if isinstance(k[4], str) else (
            "s0" if (s0 and meth in SEEDED - NC_SEEDED) or (nc_s0 and meth in NC_SEEDED) else ("agg" if meth in SEEDED else "-"))
        if meth == "none":
            md = "-"
        return S.get(run, T, st, K, wave, meth, pn, meff, md)

    def val(run, T, st, K, k, s0=False, nc_s0=False):
        """鍵が list なら（R6′ の 20 回の選択）点の値と複製をそれぞれ平均する（差の平均 ＝ 平均の差）。"""
        if not isinstance(k, list):
            return val1(run, T, st, K, k, s0, nc_s0)
        vs = [val1(run, T, st, K, k_, s0, nc_s0) for k_ in k]
        if not vs or any(v[0] is None for v in vs):
            return val1(run, T, st, K, None)
        pt = {q: float(np.mean([v[0][q] for v in vs])) for q in vs[0][0]}
        if any(v[1] is None for v in vs):
            return pt, None
        rep = {}
        for kind in Store.FILES:
            if any(v[1][kind] is None for v in vs):
                rep[kind] = None
            else:
                rep[kind] = {q: np.mean([v[1][kind][q] for v in vs], 0) for q in vs[0][1][kind]}
        return pt, rep

    NONE = ("e77", "none", "-", "0", "-")

    def dl_of(run, st, K, kind, scale):
        r = DEL.get((run, st, K))
        if r is None:
            return np.nan
        return float(r.delta_d if kind == "d" else r.delta_u) * scale

    def zone(x, dl):
        return "pos" if x >= dl else ("neg" if x <= -dl else "mid")

    def compare(family, comp, run, st, K, kx, ky, metric, variant, sens, s0=False, iem_mode="both", inactive=None, note=None, no_teacher=None,
                nc_s0=False, abl=None, abl_keep=None):
        """値 = METRIC の向き（uar は x − y，dir などは y − x）．コーパスごとに点の値と 2 段のブートストラップの区間を作り，判定する．
        区間：G_eff < 20 のターゲットを含む単位は，百分位の区間と人物の jackknife の t 区間の和集合（E76 の規則）．sens が空（主の行）のときは，
        MELD・IEMOCAP の塊の区間（MELD は対話の百分位，IEMOCAP はセッションの百分位とセッションを 1 つずつ抜く jackknife の t（自由度 セッション数 − 1）の和集合）を
        主の区間と和集合にした感度の行（cluster_boot）も出す（関門 1 の追記 stats 1．主の区間を置き換えない）．
        inactive（P4 の手法の名前）：inactive_of の基準で「働かなかった」単位の「同じ水準」「判定できない」を inactive に替える（「増やした」は替えない）．
        no_teacher："exclude" なら残す教師が 0 個の単位を「対象外（教師なし）」にして数え上げから外す，"count" なら判定を残して印だけ付ける（P5-2・P5-3u）．
        abl（除去の行の名前）：単位ごとの意味（abl_class）を列に出し，identical は数え上げから外す．abl_keep を渡すと，その意味でない単位を other_class にして外す。"""
        q, sign, dkind, has_ci = METRIC[metric]
        targets = [t for t in groups_for(run) if (args.targets is None or t in args.targets)]
        byc = {}
        for T in targets:
            if iem_mode == "impro_only" and T == "IEMOCAP-script":
                continue
            byc.setdefault(corp(T), []).append(T)
        for cp, Ts in byc.items():
            pts, reps_, aucs, des, geffs, ncls = [], [], [], [], [], []
            for T in Ts:
                kxx = kx(T) if callable(kx) else kx
                kyy = ky(T) if callable(ky) else ky
                px, bx = val(run, T, st, K, kxx, s0, nc_s0)
                py, by = val(run, T, st, K, kyy, s0, nc_s0)
                pts.append(sign * (px[q] - py[q]))
                aucs.append(sign * -(px["a_auc"] - py["a_auc"]) if metric == "dir" else np.nan)   # 向きの損の差を Φ(d_AUC/2) で測った値
                des.append(px["e_model"] - py["e_model"] if metric == "dir" else np.nan)
                rr = {}
                for kind in Store.FILES:
                    a_ = None if (not has_ci or bx is None or by is None) else (bx[kind], by[kind])
                    rr[kind] = None if (a_ is None or a_[0] is None or a_[1] is None or q not in a_[0]) else sign * (a_[0][q] - a_[1][q])
                reps_.append(rr)
                geffs.append(S.geff.get((run, T), np.nan))
                ncls.append(S.ncl.get((run, T), 0))
            x = float(np.mean(pts)) if pts else np.nan

            def pct(kind):
                if not reps_ or any(r_[kind] is None for r_ in reps_):
                    return np.nan, np.nan, 0
                b = np.mean(np.stack([r_[kind] for r_ in reps_]), 0)
                b = b[np.isfinite(b)]
                return (*(np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)), len(b))

            def union(*iv):
                los, his = [a_ for a_, _ in iv if np.isfinite(a_)], [b_ for _, b_ in iv if np.isfinite(b_)]
                return (min(los) if los else np.nan), (max(his) if his else np.nan)
            lo_p, hi_p, nvalid = pct("boot")
            gmin = float(np.nanmin(geffs)) if np.isfinite(geffs).any() else np.nan
            lo_j = hi_j = np.nan
            rule = "pct"
            if has_ci and nvalid == 0:
                rule = "no_replicates"                                         # 鍵が無い（定義から無い単位など）．欠けになる
            elif has_ci and np.isfinite(gmin) and gmin < GEFF_MIN:
                if reps_ and all(r_["jk"] is not None for r_ in reps_):
                    th = np.mean(np.stack([r_["jk"] for r_ in reps_]), 0)
                    lo_j, hi_j = jk_interval(x, th, gmin)
                    rule = "union_pct_jk"
                else:
                    rule = "pct_jk_missing"
            lo, hi = union((lo_p, hi_p), (lo_j, hi_j))
            if rule == "pct_jk_missing":
                lo = hi = np.nan                                               # jackknife が要る単位で無いときは判定しない（欠け）
            # 塊の区間（感度の行）：塊の百分位 ∪ 塊が少なければ塊を 1 つずつ抜く jackknife の t（自由度 塊の数 − 1），それを主の区間と和集合にする
            lo_c, hi_c, _ = pct("cl")
            lo_cj = hi_cj = np.nan
            if reps_ and all(r_["cljk"] is not None for r_ in reps_):
                lo_cj, hi_cj = jk_interval(x, np.mean(np.stack([r_["cljk"] for r_ in reps_]), 0), float(min(ncls)))
            lo_s, hi_s = union((lo, hi), (lo_c, hi_c), (lo_cj, hi_cj)) if np.isfinite([lo, hi]).all() else (np.nan, np.nan)
            if inactive is not None:
                is_inactive, ina_a, ina_b = inactive_of(run, Ts, st, K, inactive, variant)
            else:
                is_inactive, ina_a, ina_b = False, np.nan, np.nan
            # 残す教師が 0 個の単位（R3‴）：単位のすべての群で教師が 0 個のときだけ当て，一部の群だけのときは記録する
            nk = [(run, T, st, K) in NOKEPT for T in Ts]
            all_nt = bool(nk and all(nk))
            is_no_teacher = bool(no_teacher == "exclude" and all_nt)
            partial_no_teacher = bool(no_teacher and any(nk) and not all(nk))
            acls = None
            if abl is not None:
                cl_ = {abl_class(abl, run, T, st, K) for T in Ts}
                acls = cl_.pop() if len(cl_) == 1 else "mixed"
            xa, de = float(np.mean(aucs)), float(np.mean(des))
            ivs = [(sens, lo, hi, rule)]
            if not sens:
                ivs.append(("cluster_boot", lo_s, hi_s, "union_main_cluster" if cp in CLUSTER_COLS else rule))
            for sname, lo_, hi_, rule_ in ivs:
                for scale, stag in ((1.0, ""), (0.5, "delta_half"), (2.0, "delta_double")):
                    dl = dl_of(run, st, K, dkind, scale)
                    md = np.nan
                    if metric == "dir" and np.isfinite(x) and np.isfinite(xa) and np.isfinite(dl):   # 模型に依る単位の印（3 節の表の 1・2 行目）
                        md = bool(abs(x - xa) > MODEL_UNIT_FRAC * dl or abs(de) > MODEL_UNIT_FRAC * dl or zone(x, dl) != zone(xa, dl))
                    jd = jraw = judge_unit(x, lo_, hi_, dl) if has_ci else "point_only"
                    if is_inactive and jd in ("same", "undet"):                # 「増やした」は替えない（関門 1 の追記）
                        jd = "inactive"
                    if is_no_teacher or (acls == "no_teacher" and no_teacher == "exclude"):
                        jd = "no_teacher"
                    elif acls == "identical":
                        jd = "identical"
                    elif abl_keep is not None and acls != abl_keep:
                        jd = "other_class"
                    units.append({"family": family, "comparison": comp, "run": run, "student": st, "role": role_of(run, st), "K": K,
                                  "corpus": cp, "n_groups": len(Ts), "metric": metric, "point": x, "lo": lo_, "hi": hi_, "n_boot_valid": nvalid,
                                  "lo_pct": lo_p, "hi_pct": hi_p, "lo_jk": lo_j, "hi_jk": hi_j, "lo_cluster": lo_c, "hi_cluster": hi_c,
                                  "lo_cluster_jk": lo_cj, "hi_cluster_jk": hi_cj, "g_eff_min": gmin, "interval_rule": rule_,
                                  "delta": dl, "judgment": jd, "judgment_raw": jraw, "variant": variant,
                                  "inactive_a": ina_a, "inactive_b": ina_b, "inactive": is_inactive,
                                  "no_teacher": all_nt and bool(no_teacher), "partial_no_teacher": partial_no_teacher, "abl_kind": acls,
                                  "point_auc": xa, "model_err_diff": de, "model_dependent": md,
                                  "sensitivity": ",".join([s_ for s_ in (sname, stag) if s_]) or "main",
                                  "note": note if note is not None else NOTES.get(comp.split("-")[-1], "")})

    p4only = args.families == "p4"
    for run in dirs:
        world = run in WORLDS
        stus = sorted(set(sel[sel.run == run].student))
        refk = S.keylist.get(run)
        for st in stus:
            role = role_of(run, st)
            fam4 = "P4" if role == "kind" else ("P4'" if role == "fused" else "P4_uni_world")
            for K in KS:
                famK = fam4 + ("" if K in P4_KS or fam4 == "P4_uni_world" else "_otherK")
                # 主（excl），MOSEI・MELD の選び直し（ns_excl），entmin・IM を種 0 だけで読む感度の行（nc_seed0．関門 1 の追記 define 4．ほかの手法は主と同じ値）
                for v, s0, sens, ncs0 in (("excl", False, "", False), ("ns_excl", True, "noscript_reselect", False), ("excl", False, "nc_seed0", True)):
                    def kx_of(mm, vv=v):
                        return lambda T: pick(run, T, st, K, mm, vv if (not vv.startswith("ns") or T in ("MOSEI", "MELD")) else "excl")
                    for meth in GROUP_B + ("t3a_b", "mapctl", "coralctl"):
                        if role == "fused" and meth in ("dann", "mme", "mapctl"):
                            continue
                        if ncs0 and meth not in GROUP_B:
                            continue
                        fam = famK if meth in GROUP_B else "P4_aux"
                        ina = meth if meth in GROUP_B else None                # 「働かなかった」（inactive_of の手法ごとの基準）
                        compare(fam, f"none-{meth}", run, st, K, kx_of(meth), NONE, "dir", v, sens, s0, inactive=ina, nc_s0=ncs0)
                        if not sens:
                            compare(fam, f"none-{meth}", run, st, K, kx_of(meth), NONE, "dir_rev", v, "rev_order", s0, inactive=ina)
                            if run in ("text_roberta", "fuse-audio-video-text"):
                                compare(fam, f"none-{meth}", run, st, K, kx_of(meth), NONE, "dir", v, "iemocap_impro_only", s0, "impro_only", inactive=ina)
                    # 対照との比較（関門 1 の C-2）：CORAL は coralctl − CORAL，DANN・MME は自分で選んだ mapctl（種 3 つ）− 手法（種 3 つ）
                    ctl = [("coral", "coralctl")] + ([] if role == "fused" else [("dann", "mapctl"), ("mme", "mapctl")])
                    for meth, cm in ctl:
                        fam = (famK if fam4 != "P4_uni_world" else fam4) + "_ctl"
                        compare(fam, f"{cm}-{meth}", run, st, K, kx_of(meth), kx_of(cm), "dir", v, sens, s0, nc_s0=ncs0)
                    if sens:
                        continue
                    for meth in GROUP_B + ("massbal", "saerens", "t3a_b"):    # 記述（判定の族に入れない）：UAR，位置の損，切片だけを直した UAR，切片で取り戻せる分
                        if role == "fused" and meth in ("dann", "mme"):
                            continue
                        kx = (lambda T, m_=meth: fixkey(run, T, st, K, m_)) if meth in ("massbal", "saerens") else kx_of(meth)
                        for met in ("uar", "pos", "bfix", "ofix"):
                            compare(f"P4_desc_{met}", f"{meth}-none" if met in ("uar", "bfix", "ofix") else f"none-{meth}", run, st, K, kx, NONE,
                                    met, v, "")
                        if meth in ("coral", "dann", "mme"):
                            for cm in (("coralctl",) if meth == "coral" else ("mapctl",)):
                                compare("P4_ctl_uar", f"{meth}-{cm}", run, st, K, kx, kx_of(cm), "uar", v, "")
                    for meth in ("dann", "mme"):                               # 同じ lr・反復・μ の対照（種 0 の 1 回．種の幅と並べて読む．E78 の I5-2）
                        if role == "fused":
                            continue
                        compare("P4_dann_mme_s0", f"mapctl_match-{meth}", run, st, K, kx_of(meth),
                                lambda T, mm=meth: fixkey(run, T, st, K, f"mapctl_match_{mm}"), "dir", v, "", True)
                        compare("P4_dann_mme_s0", f"none-mapctl_match_{meth}", run, st, K, lambda T, mm=meth: fixkey(run, T, st, K, f"mapctl_match_{mm}"),
                                NONE, "dir", v, "", True)
                    for fx in ("im_fix", "t3a_fix", "lp_fix", "dann_fix", "mme_fix", "mapctl_fix", "coral_fix", "massbal", "saerens"):
                        if role == "fused" and fx in ("dann_fix", "mme_fix", "mapctl_fix"):
                            continue
                        compare("fixed", f"none-{fx}", run, st, K, lambda T, f_=fx: fixkey(run, T, st, K, f_), NONE, "dir", "fixed", "")
                        compare("fixed_uar", f"{fx}-none", run, st, K, lambda T, f_=fx: fixkey(run, T, st, K, f_), NONE, "uar", "fixed", "")
                # 参照の行（ref_*・教師）と none の差（区間つき．記述．関門 1 の S5・R10-1）
                if refk is not None:
                    rk = refk[(refk.student == st) & (refk.K == K) & refk.method.isin(REFS)][["wave", "method", "param", "meff"]].drop_duplicates()
                    for wv, meth, pn, mf in rk.itertuples(index=False):
                        nm = f"{meth}:{pn}" if meth == "teacher" else meth
                        compare("refs", f"{nm}-none", run, st, K, refkey(meth, pn, wv, mf), NONE, "uar", "ref", "")
                        compare("refs_dir", f"none-{nm}", run, st, K, refkey(meth, pn, wv, mf), NONE, "dir", "ref", "")
                if not world or K not in P5_KS or p4only:                     # --families p4 では P5 の族を流さない
                    continue
                OX = lambda T, s_="oursx", v_="incl": pick(run, T, st, K, s_, v_)  # noqa: E731
                KD = lambda T: pick(run, T, st, K, "kd", "incl")  # noqa: E731
                GB = lambda T: pick(run, T, st, K, "groupB", "incl")  # noqa: E731
                OXN = lambda T: pick(run, T, st, K, "oursx_noJ", "incl")  # noqa: E731
                KDN = lambda T: pick(run, T, st, K, "kd_noJ", "incl")  # noqa: E731
                R6 = lambda T: [pick(run, T, st, K, f"{R6_PREFIX}{i}", "incl") for i in range(R6_DRAWS)]  # noqa: E731
                # 教師なしの単位（R3‴）：P5-1・P5-3 とその記述，除去実験，5.3 節の融合の生徒，群 C，共正則化の上限，参照との比べは対象外にして外す（"exclude"）．
                # P5-2・P5-3u・群 B の手法ごとの表は oursx ＝ none として残し，数だけ数える（"count"．手順としての提案を問う行）
                compare("P5-1", "oursx-kd", run, st, K, OX, KD, "uar", "incl", "", no_teacher="exclude")
                compare("P5-2", "oursx-groupB", run, st, K, OX, GB, "uar", "incl", "", no_teacher="count")
                # R6′（縮めた格子の感度の行）：oursx の候補から KD と同じ数の部分集合を 20 回引いて選んだ値の平均と相手の差
                compare("P5-1", "oursx-kd", run, st, K, R6, KD, "uar", "incl", "r6_subsets", note=NOTES["r6"], no_teacher="exclude")
                compare("P5-2", "oursx-groupB", run, st, K, R6, GB, "uar", "incl", "r6_subsets", note=NOTES["r6"], no_teacher="count")
                # ふるわない版（関門 1 の追記 compare 1）：KD を前の設計の教師で調整した相手（kd_noJ），提案も J でふるわない版（oursx_noJ）．
                # どちらも教師なしの単位で none 以外を選びうるので数え上げに入れる
                compare("P5-1", "oursx-kd_noJ", run, st, K, OX, KDN, "uar", "incl", "kd_noJ", note=NOTES["noJ"], no_teacher="count")
                compare("P5-1", "oursx_noJ-kd_noJ", run, st, K, OXN, KDN, "uar", "incl", "noJ_both", note=NOTES["noJ"])
                compare("screening", "oursx-oursx_noJ", run, st, K, OX, OXN, "uar", "incl", "", note=NOTES["noJ"], no_teacher="count")
                compare("screening_dir", "oursx_noJ-oursx", run, st, K, OXN, OX, "dir", "incl", "", note=NOTES["noJ"], no_teacher="count")
                compare("screening", "kd-kd_noJ", run, st, K, KD, KDN, "uar", "incl", "", note=NOTES["noJ"], no_teacher="count")
                compare("P5-3", "none-oursx", run, st, K, OX, NONE, "dir", "incl", "", no_teacher="exclude")
                compare("P5-3", "none-oursx", run, st, K, OX, NONE, "dir_rev", "incl", "rev_order", no_teacher="exclude")
                compare("P5-3u", "oursx-none", run, st, K, OX, NONE, "uar", "incl", "", no_teacher="count")
                for met in ("bfix", "ofix", "pos"):                            # P5-3 を 4 クラスのまま分ける（点の値．切片だけを直した UAR）と，模型の位置の損
                    compare(f"P5-3_desc_{met}", "oursx-none" if met != "pos" else "none-oursx", run, st, K, OX, NONE, met, "incl", "",
                            no_teacher="exclude")
                # 群 B の 8 手法それぞれ（仮のターゲットで選んだもの，none を含む版）と oursx の UAR の差（記述．判定の族に入れない．模擬査読の S5-2）
                for meth in GROUP_B:
                    if role == "fused" and meth in ("dann", "mme"):
                        continue
                    compare("P5-2_by_method", f"oursx-{meth}", run, st, K, OX, lambda T, m_=meth: pick(run, T, st, K, m_, "incl"), "uar", "incl", "",
                            no_teacher="count")
                if run == "fuse-audio-video-text":
                    for fam, comp, kx, ky, met in (("P5-1", "oursx-kd", OX, KD, "uar"),
                                                   ("P5-2", "oursx-groupB", OX, GB, "uar"),
                                                   ("P5-3", "none-oursx", OX, NONE, "dir")):
                        compare(fam, comp, run, st, K, kx, ky, met, "incl", "iemocap_impro_only", False, "impro_only",
                                no_teacher="count" if fam == "P5-2" else "exclude")
                ns = lambda T: "ns_incl" if T in ("MOSEI", "MELD") else "incl"  # noqa: E731
                for fam, comp, kx, ky in (("P5-1", "oursx-kd", lambda T: pick(run, T, st, K, "oursx", ns(T)), lambda T: pick(run, T, st, K, "kd", ns(T))),
                                          ("P5-2", "oursx-groupB", lambda T: pick(run, T, st, K, "oursx", ns(T)),
                                           lambda T: pick(run, T, st, K, "groupB", ns(T)))):
                    compare(fam, comp, run, st, K, kx, ky, "uar", "ns_incl", "noscript_reselect", True,
                            no_teacher="count" if fam == "P5-2" else "exclude")
                # 配備できる参照と oursx（関門 1 の C-3）：合併（S+T）・ターゲットのみ（K = 5・25），W*（上限としてだけ），融合の生徒の教師
                if refk is not None:
                    rk = refk[(refk.student == st) & (refk.K == K) & refk.method.isin(REFS)][["wave", "method", "param", "meff"]].drop_duplicates()
                    for wv, meth, pn, mf in rk.itertuples(index=False):
                        if meth in ("ref_srconly",):
                            continue
                        nm = f"{meth}:{pn}" if meth == "teacher" else meth
                        compare("P5_refs", f"oursx-{nm}", run, st, K, OX, refkey(meth, pn, wv, mf), "uar", "incl", "", no_teacher="exclude")
                # 除去実験（今の ours を含む．総重みは行ごとに選ぶ．R4・R5）．単位ごとの意味（abl_class）で 2 つの族に分けて読む：
                # ablation ＝ 外した比べ（X の候補 ⊆ oursx の候補），ablation_replace ＝ 替えた比べ（X が oursx に無い教師を使う）．
                # 候補が同じ単位（identical）と教師なしの単位は外す（関門 1 の追記 define 1・compare 2）
                abl = ["abl_refa", "abl_refb", "abl_pF"] if role == "uni" else ["abl_ref0", "abl_poe0"]
                for a in abl + ["abl_bal", "abl_raw", "abl_hard", "abl_own", "abl_ownK", "ours"]:
                    XA = lambda T, a_=a: pick(run, T, st, K, a_, "incl")  # noqa: E731
                    for fam, keep in (("ablation", "subset"), ("ablation_replace", "replace")):
                        compare(fam, f"oursx-{a}", run, st, K, OX, XA, "uar", "incl", "", no_teacher="exclude", abl=a, abl_keep=keep)
                    for fam, keep in (("ablation_dir", "subset"), ("ablation_replace_dir", "replace")):   # 向きの損と切片も 2 つの読み方に分ける（I-abl-dir）
                        compare(fam, f"{a}-oursx", run, st, K, XA, OX, "dir", "incl", "", no_teacher="exclude", abl=a, abl_keep=keep)
                    for fam, keep in (("ablation_ofix", "subset"), ("ablation_replace_ofix", "replace")):
                        compare(fam, f"oursx-{a}", run, st, K, OX, XA, "ofix", "incl", "", no_teacher="exclude", abl=a, abl_keep=keep)
                # 群 C の探索（xMUDA 型・共訓練）
                for a in ("xmuda", "cotrain", "cotrain_prod"):
                    if (a == "xmuda" and role != "uni") or (a == "cotrain_prod" and role != "fused"):
                        continue
                    XA = lambda T, a_=a: pick(run, T, st, K, a_, "incl")  # noqa: E731
                    compare("groupC_explore", f"oursx-{a}", run, st, K, OX, XA, "uar", "incl", "", no_teacher="exclude")
                    compare("groupC_explore_dir", f"{a}-oursx", run, st, K, XA, OX, "dir", "incl", "", no_teacher="exclude")
                # 共正則化：テストで選んだ上限と比べる（短縮 b）．UAR は UAR の上限，向きの損は向きの損の上限．片側にだけ読む
                compare("groupC_coreg_oracle", "oursx-coreg_oracle", run, st, K, OX, lambda T: S.coreg.get((run, T, st, K, "uar")), "uar", "incl", "",
                        note=NOTES["coreg_oracle"], no_teacher="exclude")
                compare("groupC_coreg_oracle_dir", "coreg_oracle-oursx", run, st, K, OX, lambda T: S.coreg.get((run, T, st, K, "dir")), "dir", "incl", "",
                        note=NOTES["coreg_oracle"], no_teacher="exclude")
                if role == "uni":                                              # 5.3 節：K 本で学び直すことの寄与（単モダリティの生徒は oursx を含まない比べ）
                    compare("sec5.3", "abl_refa-abl_pF", run, st, K, lambda T: pick(run, T, st, K, "abl_refa", "incl"),
                            lambda T: pick(run, T, st, K, "abl_pF", "incl"), "uar", "incl", "")
                else:
                    compare("sec5.3", "oursx-abl_ref0", run, st, K, OX, lambda T: pick(run, T, st, K, "abl_ref0", "incl"), "uar", "incl", "",
                            no_teacher="exclude")
    U = pd.DataFrame(units)
    # 教師なしの単位を外さずに判定した感度の行（関門 1 の追記 stats 3・define 6・compare 3，再検査の I-nt-same）：教師なしの単位だけを外す前の判定
    # （judgment_raw．P5-1・P5-3 では差が定義から 0 なので「同じ水準」，参照・群 C・替えた除去の行では none と相手の本当の差の判定）に戻し，ほかの単位はそのまま
    nt = U[(U.sensitivity == "main") & (U.judgment == "no_teacher")]
    if len(nt):
        fams = set(nt.family)
        U2 = U[(U.sensitivity == "main") & U.family.isin(fams)].copy()
        m_ = U2.judgment == "no_teacher"
        U2.loc[m_, "judgment"] = U2.loc[m_, "judgment_raw"]
        U = pd.concat([U, U2.assign(sensitivity="no_teacher_unconditional")], ignore_index=True)
    U["excl_mosei_afew"] = U.corpus.isin(["MOSEI", "AFEW"])
    U.to_csv(out / "units.csv", index=False)
    noneloss = none_levels(S)
    combos, statements = summarize(U, sel, delta, noneloss)
    combos.to_csv(out / "combos.csv", index=False)
    statements.to_csv(out / "statements.csv", index=False)
    tables(args, out, S, sel, fixed, dirs, U, delta)
    conv_inputs = convergence_tables(args, out, dirs)
    write_prov(out, "judge", args, {"chain": chain, "n_units": len(U), "delta_csv_sha256": sha256(Path(args.out) / "delta" / "delta.csv"),
                                    "convergence_inputs_sha256": conv_inputs,
                                    "chosen_csv_sha256": sha256(Path(args.out) / "select" / "chosen.csv"),
                                    "fixed_csv_sha256": sha256(Path(args.out) / "select" / "fixed.csv")})
    print(f"判定：単位 {len(U)} 行，組み合わせ {len(combos)} 行，文の行 {len(statements)} 行", flush=True)


NOTES = {"noJ": "ふるわない版：oursx_noJ・kd_noJ は，その（T, 生徒, K）で行がある教師の全部（E77 の教師：dist は単モダリティ pF・pFK・ref，融合 poe0・ref・ref0，"
                "KD は単モダリティ pF・pFK・ref，融合 poe0・ref．これに E80 の段 c で計算した教師を足す）．段 c は kept で残した新しい教師しか計算しないので，"
                "残さなかった新しい教師（agree・poolC・mlp・other・ds など）は noJ に入らない．ふるいの寄与と，ふるいが KD に不利に働いたかを見る（関門 1 の追記 compare 1）",
         "r6": "R6′：oursx の候補から KD の族と同じ数の部分集合を種 0〜19 で一様に引き（none は常に含む），それぞれ仮のターゲットで選んだ値の 20 回の平均",
         "coreg_oracle": "共正則化はテストで選んだ上限（UAR は反復の平均の最大，向きの損は反復の平均の最小．none も候補）．oursx が上回った・減らしたかだけを読む",
         "ours": "今の ours（pF・poe0 の bal）．除去実験の 1 行（E80 の R4）"}


def none_levels(S):
    """(run, 生徒, K) ごとの none の向きの損の水準（コーパスの平均．IEMOCAP は 2 群の平均）。δ_d の大きさを読むために使う。"""
    out = {}
    for run, P in S.point.items():
        x = P[(P.method == "none") & (P.wave == "e77")].assign(c=lambda d: d.target.map(corp))
        g = x.groupby(["student", "K", "target", "c"]).dirloss.mean().groupby(["student", "K", "c"]).mean().groupby(["student", "K"]).mean()
        for (st, K), v in g.items():
            out[(run, st, K)] = float(v)
    return out


def summarize(U, sel, delta, noneloss):
    """7.2 節の表を組み合わせごとに当てる．感度の行（sensitivity）と MOSEI・AFEW を除いた集計も同じ規則で。"""
    combos, st_rows = [], []
    variants = [(s_, False) for s_ in sorted(U.sensitivity.unique())] + [("main", True), ("delta_half", True), ("nc_seed0", True), ("nc_seed0,delta_half", True)]
    DEL = {(r.run, r.student, r.K): r for r in delta.itertuples(index=False)}
    SELX = {(r.run, r.target, r.student, r.K, r.selector, r.variant): r.chose_none for r in sel.itertuples(index=False)}
    for sens, drop_ma in variants:
        V = U[U.sensitivity == sens]
        if drop_ma:
            V = V[~V.excl_mosei_afew]
        tag = sens + ("+excl_mosei_afew" if drop_ma else "")
        for (fam, comp, run, st, K, metric), g in V.groupby(["family", "comparison", "run", "student", "K", "metric"]):
            cps = {corp(t) for t in groups_for(run)} - ({"MOSEI", "AFEW"} if drop_ma else set())
            c = g.judgment.value_counts()
            nnt = int(c.get("no_teacher", 0))                                  # 対象外（教師なし），候補が同じ（identical），読む意味の違う除去の単位（other_class）は
            nid, noc = int(c.get("identical", 0)), int(c.get("other_class", 0))   # 数え上げから外し，過半は残りのコーパスの数で決める
            n_tot = len(cps) - nnt - nid - noc
            maj = majority(n_tot)
            npos, nneg, nsame, nund, nmis, nina = (int(c.get(k, 0)) for k in ("pos", "neg", "same", "undet", "missing", "inactive"))
            dr = DEL.get((run, st, K))
            row = {"tag": tag, "family": fam, "comparison": comp, "metric": metric, "run": run, "student": st, "K": K, "n_corpora": n_tot,
                   "majority": maj, "n_pos": npos, "n_neg": nneg, "n_same": nsame, "n_undet": nund, "n_missing": nmis, "n_inactive": nina,
                   "n_no_teacher": nnt, "n_identical": nid, "n_other_class": noc,
                   "n_oursx_none": int((g.no_teacher.map(lambda v: v is True or str(v) == "True") & ~g.judgment.isin(EXCLUDED)).sum())
                   if "no_teacher" in g.columns else 0,
                   "chance_combo_pos": float(stats.binom.sf(maj - 1, max(n_tot - nmis, 0), 0.025)) if n_tot - nmis > 0 else 0.0,
                   "expected_chance_pos": round(0.025 * (n_tot - nmis), 3),
                   "ci_halfwidth_median": float(((g.hi - g.lo) / 2).median()) if g.hi.notna().any() else np.nan,
                   "delta_d": float(dr.delta_d) if dr is not None else np.nan, "none_dirloss": noneloss.get((run, st, K), np.nan)}
            row["delta_d_ratio_to_none_dirloss"] = row["delta_d"] / row["none_dirloss"] if row["none_dirloss"] else np.nan
            md = g.model_dependent.map(lambda v: v is True or str(v) == "True")
            row["n_model_dependent"] = int(md.sum())
            g2 = g[~md]
            c2 = g2.judgment.value_counts()
            np2, nn2, ns2 = (int(c2.get(k, 0)) for k in ("pos", "neg", "same"))
            row.update({"n_pos_nomd": np2, "n_neg_nomd": nn2, "n_same_nomd": ns2})
            pts = g.point.dropna()
            nz = pts[pts != 0]
            row["sign_p"] = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else np.nan
            if fam.startswith("P4") and metric in ("dir", "dir_rev") and not fam.startswith(("P4_desc", "P4_dann_mme")):
                meth = comp.split("-", 1)[1] if comp.startswith(("none-", "coralctl-", "mapctl-")) else comp.split("-")[0]
                base = fam.split("_")[0]
                row["method"] = meth
                row["seen_before_prediction"] = meth in SEEN.get(base, set())
                row.update(rule_p4(g, npos, nneg, nsame, nmis, maj, n_tot, SELX, meth, tag))
                r_nomd = rule_p4(g2, np2, nn2, ns2, int((g2.judgment == "missing").sum()), maj, n_tot, SELX, meth, tag)
                row.update({f"{k}_nomd": v for k, v in r_nomd.items() if k in ("r1", "r2", "r2p", "r3ok", "r4")})
            combos.append(row)
    C = pd.DataFrame(combos)
    if C.empty:
        return C, pd.DataFrame()
    C["sign_p_holm"] = np.nan
    for (tag, fam), g in C.groupby(["tag", "family"]):
        ok = g.sign_p.notna()
        if ok.any():
            p = g.sign_p[ok].to_numpy()
            o = np.argsort(p)
            adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
            res = np.empty_like(adj)
            res[o] = adj
            C.loc[g.index[ok], "sign_p_holm"] = res
    for tag, g in C.groupby("tag"):
        st_rows += statements_p4(tag, g, C) + statements_p5(tag, g)
    S_ = pd.DataFrame(st_rows)
    # 種の感度（関門 1 の追記 define 4）：entmin・IM を種 0 だけで読んだ行（nc_seed0）と主の行で筋 4 の文の鍵が食い違うときは (3) にせず，両方の鍵を書く
    if len(S_) and "subset" in S_.columns:
        for base, alt in (("main", "nc_seed0"), ("main+excl_mosei_afew", "nc_seed0+excl_mosei_afew")):
            for fam in ("P4", "P4'"):
                for sub in ("all", "unseen_only"):
                    m0 = (S_.tag == base) & (S_.family == fam) & (S_.subset == sub) & ~S_.rule.isin(["chance"])
                    m1 = (S_.tag == alt) & (S_.family == fam) & (S_.subset == sub) & ~S_.rule.isin(["chance"])
                    if not m0.any() or not m1.any():
                        continue
                    r0, r1 = S_.loc[m0, "rule"].iloc[0], S_.loc[m1, "rule"].iloc[0]
                    if r0 != r1:
                        i0 = S_.index[m0][0]
                        note = f"entmin・IM の種 0 だけの判定（{alt}）では {r1}：{S_.loc[m1, 'statement_key'].iloc[0]}"
                        if r0 == "(3)":
                            S_.loc[i0, "rule"] = "(4)"
                            S_.loc[i0, "statement_key"] = (("(5) 筋の 4 を「単モダリティの特徴の空間では」と限る．" if fam == "P4'" else "")
                                                           + "ラベル無しの手法が向きの損を減らした証拠は無かった（区間の幅と判定できない数を添える）")
                            note = "種の平均と種 0 で文の鍵が食い違うので (3) にせず (4) の文にする．" + note
                        S_.loc[i0, "notes"] = " / ".join(x for x in (str(S_.loc[i0, "notes"]) if pd.notna(S_.loc[i0, "notes"]) else "", note) if x)
    return C, S_


def rule_p4(g, npos, nneg, nsame, nmis, maj, n_tot, SELX, meth, tag):
    out = {"r1": npos >= maj, "r2": npos >= 1 and nneg >= 1, "r2p": 1 <= npos < maj and nneg == 0,
           "r3ok": npos == 0 and nsame + nneg >= maj, "all_missing": nmis == n_tot}
    out["r4"] = (npos == 0 and not out["r3ok"]) and not out["all_missing"]
    rest = n_tot - npos
    out["r2p_rest_same_or_inc"] = (nsame + nneg) >= majority(rest) if rest > 0 else False
    # (2) の一致と (2′) の選択：「減らした」のコーパスで none を含む選択がその手法を選び，「増やした」のコーパスで none を選んだか．
    # IEMOCAP は即興と台本の両方でそうなったときだけ．MOSEI・MELD の選び直しの感度の行では ns_incl の選択で見る
    if g.empty:
        out.update({"r2_selection_agrees": np.nan, "r2p_selection_chose_method": np.nan})
        return out
    run, st, K = g.run.iloc[0], g.student.iloc[0], g.K.iloc[0]
    agree, chose_all = True, True
    for r in g.itertuples():
        if r.judgment not in ("pos", "neg"):
            continue
        Ts = [t for t in groups_for(run) if corp(t) == r.corpus]
        var = ["ns_incl" if ("noscript" in tag and t in ("MOSEI", "MELD")) else "incl" for t in Ts]
        cn = [str(SELX.get((run, t, st, K, meth, v_), "nan")) for t, v_ in zip(Ts, var)]
        chose = bool(cn) and all(v == "False" for v in cn)
        chose_none = bool(cn) and all(v == "True" for v in cn)
        if (r.judgment == "pos" and not chose) or (r.judgment == "neg" and not chose_none):
            agree = False
        if r.judgment == "pos" and not chose:
            chose_all = False
    out["r2_selection_agrees"] = agree if out["r2"] else np.nan
    out["r2p_selection_chose_method"] = chose_all if out["r2p"] else np.nan
    return out


def half_tag(tag):
    if "+" in tag:
        a, b = tag.split("+", 1)
        return None if "delta_" in a else ("delta_half" if a == "main" else a + ",delta_half") + "+" + b
    return "delta_half" if tag == "main" else (None if "delta_" in tag else tag + ",delta_half")


def p4_hit(g, suffix=""):
    for flag, name in (("r1", "(1)"), ("r2", "(2)"), ("r2p", "(2')"), ("r4", "(4)")):
        col = flag + suffix
        if col in g.columns:
            x = g[g[col].map(lambda v: v is True or str(v) == "True")]
            if len(x):
                return name, x
    return None


def p4_is3(g, suffix=""):
    return g[f"r3ok{suffix}"].map(lambda v: v is True or str(v) == "True").all() and (g[f"n_pos{'_nomd' if suffix else ''}"] == 0).all()


def statements_p4(tag, C, Call):
    """筋の 4 の文の鍵．(3) は δ を半分にした行でも (3) のときだけ（関門 1 の S-1）．模型に依る単位，対照との比較，予言の時点も見る。"""
    rows = []
    KEYS = {"(1)": "「ラベルを付ける発話の選び方では直らない」に狭める．直した向きの量を書く",
            "(2)": "選択との一致を見る（r2_selection_agrees）",
            "(2')": "〈コーパス〉では〈手法〉が向きの損を減らした（r2p_rest_same_or_inc で 2 つの書き方．r2p_selection_chose_method で選択を添える）",
            "(4)": "ラベル無しの手法が向きの損を減らした証拠は無かった（区間の幅と判定できない数を添える）"}
    met = "dir_rev" if "rev_order" in tag else "dir"                        # 逆の順序の感度の行は逆の順序の向きの損で文の鍵を作る（関門 1 の再検査の記録）
    for fam in ("P4", "P4'"):
        for subset in ("all", "unseen_only"):
            g = C[(C.family == fam) & (C.metric == met)]
            if subset == "unseen_only" and len(g):
                g = g[~g.seen_before_prediction.map(lambda v: v is True or str(v) == "True")]
            if g.empty:
                continue
            g = g[~g["all_missing"].map(lambda v: v is True or str(v) == "True")]
            if g.empty:
                continue
            hit = p4_hit(g)
            is3 = hit is None and p4_is3(g)
            notes = []
            ht = half_tag(tag)
            if is3 and ht:                                                     # δ を半分にした行でも (3) か（δ を変えた感度の行そのものでは見ない）
                Ch = Call[(Call.tag == ht) & (Call.family == fam) & (Call.metric == met)] if ht else Call.iloc[:0]
                if subset == "unseen_only" and len(Ch):
                    Ch = Ch[~Ch.seen_before_prediction.map(lambda v: v is True or str(v) == "True")]
                if len(Ch):
                    Ch = Ch[~Ch["all_missing"].map(lambda v: v is True or str(v) == "True")]
                if Ch.empty or p4_hit(Ch) is not None or not p4_is3(Ch):
                    is3 = False
                    notes.append("δ を半分にすると (3) に当たらないので (4) の文にする")
                if is3 and not p4_is3(g, "_nomd"):                               # 模型に依る単位を除くと (3) が成り立たない
                    is3 = False
                    notes.append("模型に依る単位を除くと (3) に当たらないので (4) の文にする")
            ctl = Call[(Call.tag == tag) & (Call.family == fam + "_ctl") & (Call.metric == "dir")]
            ctl_hit = p4_hit(ctl[~ctl["all_missing"].map(lambda v: v is True or str(v) == "True")]) if len(ctl) else None
            if ctl_hit is not None and ctl_hit[0] in ("(1)", "(2)", "(2')"):  # 対照より減らした手法がある（関門 1 の C-2）
                notes.append("〈手法〉の DA の項は同じ形の対照より向きの損を減らした（〈コーパス〉）が，none より減らしたとは言えない："
                             + ";".join(f"{a}|{b}|K{c}" for a, b, c in zip(ctl_hit[1].comparison, ctl_hit[1].student, ctl_hit[1].K)))
                if is3:
                    is3 = False
                    notes.append("対照との比較で減らした単位があるので (3) にせず (4) の文にする")
            if hit is not None:
                name, x = hit
                if p4_hit(x, "_nomd") is None:
                    notes.append("この印は模型に依る単位だけで決まった")
            elif is3:
                name, x = "(3)", g
            else:
                name, x = "(4)", g
            key = "筋の 4 を言い換えずに残す（どのコーパスでも減りは δ_d より小さかった，と δ_d の値と none の向きの損に対する割合を添える）" if name == "(3)" else KEYS[name]
            if fam == "P4'" and name != "(3)":
                key = "(5) 筋の 4 を「単モダリティの特徴の空間では」と限る．" + key
            dd = g.groupby(["run", "student", "K"])[["delta_d", "none_dirloss", "delta_d_ratio_to_none_dirloss"]].first().reset_index()
            rows.append({"tag": tag, "family": fam, "subset": subset, "rule": name, "n_combos": len(x),
                         "combos": ";".join(f"{a}|{b}|K{c}" for a, b, c in zip(x.comparison, x.student, x.K)) if name != "(3)" else "",
                         "statement_key": key, "notes": " / ".join(notes),
                         "delta_d": ";".join(f"{a}|{b}|K{c}:{d:.3g}（none の {e:.1%}）" for a, b, c, d, e in
                                             zip(dd.run, dd.student, dd.K, dd.delta_d, dd.delta_d_ratio_to_none_dirloss))})
            units = int((g.n_corpora - g.n_missing).sum())
            rows.append({"tag": tag, "family": fam, "subset": subset, "rule": "chance", "n_combos": len(g), "combos": "",
                         "statement_key": f"単位 {units}，偶然の「減らした」の見込み {0.025 * units:.2f}，実際の「減らした」{int(g.n_pos.sum())}，"
                                          f"模型に依る単位 {int(g.n_model_dependent.sum())}，働かなかった単位（手法ごとの基準で {DISAGREE_MIN:.0%} 未満：DANN・MME は同じ lr・反復・μ の "
                                          "mapctl との予測の食い違いが種の間の食い違いの最大を超えない（K ≥ 1．種の幅が無い単位も含む）か 1% 未満，CORAL は同じ (ρ_M, C′, C_b) の "
                                          "coralctl との食い違い，entmin・IM は種 0 と none の食い違い，ほかは none との食い違い．「同じ水準」「判定できない」を置き換え，"
                                          f"支持の数から除いた．「増やした」は置き換えない）{int(g.n_inactive.sum())}"})
    return rows


ABL_DROPPED = {"abl_bal": "② raw と ③ hard の重みの付け方", "abl_raw": "① bal と ③ hard の重みの付け方", "abl_hard": "① bal と ② raw の重みの付け方",
               "abl_refa": "pFK 以外の残した教師", "abl_refb": "ref 以外の残した教師", "abl_pF": "pF 以外の残した教師",
               "abl_ref0": "ref0 以外の残した教師", "abl_poe0": "poe0 以外の残した教師", "ours": "ほかの残した教師と bal 以外の重みの付け方"}
ABL_REPLACED = {"abl_refa": "教師を pFK に替えた版", "abl_refb": "教師を ref に替えた版", "abl_pF": "教師を pF に替えた版", "abl_ref0": "教師を ref0 に替えた版",
                "abl_poe0": "教師を poe0 に替えた版", "ours": "教師を pF（融合は poe0）の bal に替えた版（今の ours）",
                "abl_own": "自分の出発点を釣り合いを取って弱いラベルにした版（own）", "abl_ownK": "自分の none[K] を参照にした版（ownK）"}


def chance_rows(tag, fam, g, what="上回った"):
    """組み合わせごとの偶然の見込み（関門 1 の追記 stats 3）：真の差が 0 で単位ごとの当たりが 0.025 のとき，過半に当たる確率の和 Σ P(Binom(n, 0.025) ≥ 過半)。"""
    n_eff = (g.n_corpora - g.n_missing).clip(lower=0)
    units = int(n_eff.sum())
    ch = float(sum(stats.binom.sf(m - 1, n, 0.025) for m, n in zip(g.majority, n_eff) if n > 0))
    return units, ch


def statements_p5(tag, C):
    rows = []
    P5 = C[C.family.str.match(r"^P5-\d$|^ablation(_replace)?$|^sec5\.3$|^P5_refs$|^groupC_coreg_oracle(_dir)?$") & C.metric.isin(["uar", "dir"])]
    for (fam, comp, run, st, K, metric), g in P5.groupby(["family", "comparison", "run", "student", "K", "metric"]):
        r = g.iloc[0]
        maj = r.majority
        excl = int(r.n_no_teacher + r.n_identical + r.n_other_class) if "n_identical" in r.index else int(r.n_no_teacher)
        if r.n_corpora < 2 and excl > 0:                                      # 組み合わせの文は教師の残った（読む意味の）コーパスが 2 つ以上のときだけ（関門 1 の追記 stats 3）
            rows.append({"tag": tag, "family": fam, "rule": comp, "n_combos": 1, "combos": f"{run}|{st}|K{K}",
                         "statement_key": f"対象外（教師の残ったコーパスが {int(r.n_corpora)} つ．教師なし {int(r.n_no_teacher)}，"
                                          f"候補が同じ {int(getattr(r, 'n_identical', 0))}，意味の違う除去の単位 {int(getattr(r, 'n_other_class', 0))}）"})
            continue
        den = f"（教師の残った {int(r.n_corpora)} コーパスのうち）" if excl else ""
        if fam in ("P5-1", "P5-2"):
            if r.n_same + r.n_neg >= maj:
                key = "相手が oursx と同じ水準か上回った"
            elif r.n_pos >= maj:
                key = "oursx が相手を上回った"
            else:
                key = "判定できない（受け皿の行）"
            if fam == "P5-2" and getattr(r, "n_oursx_none", 0):
                key += f"．うち {int(r.n_oursx_none)} コーパスは教師なしで oursx ＝ none"
        elif fam == "P5-3":
            if metric != "dir":
                continue
            u = C[(C.family == "P5-3u") & (C.run == run) & (C.student == st) & (C.K == K)]
            if r.n_pos >= maj:
                key = "oursx は none より向きの損を減らした"
            elif len(u) and u.n_pos.iloc[0] >= u.majority.iloc[0]:
                key = "向きの損は減らさず UAR は上回った：筋の 5 を書き直し ours を o の補正として書く"
            else:
                key = "判定できない（受け皿の行）"
        elif fam == "ablation":
            # 外した比べ（X の候補 ⊆ oursx の候補）だけで数える（関門 1 の追記 define 1・compare 2）
            a = comp.split("-", 1)[1]
            hit = [k for k, n in (("pos", r.n_pos), ("same", r.n_same), ("neg", r.n_neg)) if n >= maj]
            drop = ABL_DROPPED.get(a, a)
            key = {"pos": f"外した候補（{drop}）は得に寄与した", "same": f"外した候補（{drop}）の寄与は見えない",
                   "neg": f"候補（{drop}）を外すと仮のターゲットの選択が良い設定を選んだ"}.get(hit[0] if hit else "", "判定できない")
            if comp == "oursx-abl_hard":                                       # 5.3 節の筋（③ だけ）は別の行にも残す
                rows.append({"tag": tag, "family": fam, "rule": comp + "|5.3", "n_combos": 1, "combos": f"{run}|{st}|K{K}",
                             "statement_key": ("5.3 節の筋で書く" if r.n_same >= maj else "（③ だけは oursx と同じ水準ではない）") + den})
        elif fam == "ablation_replace":
            # 替えた比べ（X が oursx に無い教師を使う）：部品の寄与としては書かない
            a = comp.split("-", 1)[1]
            hit = [k for k, n in (("pos", r.n_pos), ("same", r.n_same), ("neg", r.n_neg)) if n >= maj]
            rep = ABL_REPLACED.get(a, a)
            key = {"pos": f"oursx は {rep} を上回った", "same": f"oursx は {rep} と同じ水準", "neg": f"oursx は {rep} を下回った"}.get(
                hit[0] if hit else "", "判定できない")
        elif fam in ("groupC_coreg_oracle", "groupC_coreg_oracle_dir"):
            # 共正則化はテストで選んだ上限なので片側にだけ読む（短縮 b．比較の検査役の C5・C9）
            what = "上回った" if fam == "groupC_coreg_oracle" else "向きの損を減らした"
            key = (f"oursx がテストで選んだ上限の共正則化を{what}" if r.n_pos >= maj else "上限との比べなので判定しない（何も言わない）")
        elif fam == "P5_refs":
            if not comp.startswith("oursx-teacher:") or role_of(run, st) != "fused":
                continue
            key = ("融合の生徒では蒸留の得を主張せず，教師と同じ水準か教師が上回ったと書く" if r.n_same + r.n_neg >= maj
                   else ("oursx が教師を上回った" if r.n_pos >= maj else "判定できない（受け皿の行）"))
        else:
            key = "上回った" if r.n_pos >= maj else "上回っていない"
        rows.append({"tag": tag, "family": fam, "rule": comp, "n_combos": 1, "combos": f"{run}|{st}|K{K}", "statement_key": key + den})
    # 要旨の文は数え上げの形だけ（関門 1 の S-3）：生徒 × K のうち何個で「上回った」か（世界ごとの内訳）．単位ごとと組み合わせごとの偶然の見込みを並べ，
    # 教師なしの単位を外した分母と含めた分母の両方を書く（関門 1 の追記 stats 3・define 6・compare 3）
    for fam in ("P5-1", "P5-2", "P5-3"):
        g = C[(C.family == fam) & C.metric.isin(["uar", "dir"])]
        if g.empty:
            continue
        n_all_combo = int(len(g))
        n_nt_unit = int(g.n_no_teacher.sum()) if "n_no_teacher" in g.columns else 0
        n_none_unit = int(g.n_oursx_none.sum()) if "n_oursx_none" in g.columns else 0
        units_all = int((g.n_corpora + g.n_no_teacher - g.n_missing).clip(lower=0).sum())
        g = g[g.n_corpora >= 2]                                                # 組み合わせの文を書けるもの
        hit = g.n_pos >= g.majority
        by = g.assign(h=hit).groupby("run").h.agg(["sum", "size"])
        units, ch = chance_rows(tag, fam, g)
        rows.append({"tag": tag, "family": fam, "rule": "summary", "n_combos": int(len(g)), "combos": "",
                     "statement_key": f"生徒 × K {n_all_combo} のうち教師の残ったコーパスが 2 つ以上の {len(g)} で，{int(hit.sum())} で上回った（"
                                      + "，".join(f"{a} の {b} のうち {c}" for a, c, b in zip(by.index, by["sum"], by["size"])) + "）．"
                                      f"単位：全部で {units_all}（うち教師なし {n_nt_unit} を外した），数えた単位 {units}．"
                                      f"偶然の「上回った」の見込み：単位 {0.025 * units:.2f}，組み合わせ {ch:.3f}．実際の「上回った」：単位 {int(g.n_pos.sum())}．"
                                      + (f"oursx ＝ none（教師なし）のまま数えた単位 {n_none_unit}．" if n_none_unit else "")
                                      + "特定の生徒と K だけを取り出した文は書かない"})
    for fam in ("ablation", "ablation_replace"):                               # 除去実験の偶然の見込み（関門 1 の追記 stats の変更 9 の記録）
        for comp, g in C[(C.family == fam) & (C.metric == "uar")].groupby("comparison"):
            g = g[g.n_corpora >= 2]
            units, ch = chance_rows(tag, fam, g)
            rows.append({"tag": tag, "family": fam, "rule": f"{comp}|chance", "n_combos": int(len(g)), "combos": "",
                         "statement_key": f"単位 {units}，偶然の「上回った」「下回った」の見込み：単位 各 {0.025 * units:.2f}，組み合わせ 各 {ch:.3f}．"
                                          f"実際：上回った {int(g.n_pos.sum())}，下回った {int(g.n_neg.sum())}"})
    s53 = C[(C.family == "sec5.3") & C.K.isin(["5", "25"])]
    if len(s53):
        per = s53.assign(ok=(s53.n_pos >= s53.majority) & (s53.n_corpora >= 2)).groupby(["run", "student"]).ok.all()
        rows.append({"tag": tag, "family": "sec5.3", "rule": "7 生徒の過半", "n_combos": int(len(per)),
                     "combos": ";".join(f"{a}|{b}:{bool(v)}" for (a, b), v in per.items()),
                     "statement_key": "「K 本で学び直してから」を書く" if int(per.sum()) >= 4 else "その部分を外す（教師の作り方の寄与を 2 段目の参照に限る）"})
    return rows


def convergence_tables(args, out, dirs):
    """付録の記述の表（比較の検査役の追記 2 の [C6]）．E77 の `<T>_convp.csv` から，手法（what）ごとに問の数，反復の上限（logreg.MAX_ITER）に
    当たった問の数，止まったときの勾配が許容幅（logreg.TOL）を超えた問の数（そのうち上限の前に止まった問．Armijo の歩幅が決まらずに止まった問）．
    E77 の `<T>_diag.csv` の cg_iters・cg_resid から，ラベル伝播の共役勾配法が上限に当たった問の数と残差の最大．
    写した仮のターゲットの行（copied_from）は写し元と同じ問なので数えない．値の列（uar_test など）は読まない。戻り：読んだファイルの sha256。"""
    cv, cg, inputs = [], [], {}
    for run, (d77, _) in dirs.items():
        for T in [t for t in (args.targets or groups_for(run)) if t in groups_for(run)]:
            f = Path(d77) / f"{T}_convp.csv"
            if f.exists() and f.stat().st_size > 1:
                inputs[str(f)] = sha256(f)
                x = pd.read_csv(f, usecols=lambda c: c in ("pseudo", "target", "rep", "student", "what", "key", "n_iter", "grad", "copied_from"),
                                dtype={"what": str, "key": str})
                if "copied_from" in x.columns:
                    x = x[x.copied_from.isna()]
                side = np.where(x.target.notna(), "target", "pseudo") if "target" in x.columns else np.full(len(x), "pseudo")
                x = x.assign(side=side, what=x.what.fillna("-"), hit=x.n_iter >= CONV_MAX_ITER, over=x.grad > CONV_TOL)
                x["early"] = x.over & ~x.hit
                g = x.groupby(["side", "what"]).agg(n_prob=("n_iter", "size"), n_hit_max_iter=("hit", "sum"), n_grad_over_tol=("over", "sum"),
                                                   n_stopped_early=("early", "sum"), n_iter_max=("n_iter", "max"), grad_max=("grad", "max")).reset_index()
                cv.append(g.assign(run=run, eval_target=T))
            f = Path(d77) / f"{T}_diag.csv"
            if f.exists() and f.stat().st_size > 1:
                inputs[str(f)] = sha256(f)
                x = pd.read_csv(f, usecols=lambda c: c in ("pseudo", "target", "rep", "student", "method", "param", "K", "cg_iters", "cg_resid", "copied_from"),
                                dtype={"param": str, "K": str})
                if "copied_from" in x.columns:
                    x = x[x.copied_from.isna()]
                if "cg_iters" not in x.columns or "method" not in x.columns:
                    continue
                x = x[(x.method == "lp") & x.cg_iters.notna()]
                if x.empty:
                    continue
                side = np.where(x.target.notna(), "target", "pseudo") if "target" in x.columns else np.full(len(x), "pseudo")
                x = x.assign(side=side, alpha=x.param.str.extract(r"a([0-9.]+)", expand=False), hit=x.cg_iters >= LP_CG_MAX)
                g = x.groupby(["side", "K", "alpha"]).agg(n_prob=("cg_iters", "size"), n_hit_cg_max=("hit", "sum"),
                                                          cg_iters_max=("cg_iters", "max"), cg_resid_max=("cg_resid", "max")).reset_index()
                cg.append(g.assign(run=run, eval_target=T))
    if cv:
        C = pd.concat(cv, ignore_index=True)
        C.to_csv(out / "convergence.csv", index=False)
        C.groupby(["side", "what"]).agg(n_prob=("n_prob", "sum"), n_hit_max_iter=("n_hit_max_iter", "sum"), n_grad_over_tol=("n_grad_over_tol", "sum"),
                                        n_stopped_early=("n_stopped_early", "sum"), n_iter_max=("n_iter_max", "max"),
                                        grad_max=("grad_max", "max")).reset_index().to_csv(out / "convergence_by_method.csv", index=False)
    if cg:
        G = pd.concat(cg, ignore_index=True)
        G.to_csv(out / "lp_cg.csv", index=False)
        G.groupby(["side", "K", "alpha"]).agg(n_prob=("n_prob", "sum"), n_hit_cg_max=("n_hit_cg_max", "sum"), cg_iters_max=("cg_iters_max", "max"),
                                              cg_resid_max=("cg_resid_max", "max")).reset_index().to_csv(out / "lp_cg_by_alpha.csv", index=False)
    return inputs


def tables(args, out, S, sel, fixed, dirs, U, delta):
    """付録の表：none が選ばれた単位の数，選択の取りこぼし（オラクルは上限としてだけ．種の幅つき）と手法ごとの候補の数と取りこぼし，参照の行，種の幅，
    模型の当てはまり（E75 の決まり），DA の項の効き，定義から同じになる組，向きの損のオラクル，対照との食い違い。"""
    kept = {}                                                                  # (実行, T, 生徒, K) → E80 で残した教師（select の段が使ったもの）
    kf = Path(args.out) / "select" / "kept_used.csv"
    if kf.exists() and kf.stat().st_size > 1:
        kk = pd.read_csv(kf, dtype={"K": str, "teachers": str}).fillna({"teachers": ""})
        kept = {(a, b, c, d): tuple(t for t in e.split(";") if t) for a, b, c, d, e in zip(kk.run, kk.target, kk.student, kk.K, kk.teachers)}
    inc = sel[sel.variant == "incl"]
    nn = inc.assign(chose_none=inc.chose_none.map(lambda v: str(v) == "True")).groupby(["run", "role", "selector", "K"]).chose_none.agg(["sum", "size"])
    nn.reset_index().to_csv(out / "none_chosen.csv", index=False)
    CK = ["wave", "method", "pnorm", "meff"]
    rows, orows = [], []
    for run, Pall in S.point.items():
        # 種：E78 の手法は種 0 にそろえる（E78 の記録．選ばれた設定にしか種 1・2 が無い）．entmin・im は全候補に種 1・2 があるので，
        # テストで選ぶ上限も選んだ値も種 0・1・2 の平均で作る（設計の 4）
        P = Pall[(Pall.seed <= 0) | Pall.method.isin(list(NC_SEEDED))]
        per = (P.groupby(["target", "student", "K"] + CK + ["rep"]).uar.mean() * 100)
        mean = per.groupby(level=list(range(7))).mean()
        nrep = per.groupby(level=list(range(7))).size()
        dmean = P.groupby(["target", "student", "K"] + CK).dirloss.mean()
        uniq = P[["student"] + CK].drop_duplicates()
        mcache = {}

        newt = {k_: e80c_teachers(g_) for k_, g_ in P[P.wave == "e80c"].groupby(["target", "student", "K"])} if (P.wave == "e80c").any() else {}

        def members_of(role, selector, kept_u, noj_new=((), ())):
            """（役, 族, 残した教師, 段 c の教師）→ 候補の集合（評価の行のキーで作る）。"""
            k_ = (role, selector, kept_u if selector in KEPT_FAMS else (), noj_new if selector in ("oursx_noJ", "kd_noJ") else ())
            if k_ not in mcache:
                u = uniq[[role_of(run, st) == role for st in uniq.student]].drop_duplicates(CK)
                dd = sel_defs(role, kept_u, noj_new).get(selector)
                mcache[k_] = None if dd is None else set(map(tuple, u[cand_mask(u, dd)][CK].to_numpy()))
            return mcache[k_]
        for r in sel[(sel.run == run) & (sel.variant.isin(["incl", "excl"]))].itertuples():
            if r.method == EMPTY or pd.isna(r.method) or r.selector.startswith(R6_PREFIX):
                continue
            mem = members_of(r.role, r.selector, kept.get((run, r.target, r.student, r.K), ()), newt.get((r.target, r.student, r.K), ((), ())))
            if mem is None:
                continue
            try:
                cm = mean.loc[(r.target, r.student, r.K)]
                cn = nrep.loc[(r.target, r.student, r.K)]
            except KeyError:
                continue
            keep = [tuple(i) in mem for i in cm.index]
            cm, cn = cm[keep], cn[keep]
            cm = cm[cn == cn.max()] if len(cn) else cm                          # 評価の全反復に値のある候補だけ（CORAL の r の除外と同じ考え）
            if cm.empty:
                continue
            orc, best_key = cm.max(), cm.idxmax()
            if r.variant == "incl":                                             # none を含む版のオラクルは none も候補
                nonev = mean.get((r.target, r.student, r.K, "e77", "none", "-", "0"), np.nan)
                if nonev >= orc:
                    orc, best_key = nonev, ("e77", "none", "-", "0")
            pr = per.loc[(r.target, r.student, r.K)]
            ch = pr.get((r.wave, r.method, r.param, r.meff))
            bs = pr.get(tuple(best_key))
            if ch is None or bs is None:
                continue
            loss = (bs - ch).dropna()
            sd_seed, n_seed = np.nan, 1
            if r.method in SEEDED:                                              # 種の幅（E78 の I5-2：種 0 の 1 回の差は種の幅と並べて読む）
                xs = S.rows_of(run, r.target, r.student, r.K, r.wave, r.method, r.param, r.meff, "agg")
                v = xs.groupby("seed").uar.mean() * 100
                sd_seed, n_seed = (v.std(ddof=1) if len(v) > 1 else np.nan), int(len(v))
            rows.append({"run": run, "target": r.target, "student": r.student, "K": r.K, "selector": r.selector, "variant": r.variant,
                         "n_cand": int(len(cm)), "oracle_upper": orc, "chosen": ch.mean(), "selection_loss": orc - ch.mean(),
                         "loss_sd_over_reps": loss.std(ddof=1) if len(loss) > 1 else np.nan, "n_reps": int(len(loss)),
                         "chosen_seed_sd": sd_seed, "n_seeds": n_seed})
            # 向きの損で見た格子の最良（オラクル．上限としてだけ書く．関門 1 の C-4）
            if r.variant == "excl" and r.selector in GROUP_B:
                dm = dmean.loc[(r.target, r.student, r.K)]
                dm = dm[[tuple(i) in mem for i in dm.index]]
                nd = dmean.get((r.target, r.student, r.K, "e77", "none", "-", "0"), np.nan)
                if len(dm):
                    orows.append({"run": run, "target": r.target, "corpus": corp(r.target), "student": r.student, "K": r.K, "method": r.selector,
                                  "none_minus_best_dirloss": nd - dm.min(), "best": "|".join(map(str, dm.idxmin()))})
    SL = pd.DataFrame(rows)
    SL.to_csv(out / "selection_loss.csv", index=False)
    # 手法ごとの候補の数と選択の取りこぼし（E80 の記録．S5-2・S5-4）：（役, 族, K, 版）ごとに，候補の数（none を含む）と，
    # テストで選んだ上限 − 仮のターゲットで選んだ値の平均・中央値・最大と単位の数．上限は上限としてだけ書く
    cf = Path(args.out) / "select" / "candidates.csv"
    if len(SL) and cf.exists():
        cd = pd.read_csv(cf, dtype={"K": str})
        cn = cd.groupby(["run", "role", "selector", "K"]).n_cand_with_none.agg(n_cand_with_none_min="min", n_cand_with_none_median="median",
                                                                               n_cand_with_none_max="max").reset_index()
        SL = SL.merge(sel[["run", "target", "student", "K", "selector", "variant", "role"]].drop_duplicates(), how="left")
        by = SL.groupby(["run", "role", "selector", "K", "variant"]).selection_loss.agg(
            loss_mean="mean", loss_median="median", loss_max="max", n_units="size").reset_index()
        by.merge(cn, how="left").to_csv(out / "selection_loss_by_method.csv", index=False)
    if orows:
        O = pd.DataFrame(orows)
        O = O.groupby(["run", "corpus", "student", "K", "method"]).none_minus_best_dirloss.mean().reset_index()
        Dd = delta.set_index(["run", "student", "K"]).delta_d
        O["delta_d"] = [Dd.get((a, b, c), np.nan) for a, b, c in zip(O.run, O.student, O.K)]
        O["at_least_delta_d"] = O.none_minus_best_dirloss >= O.delta_d
        O.to_csv(out / "oracle_dir.csv", index=False)
    R = U[U.family.isin(["refs", "refs_dir"]) & (U.sensitivity == "main")]
    if len(R):
        R.to_csv(out / "refs.csv", index=False)
    seeds = []
    for run, P in S.point.items():
        x = P[P.method.isin(SEEDED) & (P.seed >= 0)]
        if len(x):
            g = x.groupby(["target", "student", "K", "method", "pnorm", "meff", "seed"]).agg(uar=("uar", "mean"), dirloss=("dirloss", "mean")).reset_index()
            s_ = g.groupby(["target", "student", "K", "method", "pnorm", "meff"]).agg(
                n_seeds=("seed", "nunique"), uar_sd=("uar", "std"), uar_range=("uar", lambda v: 100 * (v.max() - v.min())),
                dir_sd=("dirloss", "std")).reset_index()                       # 種の幅は UAR の差でも読む（1 つの感情だけを予測する解に崩れる組があるため）
            seeds.append(s_[s_.n_seeds > 1].assign(run=run))
    if seeds:
        pd.concat(seeds).to_csv(out / "seeds.csv", index=False)
    # 2 つの山の模型（E75 の決まり：誤差 ÷ 損 < 1/3．誤差は模型が正しいときの標本の揺れの中央値を引いてから測る）
    fit = []
    for run, P in S.point.items():
        nf = Path(args.out) / "measure" / f"model_null_{run}.csv"
        if "e_model_abs" not in P.columns or not nf.exists():
            continue
        nul = pd.read_csv(nf).set_index(["target", "student", "rep"])
        x = P[P.e_model_abs.notna()].copy()
        if x.empty:
            continue
        key = list(zip(x.target, x.student, x.rep))
        x["e_null_median"] = [nul.e_null_median.get(k, np.nan) for k in key]
        x["e_null_95"] = [nul.e_null_95.get(k, np.nan) for k in key]
        x["e_adj"] = (x.e_model_abs - x.e_null_median).clip(lower=0)
        for (st, K), g in x.groupby(["student", "K"]):
            md, mp = g.dirloss.abs().median(), g.posloss.abs().median()
            fit.append({"run": run, "student": st, "K": K, "n_rows": len(g), "err_median": g.e_model_abs.median(),
                        "err_null_median": g.e_null_median.median(), "err_adj_median": g.e_adj.median(),
                        "ratio_dir_raw": g.e_model_abs.median() / md if md else np.nan, "ratio_dir_adj": g.e_adj.median() / md if md else np.nan,
                        "ratio_pos_adj": g.e_adj.median() / mp if mp else np.nan,
                        "holds_dir": (g.e_adj.median() / md < MODEL_RATIO_MAX) if md else False,
                        "holds_pos": (g.e_adj.median() / mp < MODEL_RATIO_MAX) if mp else False,
                        "frac_over_null95": float((g.e_model_abs > g.e_null_95).mean())})
    if fit:
        pd.DataFrame(fit).to_csv(out / "model_fit.csv", index=False)
    vr = []                                                                    # 等分散（記述だけ）
    for run in dirs:
        f = Path(args.out) / "measure" / f"pairs_{run}.csv.gz"
        if f.exists():
            p = pd.read_csv(f)
            p["vr_in"] = p.var_ratio.between(*VR_RANGE)
            vr.append(p.groupby(["target", "student"]).vr_in.mean().reset_index().assign(run=run))
    if vr:
        pd.concat(vr).to_csv(out / "model_var_ratio.csv", index=False)
    da = []
    for run in dirs:
        f = Path(args.out) / "measure" / f"diag_{run}.csv"
        if not f.exists():
            continue
        dg = pd.read_csv(f, dtype={"K": str})
        for sm in ("dann", "mme"):                                             # DA の項：K ≥ 1 は a > b かつ a ≥ 1%，K = 0 は a ≥ 1%
            a = dg[dg.what == f"{sm}_vs_mapctl_disagree"].groupby(["target", "student", "K"]).value.mean().rename("a_vs_mapctl")
            b = dg[dg.what == f"{sm}_seed_disagree_max"].groupby(["target", "student", "K"]).value.mean().rename("b_seed_max")
            nd = dg[dg.what == f"{sm}_vs_none_disagree"].groupby(["target", "student", "K"]).value.mean().rename("vs_none")
            g = pd.concat([a, b, nd], axis=1).reset_index()
            if g.empty:
                continue
            for col in ("anorm", "dom_first", "dom_last", "ent_first", "ent_last"):
                x = dg[dg.what == f"{sm}_{col}"].set_index(["target", "student", "K"]).value
                if len(x):
                    g[col] = [x.get((t, s_, k), np.nan) for t, s_, k in zip(g.target, g.student, g.K)]
            g["da_changed_predictions"] = [(av >= DISAGREE_MIN) if k == "0" else (av >= DISAGREE_MIN and av > bv)
                                           for av, bv, k in zip(g.a_vs_mapctl, g.b_seed_max, g.K)]
            da.append(g.assign(run=run, method=sm))
    if da:
        pd.concat(da).to_csv(out / "da_effect.csv", index=False)
    # 「働かなかった」の表（関門 1 の追記）：P4 の族の単位ごとに，基準の量（a：DANN・MME は mapctl との食い違い，CORAL は coralctl との食い違い，
    # entmin・IM は種 0 と none の食い違い，ほかは none との食い違い．b：DANN・MME の種の間の食い違いの最大），置き換える前と後の判定，向きの損の点の値
    ia = U[U.family.str.match(r"^P4") & U.inactive_a.notna() & (U.metric.isin(["dir", "dir_rev"]))]
    if len(ia):
        ia[["family", "comparison", "metric", "sensitivity", "run", "student", "K", "corpus", "variant", "inactive_a", "inactive_b", "inactive",
            "judgment_raw", "judgment", "point"]].to_csv(out / "inactive.csv", index=False)
    # 対照との食い違い（関門 1 の C-2）：none との判定と対照との判定がコーパスで食い違う単位
    M = U[(U.sensitivity == "main") & (U.metric == "dir")]
    a = M[M.family.isin(["P4", "P4'", "P4_uni_world"]) & M.comparison.isin(["none-coral", "none-dann", "none-mme"])].assign(
        method=lambda d: d.comparison.str.split("-").str[1])
    b = M[M.family.str.endswith("_ctl")].assign(method=lambda d: d.comparison.str.split("-").str[1])
    if len(a) and len(b):
        j = a.merge(b, on=["run", "student", "K", "corpus", "method"], suffixes=("_none", "_ctl"))
        j["disagree"] = j.judgment_none != j.judgment_ctl
        j[["run", "student", "K", "corpus", "method", "judgment_none", "judgment_ctl", "point_none", "point_ctl", "disagree"]].to_csv(
            out / "ctl_disagree.csv", index=False)
    idc = []                                                                   # 定義から d が同じになる組（massbal・saerens と none，T3A の補助と主）
    dcols = [f"d{p}" for p in range(6)]
    for run, P in S.point.items():
        base = P[P.method.isin(["none", "massbal", "saerens", "t3a", "t3a_b"]) & (P.wave == "e77")]
        for (T, st, K, rp), g in base.groupby(["target", "student", "K", "rep"]):
            nn_ = g[g.method == "none"]
            for m_ in ("massbal", "saerens"):
                x = g[g.method == m_]
                if len(nn_) and len(x):
                    idc.append({"run": run, "target": T, "student": st, "K": K, "rep": rp, "pair": f"{m_}-none",
                                "d_absdiff_max": float(np.nanmax(np.abs(x[dcols].to_numpy() - nn_[dcols].to_numpy())))})
            a_, b_ = g[g.method == "t3a"].set_index("param"), g[g.method == "t3a_b"].set_index("param")
            common = a_.index.intersection(b_.index)
            if len(common):
                idc.append({"run": run, "target": T, "student": st, "K": K, "rep": rp, "pair": "t3a_b-t3a",
                            "d_absdiff_max": float(np.nanmax(np.abs(a_.loc[common, dcols].to_numpy() - b_.loc[common, dcols].to_numpy())))})
    if idc:
        pd.DataFrame(idc).to_csv(out / "identity_checks.csv", index=False)
    bf = []                                                                    # 切片で取り戻せる分（UAR − 切片だけを直した UAR）が負の行の数（関門 1 の define の D7）
    for run, P in S.point.items():
        if "uar_bfix" in P.columns:
            x = P[P.uar_bfix.notna()]
            if len(x):
                o = 100 * x.uar - x.uar_bfix
                bf.append({"run": run, "n_rows": int(len(x)), "n_negative": int((o > 0).sum()), "n_fit_failed": int((x.bias_fit_ok == 0).sum())})
    if bf:
        pd.DataFrame(bf).to_csv(out / "bias_fixed_negative.csv", index=False)


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("delta", "select", "measure", "judge"))
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=RUNS)
    ap.add_argument("--e77", nargs="+", help="名前=場所（既定 results/e77_<名前>）")
    ap.add_argument("--e78", nargs="+", help="名前=場所（既定 results/e78_<名前>）")
    ap.add_argument("--e80c", nargs="+", help="名前=場所（世界の実行だけ．既定 results/e80c_<名前>．E80 の段 c の出力．E77 の行に足して読む）")
    ap.add_argument("--targets", nargs="+", help="試走用．評価のターゲットを限る")
    ap.add_argument("--reps", nargs="+", type=int, default=list(REPS))
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=list(PSEUDO_REPS))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e79")
    ap.add_argument("--families", choices=("all", "p4"), default="all",
                    help="p4：P4・P4′ の族（群 B の 8 手法とその対照・記述・固定版）だけを選び・測り・判定する．kept.csv を読まないので，"
                         "E80 で E77 に無い教師が残っても（段 c を待たずに）流せる．select・measure・judge で同じ値を渡す（違えば止まる）")
    ap.add_argument("--e80-kept", type=Path, default=ROOT / "results" / "e80" / "kept.csv",
                    help="E80 の R3′〜R3‴ で残した教師の表（世界, target, 生徒, 教師, K, keep）．select の段が読む")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--no-bias-fixed", dest="bias_fixed", action="store_false", help="切片だけを直した UAR を計算しない（試走用）")
    ap.add_argument("--allow-incomplete", action="store_true", help="試走用．欠けと食い違いを止めずに checks.csv に記録する")
    ap.add_argument("--skip-commit-check", action="store_true", help="試走用．δ・chosen.csv・fixed.csv のコミットの確認を飛ばす（provenance に残す）")
    ap.add_argument("--allow-dirty", action="store_true", help="試走用．exp/ か measure/ に未コミットの変更があっても流す（provenance に残す）")
    args = ap.parse_args()
    args._t0 = time.time()
    args._t0_str = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dirs = parse_dirs(args)
    roots = [d for pair in dirs.values() for d in pair if d is not None] + list(E80C.values())
    check_dirty(args)
    out = prepare_out(args.out, args.stage)
    install_guard(args.stage, roots)
    if args.stage in ALLOW:
        probe = guard_selftest(args.stage, [str(d) for d in roots])
        print(f"見張り：{args.stage} の段は入力の場所で {ALLOW[args.stage]} だけを開く（{Path(probe).name} を開けないことを確かめた）", flush=True)
    {"delta": stage_delta, "select": stage_select, "measure": stage_measure, "judge": stage_judge}[args.stage](args, dirs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
