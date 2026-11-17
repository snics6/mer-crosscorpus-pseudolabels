"""【本番の解析】E76（DAI 論文のステップ 4）：筋 1〜3 の学び直しの出力（`results/e76/`）から，設計 2.4 節の解析をすべて CPU で計算する。

設計は `docs/exploratory-analyses.md` の「## E76．」の節（2.4.0〜2.4.6．関門 1 の直しを含む版．`DAI_paper/steps/step2_design.md`
の 2.4 節と食い違うところは前者に従う）。判定と書き方の決まりは値を見る前に書かれたもので，この台本はそれを機械的に当てる。
学び直しはしない（GPU を使わない．起動の最初に CUDA_VISIBLE_DEVICES を空にする）。

節（出力の名前の頭の数字）
  0  共通（2.4.0）：人物の数 G と実効の数 G_eff，テストの集合の数，引き直しと除いた回数，A(HuBERT) の注の一覧
  1  地図の区間（2.4.1）：組 × K の利得（③ − ①）の 2 段のブートストラップ（人物 × 反復，B = 20,000）と，G_eff < 20 のターゲットでの
     jackknife の t 区間の和集合，Holm（種類 × K の族），「上」「下」，K*（主：続く最後の K．補足：最大の K），副の区間（補正した
     resampled t），反復を独立とみなした t 区間，1 段目だけの区間，反復 0〜29 の平均（地図の CSV）
  2  性別とテストの部分集合（2.4.2）：性別ごとの利得と女性 − 男性の差（記述），MOSEI のチャンネルの分かる発話，IEMOCAP 台本の台詞の重なり
  3  ターゲットのみの C の規則（2.4.3）：主・仮のターゲット・C 固定 4 本の学習曲線，K=1・5 の利得，K*，q（D と D′，3 点と 5 点，
     D を固定した区間，M1・M2，K を 1 つ抜いた予測），d_T(K) > D の数，n の実測と予言，順位相関
  4  主張 1 の中心合わせの手（2.4.4）：G_crit 〜 G_merge_c，d と o，floor，R_o*・R_o・R_d，o と UAR の橋，l2 の感度，プールの平均の版，
     6 対を同時に直す切片（オラクル）
  5  手の比較の表（2.4.5）：手 1〜11 の「ソースのみからの差」，h7・h11 とほかの手の差の区間，WEIGHTED の α（メタ検証・オラクル）
  6  仮定の表（2.4.6）：H1〜H6
  判定（e76a_judgments.csv）：各節の「判定と書き方の決まり」の表の行ごとに，条件が成り立つかと値

    CUDA_VISIBLE_DEVICES= ./.venv/bin/python exp/analyze_e76.py --out results/e76_analysis           # 本番（8 種類）
    CUDA_VISIBLE_DEVICES= ./.venv/bin/python exp/analyze_e76.py --kinds audio_wavlm --targets IEMOCAP-script RAVDESS \
        --B 2000 --out <scratchpad>/e76a_test --allow-dirty                                               # 小さく試す

入力：`--e76`（既定 `results/e76/`：<kind>/<T>.npz・.csv・.json と check_rule_finish.json），`--fixc-map`（反復 0〜29 の補足の行），
`--e13`（E13 の own_d・d．A(HuBERT) は `explore_pairs.measure('audio_hubert')` で測り直して出力に残す），`--compare-prefix`
（E31 の P1 の UAR．n の実測），`data/metadata/`（性別，チャンネル，転写），特徴（`measure/data.load_any`．プールの平均と係数の検算）。
E76 の json の code_hash は，E76 の本番（9891f0668848…）と，A(HuBERT) の包み台本の 2 回（`check_rule_finish.json` の written）だけを
受け入れ，照合に落ちたまま使うファイルは `check_rule_finish.json` の accepted_existing・waived に載ったものだけを受け入れる。
出力（`--out`．既にファイルがあれば止まる）：`e76a_*.csv`（1 行目に `#` で設定とコミット番号），`checks.csv`（検算），
`records.csv`（数え上げの記録），`provenance.json`（コミット，台本と部品の sha256，入力のファイルの sha256，所要時間）。
画面には行の数，検算の結果，所要時間だけを出す（UAR・利得の値は出さない）。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                         # GPU を使わない．import より前に置く
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

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
from scipy import optimize, sparse, stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _v in ("MERDA_OUT_ROOT", "MERDA_MAP_ROOT"):
    if os.environ.get(_v):
        raise SystemExit(f"環境変数 {_v} が与えられている．外してから流す")
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(ROOT / "features"))
import explore_compare as EC  # noqa: E402   E31 の P1 の UAR（n の実測．E32 と同じ読み方）
import map as M  # noqa: E402               choose_c（C の同点の規則）と C_GRID・K_GRID だけを使う
from ablate import fit_class_bias  # noqa: E402
from analyze_e79 import jk_interval  # noqa: E402
from data import GROUP_SPEC, groups_for, load_any  # noqa: E402
from explore_direction_bound import q_fit  # noqa: E402
from explore_pairs import ba, measure  # noqa: E402
from explore_source_value import KS_CURVE, n_model, n_of, rankable  # noqa: E402
from explore_synergy import PAIRS, d_o  # noqa: E402
from pair_table import KSTAR_RANK, holm  # noqa: E402
from reanalysis_e75 import Q_GRID, joint_offset, loc_loss_clipped, pair_acc, spear, within_target_perm  # noqa: E402
from summarize_map import t_interval  # noqa: E402

KINDS_ALL = ["audio_wavlm", "video", "text_roberta", "fuse-audio-video", "fuse-audio-text",
             "fuse-video-text", "fuse-audio-video-text", "audio_hubert"]
SHORT = {"audio_wavlm": "A", "audio_hubert": "A(HuBERT)", "video": "V", "fuse-audio-video": "AV",
         "text_roberta": "T", "fuse-audio-text": "AT", "fuse-video-text": "VT", "fuse-audio-video-text": "AVT"}
TEXT_KINDS = ("text_roberta", "fuse-audio-text", "fuse-video-text", "fuse-audio-video-text")
REPS = tuple(range(10))
KS = M.K_GRID                                # ("1", "5", "10", "25", "50", "all")
KH = ("1", "5", "10")                        # 手の K
C_GRID = M.C_GRID                            # (0.01, 0.1, 1.0, 10.0)
ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)         # 手 10 の α（relearn_e76.ALPHAS と同じ．読んだ npz の設定と照合する）
L2_MAIN, L2_ALT = 1e-2, (1e-3, 1e-1)         # 切片のずらしの罰則（2.4.4 の B8）
HANDS = ("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11")
SUBJECTS = ("h7", "h11")
RULES = ("main", "pseudo", "C0.01", "C0.1", "C1", "C10")
B_DEFAULT = 20_000
GEFF_MIN = 20                                 # これ未満のターゲットを含む単位は百分位と jackknife の t 区間の和集合
GENDER_CORPORA = {"IEMOCAP-impro", "IEMOCAP-script", "AFEW", "RAVDESS"}
SCRIPT = "IEMOCAP-script"
E76_MAIN_HASH = "9891f06688489b66ee97f5b04814f87a7f888454f1bcbb0eda15b969671b7ee6"
EXPECTED_SETTINGS = {"reps": list(REPS), "K": list(KS), "hands": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "merged_all_c": False,
                     "sources": None, "C_grid": list(C_GRID), "K_hands": list(KH), "ablate_l2": L2_MAIN, "alphas": list(ALPHAS)}
EXPECTED_PAIRS = {"audio_wavlm": 40, "audio_hubert": 40, "video": 28, "fuse-audio-video": 28,
                  "text_roberta": 10, "fuse-audio-text": 10, "fuse-video-text": 10, "fuse-audio-video-text": 10}
EXPECTED_TARGETS = {"audio_wavlm": 7, "audio_hubert": 7, "video": 6, "fuse-audio-video": 6,
                    "text_roberta": 4, "fuse-audio-text": 4, "fuse-video-text": 4, "fuse-audio-video-text": 4}
M2_B_GRID = np.linspace(-1.0, 2.0, 3001)     # M2（d = a K^b）の b の格子（a は閉じた形）
NAME_PAIRS = [f"{a}-{b}" for a, b in PAIRS]
EMO = ["neutral", "happy", "sad", "angry"]
E13_NAME_TO_IDX = {f"{EMO[a]}-{EMO[b]}": f"{a}-{b}" for a, b in PAIRS}


# ---------------------------------------------------------------- 記録

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def module_state() -> dict:
    """読み込んだ MER-DA の部品の git の状態と sha256。"""
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
        out[str(rel)] = {"status": git("status", "--porcelain", "--", str(rel)) or "clean",
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


class Ctx:
    """引数，出力の表，検算，記録，入力のファイルの sha256 を持つ。"""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(a.out)
        self.tables: dict[str, list[dict]] = {}
        self.checks: list[dict] = []
        self.records: list[dict] = []
        self.inputs: dict[str, str] = {}
        self.timing: dict[str, float] = {}
        self.header = {}

    def note_input(self, path: Path) -> None:
        p = str(Path(path).resolve())
        if p not in self.inputs and Path(p).is_file():
            self.inputs[p] = sha256(Path(p))

    def add(self, table: str, rows) -> None:
        if isinstance(rows, dict):
            rows = [rows]
        self.tables.setdefault(table, []).extend(rows)

    def check(self, section: str, kind: str, what: str, value: float, limit: float, note: str = "") -> None:
        ok = bool(np.isfinite(value) and value <= limit)
        self.checks.append({"section": section, "kind": kind, "what": what, "value": float(value), "limit": limit,
                            "ok": ok, "note": note})

    def record(self, section: str, kind: str, what: str, value, note: str = "") -> None:
        self.records.append({"section": section, "kind": kind, "what": what, "value": value, "note": note})

    def write_all(self) -> list[tuple[str, int]]:
        written = []
        head = "# " + json.dumps(self.header, ensure_ascii=False) + "\n"
        for name, rows in sorted(self.tables.items()):
            df = pd.DataFrame(rows)
            path = self.out / f"{name}.csv"
            with open(path, "w", encoding="utf-8") as f:
                f.write(head)
                df.to_csv(f, index=False)
            written.append((path.name, len(df)))
        for name, rows in (("checks", self.checks), ("records", self.records)):
            with open(self.out / f"{name}.csv", "w", encoding="utf-8") as f:
                f.write(head)
                pd.DataFrame(rows).to_csv(f, index=False)
            written.append((f"{name}.csv", len(rows)))
        return written


def corpus(g: str) -> str:
    return GROUP_SPEC[g][0]


def same_corpus(t: str, s: str) -> bool:
    return corpus(t) == corpus(s)


# ---------------------------------------------------------------- 入力（E76 の出力）

class Target:
    """1 つの（種類，ターゲット）の E76 の出力。npz は遅延で読む。"""

    def __init__(self, ctx: Ctx, kind: str, name: str, e76: Path, load_npz: bool = True):
        self.kind, self.name, self.corpus = kind, name, corpus(name)
        d = e76 / kind
        for ext in ("csv", "json") + (("npz",) if load_npz else ()):
            ctx.note_input(d / f"{name}.{ext}")
        self.meta = json.loads((d / f"{name}.json").read_text())
        c = pd.read_csv(d / f"{name}.csv", comment="#", dtype={"K": str, "source": str})
        c["K"], c["source"] = c.K.fillna(""), c.source.fillna("")
        self.csv = c
        self.sources = list(self.meta["sources"])
        if not load_npz:
            return
        self.z = np.load(d / f"{name}.npz")
        z = self.z
        self.y = z["y"].astype(np.int64)
        self.sid = z["sample_id"].astype(str)
        self.sg = z["speaker_group"].astype(str)
        self.person = np.char.add(self.corpus + ":", self.sg)
        if list(z["reps"]) != list(REPS):
            raise SystemExit(f"{kind} {name}：反復が {list(z['reps'])}")
        self.te = {r: np.asarray(z[f"test_idx_r{r}"], np.int64) for r in REPS}
        self.kidx = {(r, K): np.asarray(z[f"kidx_r{r}_K{K}"], np.int64) for r in REPS for K in KS if K != "all"}
        self._cache: dict = {}

    def a(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = self.z[key]
        return self._cache[key]

    def drop(self) -> None:
        self._cache.clear()

    def src_ver(self, s: str) -> np.ndarray:
        return np.asarray(self.a(f"s_{s}__ver"), np.int64)

    def src_C(self, s: str, j: int) -> float:
        return float(self.a(f"s_{s}__v{j}_C"))


def accepted_hashes(ctx: Ctx, e76: Path) -> tuple[dict, dict]:
    """A(HuBERT) の包み台本が書いたファイルのハッシュと，照合に落ちたまま使うファイルの一覧（`check_rule_finish.json`）。"""
    p = e76 / "check_rule_finish.json"
    ctx.note_input(p)
    fin = json.loads(p.read_text())
    written, accepted = {}, {}
    for run in fin["runs"]:
        if run.get("code_hash_without_wrapper") != E76_MAIN_HASH:
            raise SystemExit(f"check_rule_finish.json の包み台本を除いたハッシュが E76 の本番と違う：{run.get('code_hash_without_wrapper')}")
        written.update(run.get("written", {}))
        for k, v in list(run.get("accepted_existing", {}).items()) + list(run.get("waived", {}).items()):
            accepted.setdefault(k, set()).update(v)
    return written, accepted


def check_target_meta(ctx: Ctx, t: Target, written: dict, accepted: dict) -> str:
    """コードのハッシュ，設定，照合に落ちた記録を確かめる。受け入れたハッシュの種類を返す。"""
    m = t.meta
    key = f"{t.kind}/{t.name}"
    h = m.get("code_hash")
    if h == E76_MAIN_HASH:
        tag = "e76_main"
    elif written.get(key) == h:
        tag = "e76_finish_wrapper"
    else:
        raise SystemExit(f"{key}：受け入れないコードのハッシュ {h}")
    if m.get("code_dirty"):
        raise SystemExit(f"{key}：code_dirty が真")
    s = m.get("settings", {})
    for k, v in EXPECTED_SETTINGS.items():
        if s.get(k) != v:
            raise SystemExit(f"{key}：設定 {k} が {s.get(k)}（期待 {v}）")
    bad = m.get("check_failed") or []
    if bad:
        if key not in accepted or not set(bad) <= accepted[key]:
            raise SystemExit(f"{key}：照合に落ちた記録があり，check_rule_finish.json で受け入れられていない：{bad}")
    return tag


def hubert_notes(t: Target) -> pd.DataFrame:
    """照合で tol を超えた条件と，① の C が地図と違う条件（E76 の 10 節の A(HuBERT) の注）。どの種類でも作る（A(HuBERT) 以外は空のはず）。"""
    c = t.csv
    over = c[pd.to_numeric(c["diff"], errors="coerce") > pd.to_numeric(c["tol"], errors="coerce")]
    rows = [{"kind": t.kind, "target": t.name, "source": x.source, "method": x.method, "K": x.K, "rep": int(x.rep),
             "alpha": x.alpha, "C": x.C, "what": "tol を超えた照合"} for x in over.itertuples()]
    tg = c[(c.method == "tgt_only") & c.ref_C.notna()]
    mis = tg[~np.isclose(tg.C.astype(float), tg.ref_C.astype(float))]
    rows += [{"kind": t.kind, "target": t.name, "source": "", "method": "tgt_only", "K": x.K, "rep": int(x.rep),
              "alpha": np.nan, "C": x.C, "what": f"① の C が地図と違う（地図 {x.ref_C}，規則 {x.ref_rule}）"} for x in mis.itertuples()]
    return pd.DataFrame(rows)


def kind_note(notes: pd.DataFrame) -> str:
    if notes is None or notes.empty:
        return ""
    srcs = sorted(set(notes.source.dropna()) - {""})
    return (f"照合で tol を超えた {int((notes.method != 'tgt_only').sum())} 条件（ソース {','.join(srcs)}）と ① の C が地図と違う "
            f"{int((notes.method == 'tgt_only').sum())} 条件を含む．e76a_0_hubert_notes を参照（E76 の 10 節）")


def note_for(notes: pd.DataFrame, target: str, source: str | None = None, K: str | None = None) -> str:
    if notes is None or notes.empty:
        return ""
    n = notes[notes.target == target]
    if source is not None:
        n = n[(n.source == source) | ((n.method == "tgt_only") & ((K is None) | (n.K == K)))]
    if n.empty:
        return ""
    return "; ".join(f"{x.method} {x.source or '-'} K={x.K or '-'} r{x.rep}"
                     + (f" a={x.alpha:g}" if isinstance(x.alpha, float) and np.isfinite(x.alpha) else "")
                     + f" ({x.what})" for x in n.itertuples())


# ---------------------------------------------------------------- 引き方（2.4.0）

class Draws:
    """1 つの種類の 2 段の引き方．1 段目：コーパスごとに人物を戻して引く（IEMOCAP の 2 群は同じ人物で同じ重み）．
    2 段目：反復 0〜9 を戻して引く（種類のすべてのターゲットで共通）．引いた反復がすべて除かれるターゲットがあれば 1 段目を引き直す。"""

    def __init__(self, kind: str, targets: list[Target], B: int, seed: int, tag: str = ""):
        self.B = B
        self.corpora = sorted({t.corpus for t in targets})
        self.persons: dict[str, np.ndarray] = {}
        for c in self.corpora:
            ps = set()
            for t in targets:
                if t.corpus == c:
                    for r in REPS:
                        ps.update(t.person[t.te[r]].tolist())
            self.persons[c] = np.array(sorted(ps))
        self.pindex = {c: {p: i for i, p in enumerate(v)} for c, v in self.persons.items()}
        for t in targets:
            idx = self.pindex[t.corpus]
            t.pidx = np.array([idx.get(p, -1) for p in t.person], np.int64)
            t.tp = {r: np.unique(t.pidx[t.te[r]]) for r in REPS}
        base = [seed, zlib.crc32(kind.encode())] + ([zlib.crc32(tag.encode())] if tag else [])
        self.W = {c: np.random.default_rng(base + [zlib.crc32(c.encode())]).multinomial(
            len(self.persons[c]), np.full(len(self.persons[c]), 1 / len(self.persons[c])), size=B).astype(np.float32)
            for c in self.corpora}
        self.A = np.random.default_rng(base + [zlib.crc32(b"reps")]).multinomial(
            len(REPS), np.full(len(REPS), 1 / len(REPS)), size=B).astype(np.float64)
        rng_re = np.random.default_rng(base + [zlib.crc32(b"redraw")])
        self.redraws = {t.name: 0 for t in targets}
        for _ in range(1000):
            bad = np.zeros(B, bool)
            for t in targets:
                v = self.valid(t)
                bt = ((self.A > 0) & v).sum(1) == 0
                self.redraws[t.name] += int(bt.sum())
                bad |= bt
            if not bad.any():
                break
            nb = int(bad.sum())
            for c in self.corpora:
                P = len(self.persons[c])
                self.W[c][bad] = rng_re.multinomial(P, np.full(P, 1 / P), size=nb).astype(np.float32)
        else:
            raise SystemExit(f"{kind}：1 段目の引き直しが終わらない")

    def valid(self, t: Target) -> np.ndarray:
        """(B, R)：反復のテストの人物の重みの和が正か。"""
        W = self.W[t.corpus]
        return np.stack([W[:, t.tp[r]].sum(1) > 0 for r in REPS], 1)


# ---------------------------------------------------------------- 反復 1 つの重みつきの量

def person_indicator(pl: np.ndarray, y: np.ndarray, P: int) -> sparse.csr_matrix:
    """(P × 4) 行 × 発話 の指示行列（人物 × 感情）。"""
    n = len(y)
    return sparse.csr_matrix((np.ones(n), (pl * 4 + y, np.arange(n))), shape=(P * 4, n))


def uar_from_counts(Cc: np.ndarray, Nc: np.ndarray, axis: int = -2) -> np.ndarray:
    """axis が感情の軸．重みが 0 の感情を除いた感情で再現率を平均する．すべて 0 なら nan。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = Nc > 0
        rec = np.where(ok, Cc / np.where(ok, Nc, 1), 0.0)
        n_ok = ok.sum(axis)
        return np.where(n_ok > 0, rec.sum(axis) / np.maximum(n_ok, 1), np.nan)


class Acc:
    """条件ごとの反復をまたぐ集計：2 段の値，1 段目だけの値，全データの値，人物を 1 人ずつ抜いた値（コーパスの人物の並び）。"""

    def __init__(self, B: int, P: int):
        self.n2 = np.zeros(B)
        self.d2 = np.zeros(B)
        self.n1 = np.zeros(B)
        self.d1 = np.zeros(B)
        self.full = np.full(len(REPS), np.nan)
        self.jn = np.zeros(P)
        self.jd = np.zeros(P)

    def finish(self) -> dict:
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = np.isfinite(self.full)
            th = float(self.full[ok].mean()) if ok.any() else np.nan
            return {"theta": th, "b2": (self.n2 / self.d2).astype(np.float32), "b1": (self.n1 / self.d1).astype(np.float32),
                    "jk": self.jn / self.jd, "full": self.full.copy()}


def rep_engine(Wr: np.ndarray, pl: np.ndarray, y: np.ndarray, preds: np.ndarray, masks: np.ndarray, mask_of: np.ndarray,
               chunk: int = 96):
    """1 つの（ターゲット，反復）のテストで，条件ごとの重みつき UAR を引き方ごとに出す。
    Wr：(B, P_r) 人物の重み．pl：発話ごとの人物の番号（0..P_r−1）．preds：(M, n) 予測．masks：(nm, n) 使う発話．mask_of：(M,)。
    戻り：U (B, M)，full (M,)，loo (P_r, M)（その人物を除いた値），Nm（(P_r, 4, nm) 人物 × 感情の本数）。"""
    P = Wr.shape[1]
    S = person_indicator(pl, y, P)
    Nm = np.asarray((S @ masks.T.astype(np.float64))).reshape(P, 4, -1)                      # (P, 4, nm)
    WN = (Wr @ Nm.reshape(P, -1).astype(np.float32)).reshape(len(Wr), 4, -1)               # (B, 4, nm)
    M_ = len(preds)
    U = np.empty((len(Wr), M_), np.float32)
    corr = (preds == y[None, :]) & masks[mask_of]
    Cm = np.asarray((S @ corr.T.astype(np.float64))).reshape(P, 4, M_)                       # (P, 4, M)
    for i in range(0, M_, chunk):
        sl = slice(i, i + chunk)
        WC = (Wr @ Cm[:, :, sl].reshape(P, -1).astype(np.float32)).reshape(len(Wr), 4, -1)
        U[:, sl] = uar_from_counts(WC, WN[:, :, mask_of[sl]])
    Ct, Nt = Cm.sum(0), Nm.sum(0)                                                              # (4, M), (4, nm)
    full = uar_from_counts(Ct, Nt[:, mask_of])
    loo = uar_from_counts(Ct[None] - Cm, (Nt[None] - Nm)[:, :, mask_of])                     # (P, M)
    return U, full, loo, Nm, WN


def accumulate(accs: list[Acc], U: np.ndarray, full: np.ndarray, loo: np.ndarray, A_r: np.ndarray, tp: np.ndarray, r: int) -> None:
    for m, acc in enumerate(accs):
        u = U[:, m].astype(np.float64)
        ok = np.isfinite(u)
        uu = np.where(ok, u, 0.0)
        acc.n2 += A_r * uu
        acc.d2 += A_r * ok
        acc.n1 += uu
        acc.d1 += ok
        f = full[m]
        acc.full[r] = f
        if not np.isfinite(f):
            continue
        acc.jn += f
        acc.jd += 1
        lo = loo[:, m]
        okl = np.isfinite(lo)
        acc.jn[tp] += np.where(okl, lo - f, -f)
        acc.jd[tp] -= (~okl).astype(float)


def weighted_do(Wr: np.ndarray, pl: np.ndarray, y: np.ndarray, Ls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.4.0 の重みつきの d（引き方ごと）と全データの d。Ls：(Q, n, 4)。戻り：d_b (B, Q, 6)，d_full (Q, 6)。"""
    P = Wr.shape[1]
    S = person_indicator(pl, y, P)
    Q = len(Ls)
    feats = []
    for k, j in PAIRS:
        z = Ls[:, :, k].astype(np.float64) - Ls[:, :, j]                                    # (Q, n)
        for c in (k, j):
            m = (y == c).astype(np.float64)
            feats += [z * m, z * z * m]
    F = np.stack(feats, 1)                                                                    # (Q, 24, n)
    G = np.asarray((S @ F.reshape(-1, F.shape[-1]).T))                                        # (P*4, Q*24)
    G = G.reshape(P, 4, Q * 24)
    # 人物 × 感情の和のうち，その特徴の感情の行だけが非零（z·1{y=c} は感情 c の行にしか無い）ので感情の軸を足してよい
    Gp = G.sum(1)                                                                             # (P, Q*24)
    Np = np.asarray((S @ np.ones(len(y)))).reshape(P, 4)
    Wd = Wr.astype(np.float64)
    WG = (Wd @ Gp).reshape(len(Wr), Q, 6, 4)
    WN = Wd @ Np                                                                              # (B, 4)
    Gt = Gp.sum(0).reshape(Q, 6, 4)
    Nt = Np.sum(0)

    def dval(Sm, N):
        out = []
        for p, (k, j) in enumerate(PAIRS):
            nk, nj = N[..., k], N[..., j]
            s1k, s2k, s1j, s2j = Sm[..., p, 0], Sm[..., p, 1], Sm[..., p, 2], Sm[..., p, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                mk, mj = s1k / nk, s1j / nj
                vk = (s2k - nk * mk ** 2) / (nk - 1)
                vj = (s2j - nj * mj ** 2) / (nj - 1)
                sd = np.sqrt(np.clip((vk + vj) / 2, 0, None))
                ok = (nk >= 2) & (nj >= 2) & (sd > 0)
                out.append(np.where(ok, (mk - mj) / sd, np.nan))
        return np.stack(out, -1)
    return dval(WG, WN[:, None, :]).astype(np.float32), dval(Gt, Nt[None, :])


def do_stats(L: np.ndarray, y: np.ndarray) -> dict:
    """全データの d，o，s，実際の対の正解率（E13 の定義．ddof = 1）。"""
    d, o, s, acc = [], [], [], []
    for k, j in PAIRS:
        zk, zj = L[y == k, k] - L[y == k, j], L[y == j, k] - L[y == j, j]
        sd = np.sqrt((zk.var(ddof=1) + zj.var(ddof=1)) / 2)
        d.append((zk.mean() - zj.mean()) / sd)
        o.append(-(zk.mean() + zj.mean()) / 2 / sd)
        s.append(sd)
        acc.append(0.5 * ((zk > 0).mean() + (zj < 0).mean()))
    return {"d": np.array(d), "o": np.array(o), "s": np.array(s), "acc": np.array(acc)}


def uar(L: np.ndarray, y: np.ndarray) -> float:
    pred = L.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(4) if (y == c).any()]))


def floor_o(o: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray]:
    """切片のずらし c（c_0 = 0）で 6 対の平均の |o_kl − (c_k − c_l)/s_kl| を最小にした値（線形計画）。"""
    A = np.zeros((6, 3))
    for i, (a, b) in enumerate(PAIRS):
        if a > 0:
            A[i, a - 1] += 1 / s[i]
        if b > 0:
            A[i, b - 1] -= 1 / s[i]
    # 変数 [c (3), t (6)]，t ≥ ±(o − A c)
    cvec = np.r_[np.zeros(3), np.ones(6) / 6]
    Aub = np.block([[-A, -np.eye(6)], [A, -np.eye(6)]])
    bub = np.r_[-o, o]
    r = optimize.linprog(cvec, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * 3 + [(0, None)] * 6, method="highs")
    if not r.success:
        return np.nan, np.full(3, np.nan)
    return float(r.fun), r.x[:3]


# ---------------------------------------------------------------- 区間

def interval(theta: float, b2: np.ndarray, jk: np.ndarray | None, geff_min: float, use_jk: bool) -> dict:
    """2.4.0：百分位の 95% 区間と p 値（(数 + 1)/(B + 1)）．use_jk なら jackknife の t 区間との和集合と p の大きい方。"""
    v = b2[np.isfinite(b2)].astype(np.float64)
    nb = len(v)
    out = {"theta": theta, "n_draws": nb}
    if nb == 0 or not np.isfinite(theta):
        out.update(boot_lo=np.nan, boot_hi=np.nan, p_boot=np.nan, jk_lo=np.nan, jk_hi=np.nan, p_jk=np.nan,
                   lo=np.nan, hi=np.nan, p=np.nan, use_jk=use_jk)
        return out
    lo, hi = np.quantile(v, [0.025, 0.975])
    p = min(1.0, 2 * min(((v <= 0).sum() + 1) / (nb + 1), ((v >= 0).sum() + 1) / (nb + 1)))
    out.update(boot_lo=float(lo), boot_hi=float(hi), p_boot=float(p))
    jlo = jhi = pj = np.nan
    if jk is not None:
        th = jk[np.isfinite(jk)]
        if len(th) >= 2 and np.isfinite(geff_min):
            jlo, jhi = jk_interval(theta, th, geff_min)
            G = len(th)
            se = np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum())
            df = max(1, int(np.floor(geff_min)) - 1)
            pj = float(2 * stats.t.sf(abs(theta) / se, df)) if se > 0 else (0.0 if theta != 0 else 1.0)
    out.update(jk_lo=jlo, jk_hi=jhi, p_jk=pj, use_jk=use_jk)
    if use_jk and np.isfinite(jlo):
        out.update(lo=min(lo, jlo), hi=max(hi, jhi), p=max(p, pj))
    else:
        out.update(lo=float(lo), hi=float(hi), p=float(p))
    return out


def pct(b: np.ndarray) -> tuple[float, float]:
    v = b[np.isfinite(b)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


class Store:
    """1 つの種類の条件ごとの集計（Acc.finish の結果）と，その線形結合の区間。"""

    def __init__(self, draws: Draws, geff: dict):
        self.draws = draws
        self.res: dict[tuple, dict] = {}
        self.geff = geff                                # ターゲット → G_eff
        self.tcorp: dict[str, str] = {}

    def put(self, T: str, key: tuple, r: dict, corp: str) -> None:
        self.res[(T,) + key] = r
        self.tcorp[T] = corp

    def has(self, T: str, key: tuple) -> bool:
        return (T,) + key in self.res

    def combo(self, terms: list[tuple[float, str, tuple]]) -> dict:
        """terms：(係数, ターゲット, 条件の鍵)．値，2 段の値，1 段目だけの値，jackknife の値（関わるコーパスの人物を並べたもの）。"""
        theta = 0.0
        b2 = np.zeros(self.draws.B)
        b1 = np.zeros(self.draws.B)
        full = np.zeros(len(REPS))
        corps = sorted({self.tcorp[T] for _, T, _ in terms})
        jk = {c: np.zeros(len(self.draws.persons[c])) for c in corps}
        for coef, T, key in terms:
            r = self.res[(T,) + key]
            theta += coef * r["theta"]
            b2 += coef * r["b2"]
            b1 += coef * r["b1"]
            full += coef * r["full"]
            for c in corps:
                jk[c] += coef * (r["jk"] if c == self.tcorp[T] else r["theta"])
        targets = sorted({T for _, T, _ in terms})
        gmin = min(self.geff[T] for T in targets)
        return {"theta": theta, "b2": b2, "b1": b1, "full": full, "jk": np.concatenate([jk[c] for c in corps]),
                "geff_min": gmin, "use_jk": bool(gmin < GEFF_MIN), "targets": targets}

    def ival(self, terms, scale: float = 100.0) -> dict:
        c = self.combo(terms)
        iv = interval(c["theta"], c["b2"], c["jk"], c["geff_min"], c["use_jk"])
        l1, h1 = pct(c["b1"])
        out = {k: (v * scale if isinstance(v, float) and k not in ("p_boot", "p_jk", "p") and k != "n_draws" else v)
               for k, v in iv.items()}
        out.update(stage1_lo=l1 * scale, stage1_hi=h1 * scale, geff_min=c["geff_min"])
        out["_full"] = c["full"] * scale
        return out


# ---------------------------------------------------------------- 1 つのターゲットの計算（反復ごと）

def pseudo_rule_C(allcsv: dict, kind: str, T: str) -> dict:
    """2.4.3 の仮のターゲットの規則：T と K ごとに，コーパスが T と違う群 G の ① のテストの UAR（反復 0〜9 の平均）を C ごとに平均し，
    最大の C（同点は map.choose_c と同じ）。T のラベルは使わない。"""
    Gs = [g for g in allcsv if corpus(g) != corpus(T)]
    out = {}
    for K in KS:
        sc = {}
        for c in C_GRID:
            v = []
            for g in Gs:
                d = allcsv[g]
                d = d[(d.method == "tgt_only") & (d.K == K) & d.rep.isin(REPS)]
                v.append(float(d[f"test_uar_C{c:g}"].mean()))
            sc[c] = float(np.mean(v))
        out[K] = M.choose_c(sc)[0]
    return out


def h10_meta_alpha(allcsv: dict, T: str, S: str, K: str) -> float:
    """2.4.5：E31 のメタ検証と同じ規則で α を選ぶ（S をほかのターゲット T′ に使った組の反復 0〜9 の平均の UAR を T′ で平均）。"""
    tp = [u for u in allcsv if u != T and u != S and not same_corpus(u, S) and not same_corpus(u, T)
          and S in set(allcsv[u].source)]
    if not tp:
        return np.nan
    sc = {}
    for a in ALPHAS:
        v = []
        for u in tp:
            d = allcsv[u]
            d = d[(d.method == "h10") & (d.source == S) & (d.K == K) & d.rep.isin(REPS)]
            d = d[np.isclose(d.alpha.astype(float), a)]
            v.append(float(d.uar.mean()))
        sc[a] = float(np.mean(v))
    best = max(sc.values())
    tied = [a for a in ALPHAS if abs(sc[a] - best) <= 1e-12]
    return sorted(tied, key=lambda a: (round(abs(a - 0.5), 9), a))[0]


def script_norm(s: str) -> str:
    """転写を小文字にし，句読点を除き，空白をそろえる（2.4.2 の台詞の重なり）。"""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def run_target(ctx: Ctx, kind: str, t: Target, X: np.ndarray, dr: Draws, store: Store, pseudo: dict,
               notes: pd.DataFrame, meta_tables: dict) -> dict:
    """反復ごとに条件の予測を作り，引き方の集計を Store に入れる。点の値の表（d・o，橋，オラクルの切片，l2）もここで作る。"""
    z = t.z
    y_all = t.y
    P_c = len(dr.persons[t.corpus])
    A = dr.A
    srcs = t.sources
    csvk = {}
    tcsv = {(x["K"], int(x["rep"])): x for x in t.csv[t.csv.method == "tgt_only"].to_dict("records")}
    for x in t.csv.itertuples():
        a = float(x.alpha) if x.method == "h10" else np.nan
        csvk[(x.method, x.source, x.K, int(x.rep), None if np.isnan(a) else round(a, 6))] = float(x.uar)
    diffs = {"uar_vs_csv": [], "h4_b_refit": [], "mean_t": []}
    excl = {"emotion_cells": 0, "emotion_cells_weighted": 0}
    accs: dict[tuple, Acc] = {}
    dR = np.full((len(REPS), dr.B, len(KS), len(C_GRID), 6), np.nan, np.float32)    # ① の重みつき d（2.4.3）
    dF = np.full((len(REPS), len(KS), len(C_GRID), 6), np.nan)
    tC = np.zeros((len(REPS), len(KS)))
    point_rows, joint_rows, h5acc_rows = [], [], []
    mean_t = np.asarray(z["mean_t"], np.float64)
    diffs["mean_t"].append(float(np.abs(mean_t - X.mean(0)).max()))
    ch_known = meta_tables.get("mosei_channel")
    lines = meta_tables.get("script_lines")
    # ソースの点数（全発話．版ごと）
    src = {}
    for s in srcs:
        ver = t.src_ver(s)
        vs = {}
        for j in sorted(set(ver.tolist())):
            C = t.src_C(s, j)
            jc = C_GRID.index(C)
            vs[j] = {"C": C, "L": np.asarray(z[f"s_{s}__v{j}_logit"][jc], np.float64),
                     "h2": np.asarray(z[f"h2_{s}__v{j}_logit"], np.float64),
                     "h2A": np.asarray(z[f"h2_{s}__v{j}_A"], np.float64)}
        src[s] = (ver, vs)
    nall = len(y_all)
    for r in REPS:
        te = t.te[r]
        y = y_all[te]
        pool = np.setdiff1d(np.arange(nall), te)
        tp = t.tp[r]
        loc = {p: i for i, p in enumerate(tp)}
        pl = np.array([loc[p] for p in t.pidx[te]], np.int64)
        Wr = dr.W[t.corpus][:, tp]
        conds: dict[tuple, tuple[np.ndarray, int]] = {}
        masks = [np.ones(len(te), bool)]

        def add(key, L, mask_id=0, csv_key=None):
            pred = L.argmax(1).astype(np.int8)
            conds[key] = (pred, mask_id)
            if csv_key is not None and csv_key in csvk:
                diffs["uar_vs_csv"].append(abs(uar(L, y) - csvk[csv_key]))

        tl = np.asarray(z[f"t_logit_r{r}"], np.float64)                                # (6, 4, n, 4)
        tCr = np.asarray(z[f"t_C_r{r}"], np.float64)
        tC[r] = tCr
        for kk, K in enumerate(KS):
            jc = C_GRID.index(float(tCr[kk]))
            for jj, c in enumerate(C_GRID):
                add(("t", K, c), tl[kk, jj])
                diffs["uar_vs_csv"].append(abs(uar(tl[kk, jj], y) - float(tcsv[(K, r)][f"test_uar_C{c:g}"])))
            add(("tm", K), tl[kk, jc], csv_key=("tgt_only", "", K, r, None))
        # 部分集合（2.4.2）
        ch_mid = None
        if ch_known is not None:
            masks.append(ch_known[te])
            ch_mid = len(masks) - 1
            for kk, K in enumerate(KS):
                add(("sub_ch", "tm", K), tl[kk, C_GRID.index(float(tCr[kk]))], ch_mid)
        line_mid = {}
        if lines is not None:
            for kk, K in enumerate(KS):
                if K == "all":
                    continue
                shot = set(lines[t.kidx[(r, K)]].tolist())
                masks.append(~np.isin(lines[te], list(shot)))
                line_mid[K] = len(masks) - 1
                add(("sub_ln", "tm", K), tl[kk, C_GRID.index(float(tCr[kk]))], line_mid[K])
        # ① の重みつき d（2.4.3）
        Ls = tl.reshape(len(KS) * len(C_GRID), len(te), 4)
        db, df_ = weighted_do(Wr, pl, y, Ls)
        dR[r] = db.reshape(dr.B, len(KS), len(C_GRID), 6)
        dF[r] = df_.reshape(len(KS), len(C_GRID), 6)
        if r == 0:
            ref = np.array([[d for d, _ in d_o(tl[kk, jj], y)] for kk in range(len(KS)) for jj in range(len(C_GRID))])
            diffs["d_vs_d_o"] = float(np.nanmax(np.abs(ref - df_)))
        # ① の H5（実際の対の正解率と模型の正解率）
        for kk, K in enumerate(KS):
            st = do_stats(tl[kk, C_GRID.index(float(tCr[kk]))], y)
            h5acc_rows.append({"method": "tgt_only", "K": K, "rep": r,
                               "absdiff": np.abs(st["acc"] - ba(st["d"], st["o"])).tolist()})
        m_pool = X[pool].mean(0)
        for s in srcs:
            ver, vs = src[s]
            v = vs[int(ver[r])]
            C = v["C"]
            Ls2 = v["L"][te]
            add((s, "src"), Ls2, csv_key=("src_only", s, "", r, None))
            L2c = v["h2"][te]
            add((s, "h2"), L2c, csv_key=("h2", s, "", r, None))
            shift = (mean_t - m_pool) @ v["h2A"].T                                    # プールの平均で中心合わせした版（2.4.4）
            add((s, "h2pool"), L2c + shift)
            ml = np.asarray(z[f"m_{s}_r{r}_logit"], np.float64)                       # (6, n, 4)
            for kk, K in enumerate(KS):
                add((s, "m", K), ml[kk], csv_key=("merged", s, K, r, None))
                if ch_mid is not None:
                    add(("sub_ch", "m", s, K), ml[kk], ch_mid)
                if K in line_mid:
                    add(("sub_ln", "m", s, K), ml[kk], line_mid[K])
            b4 = np.asarray(z[f"h4_{s}_r{r}_b"], np.float64)
            b5 = np.asarray(z[f"h5_{s}_r{r}_b"], np.float64)
            for kh, K in enumerate(KH):
                ki = t.kidx[(r, K)]
                add((s, "h4", K), Ls2 + b4[kh], csv_key=("h4", s, K, r, None))
                add((s, "h5", K), L2c + b5[kh], csv_key=("h5", s, K, r, None))
                if r == 0:
                    diffs["h4_b_refit"].append(float(np.abs(fit_class_bias(v["L"][ki], y_all[ki], l2=L2_MAIN) - b4[kh]).max()))
                    diffs["h4_b_refit"].append(float(np.abs(fit_class_bias(v["h2"][ki], y_all[ki], l2=L2_MAIN) - b5[kh]).max()))
                for l2 in L2_ALT:
                    add((s, "h4l2", l2, K), Ls2 + fit_class_bias(v["L"][ki], y_all[ki], l2=l2))
                    add((s, "h5l2", l2, K), L2c + fit_class_bias(v["h2"][ki], y_all[ki], l2=l2))
                bp = fit_class_bias(v["h2"][ki] + shift, y_all[ki], l2=L2_MAIN)
                add((s, "h5pool", K), L2c + shift + bp)
                for h in (3, 6, 7, 8, 9, 11):
                    add((s, f"h{h}", K), np.asarray(z[f"h{h}_{s}_r{r}_logit"][kh], np.float64),
                        csv_key=(f"h{h}", s, K, r, None))
                h10 = z[f"h10_{s}_r{r}_logit"]
                for ia, a in enumerate(ALPHAS):
                    add((s, "h10", a, K), np.asarray(h10[ia, kh], np.float64), csv_key=("h10", s, K, r, round(a, 6)))
            # 2.4.4 の点の d と o（テスト．反復ごと）
            if not same_corpus(t.name, s):
                pr = {"src": do_stats(Ls2, y), "h2": do_stats(L2c, y)}
                for kh, K in enumerate(KH):
                    pr[f"h4_{K}"] = do_stats(Ls2 + b4[kh], y)
                    pr[f"h5_{K}"] = do_stats(L2c + b5[kh], y)
                    pr[f"m_{K}"] = do_stats(ml[KS.index(K)], y)
                    pr[f"h3_{K}"] = do_stats(np.asarray(z[f"h3_{s}_r{r}_logit"][kh], np.float64), y)
                for meth, st in pr.items():
                    point_rows.append({"source": s, "rep": r, "method": meth, **{f"{q}_{p}": st[q][i] for q in ("d", "o", "s", "acc")
                                                                                   for i, p in enumerate(NAME_PAIRS)}})
                for meth in ("src", "h2"):
                    h5acc_rows.append({"method": meth, "K": "", "rep": r, "source": s,
                                       "absdiff": np.abs(pr[meth]["acc"] - ba(pr[meth]["d"], pr[meth]["o"])).tolist()})
                # 6 対を同時に直せる切片（オラクル．プールの全ラベル）
                Lp, yp = v["L"][pool], y_all[pool]
                b_ce = fit_class_bias(Lp, yp, l2=L2_MAIN)
                stp = do_stats(Lp, yp)
                resid, cj, okj, _ = joint_offset(stp["d"], stp["o"], stp["s"])
                cpad = np.r_[0.0, cj]
                st0 = pr["src"]
                Am = np.zeros((6, 3))
                for i, (a_, b_) in enumerate(PAIRS):
                    if a_ > 0:
                        Am[i, a_ - 1] += 1 / st0["s"][i]
                    if b_ > 0:
                        Am[i, b_ - 1] -= 1 / st0["s"][i]
                joint_rows.append({"source": s, "rep": r,
                                   "uar_src": uar(Ls2, y), "uar_ce": uar(Ls2 + b_ce, y), "uar_joint": uar(Ls2 + cpad, y),
                                   "acc_src": float(pair_acc(Ls2, y).mean()), "acc_ce": float(pair_acc(Ls2 + b_ce, y).mean()),
                                   "acc_joint": float(pair_acc(Ls2 + cpad, y).mean()),
                                   "loc_src_test": float(np.mean(ba(st0["d"], 0) - ba(st0["d"], st0["o"]))),
                                   "loc_src_test_clipped": float(np.mean(loc_loss_clipped(st0["d"], st0["o"]))),
                                   "loc_after_joint_test_model": float(np.mean(loc_loss_clipped(st0["d"], st0["o"] - Am @ cj))),
                                   "loc_after_joint_pool_model": resid, "joint_success": okj})
        masks_a = np.stack(masks)
        keys = list(conds)
        preds = np.stack([conds[k][0] for k in keys])
        mof = np.array([conds[k][1] for k in keys])
        U, full, loo, Nm, WN = rep_engine(Wr, pl, y, preds, masks_a, mof)
        for k in keys:
            if k not in accs:
                accs[k] = Acc(dr.B, P_c)
        accumulate([accs[k] for k in keys], U, full, loo, A[:, r], tp, r)
        # 感情を除いた回数（全テスト）
        miss = ((WN[:, :, 0] == 0).any(1)) & (WN[:, :, 0].sum(1) > 0)
        excl["emotion_cells"] += int(miss.sum())
        excl["emotion_cells_weighted"] += int((miss * A[:, r]).sum())
        del U, loo
    for k, acc in accs.items():
        store.put(t.name, k, acc.finish(), t.corpus)
    vmask = dr.valid(t)
    ctx.add("e76a_0_exclusions", {"kind": kind, "short": SHORT[kind], "target": t.name, "B": dr.B,
                                  "emotion_excluded_draw_rep_cells": excl["emotion_cells"],
                                  "emotion_excluded_weighted_by_stage2": excl["emotion_cells_weighted"],
                                  "rep_excluded_draw_rep_cells": int((~vmask).sum()),
                                  "rep_excluded_weighted_by_stage2": int(((~vmask) * A).sum()),
                                  "stage1_redraws": dr.redraws[t.name]})
    ctx.check("I", kind, f"保存した点数から作り直した UAR と csv の uar の差の最大 {t.name}", max(diffs["uar_vs_csv"]), 1e-9,
              f"{len(diffs['uar_vs_csv'])} 条件．float32 の点数の argmax の同点で 1 本ぶん変わりうる")
    ctx.record("I", kind, f"csv の uar と 1e-9 を超えて違う条件の数 {t.name}", int((np.array(diffs['uar_vs_csv']) > 1e-9).sum()))
    ctx.check("I", kind, f"保存した h4・h5 のずらしと l2=1e-2 で解き直した値の差の最大（反復 0）{t.name}", max(diffs["h4_b_refit"]), 1e-4,
              "解き直しは float32 で保存した点数から")
    ctx.check("I", kind, f"mean_t と特徴の平均の差 {t.name}", diffs["mean_t"][0], 1e-6)
    ctx.check("3", kind, f"重みつきの d（重み 1）と explore_synergy.d_o の差の最大（反復 0，① の K 6 × C 4）{t.name}",
              diffs["d_vs_d_o"], 1e-9)
    return {"dR": dR, "dF": dF, "tC": tC, "point": pd.DataFrame(point_rows), "joint": pd.DataFrame(joint_rows),
            "h5acc": h5acc_rows}


def coef_check(ctx: Ctx, kind: str, t: Target, X: np.ndarray) -> None:
    """係数から作り直した点数と保存した点数の差（反復 0：①（K 6 × C 4），② と h2（全発話），③（K 6），h3・h9（K 3））。"""
    z = t.z
    te = t.te[0]
    Xt = X[te]
    mx, agree = 0.0, 1.0

    def upd(Ls, Ln):
        nonlocal mx, agree
        mx = max(mx, float(np.abs(Ls - Ln).max()))
        agree = min(agree, float((Ls.argmax(-1) == Ln.argmax(-1)).mean()))
    tA, ta0, tl = z["t_A_r0"], z["t_a0_r0"], z["t_logit_r0"]
    for kk in range(len(KS)):
        for jc in range(len(C_GRID)):
            upd(tl[kk, jc], Xt @ tA[kk, jc].T.astype(np.float64) + ta0[kk, jc])
    mean_t = np.asarray(z["mean_t"], np.float64)
    for s in t.sources:
        j = int(t.src_ver(s)[0])
        sA, sa0, sl = z[f"s_{s}__v{j}_A"], z[f"s_{s}__v{j}_a0"], z[f"s_{s}__v{j}_logit"]
        for jc in range(len(C_GRID)):
            upd(sl[jc], X @ sA[jc].T.astype(np.float64) + sa0[jc])
        upd(z[f"h2_{s}__v{j}_logit"], (X - mean_t) @ z[f"h2_{s}__v{j}_A"].T.astype(np.float64) + z[f"h2_{s}__v{j}_a0"])
        mA, ma0, ml = z[f"m_{s}_r0_A"], z[f"m_{s}_r0_a0"], z[f"m_{s}_r0_logit"]
        for kk in range(len(KS)):
            upd(ml[kk], Xt @ mA[kk].T.astype(np.float64) + ma0[kk])
        for h, Xh in ((3, Xt - mean_t), (9, Xt)):
            hA, ha0, hl = z[f"h{h}_{s}_r0_A"], z[f"h{h}_{s}_r0_a0"], z[f"h{h}_{s}_r0_logit"]
            for kh in range(len(KH)):
                upd(hl[kh], Xh @ hA[kh].T.astype(np.float64) + ha0[kh])
    ctx.check("I", kind, f"係数から作り直した点数と保存した点数の差の最大（反復 0）{t.name}", mx, 1e-3, "係数は float32")
    ctx.check("I", kind, f"係数から作り直した点数の argmax の不一致の割合（反復 0）{t.name}", 1 - agree, 1e-3)


# ---------------------------------------------------------------- 2.4.1

def kstar_main(lo: dict) -> str:
    """K=1 から順に見て下端が 0 を上回り続ける最後の K（なし・全件の打ち切り）。"""
    last = "none"
    for K in KS:
        if np.isfinite(lo.get(K, np.nan)) and lo[K] > 0:
            last = K
        else:
            break
    return last


def kstar_max(lo: dict) -> str:
    sig = [K for K in KS if np.isfinite(lo.get(K, np.nan)) and lo[K] > 0]
    return sig[-1] if sig else "none"


def resampled_t(x: np.ndarray, rho: float) -> dict:
    """反復ごとの値 x から，反復を独立とみなした t 区間と補正した resampled t（Nadeau & Bengio 2003）。"""
    J = len(x)
    m, lo, hi = t_interval(x)
    sd = float(np.std(x, ddof=1)) if J >= 2 else np.nan
    se = sd * np.sqrt(1.0 / J + rho)
    tq = stats.t.ppf(0.975, J - 1) if J >= 2 else np.nan
    return {"naive_lo": lo, "naive_hi": hi, "corr_lo": m - tq * se, "corr_hi": m + tq * se,
            "corr_p": float(2 * stats.t.sf(abs(m) / se, J - 1)) if se > 0 else np.nan}


def section1(ctx: Ctx, kind: str, store: Store, tgts: list[Target], notes: pd.DataFrame, fixc_gain: pd.DataFrame | None,
             rule_terms: dict) -> dict:
    """地図の区間．rule_terms：規則 → (T, K) → ① の項（2.4.3 でも同じ関数を使う）。戻り：規則ごとの組 × K の表。"""
    out = {}
    for rule, tterm in rule_terms.items():
        rows = []
        for t in tgts:
            T = t.name
            rho = {}
            tg = t.csv[t.csv.method == "tgt_only"]
            for K in KS:
                g = tg[tg.K == K]
                rho[K] = float((g.n_test / g.n_tgt).mean())
            for s in t.sources:
                if same_corpus(T, s):
                    continue
                for K in KS:
                    terms = [(1.0, T, (s, "m", K))] + [(-c, TT, k) for c, TT, k in tterm[(T, K)]]
                    iv = store.ival(terms)
                    full = iv.pop("_full")
                    rt = resampled_t(full, rho[K]) if rule == "main" else {}
                    rows.append({"kind": kind, "short": SHORT[kind], "rule": rule, "target": T, "source": s, "K": K,
                                 "gain": iv["theta"], **{k: v for k, v in iv.items() if k != "theta"}, **rt,
                                 "rho": rho[K], "note": note_for(notes, T, s, K)})
        P = pd.DataFrame(rows)
        if rule == "main":
            ref = []
            for t in tgts:
                c = t.csv
                tg = c[c.method == "tgt_only"].set_index(["K", "rep"]).uar
                mg = c[c.method == "merged"]
                g = mg.uar.to_numpy() - tg.loc[list(zip(mg.K, mg.rep))].to_numpy()
                ref.append(pd.DataFrame({"target": t.name, "source": mg.source.to_numpy(), "K": mg.K.to_numpy(), "g": g})
                           .groupby(["target", "source", "K"]).g.mean() * 100)
            ref = pd.concat(ref)
            dv = np.abs(P.set_index(["target", "source", "K"]).gain - ref.reindex(P.set_index(["target", "source", "K"]).index))
            ctx.check("1", kind, "組 × K の利得（点数から）と csv の uar から作った利得の差の最大（点）", float(dv.max()), 1e-9)
        P["p_holm"] = np.nan
        for K, h in P.groupby("K"):
            ok = h.p.notna()
            P.loc[h.index[ok], "p_holm"] = holm(h.p[ok].to_numpy())
        P["up"] = (P.p_holm < 0.05) & (P.gain > 0)
        P["down"] = (P.p_holm < 0.05) & (P.gain < 0)
        P["up_unc"] = P.lo > 0
        P["down_unc"] = P.hi < 0
        out[rule] = P
    P = out["main"]
    n_pairs = P.groupby("K").size()
    ctx.check("1", kind, "組の数（同じコーパスを除く）と設計の数の差", abs(int(n_pairs.iloc[0]) - EXPECTED_PAIRS[kind])
              if len(tgts) == EXPECTED_TARGETS[kind] else 0, 0, f"ターゲット {len(tgts)}")
    ctx.add("e76a_1_pairs", P.drop(columns=["rule"]).to_dict("records"))
    # 種類 × K
    for K in KS:
        h = P[P.K == K]
        terms = []
        for x in h.itertuples():
            terms += [(1.0 / len(h), x.target, (x.source, "m", K))] + [(-c / len(h), TT, k) for c, TT, k in rule_terms["main"][(x.target, K)]]
        iv = store.ival(terms)
        iv.pop("_full")
        tl = h.groupby("target").agg(n=("up", "size"), n_up=("up", "sum"))
        row = {"kind": kind, "short": SHORT[kind], "K": K, "n_targets": h.target.nunique(), "n_pairs": len(h),
               "mean_gain": iv["theta"], **{k: v for k, v in iv.items() if k != "theta"},
               "n_up_holm": int(h.up.sum()), "n_down_holm": int(h.down.sum()),
               "n_up_unc": int(h.up_unc.sum()), "n_down_unc": int(h.down_unc.sum()),
               "share_up_holm": float(h.up.mean()), "share_down_holm": float(h.down.mean()),
               "n_targets_all_up": int((tl.n_up == tl.n).sum()),
               "targets_up_below_half": ";".join(tl.index[tl.n_up < tl.n / 2]), "note": kind_note(notes)}
        if fixc_gain is not None:
            g = fixc_gain[(fixc_gain.K == K)]
            row["mean_gain_reps0_29_map"] = float(g.gain.mean() * 100) if len(g) else np.nan
            row["n_pairs_reps0_29_map"] = len(g)
        ctx.add("e76a_1_kind", row)
        for T, x in tl.iterrows():
            ctx.add("e76a_1_targets", {"kind": kind, "short": SHORT[kind], "K": K, "target": T, "n_pairs": int(x.n),
                                       "n_up_holm": int(x.n_up), "share_up_holm": float(x.n_up / x.n),
                                       "all_up": bool(x.n_up == x.n)})
    # K*
    ks = kstar_rows(kind, P, {"main": ("lo",), "boot_only": ("boot_lo",), "naive_t": ("naive_lo",), "corrected_t": ("corr_lo",),
                              "stage1": ("stage1_lo",)})
    ctx.add("e76a_1_kstar", ks.to_dict("records"))
    for (iv_name, dfn), g in ks.groupby(["interval", "definition"]):
        ctx.add("e76a_1_kstar_dist", {"kind": kind, "short": SHORT[kind], "interval": iv_name, "definition": dfn,
                                      "n_pairs": len(g), "median_rank": float(g.rank_.median()),
                                      **{f"n_{k}": int((g.K_star == k).sum()) for k in ("none",) + KS}})
    return out


def kstar_rows(kind: str, P: pd.DataFrame, cols: dict) -> pd.DataFrame:
    rows = []
    for (T, s), g in P.groupby(["target", "source"]):
        g = g.set_index("K")
        for name, (col,) in cols.items():
            lo = {K: float(g.loc[K, col]) if K in g.index else np.nan for K in KS}
            for dfn, f in (("last_continuing", kstar_main), ("max_K", kstar_max)):
                k = f(lo)
                rows.append({"kind": kind, "short": SHORT[kind], "target": T, "source": s, "interval": name, "definition": dfn,
                             "K_star": k, "rank_": KSTAR_RANK[k]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2.4.2

def gender_tables(ctx: Ctx, B: int, seed: int, kind: str, tgts: list[Target], gender_of: dict) -> None:
    """性別ごとの利得（反復 0〜9 のテストの予測をまとめた UAR の ③ − ①，ソースで平均）と女性 − 男性の差（記述）。"""
    gt = [t for t in tgts if t.name in GENDER_CORPORA]
    if not gt:
        return
    for t in gt:
        g = np.array([gender_of.get((t.corpus, sid), "") for sid in t.sid])
        pg = pd.DataFrame({"p": t.person, "g": g}).groupby("p").g.agg(lambda v: v.iloc[0] if v.nunique() == 1 else "mixed")
        t.gender_p = pg
        ctx.record("2", kind, f"性別が混ざるか分からない人物の数 {t.name}", int((~pg.isin(["F", "M"])).sum()))
    corpora = sorted({t.corpus for t in gt})
    persons = {c: {g: sorted(set().union(*[set(t.gender_p[t.gender_p == g].index) for t in gt if t.corpus == c])) for g in "FM"}
               for c in corpora}
    base = [seed, zlib.crc32(kind.encode()), zlib.crc32(b"gender")]
    W = {}
    for c in corpora:
        rng = np.random.default_rng(base + [zlib.crc32(c.encode())])
        W[c] = {g: rng.multinomial(len(persons[c][g]), np.full(len(persons[c][g]), 1 / len(persons[c][g])), size=B).astype(np.float64)
                for g in "FM"}
    Ar = np.random.default_rng(base + [zlib.crc32(b"reps")]).multinomial(len(REPS), np.full(len(REPS), 1 / len(REPS)), size=B).astype(np.float64)
    rng_re = np.random.default_rng(base + [zlib.crc32(b"redraw")])
    pidx = {c: {g: {p: i for i, p in enumerate(persons[c][g])} for g in "FM"} for c in corpora}
    # 反復ごとのテストの性別ごとの人物
    tp = {}
    for t in gt:
        for g in "FM":
            for r in REPS:
                ps = [p for p in np.unique(t.person[t.te[r]]) if t.gender_p.get(p) == g]
                tp[(t.name, g, r)] = np.array([pidx[t.corpus][g][p] for p in ps], np.int64)
    redraw = {t.name: 0 for t in gt}
    for _ in range(1000):
        bad = np.zeros(B, bool)
        for t in gt:
            for g in "FM":
                wsum = np.stack([W[t.corpus][g][:, tp[(t.name, g, r)]].sum(1) for r in REPS], 1)
                bt = (Ar * wsum).sum(1) == 0
                redraw[t.name] += int(bt.sum())
                bad |= bt
        if not bad.any():
            break
        nb = int(bad.sum())
        for c in corpora:
            for g in "FM":
                P = len(persons[c][g])
                W[c][g][bad] = rng_re.multinomial(P, np.full(P, 1 / P), size=nb)
        Ar[bad] = rng_re.multinomial(len(REPS), np.full(len(REPS), 1 / len(REPS)), size=nb)
    for t in gt:
        ctx.record("2", kind, f"性別の引き方で 1・2 段目を引き直した回数 {t.name}", redraw[t.name])
        z = t.z
        srcs = [s for s in t.sources if not same_corpus(t.name, s)]
        res = {}
        for K in KH:
            kk = KS.index(K)
            for g in "FM":
                P = len(persons[t.corpus][g])
                TC = {}
                TN = np.zeros((B, 4))
                pc = {}
                pn = np.zeros((P, 4))
                for r in REPS:
                    te = t.te[r]
                    y = t.y[te]
                    sel = np.array([t.gender_p.get(p) == g for p in t.person[te]])
                    if not sel.any():
                        continue
                    pl = np.array([pidx[t.corpus][g][p] for p in t.person[te][sel]], np.int64)
                    S = person_indicator(pl, y[sel], P)
                    nn = np.asarray(S @ np.ones(sel.sum())).reshape(P, 4)
                    TN += Ar[:, [r]] * (W[t.corpus][g] @ nn)
                    pn += nn
                    tl = z[f"t_logit_r{r}"]
                    C = float(z[f"t_C_r{r}"][kk])
                    preds = {"t": tl[kk, C_GRID.index(C)][sel].argmax(1)}
                    for s in srcs:
                        preds[s] = z[f"m_{s}_r{r}_logit"][kk][sel].argmax(1)
                    for key, pr in preds.items():
                        cc = np.asarray(S @ (pr == y[sel]).astype(np.float64)).reshape(P, 4)
                        TC[key] = TC.get(key, 0) + Ar[:, [r]] * (W[t.corpus][g] @ cc)
                        pc[key] = pc.get(key, 0) + cc
                vals = {}
                for key in ["t"] + srcs:
                    b = uar_from_counts(TC[key], TN, axis=-1)
                    f = uar_from_counts(pc[key].sum(0), pn.sum(0), axis=-1)
                    loo = uar_from_counts(pc[key].sum(0)[None] - pc[key], pn.sum(0)[None] - pn, axis=-1)
                    vals[key] = (b, f, loo)
                res[(K, g)] = vals
                res[(K, g, "n")] = (pn, P)
            # 女性 − 男性の差と性別ごとの利得（jackknife は両方の性別の人物を 1 人ずつ抜く）
            row = {"kind": kind, "short": SHORT[kind], "target": t.name, "K": K}
            gains = {}
            for g in "FM":
                vals = res[(K, g)]
                gb = np.mean([vals[s][0] - vals["t"][0] for s in srcs], 0)
                gf = float(np.mean([vals[s][1] - vals["t"][1] for s in srcs]))
                gl = np.mean([vals[s][2] - vals["t"][2] for s in srcs], 0)                # 同じ性別の人物を抜いた値
                gains[g] = (gb, gf, gl)
                pn, P = res[(K, g, "n")]
                n_p = pn.sum(1)
                geff = float(n_p.sum() ** 2 / (n_p ** 2).sum()) if (n_p > 0).any() else np.nan
                row.update({f"n_persons_{g}": int((n_p > 0).sum()), f"geff_{g}": geff,
                            f"uar_tgt_{g}": float(vals["t"][1] * 100),
                            f"uar_merged_mean_{g}": float(np.mean([vals[s][1] for s in srcs]) * 100)})
            gmin = min(row["geff_F"], row["geff_M"])
            use_jk = bool(gmin < GEFF_MIN)
            for g in "FM":
                gb, gf, gl = gains[g]
                other = "M" if g == "F" else "F"
                jk = np.r_[gl, np.full(len(gains[other][2]), gf)]                           # 別の性別の人物を抜いても値は変わらない
                iv = interval(gf, gb, jk, gmin, use_jk)
                row.update({f"gain_{g}": gf * 100, f"gain_{g}_lo": iv["lo"] * 100, f"gain_{g}_hi": iv["hi"] * 100})
            db = gains["F"][0] - gains["M"][0]
            dfull = gains["F"][1] - gains["M"][1]
            jk = np.r_[gains["F"][2] - gains["M"][1], gains["F"][1] - gains["M"][2]]
            iv = interval(dfull, db, jk, gmin, use_jk)
            both = sum(1 for r in REPS if len(tp[(t.name, "F", r)]) and len(tp[(t.name, "M", r)]))
            row.update({"diff_F_minus_M": dfull * 100, "diff_lo": iv["lo"] * 100, "diff_hi": iv["hi"] * 100,
                        "diff_boot_lo": iv["boot_lo"] * 100, "diff_boot_hi": iv["boot_hi"] * 100,
                        "diff_jk_lo": iv["jk_lo"] * 100 if np.isfinite(iv["jk_lo"]) else np.nan,
                        "diff_jk_hi": iv["jk_hi"] * 100 if np.isfinite(iv["jk_hi"]) else np.nan,
                        "p": iv["p"], "use_jk": use_jk, "geff_min": gmin, "n_reps_both_genders": both,
                        "n_sources": len(srcs), "main": K == "5"})
            ctx.add("e76a_2_gender", row)


def subset_tables(ctx: Ctx, kind: str, store: Store, t: Target, tag: str, Ks: tuple, frac_removed: dict) -> None:
    """MOSEI のチャンネルの分かる発話（sub_ch）と IEMOCAP 台本の台詞の重なりを除いた発話（sub_ln）の利得と，全テストの利得との差。"""
    srcs = [s for s in t.sources if not same_corpus(t.name, s)]
    for K in Ks:
        full_terms_all, sub_terms_all = [], []
        for s in srcs:
            full_terms = [(1.0, t.name, (s, "m", K)), (-1.0, t.name, ("tm", K))]
            sub_terms = [(1.0, t.name, (tag, "m", s, K)), (-1.0, t.name, (tag, "tm", K))]
            full_terms_all += [(c / len(srcs), T, k) for c, T, k in full_terms]
            sub_terms_all += [(c / len(srcs), T, k) for c, T, k in sub_terms]
            for unit, ft, st in (("pair", full_terms, sub_terms),):
                f = store.ival(ft)
                sb = store.ival(st)
                ctx.add("e76a_2_subsets", subset_row(kind, t.name, tag, unit, s, K, f, sb, frac_removed.get(K)))
        f = store.ival(full_terms_all)
        sb = store.ival(sub_terms_all)
        ctx.add("e76a_2_subsets", subset_row(kind, t.name, tag, "target_mean", "", K, f, sb, frac_removed.get(K)))


def subset_row(kind, T, tag, unit, s, K, f, sb, frac) -> dict:
    half = (f["hi"] - f["lo"]) / 2
    diff = sb["theta"] - f["theta"]
    return {"kind": kind, "short": SHORT[kind], "target": T, "subset": tag, "unit": unit, "source": s, "K": K,
            "gain_full": f["theta"], "full_lo": f["lo"], "full_hi": f["hi"], "full_half_width": half,
            "gain_subset": sb["theta"], "subset_lo": sb["lo"], "subset_hi": sb["hi"], "subset_use_jk": sb["use_jk"],
            "diff_subset_minus_full": diff, "within_half_width": bool(abs(diff) <= half),
            "frac_test_removed": frac, "n_draws_subset": sb["n_draws"]}


# ---------------------------------------------------------------- 2.4.3

def fit_f0(Mv: np.ndarray, D: np.ndarray, Ks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d = D/√(1 + q/K) を Q_GRID で当てる．Mv：(..., 6, nK)．戻り：q (...)，最小の 2 乗和 (...)。"""
    f = 1 / np.sqrt(1 + Q_GRID[:, None] / Ks[None, :])                                       # (q, nK)
    u = np.einsum("...pk,p->...k", Mv, D)                                                      # (..., nK)
    c0 = (Mv ** 2).sum((-1, -2))
    sse = c0[..., None] - 2 * u @ f.T + (D ** 2).sum() * (f ** 2).sum(1)                       # (..., q)
    i = sse.argmin(-1)
    return Q_GRID[i], np.take_along_axis(sse, i[..., None], -1)[..., 0]


def fit_m1(v: np.ndarray, Ks: np.ndarray) -> tuple[float, float]:
    """M1：対の平均の曲線に d = D_s/√(1 + q/K)（D_s と q を当てる．D_s は閉じた形，q は Q_GRID）。"""
    f = 1 / np.sqrt(1 + Q_GRID[:, None] / Ks[None, :])
    Ds = (f @ v) / (f ** 2).sum(1)
    sse = ((v[None] - Ds[:, None] * f) ** 2).sum(1)
    i = int(sse.argmin())
    return float(Ds[i]), float(Q_GRID[i])


def fit_m2(v: np.ndarray, Ks: np.ndarray) -> tuple[float, float]:
    """M2：対の平均の曲線に d = a K^b（b は格子，a は閉じた形）。"""
    g = Ks[None, :] ** M2_B_GRID[:, None]
    a = (g @ v) / (g ** 2).sum(1)
    sse = ((v[None] - a[:, None] * g) ** 2).sum(1)
    i = int(sse.argmin())
    return float(a[i]), float(M2_B_GRID[i])


def loko_all(Mv: np.ndarray, D: np.ndarray, Ks: np.ndarray) -> dict:
    """延長の 5 点で K を 1 つ抜き，残りで当てて，抜いた K の対の平均の d を予測した 2 乗誤差（F0・M1・M2）。端（K=1，50）と内側に分ける。"""
    err = {"F0": [], "M1": [], "M2": []}
    vbar = Mv.mean(0)
    for k in range(len(Ks)):
        keep = [i for i in range(len(Ks)) if i != k]
        q0, _ = fit_f0(Mv[:, keep], D, Ks[keep])
        p0 = float((D / np.sqrt(1 + q0 / Ks[k])).mean())
        Ds, q1 = fit_m1(vbar[keep], Ks[keep])
        p1 = Ds / np.sqrt(1 + q1 / Ks[k])
        a, b = fit_m2(vbar[keep], Ks[keep])
        p2 = a * Ks[k] ** b
        for name, p in (("F0", p0), ("M1", p1), ("M2", p2)):
            err[name].append((p - vbar[k]) ** 2)
    edge = [0, len(Ks) - 1]
    inner = list(range(1, len(Ks) - 1))
    out = {}
    for name, e in err.items():
        e = np.array(e)
        out[f"loko_mse_{name}"] = float(e.mean())
        out[f"loko_mse_{name}_edge"] = float(e[edge].mean())
        out[f"loko_mse_{name}_inner"] = float(e[inner].mean())
    return out


def rule_C_table(t: Target, pseudo: dict, tC: np.ndarray) -> dict:
    """規則 → K → 反復ごとの C の番号（主は反復ごと，ほかは K ごとに 1 つ）。"""
    out = {}
    out["main"] = {K: [C_GRID.index(float(tC[r, kk])) for r in REPS] for kk, K in enumerate(KS)}
    out["pseudo"] = {K: [C_GRID.index(pseudo[K])] * len(REPS) for K in KS}
    for c in C_GRID:
        out[f"C{c:g}"] = {K: [C_GRID.index(c)] * len(REPS) for K in KS}
    return out


def rule_terms_for(t: Target, rc: dict) -> dict:
    """規則 → (T, K) → ① の項．主は ("tm", K)，ほかは ("t", K, C)。"""
    out = {}
    for rule, byK in rc.items():
        out[rule] = {}
        for K in KS:
            if rule == "main":
                out[rule][(t.name, K)] = [(1.0, t.name, ("tm", K))]
            else:
                out[rule][(t.name, K)] = [(1.0, t.name, ("t", K, C_GRID[byK[K][0]]))]
    return out


def reduce_d(dR: np.ndarray, A: np.ndarray, sel: list[int], kk: int) -> np.ndarray:
    """(B, 6)：反復ごとに選んだ C の重みつき d を，2 段目の重みで平均（d が定まらない反復は除く）。"""
    num = np.zeros((A.shape[0], 6))
    den = np.zeros((A.shape[0], 6))
    for r in REPS:
        v = dR[r, :, kk, sel[r], :].astype(np.float64)
        ok = np.isfinite(v)
        num += A[:, [r]] * np.where(ok, v, 0.0)
        den += A[:, [r]] * ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def section3_target(ctx: Ctx, kind: str, t: Target, td: dict, dr: Draws, D: np.ndarray, rc: dict) -> list[dict]:
    """q の当てはめ（規則 × D と D′ × 3 点と 5 点），区間，残差，M1・M2，K を 1 つ抜いた予測。"""
    rows = []
    K3, K5 = np.array([1.0, 5.0, 10.0]), np.array([1.0, 5.0, 10.0, 25.0, 50.0])
    for rule, byK in rc.items():
        # 点の d_T(K)（反復の平均）
        Mpt = np.stack([np.nanmean(np.stack([td["dF"][r, kk, byK[K][r]] for r in REPS]), 0) for kk, K in enumerate(KS)], 1)  # (6, 6K)
        Mb = np.stack([reduce_d(td["dR"], dr.A, byK[K], kk) for kk, K in enumerate(KS)], 2)                               # (B, 6, 6K)
        Dp = Mpt[:, KS.index("all")]
        se = np.nanstd(Mb, 0, ddof=1)                                                                                    # (6, 6K)
        se_bar = np.nanstd(np.nanmean(Mb, 1), 0, ddof=1)                                                                # (6K,)
        for dname, Dv in (("D", D), ("Dprime", Dp)):
            for fname, Ks_ in (("3pt", K3), ("5pt", K5)):
                cols = [KS.index(str(int(k))) for k in Ks_]
                Mv = Mpt[:, cols]
                q, sse = fit_f0(Mv, Dv, Ks_)
                okb = np.isfinite(Mb[:, :, cols]).all((1, 2))
                qb, _ = fit_f0(Mb[okb][:, :, cols], Dv, Ks_)
                fit = Dv[:, None] / np.sqrt(1 + q / Ks_[None, :])
                res = Mv - fit
                thr = max(0.1 * float(np.mean(Dv)), 2 * float(np.mean(se[:, cols])))
                row = {"kind": kind, "short": SHORT[kind], "target": t.name, "rule": rule, "D_version": dname, "fit": fname,
                       "q": float(q), "q_at_edge": bool(q <= Q_GRID[0] or q >= Q_GRID[-1]),
                       "q_lo": float(np.quantile(qb, 0.025)) if len(qb) else np.nan,
                       "q_hi": float(np.quantile(qb, 0.975)) if len(qb) else np.nan,
                       "n_boot_q": int(len(qb)), "n_boot_q_at_edge": int(((qb <= Q_GRID[0]) | (qb >= Q_GRID[-1])).sum()),
                       "mean_abs_resid": float(np.abs(res).mean()), "resid_threshold": thr,
                       "resid_exceeds": bool(np.abs(res).mean() > thr), "D_mean": float(np.mean(Dv)),
                       "se_mean": float(np.mean(se[:, cols])),
                       **{f"mean_resid_K{int(k)}": float(res[:, i].mean()) for i, k in enumerate(Ks_)}}
                if fname == "5pt":
                    vbar = Mv.mean(0)
                    Ds, q1 = fit_m1(vbar, Ks_)
                    r1 = vbar - Ds / np.sqrt(1 + q1 / Ks_)
                    a2, b2 = fit_m2(vbar, Ks_)
                    r2 = vbar - a2 * Ks_ ** b2
                    thr1 = max(0.1 * float(np.mean(Dv)), 2 * float(np.mean(se_bar[cols])))
                    lk = loko_all(Mv, Dv, Ks_)
                    row.update({"M1_D": Ds, "M1_q": q1, "M1_q_at_edge": bool(q1 <= Q_GRID[0] or q1 >= Q_GRID[-1]),
                                "M1_mean_abs_resid": float(np.abs(r1).mean()), "M1_resid_threshold": thr1,
                                "M1_resid_exceeds": bool(np.abs(r1).mean() > thr1),
                                "M2_a": a2, "M2_b": b2, "M2_mean_abs_resid": float(np.abs(r2).mean()), **lk,
                                "M1_better_loko": lk["loko_mse_M1"] < lk["loko_mse_F0"],
                                "M2_better_loko": lk["loko_mse_M2"] < lk["loko_mse_F0"]})
                    row["M1orM2_better_loko"] = row["M1_better_loko"] or row["M2_better_loko"]
                if dname == "D" and fname == "3pt":
                    # H2：対ごとの q_kl（主と同じ格子）
                    qkl = [float(fit_f0(Mv[[p]], Dv[[p]], Ks_)[0]) for p in range(6)]
                    row.update({f"q_pair_{NAME_PAIRS[p]}": qkl[p] for p in range(6)})
                    row["q_pair_max_over_min"] = float(max(qkl) / min(qkl))
                rows.append(row)
        for kk, K in enumerate(KS):
            for p in range(6):
                ctx.add("e76a_3_dT", {"kind": kind, "short": SHORT[kind], "target": t.name, "rule": rule, "K": K, "pair": NAME_PAIRS[p],
                                      "d_T": float(Mpt[p, kk]), "D": float(D[p]), "Dprime": float(Dp[p]), "se_boot": float(se[p, kk]),
                                      "d_T_gt_D": bool(Mpt[p, kk] > D[p])})
        # 当てはめの検算：主の規則の 3 点を E28-9 と同じ関数で
        if rule == "main":
            rec = [{"rep": r, "K": K, "pair": NAME_PAIRS[p], "d": float(td["dF"][r, KS.index(K), byK[K][r], p])}
                   for r in REPS for K in ("1", "5", "10") for p in range(6)]
            q_ref = q_fit(pd.DataFrame(rec), pd.Series(D, index=NAME_PAIRS))
            ctx.check("3", kind, f"q の当てはめ（まとめて解く版）と explore_direction_bound.q_fit の差 {t.name}",
                      abs(rows[0]["q"] - q_ref) if rows[0]["rule"] == "main" else np.nan, 0.0)
    return rows


def n_tables(ctx: Ctx, kind: str, curves: dict, P1: pd.Series, e13: pd.DataFrame, qtab: pd.DataFrame, srcs: dict) -> pd.DataFrame:
    """E32 と同じ手順の n：測った n（P1 の UAR が ① の学習曲線で何本に当たるか − K），予言した n（模型．q は規則ごとの E76 の q）。"""
    rows = []
    for rule in RULES:
        for T, us in curves[rule].items():
            own = e13[e13.target == T].groupby("pair_idx").own_d.mean().reindex(NAME_PAIRS).to_numpy()
            qrow = qtab[(qtab.target == T) & (qtab.rule == rule) & (qtab.D_version == "D") & (qtab.fit == "3pt")]
            q = float(qrow.q.iloc[0]) if len(qrow) else np.nan
            for s in srcs[T]:
                if same_corpus(T, s):
                    continue
                ds = e13[(e13.target == T) & (e13.source == s)].set_index("pair_idx").d.reindex(NAME_PAIRS).to_numpy()
                e = np.clip(ds / own, 0.0, 0.999)
                for K in ("1", "5", "10", "25"):
                    u = P1.get((T, s, K), np.nan)
                    nm, fl = n_of(us, u, int(K)) if np.isfinite(u) else (np.nan, 9)
                    npd, fp = n_model(own, e, q, int(K), pos=True) if np.isfinite(q) and np.isfinite(e).all() else (np.nan, 9)
                    rows.append({"kind": kind, "short": SHORT[kind], "rule": rule, "target": T, "source": s, "K": K,
                                 "uar_P1": u, "n_meas": nm, "flag_meas": fl, "n_pred": npd, "flag_pred": fp, "q": q})
    return pd.DataFrame(rows)


def rank_tables(ctx: Ctx, kind: str, N: pd.DataFrame, seed: int) -> None:
    for (rule, K), g in N.groupby(["rule", "K"]):
        g = g[(g.flag_meas != 9) & (g.flag_pred != 9)]
        if g.empty:
            continue
        meas = rankable(g.n_meas.to_numpy(), g.flag_meas.to_numpy())
        pred = rankable(g.n_pred.to_numpy(), g.flag_pred.to_numpy())
        rho, p, n = spear(pred, meas)
        ok = (g.flag_meas == 0) & (g.flag_pred == 0) & (g.n_pred != 0)
        groups, few = [], []
        for T, h in g.groupby("target"):
            if len(h) >= 4:
                xp, xm = rankable(h.n_pred.to_numpy(), h.flag_pred.to_numpy()), rankable(h.n_meas.to_numpy(), h.flag_meas.to_numpy())
                groups.append((xp, xm))
                rt = spear(xp, xm)
                ctx.add("e76a_3_rank_targets", {"kind": kind, "short": SHORT[kind], "rule": rule, "K": K, "target": T,
                                                "n_sources": len(h), "spearman_within": rt[0]})
            else:
                few.append(f"{T}({len(h)})")
        rng = np.random.default_rng([seed, zlib.crc32(f"perm:{kind}:{rule}:{K}".encode())])
        wt = within_target_perm(groups, 1000, rng) if groups else {"n_targets": 0}
        ctx.add("e76a_3_rank", {"kind": kind, "short": SHORT[kind], "rule": rule, "K": K, "n_pairs": n, "spearman_across": rho,
                                "p_across": p, "ratio_meas_over_pred_median": float(np.median(g.n_meas[ok] / g.n_pred[ok])) if ok.any() else np.nan,
                                "n_ratio": int(ok.sum()), "targets_with_3_or_fewer_sources": ";".join(few),
                                **{f"within_{k}": v for k, v in wt.items()}})


# ---------------------------------------------------------------- 本体（1 つの種類）

def run_kind(ctx: Ctx, kind: str, e13_all: pd.DataFrame, written: dict, accepted: dict, meta: dict) -> None:
    a = ctx.a
    t0 = time.perf_counter()
    groups = groups_for(kind)
    e76 = Path(a.e76)
    names = [g for g in groups if (a.targets is None or g in a.targets)]
    allT = {g: Target(ctx, kind, g, e76, load_npz=g in names) for g in groups}
    for g, t in allT.items():
        tag = check_target_meta(ctx, t, written, accepted)
        ctx.record("I", kind, f"コードのハッシュの受け入れ {g}", tag, t.meta["code_hash"][:12])
    tgts = [allT[g] for g in names]
    allcsv = {g: allT[g].csv for g in groups}
    notes = pd.concat([hubert_notes(t) for t in allT.values()], ignore_index=True)
    if len(notes):
        ctx.add("e76a_0_hubert_notes", notes.to_dict("records"))
    if kind != "audio_hubert":
        ctx.check("I", kind, "tol を超えた照合と ① の C の不一致の数（A(HuBERT) 以外は 0）", float(len(notes)), 0)
    dr = Draws(kind, tgts, a.B, a.seed)
    # G と G_eff
    geff = {}
    for t in tgts:
        n_p = np.bincount(np.concatenate([t.pidx[t.te[r]] for r in REPS]), minlength=len(dr.persons[t.corpus]))
        n_p = n_p[n_p > 0]
        geff[t.name] = float(n_p.sum() ** 2 / (n_p ** 2).sum())
        sets = {r: frozenset(t.tp[r].tolist()) for r in REPS}
        ctx.add("e76a_0_units", {"kind": kind, "short": SHORT[kind], "target": t.name, "G": int(len(n_p)), "G_eff": geff[t.name],
                                 "use_jk": geff[t.name] < GEFF_MIN, "n_reps": len(REPS), "n_test_sets": len(set(sets.values())),
                                 "test_persons_min": min(len(v) for v in sets.values()),
                                 "test_persons_max": max(len(v) for v in sets.values()),
                                 "test_persons_by_rep": ",".join(str(len(sets[r])) for r in REPS),
                                 "n_sources": len(t.sources), "n_pairs_excl_same": sum(not same_corpus(t.name, s) for s in t.sources),
                                 "stage1_redraws": dr.redraws[t.name], "n_test_by_rep": ",".join(str(len(t.te[r])) for r in REPS)})
    store = Store(dr, geff)
    pseudo = {T: pseudo_rule_C(allcsv, kind, T) for T in groups}
    tdata = {}
    for t in tgts:
        t1 = time.perf_counter()
        df, X = load_any(t.name, kind)
        X = np.asarray(X, np.float64)                    # relearn_e76 は float64 で中心を取った（logreg.to_device）
        if not (df.sample_id.astype(str).to_numpy() == t.sid).all():
            raise SystemExit(f"{kind} {t.name}：特徴の sample_id の並びが npz と違う")
        ctx.record("I", kind, f"特徴の行列の sha256 {t.name}", hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:16],
                   f"{X.shape}")
        mt = {}
        if t.name == "MOSEI":
            mt["mosei_channel"] = np.array([meta["mosei_known"].get(s, False) for s in t.sid])
        if t.name == SCRIPT and kind in TEXT_KINDS:
            mt["script_lines"] = np.array([meta["iemocap_lines"][s] for s in t.sid], dtype=object)
        td = run_target(ctx, kind, t, X, dr, store, pseudo, notes, mt)
        td["mt"] = mt
        coef_check(ctx, kind, t, X)
        tdata[t.name] = td
        t.drop()
        ctx.timing[f"{kind}/{t.name}"] = time.perf_counter() - t1
        print(f"  {kind} {t.name}：{time.perf_counter() - t1:.0f} 秒（条件 {sum(1 for k in store.res if k[0] == t.name)}）", flush=True)
    # 規則ごとの ① の項
    rc = {t.name: rule_C_table(t, pseudo[t.name], tdata[t.name]["tC"]) for t in tgts}
    rterms = {}
    for t in tgts:
        for rule, d in rule_terms_for(t, rc[t.name]).items():
            rterms.setdefault(rule, {}).update(d)
    # 2.4.1（と 2.4.3 の規則ごとの利得・K*）
    fixc_gain = fixc_gains(ctx, kind, names)
    pt = section1(ctx, kind, store, tgts, notes, fixc_gain, rterms)
    for rule, P in pt.items():
        P = P.copy()
        ctx.add("e76a_3_pairs", P[["kind", "short", "rule", "target", "source", "K", "gain", "lo", "hi", "p", "use_jk"]].to_dict("records"))
        ks = kstar_rows(kind, P, {"main": ("lo",)})
        ks = ks[ks.definition == "last_continuing"]
        for K in ("1", "5"):
            h = P[P.K == K]
            terms = []
            for x in h.itertuples():
                terms += [(1.0 / len(h), x.target, (x.source, "m", K))] + [(-c / len(h), TT, k) for c, TT, k in rterms[rule][(x.target, K)]]
            iv = store.ival(terms)
            ctx.add("e76a_3_gain", {"kind": kind, "short": SHORT[kind], "rule": rule, "K": K, "n_pairs": len(h), "note": kind_note(notes),
                                    "mean_gain": iv["theta"], "lo": iv["lo"], "hi": iv["hi"], "use_jk": iv["use_jk"],
                                    "kstar_median_rank": float(ks.rank_.median()),
                                    **{f"kstar_n_{k}": int((ks.K_star == k).sum()) for k in ("none",) + KS}})
    # 学習曲線と C（2.4.3）
    curves = {rule: {} for rule in RULES}
    for t in tgts:
        for rule in RULES:
            us = []
            for K in KS:
                key = ("tm", K) if rule == "main" else ("t", K, C_GRID[rc[t.name][rule][K][0]])
                us.append(store.res[(t.name,) + key]["theta"] * 100)
                Cs = sorted({C_GRID[i] for i in rc[t.name][rule][K]})
                ctx.add("e76a_3_curves", {"kind": kind, "short": SHORT[kind], "target": t.name, "rule": rule, "K": K,
                                          "uar_tgt": us[-1], "C": ";".join(f"{c:g}" for c in Cs)})
            curves[rule][t.name] = np.maximum.accumulate(np.array(us[:5]))
    # q（2.4.3）
    e13 = e13_all[e13_all.kind == kind]
    qrows = []
    for t in tgts:
        D = e13[e13.target == t.name].groupby("pair_idx").own_d.mean().reindex(NAME_PAIRS).to_numpy()
        if not np.isfinite(D).all():
            ctx.record("3", kind, f"own_d が無いターゲット {t.name}", 1)
            continue
        qrows += section3_target(ctx, kind, t, tdata[t.name], dr, D, rc[t.name])
    qtab = pd.DataFrame(qrows)
    ctx.add("e76a_3_q_targets", qrows)
    # n と順位相関（2.4.3）
    P1 = p1_uar(ctx, kind)
    N = n_tables(ctx, kind, curves, P1, e13, qtab, {t.name: t.sources for t in tgts})
    ctx.add("e76a_3_n_pairs", N.to_dict("records"))
    rank_tables(ctx, kind, N, a.seed)
    # 2.4.2
    gender_tables(ctx, a.B, a.seed, kind, tgts, meta["gender"])
    for t in tgts:
        mt = tdata[t.name]["mt"]
        if "mosei_channel" in mt:
            fr = {K: float(np.mean([1 - mt["mosei_channel"][t.te[r]].mean() for r in REPS])) for K in KS}
            subset_tables(ctx, kind, store, t, "sub_ch", KS, fr)
        if "script_lines" in mt:
            lines = mt["script_lines"]
            fr = {K: float(np.mean([np.isin(lines[t.te[r]], list(set(lines[t.kidx[(r, K)]].tolist()))).mean() for r in REPS]))
                  for K in KS if K != "all"}
            subset_tables(ctx, kind, store, t, "sub_ln", tuple(K for K in KS if K != "all"), fr)
    # 2.4.4
    section4(ctx, kind, store, tgts, tdata, notes)
    # 2.4.5
    section5(ctx, kind, store, tgts, allcsv, notes)
    ctx.timing[kind] = time.perf_counter() - t0
    del store, tdata, dr


def fixc_gains(ctx: Ctx, kind: str, names: list[str]) -> pd.DataFrame | None:
    """補足の行：C を固定した地図（反復 0〜29）の組 × K の平均の利得（同じコーパスの組を除く）。"""
    d = Path(ctx.a.fixc_map) / kind
    rows = []
    for T in names:
        p = d / f"{T}.csv"
        if not p.exists():
            return None
        ctx.note_input(p)
        m = pd.read_csv(p, comment="#", dtype={"K": str, "source": str}, usecols=["source", "method", "K", "rep", "uar"])
        m["K"], m["source"] = m.K.fillna(""), m.source.fillna("")
        tg = m[m.method == "tgt_only"].set_index(["K", "rep"]).uar
        mg = m[(m.method == "merged") & (m.rep < 30)]
        mg = mg[~mg.source.map(lambda s: same_corpus(T, s))]
        g = mg.uar.to_numpy() - tg.loc[list(zip(mg.K, mg.rep))].to_numpy()
        x = pd.DataFrame({"target": T, "source": mg.source.to_numpy(), "K": mg.K.to_numpy(), "rep": mg.rep.to_numpy(), "g": g})
        ctx.record("1", kind, f"地図の反復の数（反復 0〜29 の補足）{T}", int(x.rep.nunique()))
        rows.append(x.groupby(["target", "source", "K"]).g.mean().rename("gain").reset_index())
    return pd.concat(rows, ignore_index=True) if rows else None


def p1_uar(ctx: Ctx, kind: str) -> pd.Series:
    """E31 の P1（仮説転移．中心合わせなし，C_b 自由，C′ = 0.01）の UAR（反復 0〜9 の平均．E32 と同じく explore_compare の読み方）。"""
    d = Path(f"{ctx.a.compare_prefix}{kind}")
    for f in sorted(d.glob("*.csv")):
        ctx.note_input(f)
    x = EC.FAMILIES[0].rows(EC.load_compare(kind))
    x = x[np.isclose(x.setting.astype(float), EC.DEF_CP)]
    return x.groupby(["target", "source", "K"]).uar.mean()


# ---------------------------------------------------------------- 2.4.4

def section4(ctx: Ctx, kind: str, store: Store, tgts: list[Target], tdata: dict, notes: pd.DataFrame) -> None:
    Q = {"G_crit": lambda s, K: [(1, (s, "h4", K)), (-1, (s, "src"))],
         "G_merge": lambda s, K: [(1, (s, "m", K)), (-1, (s, "src"))],
         "G_center": lambda s, K: [(1, (s, "h2")), (-1, (s, "src"))],
         "G_crit_c": lambda s, K: [(1, (s, "h5", K)), (-1, (s, "h2"))],
         "G_merge_c": lambda s, K: [(1, (s, "h3", K)), (-1, (s, "h2"))],
         "G_center_pool": lambda s, K: [(1, (s, "h2pool")), (-1, (s, "src"))],
         "G_crit_c_pool": lambda s, K: [(1, (s, "h5pool", K)), (-1, (s, "h2pool"))]}
    for l2 in L2_ALT:
        Q[f"G_crit_l2_{l2:g}"] = (lambda l2_: lambda s, K: [(1, (s, "h4l2", l2_, K)), (-1, (s, "src"))])(l2)
        Q[f"G_crit_c_l2_{l2:g}"] = (lambda l2_: lambda s, K: [(1, (s, "h5l2", l2_, K)), (-1, (s, "h2"))])(l2)
    pairs = [(t.name, s) for t in tgts for s in t.sources if not same_corpus(t.name, s)]
    prow = {}
    for K in KH:
        for qn, f in Q.items():
            kterms = []
            for T, s in pairs:
                terms = [(c, T, k) for c, k in f(s, K)]
                kterms += [(c / len(pairs), T, k) for c, T, k in terms]
                v = store.res[(T,) + terms[0][2]]["theta"] - store.res[(T,) + terms[1][2]]["theta"]
                prow.setdefault((T, s, K), {})[qn] = v * 100
            iv = store.ival(kterms)
            iv.pop("_full")
            ctx.add("e76a_4_kind", {"kind": kind, "short": SHORT[kind], "K": K, "quantity": qn, "n_pairs": len(pairs),
                                    "mean": iv["theta"], **{k: v for k, v in iv.items() if k != "theta"}, "note": kind_note(notes)})
    # d・o・floor・R・橋（点の値．反復の平均）
    for t in tgts:
        pt = tdata[t.name]["point"]
        for s, g in pt.groupby("source"):
            mean = g.drop(columns=["source", "rep"]).groupby("method").mean()
            base = {}
            for meth in mean.index:
                d = mean.loc[meth, [f"d_{p}" for p in NAME_PAIRS]].to_numpy(float)
                o = mean.loc[meth, [f"o_{p}" for p in NAME_PAIRS]].to_numpy(float)
                sd = mean.loc[meth, [f"s_{p}" for p in NAME_PAIRS]].to_numpy(float)
                base[meth] = (d, o, sd)
                for i, p in enumerate(NAME_PAIRS):
                    ctx.add("e76a_4_do", {"kind": kind, "short": SHORT[kind], "target": t.name, "source": s, "method": meth,
                                          "pair": p, "d": d[i], "o": o[i], "s": sd[i]})
            fl = {m: floor_o(base[m][1], base[m][2])[0] for m in ("src", "h2")}
            ao = {m: float(np.abs(base[m][1]).mean()) for m in ("src", "h2")}
            ad = {m: float(np.abs(base[m][0]).mean()) for m in ("src", "h2")}
            den = ao["src"] - fl["src"]
            br = {m: float(np.mean(ba(base[m][0], 0) - ba(base[m][0], base[m][1]))) * 100 for m in ("src", "h2")}
            for K in KH:
                r = prow[(t.name, s, K)]
                r.update({"floor_src": fl["src"], "floor_h2": fl["h2"], "abs_o_src": ao["src"], "abs_o_h2": ao["h2"],
                          "abs_d_src": ad["src"], "abs_d_h2": ad["h2"],
                          "R_o_star": (ao["h2"] - fl["h2"]) / den if den >= 0.05 else np.nan, "R_o_star_den_small": bool(den < 0.05),
                          "R_o": ao["h2"] / ao["src"] if ao["src"] > 0 else np.nan, "R_d": ad["h2"] / ad["src"] if ad["src"] > 0 else np.nan,
                          "floor_over_abs_o_src": fl["src"] / ao["src"] if ao["src"] > 0 else np.nan,
                          "floor_over_abs_o_h2": fl["h2"] / ao["h2"] if ao["h2"] > 0 else np.nan,
                          "bridge_pred_src": br["src"], "bridge_pred_h2": br["h2"]})
    rows = []
    for (T, s, K), r in prow.items():
        rows.append({"kind": kind, "short": SHORT[kind], "target": T, "source": s, "K": K, **r, "note": note_for(notes, T, s, K)})
    R = pd.DataFrame(rows)
    ctx.add("e76a_4_pairs", rows)
    for K, g in R.groupby("K"):
        ctx.add("e76a_4_summary", {"kind": kind, "short": SHORT[kind], "K": K, "n_pairs": len(g),
                                   "median_R_o_star": float(g.R_o_star.median()), "n_R_o_star_excluded": int(g.R_o_star_den_small.sum()),
                                   "median_R_o": float(g.R_o.median()), "median_R_d": float(g.R_d.median()),
                                   "median_floor_src": float(g.floor_src.median()), "median_floor_h2": float(g.floor_h2.median()),
                                   "median_floor_over_abs_o_src": float(g.floor_over_abs_o_src.median()),
                                   "median_floor_over_abs_o_h2": float(g.floor_over_abs_o_h2.median()),
                                   "median_bridge_ratio_src": float((g.bridge_pred_src / g.G_crit).median()),
                                   "median_bridge_ratio_h2": float((g.bridge_pred_h2 / g.G_crit_c).median()),
                                   "spearman_bridge_src": spear(g.bridge_pred_src, g.G_crit)[0],
                                   "spearman_bridge_h2": spear(g.bridge_pred_h2, g.G_crit_c)[0]})
    # 6 対を同時に直せる切片（オラクル）
    for t in tgts:
        j = tdata[t.name]["joint"]
        if j.empty:
            continue
        m = j.groupby("source").mean(numeric_only=True).reset_index()
        for x in m.to_dict("records"):
            ctx.add("e76a_4_joint", {"kind": kind, "short": SHORT[kind], "target": t.name,
                                     **{k: (v * 100 if k not in ("source", "rep", "joint_success") else v) for k, v in x.items()},
                                     "n_joint_not_success": int((~j[j.source == x["source"]].joint_success.astype(bool)).sum())})
    # H5
    for t in tgts:
        h = pd.DataFrame(tdata[t.name]["h5acc"])
        for meth, g in h.groupby("method"):
            v = np.concatenate([np.array(a) for a in g.absdiff])
            ctx.add("e76a_6_h5_rows", {"kind": kind, "short": SHORT[kind], "target": t.name, "method": meth, "n": len(v),
                                       "median_absdiff": float(np.median(v))})


# ---------------------------------------------------------------- 2.4.5

def section5(ctx: Ctx, kind: str, store: Store, tgts: list[Target], allcsv: dict, notes: pd.DataFrame) -> None:
    for incl_same in (False, True):
        pairs = [(t.name, s) for t in tgts for s in t.sources if incl_same or not same_corpus(t.name, s)]
        for K in KH:
            alpha = {}
            for T, s in pairs:
                am = h10_meta_alpha(allcsv, T, s, K)
                orc = max(ALPHAS, key=lambda a: store.res[(T, s, "h10", a, K)]["theta"])
                alpha[(T, s)] = am
                if not incl_same:
                    ctx.add("e76a_5_alpha", {"kind": kind, "short": SHORT[kind], "target": T, "source": s, "K": K,
                                             "alpha_meta": am, "alpha_oracle": orc, "note": note_for(notes, T, s, K)})
            if any(not np.isfinite(a) for a in alpha.values()):
                ctx.record("5", kind, f"メタ検証の T′ が無い組の数 K={K} 同じコーパス含む={incl_same}",
                           int(sum(not np.isfinite(a) for a in alpha.values())))

            def hkey(h, s, T):
                if h == "h1":
                    return (s, "m", K)
                if h == "h2":
                    return (s, "h2")
                if h == "h10":
                    return (s, "h10", alpha[(T, s)], K)
                if h == "h10_oracle":
                    return (s, "h10", max(ALPHAS, key=lambda a: store.res[(T, s, "h10", a, K)]["theta"]), K)
                return (s, h, K)
            ok_pairs = [(T, s) for T, s in pairs if np.isfinite(alpha[(T, s)])]
            n = len(ok_pairs)
            means = {}
            for h in HANDS + ("h10_oracle",):
                terms = []
                for T, s in ok_pairs:
                    terms += [(1 / n, T, hkey(h, s, T)), (-1 / n, T, (s, "src"))]
                iv = store.ival(terms)
                iv.pop("_full")
                means[h] = iv["theta"]
                ctx.add("e76a_5_hands", {"kind": kind, "short": SHORT[kind], "K": K, "incl_same_corpus": incl_same, "hand": h,
                                         "n_pairs": n, "mean_diff_from_src": iv["theta"], **{k: v for k, v in iv.items() if k != "theta"},
                                         "note": kind_note(notes)})
            if incl_same:
                continue
            best = max(HANDS, key=lambda h: means[h])
            for sub in SUBJECTS:
                for other in HANDS:
                    if other == sub:
                        continue
                    terms = []
                    for T, s in ok_pairs:
                        terms += [(1 / n, T, hkey(sub, s, T)), (-1 / n, T, hkey(other, s, T))]
                    iv = store.ival(terms)
                    iv.pop("_full")
                    ctx.add("e76a_5_compare", {"kind": kind, "short": SHORT[kind], "K": K, "subject": sub, "other": other,
                                               "n_pairs": n, "diff": iv["theta"], **{k: v for k, v in iv.items() if k != "theta"},
                                               "best_by_mean": best, "note": kind_note(notes)})
            # ターゲットごとに最も高い手
            for T in sorted({T for T, _ in ok_pairs}):
                ss = [s for TT, s in ok_pairs if TT == T]
                v = {h: np.mean([store.res[(T,) + hkey(h, s, T)]["theta"] for s in ss]) for h in HANDS}
                ctx.add("e76a_5_targets", {"kind": kind, "short": SHORT[kind], "K": K, "target": T, "best_hand": max(v, key=v.get),
                                           "n_sources": len(ss)})


# ---------------------------------------------------------------- 判定（値を見る前に書いた表を当てる）

def judgments(ctx: Ctx) -> None:
    J = []

    def add(section, rule, kind, holds, values, writing):
        J.append({"section": section, "rule": rule, "kind": kind, "holds": holds, "values": values, "writing": writing})

    T = {k: pd.DataFrame(v) for k, v in ctx.tables.items()}
    kinds = [k for k in KINDS_ALL if k in set(T.get("e76a_1_kind", pd.DataFrame({"kind": []})).kind)]
    # 2.4.1
    if "e76a_1_kind" in T:
        k1 = T["e76a_1_kind"]
        tg = T["e76a_1_targets"]
        kd = T["e76a_1_kstar_dist"]
        k1_1 = k1[k1.K == "1"].set_index("kind")
        below = [k for k in kinds if k1_1.loc[k, "share_up_holm"] < 0.75]
        add("2.4.1", "どの種類でも（8 種類），K=1 で「上」の組が 75% 以上", "all", len(kinds) == 8 and not below,
            json.dumps({SHORT[k]: f"{int(k1_1.loc[k, 'n_up_holm'])}/{int(k1_1.loc[k, 'n_pairs'])}" for k in kinds}, ensure_ascii=False),
            "「どの種類でも K=1 で上がる」を残し，数を新しい数に替える．8 種類は同じコーパス・同じ話者・同じ分割の上の別の特徴で，8 回の独立な確かめではないことを 1 文添える")
        add("2.4.1", "75% 未満の種類がある", "all", bool(below), ";".join(SHORT[k] for k in below),
            "その種類の名前と数を書き，「どの種類でも」を満たす種類に限る")
        t1 = tg[(tg.K == "1") & (tg.share_up_holm < 0.5)]
        add("2.4.1", "「上」の組が半分未満のターゲットがある（どの種類でも．K=1 で数えた）", "all", len(t1) > 0,
            ";".join(f"{SHORT[x.kind]}:{x.target}({x.n_up_holm}/{x.n_pairs})" for x in t1.itertuples()),
            "そのターゲットの名前を書く（組の割合は実質ターゲットの数で決まるため）")
        ka = k1[k1.K == "all"].set_index("kind")
        dn = [k for k in kinds if ka.loc[k, "share_down_holm"] < 0.5]
        add("2.4.1", "全件で「下」の組が半分未満の種類がある", "all", bool(dn),
            ";".join(f"{SHORT[k]}({int(ka.loc[k, 'n_down_holm'])}/{int(ka.loc[k, 'n_pairs'])})" for k in dn),
            "その種類では「大きい K で下がる」を「下がる組がある（x/y 組）」に替える")
        mv = []
        for k in kinds:
            a = kd[(kd.kind == k) & (kd.interval == "main") & (kd.definition == "last_continuing")].median_rank
            b = kd[(kd.kind == k) & (kd.interval == "naive_t") & (kd.definition == "last_continuing")].median_rank
            if len(a) and len(b) and abs(float(a.iloc[0]) - float(b.iloc[0])) >= 2:
                mv.append(f"{SHORT[k]}({float(a.iloc[0])} vs {float(b.iloc[0])})")
        add("2.4.1", "K* の中央値が，同じ反復 0〜9 の反復を独立とみなした t 区間の K* から 2 段以上動く種類がある", "all", bool(mv),
            ";".join(mv), "本文に「K* は区間の作り方に依る」と書き，両方の分布を補足に置く")
    # 2.4.2
    if "e76a_2_gender" in T:
        g = T["e76a_2_gender"]
        g5 = g[g.K == "5"]
        add("2.4.2", "性別の差（記述．K=5）", "all", np.nan,
            json.dumps({"n_diffs": len(g5), "lo_min": float(g5.diff_lo.min()), "hi_max": float(g5.diff_hi.max())}),
            "倫理の節に，差と区間をそのまま表で示し，「差の 95% 区間は −a〜+b 点で，この幅より小さい差はこの設計では検出できない」と書く．"
            "区間が 0 を含まない差があっても，性別で利得が違うとは書かない（記述）")
    if "e76a_2_subsets" in T:
        s = T["e76a_2_subsets"]
        s = s[s.unit == "target_mean"]
        for tag, name in (("sub_ch", "MOSEI"), ("sub_ln", "台詞")):
            h = s[s.subset == tag]
            if h.empty:
                continue
            outk = h[~h.within_half_width]
            add("2.4.2", f"{name}の部分集合での利得と全テストの利得の差が，全テストの区間の半幅以内（ターゲットの平均．全種類 × K）", "all",
                outk.empty, ";".join(f"{SHORT[x.kind]} K={x.K}" for x in outk.itertuples()),
                "「部分集合でも結論は変わらない」と 1 文" if outk.empty else
                ("両方の値を書き，言葉系の主張を IEMOCAP-script を除いた集計に限る（台詞のとき）．MOSEI はその向きを書き，部分集合の利得が大きい K で"
                 "全テストより高ければ，6 節の (a) と同じ向き（同じ人物がプールに残ると ① が上がり利得が低めに出る）だと書く"))
            if tag == "sub_ch":
                big = h[h.K.isin(["25", "50", "all"])]
                add("2.4.2", "MOSEI：大きい K（25・50・全件）で部分集合の利得が全テストより高い（向き）", "all",
                    bool((big.diff_subset_minus_full > 0).all()) if len(big) else np.nan,
                    ";".join(f"{SHORT[x.kind]} K={x.K}:{'+' if x.diff_subset_minus_full > 0 else '-'}" for x in big.itertuples()),
                    "半幅を超えたとき，6 節の (a) と同じ向きだと書く")
    # 2.4.3
    if "e76a_3_gain" in T:
        gtab = T["e76a_3_gain"]
        q = T["e76a_3_q_targets"]
        k1 = gtab[gtab.K == "1"].set_index(["kind", "rule"])
        dev = {k: float(k1.loc[(k, "pseudo"), "mean_gain"] - k1.loc[(k, "main"), "mean_gain"]) for k in kinds}
        within = all(abs(v) <= 1.0 for v in dev.values())
        add("2.4.3", "どの種類でも，仮のターゲットの規則の K=1 の利得が主の規則の値から 1.0 点以内", "all", within and len(kinds) == 8,
            json.dumps({SHORT[k]: round(v, 6) for k, v in dev.items()}, ensure_ascii=False),
            "「K=1 の利得はターゲットのみの C の規則に依らない（合併の C はソースのみの C に固定したまま．範囲 a〜b）」と 1 文．数は主の規則のまま")
        over_pos = [k for k, v in dev.items() if abs(v) > 1.0 and k1.loc[(k, "pseudo"), "lo"] > 0]
        add("2.4.3", "1.0 点を超えて動くが正のまま", "all", bool(over_pos), ";".join(SHORT[k] for k in over_pos),
            "本文の K=1 の利得を 2 つの規則の範囲で書き，要旨は小さい方の値を書く")
        zero = [k for k in kinds if k1.loc[(k, "pseudo"), "lo"] <= 0]
        add("2.4.3", "ある種類で仮のターゲットの規則の K=1 の利得の区間が 0 を含むか 0 より下", "all", bool(zero), ";".join(SHORT[k] for k in zero),
            "その種類の K=1 を主張から外し，「K=1 の利得はターゲットのみの C の規則に依る」と書く．要旨の K=1 の数を K=5 の数に替える")
        prim = q[(q.D_version == "D") & (q.fit == "3pt")]
        for k in kinds:
            a = prim[(prim.kind == k) & (prim.rule == "main")].q.median()
            b = prim[(prim.kind == k) & (prim.rule == "pseudo")].q.median()
            ratio = b / a if a > 0 else np.nan
            ok = bool(0.5 <= ratio <= 2) if np.isfinite(ratio) else np.nan
            add("2.4.3", "仮のターゲットの規則の q の中央値が，主の q の 0.5〜2 倍の中にある（種類ごと．D，3 点）", k, ok, f"ratio={ratio}",
                "「q は C の規則で x〜y」と 1 文" if ok else "q を規則ごとの表にし，「q は正則化の効果を含み，1 つの値に決まらない」と書く")
            ext = q[(q.kind == k) & (q.rule == "main") & (q.fit == "5pt")]
            if len(ext):
                eD = ext[ext.D_version == "D"]
                eP = ext[ext.D_version == "Dprime"]
                pooled = lambda e, c, th: bool(e[c].mean() > e[th].mean())  # noqa: E731
                exD, exP = pooled(eD, "mean_abs_resid", "resid_threshold"), pooled(eP, "mean_abs_resid", "resid_threshold")
                exM1 = pooled(eD, "M1_mean_abs_resid", "M1_resid_threshold")
                loD = float(eD.M1orM2_better_loko.mean())
                loP = float(eP.M1orM2_better_loko.mean())
                cond1 = (exD and exP and exM1) or (loD >= 2 / 3 and loP >= 2 / 3)
                add("2.4.3", "延長の当てはめで，D でも D′ でも残差が基準を超え，かつ M1 でも基準を超える種類，または M1・M2 の方が抜いた予測の誤差が小さい"
                    "ターゲットが 3 分の 2 以上で，D′ でも同じになる種類（主の規則）", k, cond1,
                    json.dumps({"exceed_D": exD, "exceed_Dprime": exP, "exceed_M1": exM1, "share_M1orM2_better_D": loD,
                                "share_M1orM2_better_Dprime": loP, "n_targets": len(eD),
                                "n_targets_exceed_D": int(eD.resid_exceeds.sum()), "n_targets_exceed_Dprime": int(eP.resid_exceeds.sum()),
                                "n_targets_exceed_M1": int(eD.M1_resid_exceeds.sum())}),
                    "その種類では「1 母数の形は K ≤ 50 の曲線を表さない」と書き，q の使い道を K ≤ 10 に限る．言葉系で当てはまらなければ，q を出さない理由として書く")
                cond2 = exD and (not exP or not exM1)
                add("2.4.3", "D では基準を超えるが，D′ か M1 では超えない（主の規則）", k, cond2,
                    json.dumps({"exceed_D": exD, "exceed_Dprime": exP, "exceed_M1": exM1}),
                    "「形は合うが，own_d は ① の漸近の値ではない」と書き，q は D′ の当てはめの値も並べる")
    if "e76a_3_n_pairs" in T:
        N = T["e76a_3_n_pairs"]
        for k in kinds:
            for K in ("5", "10"):
                a = N[(N.kind == k) & (N.rule == "main") & (N.K == K)].set_index(["target", "source"])
                b = N[(N.kind == k) & (N.rule == "pseudo") & (N.K == K)].set_index(["target", "source"])
                both = a.index[(a.flag_meas == 0) & (b.reindex(a.index).flag_meas == 0)]
                ma, mb = float(a.loc[both].n_meas.median()), float(b.loc[both].n_meas.median())
                rel = abs(mb - ma) / abs(ma) if ma != 0 else np.nan
                add("2.4.3", f"n の実測の中央値（K={K}．両方の規則で打ち切られない組）が規則で 30% を超えて動く", k,
                    bool(rel > 0.3) if np.isfinite(rel) else np.nan,
                    json.dumps({"n_pairs_both": len(both), "rel_change": rel,
                                "censored_main": int((a.flag_meas != 0).sum()), "censored_pseudo": int((b.flag_meas != 0).sum())}),
                    "両方の値を書く（30% 以内なら 1 文）")
    # 2.4.4
    if "e76a_4_summary" in T:
        s4 = T["e76a_4_summary"]
        s5 = s4[s4.K == "5"].set_index("kind")
        w = [k for k in ("audio_wavlm", "audio_hubert") if k in s5.index]
        vals = {SHORT[k]: float(s5.loc[k, "median_R_o_star"]) for k in w}
        if len(w) == 2:
            le = all(v <= 0.5 for v in vals.values())
            add("2.4.4", "W7 と Hu7 の両方で R_o* の組の中央値が 0.5 以下（K=5．V・AV・言葉系は記述）", "W7,Hu7", le, json.dumps(vals, ensure_ascii=False),
                "P29-4 の読みを「中心合わせは K 本が直せる o の大部分を先に消すので，その後の切片直しの得は小さい」と書く（探索の結果として）" if le else
                "中心合わせの後に切片直しの得が小さくなることは，切片で直せる o が小さくなったことでは説明できないと書き，P29-4 を UAR の平均の記述に限る．"
                "W7 と Hu7 で分かれたら種類ごとに書く")
        kq = T["e76a_4_kind"]
        for k in kinds:
            x = kq[(kq.kind == k) & (kq.K == "5")].set_index("quantity")
            if "G_crit_c" not in x.index:
                continue
            gc = x.loc["G_crit_c"]
            add("2.4.4", "G_crit_c の区間が 0 より上（K=5）", k, bool(gc.lo > 0),
                json.dumps({"mean": gc["mean"], "lo": gc.lo, "hi": gc.hi, "half_width": (gc.hi - gc.lo) / 2}),
                "「中心合わせの後も切片直しの得が x 点残る」" if gc.lo > 0 else "「中心合わせの後の切片直しの得は x 点で，区間は ± y 点」（幅を必ず添える）")
            dl = {l2: float(x.loc[f"G_crit_l2_{l2:g}", "mean"] - x.loc["G_crit", "mean"]) for l2 in L2_ALT}
            add("2.4.4", "l2 を 10 倍・1/10 にして G_crit（K=5）の種類の平均が 0.5 点を超えて動く", k, any(abs(v) > 0.5 for v in dl.values()),
                json.dumps({f"{l2:g}": v for l2, v in dl.items()}), "補足に l2 の表を置き，本文で触れる（0.5 点以内なら補足の 1 文）")
            dc = float(x.loc["G_center_pool", "mean"] - x.loc["G_center", "mean"])
            add("2.4.4", "プールの平均の版の G_center（種類の平均）が全発話の平均の版から 0.5 点以内（K=5）", k, abs(dc) <= 0.5,
                json.dumps({"diff": dc}),
                "「中心合わせの平均にテストの発話を含めなくても，中心合わせの得はほぼ同じ（差 x 点）」と 1 文" if abs(dc) <= 0.5 else
                "両方の値を書き，主張 1 の中心合わせの数値は「テストを含む全発話の平均（変換的な設定）」についてのものだと本文に書く．h3 の数値も同じ限りで書く")
    # 2.4.5
    if "e76a_5_compare" in T:
        c = T["e76a_5_compare"]
        for k in kinds:
            for sub in SUBJECTS:
                h = c[(c.kind == k) & (c.subject == sub) & (c.K.isin(["5", "10"]))]
                allup = bool((h.lo > 0).all())
                bad = h[h.lo <= 0]
                best = h.best_by_mean.unique().tolist()
                add("2.4.5", f"K=5・10 で，主語の手（{sub}）とほかのすべての手との差の区間が 0 より上（記述）", k, allup,
                    json.dumps({"not_above": [f"K={x.K}:{x.other}(±{(x.hi - x.lo) / 2:.6f})" for x in bad.itertuples()],
                                "best_by_mean": best}, ensure_ascii=False),
                    f"「{sub} が手 1〜11 の中で最も高い」を残す" if allup else
                    f"「{sub} は平均で最も高いが，（手の名前）との差は区間の中（± x 点）」")
                add("2.4.5", f"平均で最も高い手が主語の手（{sub}）でない（K=5・10 のどちらか）", k, any(b != sub for b in best),
                    json.dumps({"best_by_mean": best}), "最も高い手の名前を書き，主語の手との差と区間を書く")
    ctx.tables["e76a_judgments"] = J


def assumptions(ctx: Ctx) -> None:
    """2.4.6 の H1〜H6（種類ごと）。"""
    T = {k: pd.DataFrame(v) for k, v in ctx.tables.items()}
    rows = []
    if "e76a_3_dT" in T:
        dT = T["e76a_3_dT"]
        q = T["e76a_3_q_targets"]
        for k, g in dT[dT.rule == "main"].groupby("kind", sort=False):
            big = g[g.K.isin(["25", "50", "all"])]
            share = float(big.d_T_gt_D.mean())
            dd = g[g.K == "all"]
            ratio = float((dd.Dprime / dd.D).median())
            rows.append({"kind": k, "assumption": "H1 D は ① の d_T(K) の K → ∞ の値である",
                         "measure": "d_T(K) > D の割合（K=25・50・全件）と D′/D の中央値", "value": json.dumps({"share_gt": share, "Dprime_over_D_median": ratio}),
                         "holds": bool(share <= 0.1 and 0.9 <= ratio <= 1.1),
                         "if_not": "q の主の値を D′ の当てはめに替えて並べる（2.4.3 節の表の 7 行目）"})
            for K, h in g.groupby("K"):
                ctx.add("e76a_3_dT_gt_D", {"kind": k, "short": SHORT[k], "K": K, "n_cells": len(h), "n_gt_D": int(h.d_T_gt_D.sum())})
            qp = q[(q.kind == k) & (q.rule == "main") & (q.D_version == "D") & (q.fit == "3pt")]
            rows.append({"kind": k, "assumption": "H2 6 対で q が共通である", "measure": "q_kl の最大 ÷ 最小のターゲットの中央値",
                         "value": float(qp.q_pair_max_over_min.median()), "holds": bool(qp.q_pair_max_over_min.median() <= 3),
                         "if_not": "q を「6 対の平均の曲線の係数」と書く"})
            ext = q[(q.kind == k) & (q.rule == "main") & (q.fit == "5pt")]
            rows.append({"kind": k, "assumption": "H3 d_T は 1/K の形で D に近づく", "measure": "延長の残差，M1・M2 との抜いた予測（判定は e76a_judgments の 2.4.3）",
                         "value": json.dumps({"n_targets_exceed_D": int(ext[ext.D_version == "D"].resid_exceeds.sum()),
                                              "n_targets_exceed_Dprime": int(ext[ext.D_version == "Dprime"].resid_exceeds.sum()),
                                              "n_targets": int((ext.D_version == "D").sum())}),
                         "holds": np.nan, "if_not": "2.4.3 節の判定の表のとおり"})
            pr = q[(q.kind == k) & (q.D_version == "D") & (q.fit == "3pt")]
            a = pr[pr.rule == "main"].q.median()
            fixed = {r: float(pr[pr.rule == r].q.median()) for r in RULES[2:]}
            ratio = float(pr[pr.rule == "pseudo"].q.median() / a) if a > 0 else np.nan
            rows.append({"kind": k, "assumption": "H4 q は C の規則に依らない", "measure": "仮のターゲット ÷ 主の q の中央値と，C を固定した 4 本の q の中央値",
                         "value": json.dumps({"pseudo_over_main": ratio, **fixed}), "holds": bool(0.5 <= ratio <= 2) if np.isfinite(ratio) else np.nan,
                         "if_not": "q は正則化の効果を含むと書く"})
    if "e76a_6_h5_rows" in T:
        h5 = T["e76a_6_h5_rows"]
        for k, g in h5.groupby("kind", sort=False):
            v = {m: float(h.median_absdiff.median()) for m, h in g.groupby("method")}
            rows.append({"kind": k, "assumption": "H5 対の点数の差 z は 2 つの等分散の正規分布である",
                         "measure": "|acc_actual − acc_model| の中央値（ターゲットごとの中央値の中央値．②，h2，①）", "value": json.dumps(v),
                         "holds": bool(all(x <= 0.01 for x in v.values())), "if_not": "R_o* と模型の予言の UAR への読みを記述にとどめる"})
    if "e76a_4_pairs" in T:
        p = T["e76a_4_pairs"]
        for k, g in p[p.K == "5"].groupby("kind", sort=False):
            a, b = float(g.floor_over_abs_o_src.median()), float(g.floor_over_abs_o_h2.median())
            rows.append({"kind": k, "assumption": "H6 切片で直せる o の分と直せない分を分けられる",
                         "measure": "floor ÷ 6 対の平均の |o| の組の中央値（② と h2）", "value": json.dumps({"src": a, "h2": b}),
                         "holds": bool(a <= 0.2 and b <= 0.2), "if_not": "R_o* で判定する（もともと主）"})
    ctx.tables["e76a_6_assumptions"] = rows


def unit_checks(ctx: Ctx) -> None:
    """設計の表の単位の数（全種類・全ターゲットで流したときだけ数を比べる）と，値に NaN が無いこと。"""
    T = {k: pd.DataFrame(v) for k, v in ctx.tables.items()}
    full = ctx.a.targets is None
    if full:
        u = T["e76a_0_units"]
        for k in ctx.a.kinds:
            ctx.check("U", k, "ターゲットの数と設計の数の差", abs(int((u.kind == k).sum()) - EXPECTED_TARGETS[k]), 0)
            for name, col in (("e76a_4_pairs", "K"), ("e76a_5_alpha", "K")):
                n = T[name][(T[name].kind == k)].groupby(col).size()
                ctx.check("U", k, f"{name} の組の数（K ごとの最大）と設計の数の差", float((n - EXPECTED_PAIRS[k]).abs().max()), 0)
        if set(ctx.a.kinds) == set(KINDS_ALL):
            g = T["e76a_2_gender"]
            ctx.check("U", "all", "性別の差の数（K=5）と設計の 22 の差", abs(int((g.K == "5").sum()) - 22), 0)
            sb = T["e76a_2_subsets"]
            ctx.check("U", "all", "MOSEI のチャンネルの部分集合の種類の数と 8 の差", abs(sb[sb.subset == "sub_ch"].kind.nunique() - 8), 0)
            ctx.check("U", "all", "台詞の重なりの部分集合の種類の数と 4 の差", abs(sb[sb.subset == "sub_ln"].kind.nunique() - 4), 0)
    for name, cols in (("e76a_1_pairs", ["gain", "lo", "hi", "p", "p_holm"]), ("e76a_1_kind", ["mean_gain", "lo", "hi"]),
                       ("e76a_3_pairs", ["gain", "lo", "hi"]), ("e76a_3_q_targets", ["q", "q_lo", "q_hi"]),
                       ("e76a_4_kind", ["mean", "lo", "hi"]), ("e76a_5_hands", ["mean_diff_from_src", "lo", "hi"]),
                       ("e76a_5_compare", ["diff", "lo", "hi"]), ("e76a_2_gender", ["diff_F_minus_M", "diff_lo", "diff_hi"]),
                       ("e76a_2_subsets", ["gain_full", "gain_subset", "subset_lo", "subset_hi"])):
        if name in T:
            ctx.check("U", "all", f"{name} の {','.join(cols)} の NaN の数", float(T[name][cols].isna().sum().sum()), 0)


def q_n_summaries(ctx: Ctx) -> None:
    """表 2 の q と n の列（種類 × 規則）。"""
    q = pd.DataFrame(ctx.tables.get("e76a_3_q_targets", []))
    for (k, rule, dv, fit), g in (q.groupby(["kind", "rule", "D_version", "fit"], sort=False) if len(q) else []):
        ne = g[~g.q_at_edge]
        ctx.add("e76a_3_q_summary", {"kind": k, "short": SHORT[k], "rule": rule, "D_version": dv, "fit": fit, "n_targets": len(g),
                                     "median_q_incl_edge": float(g.q.median()), "q25": float(g.q.quantile(0.25)), "q75": float(g.q.quantile(0.75)),
                                     "n_at_edge": int(g.q_at_edge.sum()), "median_q_excl_edge": float(ne.q.median()) if len(ne) else np.nan,
                                     "n_targets_resid_exceeds": int(g.resid_exceeds.sum())})
    N = pd.DataFrame(ctx.tables.get("e76a_3_n_pairs", []))
    for (k, rule, K), g in (N.groupby(["kind", "rule", "K"], sort=False) if len(N) else []):
        ok = g[g.flag_meas == 0]
        ctx.add("e76a_3_n_summary", {"kind": k, "short": SHORT[k], "rule": rule, "K": K, "n_pairs": len(g),
                                     "median_n_meas_uncensored": float(ok.n_meas.median()) if len(ok) else np.nan,
                                     "n_censored_high": int((g.flag_meas == 1).sum()), "n_censored_low": int((g.flag_meas == -1).sum()),
                                     "median_n_pred": float(g[g.flag_pred == 0].n_pred.median()) if (g.flag_pred == 0).any() else np.nan})


# ---------------------------------------------------------------- 発話一覧

def load_meta(ctx: Ctx) -> dict:
    md = ROOT / "data" / "metadata"
    gender = {}
    p = md / "iemocap_utterance_manifest.csv"
    ctx.note_input(p)
    m = pd.read_csv(p, usecols=["sample_id", "utterance_speaker", "transcript"])
    gender.update({("IEMOCAP", s): g for s, g in zip(m.sample_id.astype(str), m.utterance_speaker.astype(str))})
    lines = {s: script_norm(x) for s, x in zip(m.sample_id.astype(str), m.transcript.fillna(""))}
    p = md / "afew_utterance_manifest.csv"
    ctx.note_input(p)
    a = pd.read_csv(p, usecols=["sample_id", "gender"])
    gender.update({("AFEW", s): {"Female": "F", "Male": "M"}.get(g, "") for s, g in zip(a.sample_id.astype(str), a.gender)})
    p = md / "ravdess_utterance_manifest.csv"
    ctx.note_input(p)
    rv = pd.read_csv(p, usecols=["sample_id", "speaker_id"])
    gender.update({("RAVDESS", s): ("M" if int(sp.split("_")[1]) % 2 == 1 else "F") for s, sp in zip(rv.sample_id.astype(str), rv.speaker_id)})
    p = md / "mosei_utterance_manifest.csv"
    ctx.note_input(p)
    mo = pd.read_csv(p, usecols=["sample_id", "speaker_source"])
    known = {s: v == "youtube_channel" for s, v in zip(mo.sample_id.astype(str), mo.speaker_source)}
    return {"gender": gender, "iemocap_lines": lines, "mosei_known": known}


def e13_table(ctx: Ctx, kinds: list[str]) -> pd.DataFrame:
    p = Path(ctx.a.e13)
    ctx.note_input(p)
    e = pd.read_csv(p)
    out = [e[e.kind.isin(kinds)]]
    for k in kinds:
        if k not in set(e.kind):
            t0 = time.perf_counter()
            h = measure(k).assign(kind=k)
            ctx.tables.setdefault(f"e76a_3_e13_{k}", []).extend(h.to_dict("records"))
            ctx.timing[f"e13_measure_{k}"] = time.perf_counter() - t0
            out.append(h)
    e = pd.concat(out, ignore_index=True)
    e["pair_idx"] = e.pair.map(E13_NAME_TO_IDX)
    return e


# ---------------------------------------------------------------- 起動

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=KINDS_ALL)
    ap.add_argument("--targets", nargs="+", help="試し用：ターゲットを絞る（仮のターゲットの規則と α のメタ検証は全ターゲットの csv を読む）")
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--e76", default=str(ROOT / "results" / "e76"))
    ap.add_argument("--fixc-map", default=str(ROOT / "results" / "fixc_map"))
    ap.add_argument("--e13", default=str(ROOT / "results" / "fixc_analysis" / "explore_relation" / "e13_pairs.csv"))
    ap.add_argument("--compare-prefix", default=str(ROOT / "results" / "compare_"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e76_analysis"))
    ap.add_argument("--allow-dirty", action="store_true", help="試しだけ：台本か読み込む部品にコミットしていない変更があっても流す")
    a = ap.parse_args()
    if any(k not in KINDS_ALL for k in a.kinds):
        raise SystemExit(f"種類は {KINDS_ALL} から")
    if a.compare_prefix != str(ROOT / "results" / "compare_"):
        raise SystemExit("explore_compare.load_compare は results/compare_<kind> を固定で読むので，--compare-prefix は既定のままにする")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} にファイルがある．上書きしない")
    mods = module_state()
    me = str(Path(__file__).resolve().relative_to(ROOT))
    dirty = {k: v["status"] for k, v in mods.items() if v["status"] != "clean"}
    if dirty and not a.allow_dirty:
        raise SystemExit(f"コミットしていない変更がある（--allow-dirty は試しだけ）：{dirty}")
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(a)
    t_all = time.perf_counter()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    commit = git("rev-parse", "HEAD")
    code_hash = hashlib.sha256("".join(f"{k}:{v['sha256']};" for k, v in sorted(mods.items())).encode()).hexdigest()
    ctx.header = {"script": me, "commit": commit, "script_sha256": mods.get(me, {}).get("sha256"), "code_hash": code_hash,
                  "dirty": bool(dirty), "B": a.B, "seed": a.seed, "kinds": a.kinds, "targets": a.targets, "started": started}
    written, accepted = accepted_hashes(ctx, Path(a.e76))
    meta = load_meta(ctx)
    e13_all = e13_table(ctx, a.kinds)
    for kind in a.kinds:
        print(f"{kind}：開始", flush=True)
        run_kind(ctx, kind, e13_all, written, accepted, meta)
        print(f"{kind}：{ctx.timing[kind]:.0f} 秒", flush=True)
    judgments(ctx)
    assumptions(ctx)
    q_n_summaries(ctx)
    unit_checks(ctx)
    files = ctx.write_all()
    elapsed = time.perf_counter() - t_all
    prov = {**ctx.header, "argv": sys.argv[1:], "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_sec": elapsed,
            "timing_sec": ctx.timing, "modules": mods, "inputs_sha256": ctx.inputs,
            "constants": {"REPS": list(REPS), "KS": list(KS), "KH": list(KH), "C_GRID": list(C_GRID), "ALPHAS": list(ALPHAS),
                          "L2_MAIN": L2_MAIN, "L2_ALT": list(L2_ALT), "GEFF_MIN": GEFF_MIN, "Q_GRID": [float(Q_GRID[0]), float(Q_GRID[-1]), len(Q_GRID)],
                          "M2_B_GRID": [float(M2_B_GRID[0]), float(M2_B_GRID[-1]), len(M2_B_GRID)], "E76_MAIN_HASH": E76_MAIN_HASH,
                          "accepted_finish_hashes": written},
            "outputs": dict(files),
            "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str))
    for name, n in files:
        print(f"  {name}：{n} 行", flush=True)
    bad = [c for c in ctx.checks if not c["ok"]]
    print(f"検算：{len(ctx.checks)} 件，基準を外れた {len(bad)} 件", flush=True)
    for c in bad:
        print(f"    外れ：[{c['section']}] {c['kind']} {c['what']}（値 {c['value']:.3g}，基準 {c['limit']}）", flush=True)
    print(f"終わり：{elapsed:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
