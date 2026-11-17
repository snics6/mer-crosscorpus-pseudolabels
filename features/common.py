"""発話一覧（マニフェスト）の読み込みと，置き場の定義。

マニフェストは `data/metadata/<corpus>_utterance_manifest.csv`。
1 行が 1 発話で，動画のパス・話者・ラベルが入っている。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META_DIR = ROOT / "data" / "metadata"
CACHE_DIR = ROOT / "data" / "cache"
PARTS_DIR = CACHE_DIR / "parts"
WEIGHTS_DIR = ROOT / "weights"

# 元からある 4 つ。映像・音声・言葉が揃っている（AFEW の言葉は Whisper の書き起こし．2026-11-04〜）。
CORPORA = ("IEMOCAP", "MOSEI", "MELD", "AFEW")

# 検出力のために足した 2 つ。俳優が同じ文を感情だけ変えて読むコーパス。
# 足した時点では順序つきの対が 6 → 20 に増えた（今は IEMOCAP を 2 群に分けて 7 群・42 組）。
CORPORA_EXTRA = ("RAVDESS", "CREMA-D")

CORPORA_ALL = CORPORA + CORPORA_EXTRA

# **言葉の特徴を作るのはこの 4 つ。**IEMOCAP・MOSEI・MELD は人の書き起こし，AFEW は
# Whisper large-v3 の書き起こし（features/build_transcript_afew.py．2026-11-04 に足した）。
# RAVDESS と CREMA-D は読む文が 2 種類・12 種類しかないので，言葉からは感情の手がかりが得られない。
CORPORA_TEXT = ("IEMOCAP", "MOSEI", "MELD", "AFEW")

# label_4 の整数値と名前の対応。0=Neutral, 1=Happy, 2=Sad, 3=Angry。
LABELS = ("Neutral", "Happy", "Sad", "Angry")

# 元の 4 コーパスの群は 5 つ（CORPORA_EXTRA を足すと 7 群）。IEMOCAP は即興と台本を分ける（data/README.md 取り決め 6）。
# 台本側のラベルは注釈ではなく芝居の番号で決まっているので，即興と混ぜられない。
GROUPS = ("IEMOCAP-impro", "IEMOCAP-script", "MOSEI", "MELD", "AFEW")

# マニフェストから読む列。無い列は空で埋める。
_COLS = (
    "sample_id", "corpus", "split", "speaker_id", "label_4", "dialog_type",
    "marker_side", "path_video", "path_audio", "transcript", "exclude",
)


def rng_for(name: str, *key: int) -> np.random.Generator:
    """**用途の名前と整数の組から乱数生成器を作る。種を算術で混ぜない。**

    混ぜると衝突する。実際に衝突していた 2 例：

      * `seed·1000 + k·10 + r` … (seed 0, k 100, r 10) と (seed 1, k 1, r 0)
        がともに 1010。600 通り中 160 件（27%）が衝突していた
      * `seed·104729 + c·131 + k·11 + r` … (0, クラス 0, k 25, r 0) と
        (0, クラス 2, k 1, r 2) がともに 275。2400 通り中 560 件（23%）

    衝突すると**同じ支持話者・同じ標本が選ばれる**ので，繰り返しの独立性が
    壊れ，ばらつきの推定が過小になる。`SeedSequence` は組をそのまま受けて
    混ぜるので，**組が違えば必ず違う列**になる（numpy の推奨する方法）。

    **`name` は用途を分ける。**同じ (seed, k, r) でも用途が違えば別の列にする
    ため。逆に**台本をまたいで分割を揃えたいときは同じ `name` を使う**。

    引数の整数は**非負**に限る（`SeedSequence` の制約）。
    """
    ks = [int(k) for k in key]
    if any(k < 0 for k in ks):
        raise ValueError(f"種の要素は非負整数に限る: {name} {ks}")
    h = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")
    return np.random.default_rng(np.random.SeedSequence([h, *ks]))


def load_manifest(corpus: str) -> pd.DataFrame:
    """1 コーパスぶんの発話一覧を返す。

    落とすのは次の 2 種類である。

      * `exclude` が false でない行（マニフェスト作成時に除外と判定されたもの）
      * `label_4` が空の行（中立・喜び・悲しみ・怒りのどれにも入らない感情）

    返す表の行順は，以降すべての出力の行順になる。**並べ替えない。**
    """
    path = META_DIR / f"{corpus.lower()}_utterance_manifest.csv"
    df = pd.read_csv(path, low_memory=False)
    for c in _COLS:
        if c not in df.columns:
            df[c] = pd.NA

    keep = df["exclude"].astype(str).str.strip().str.lower() == "false"
    keep &= df["label_4"].notna()
    # 映像か音声のどちらかがあればよい（RAVDESS はいま音声しか無い）
    has_media = (df["path_video"].fillna("").astype(str).str.len() > 0) | (
        df["path_audio"].fillna("").astype(str).str.len() > 0)
    keep &= has_media
    df = df.loc[keep, list(_COLS)].reset_index(drop=True)
    df["label_4"] = df["label_4"].astype(int)
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def media_paths(df: pd.DataFrame, prefer_audio: bool) -> list[str]:
    """発話ごとに，読むべきメディアファイルの場所を返す。

    音声を取るときは `path_audio`（IEMOCAP・RAVDESS・CREMA-D の wav）を優先し，
    無ければ `path_video` から ffmpeg で取り出す。映像を取るときは
    `path_video` だけを見る（音声しか無い発話は空文字になる）。
    """
    v = df["path_video"].fillna("").astype(str)
    a = df["path_audio"].fillna("").astype(str)
    if not prefer_audio:
        return v.tolist()
    return [x if x else y for x, y in zip(a, v)]


def speaker_side(sample_id: str, marker_side: str) -> str:
    """IEMOCAP の発話者が画面の左右どちらに映っているかを返す。

    IEMOCAP の動画は 2 人が左右に並ぶ。**左がモーションキャプチャの
    マーカーを付けた人，右がその相手**である（5 セッション × 2 通りの
    計 10 本を目視で確認した。例外なし）。

      marker_side  `Ses01F` の 6 文字目。マーカーを付けた人の性別
      発話者       `sample_id` 末尾の `F000` / `M001` の先頭文字

    両者が同じなら発話者は左，違えば右。

    **前身の実装はここが壊れていた。**パスを `'/'` ではなく `'.'` で分けて
    いたため比較対象が拡張子 `"mp4"` になり，判定が常に偽（＝常に右側）に
    なっていた。発話者が左にいる約半数の発話で，相手の顔を見ていたことになる。
    """
    spk = sample_id.rsplit("_", 1)[-1][:1]
    return "left" if spk == str(marker_side)[:1] else "right"


def part_path(kind: str, corpus: str, index: int) -> Path:
    """途中保存の置き場。長い抽出が落ちても頭からやり直さずに済む。"""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    return PARTS_DIR / f"{kind}_{corpus}_{index:04d}.npz"


def out_path(kind: str, corpus: str) -> Path:
    """まとめたあとの置き場。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{kind}_{corpus}.npz"


def load_cache(kind: str, corpus: str) -> dict[str, np.ndarray]:
    """特徴を読む。まとめた npz があればそれを，無ければ途中保存をつないで返す。

    **書き出さない。**抽出の途中でも，そこまでのぶんを読んで測定を試せるようにする。
    """
    dest = out_path(kind, corpus)
    if dest.is_file():
        z = np.load(dest, allow_pickle=False)
        return {k: z[k] for k in z.files}
    parts = sorted(PARTS_DIR.glob(f"{kind}_{corpus}_*.npz"))
    if not parts:
        raise SystemExit(f"{kind}/{corpus} の特徴が無い。先に features/ の抽出を走らせること")
    zs = [np.load(p, allow_pickle=False) for p in parts]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


def merge_npz(kind: str, corpus: str, verbose: bool = True) -> Path | None:
    """途中保存を 1 つの npz にまとめる。行順は途中保存の番号順（＝マニフェスト順）。

    **途中保存が無ければ飛ばして None を返す（落とさない）。**
    `--corpus all` で回すとき，あるコーパスにその媒体が無いのは正常である
    （RAVDESS には映像が無い）。ここで落とすと**後ろのコーパスが結合されない**。
    実際 2026-09-21 に，RAVDESS で落ちて CREMA-D の映像が結合されなかった。
    """
    parts = sorted(PARTS_DIR.glob(f"{kind}_{corpus}_*.npz"))
    if not parts:
        print(f"  {kind}/{corpus}: 途中保存が無いので飛ばす")
        return None
    zs = [np.load(p, allow_pickle=False) for p in parts]
    out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].keys()}
    dest = out_path(kind, corpus)
    np.savez(dest, **out)
    if verbose:
        ok = out["ok"]
        print(f"{kind}/{corpus}: {len(ok)} 件（成功 {int(ok.sum())} / 失敗 {int((~ok).sum())}）"
              f" → {dest.name} {dest.stat().st_size / 1e6:.1f} MB")
        if (~ok).any():
            print("  失敗の内訳:")
            print(pd.Series(out["fail_reason"][~ok]).value_counts().to_string())
    return dest


# 事前学習済みモデルと，出力ファイル名に使う短い名前。
# **どれも本研究の 6 コーパスで微調整していない**（data/README.md「エンコーダ」）。
AUDIO_MODELS = {
    "microsoft/wavlm-large": "wavlm",            # 第 1
    "facebook/hubert-large-ll60k": "hubert",     # 第 2
}
TEXT_MODELS = {
    "roberta-base": "roberta",                   # 第 1
    "microsoft/deberta-v3-base": "deberta",      # 第 2
}


def fix_mojibake(s: str) -> str:
    """Windows-1252 の文字が壊れて混入しているのを直す。

    Windows-1252 で符号化された文字列を Latin-1 として読むと，C1 制御域
    （0x80〜0x9F）に本来の記号が落ちる。実測（2026-09-20）：

      IEMOCAP  0 件
      MOSEI    0 件
      MELD     3,191 件（全 11,352 発話の 28.1%）

    混入している文字は `\\x92`（本来 ’）が 4,192 回で最多，以下 `\\x85`（…）157，
    `\\x97`（—）134，`\\x91`（‘）101。**直さないと `don\\x92t` が未知の文字を含む
    トークン列になり，MELD だけ言葉の特徴が歪む。**
    """
    if not any(0x80 <= ord(ch) <= 0x9F for ch in s):
        return s
    return "".join(
        bytes([ord(ch)]).decode("cp1252", errors="replace") if 0x80 <= ord(ch) <= 0x9F else ch
        for ch in s
    )


# 装飾的な記号を ASCII に落とす対応表。
_PUNCT = {"‘": "'", "’": "'", "“": '"', "”": '"',
          "–": "-", "—": "-", "…": "..."}


def normalize_punct(s: str) -> str:
    """引用符・ダッシュ・三点リーダを ASCII に揃える。

    3 コーパスで表記の規則が違う。IEMOCAP は ASCII の ' だけで装飾記号が無く，MOSEI は ASCII と
    Unicode の記号が混在し，MELD は壊れた \\x92（修復後は ’）を含む（件数は data/README.md 4 節
    「記号の表記の違い」）

    **表記の違いが「コーパスの違い」として検出されるのを防ぐために使う。**
    ただし元のテキストを変える操作なので，**既定では使わない。**
    使うかどうかは，言葉でのコーパス当ての測定結果を見て決める。
    """
    for a, b in _PUNCT.items():
        s = s.replace(a, b)
    return s


# 注記の記号。IEMOCAP にだけ [LAUGHTER] 140 回・[BREATHING] 52 回など現れる
_ANNOT = re.compile(r"\[[A-Za-z /]+\]")
# 文末に見なす記号（**感嘆符と疑問符は消さない**．感情の手がかりそのものだから）
_END = ".!?…。！？"


def normalize_text(s: str, level: int) -> str:
    """表記を段階的に揃える（`data/README.md` 4 節「言葉の正規化の段階」）。

    **表記の流儀だけでコーパスを判別できてしまう**（意味を使わない 16 次元で
    均衡精度 0.7047・でたらめ 0.333）。どこまで揃えるべきかを測るための段階である。

    | 段階 | やること |
    |---|---|
    | **L0** | 何もしない（生のまま） |
    | **L1** | 文字化けの修復のみ．**無条件で常に入れる** |
    | **L2** | L1 ＋ 引用符・ダッシュ・省略記号を ASCII に統一 |
    | **L3** | L2 ＋ `[LAUGHTER]` のような注記を削除 |
    | **L4** | L3 ＋ 文末に句点を補う（MOSEI の 60.8% が句読点を持たない） |

    **感嘆符と疑問符は L4 でも消さない。**MELD の感嘆符 26.9%（他は 0.5〜0.8%）は
    **感情の手がかりそのもの**で，消すと分類に効く情報まで落ちる。
    """
    if level <= 0:
        return s
    s = fix_mojibake(s)
    if level >= 2:
        s = normalize_punct(s)
    if level >= 3:
        s = _ANNOT.sub(" ", s)
        s = re.sub(r"\s+", " ", s)
    s = s.strip()
    if level >= 4 and s and s[-1] not in _END:
        s += "."
    return s


def batches_by_budget(lengths, budget: int, max_size: int):
    """長さのばらつく標本を，計算量の予算でまとめる。

    注意機構の記憶使用量は「系列長 × 件数」に比例して効くので，
    件数だけで区切ると長い発話が集まったときに足りなくなる。

    **長さの順にまとめる（2026-09-21 変更）。**バッチの中では最も長い発話に合わせて
    詰め物をするので，長さを混ぜると短い発話まで長いものとして計算することになる。
    IEMOCAP の実測では，並べ替えないと**計算量が 2.35 倍**になっていた
    （中央値 3.7 秒の発話 4,000 本に対し，実データ 306 分が 718 分ぶんの計算になる）。
    並べ替えると 1.01 倍まで下がる。

    **返すのは元の添字**なので，呼び出し側が `out[idx] = ...` と書く限り
    出力の行順はマニフェストと一致したままである（旧版の docstring はここを
    誤解して「並べ替えない」理由にしていた）。
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch: list[int] = []
    longest = 0
    for i in order:
        n = lengths[i]
        nxt = max(longest, n)
        if batch and (nxt * (len(batch) + 1) > budget or len(batch) >= max_size):
            yield batch
            batch, longest, nxt = [], 0, n
        batch.append(i)
        longest = nxt
    if batch:
        yield batch
