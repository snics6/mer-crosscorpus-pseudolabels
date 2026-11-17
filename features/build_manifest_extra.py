"""RAVDESS と CREMA-D の発話一覧（マニフェスト）を作る。

この 2 つは**検出力を上げるために足したコーパス**である。コーパスが 4 つ
（うち言葉が使えるのは 3 つ）だと (ソース, 対象) の対が 6 つしかなく，
片側検定の p の下限が 0.0156 になって「差が無い」と「検出できない」を区別できない。
5 つにすれば対は 20 になる。

**どちらも俳優が決まった文を感情だけ変えて読むコーパスである。**
読む文は RAVDESS が 2 種類，CREMA-D が 12 種類しかないので，**言葉は測らない**
（`common.CORPORA_TEXT` に入れていない）。映像と音声だけの点として使う。

## ラベルの決め方

**全感情の行を残す**（2026-09-29 から．元データのラベルを尊重する）。
元の感情は `emotion_raw`，強さは `emotion_intensity` に入れる。
`label_4`（中立・喜び・悲しみ・怒り）は当てはまる行にだけ付け，ほかは空にする。

  RAVDESS  01 neutral → 中立 / 03 happy → 喜び / 04 sad → 悲しみ / 05 angry → 怒り
           **02 calm（平静）は中立に混ぜない。**別の感情として設計されている
           06 fearful・07 disgust・08 surprised は label_4 を空にする
  CREMA-D  NEU / HAP / SAD / ANG に label_4 を付ける。DIS（嫌悪）・FEA（恐れ）は空

## ファイル名の規則

RAVDESS は 7 つの数字をハイフンでつなぐ（公式の規定）。

    03-01-04-01-02-01-12.wav
    │  │  │  │  │  │  └─ 俳優 01〜24（奇数が男性，偶数が女性）
    │  │  │  │  │  └──── 繰り返し 01/02
    │  │  │  │  └─────── 読む文 01/02
    │  │  │  └────────── 強さ 01 通常 / 02 強い
    │  │  └───────────── 感情 01〜08
    │  └──────────────── 発声 01 話し / 02 歌
    └─────────────────── 種類 01 映像+音声 / 02 映像のみ / 03 音声のみ

**種類の桁だけが音声版と映像版で違う**ので，`sample_id` は先頭を除いた 6 つ組にする。
こうすると同じ発話の音声と映像が同じ `sample_id` を持つ。

CREMA-D は下線で区切る。

    1001_DFA_ANG_XX.wav
    │    │   │   └─ 強さ XX 指定なし / LO / MD / HI
    │    │   └───── 感情
    │    └───────── 読む文（12 種類）
    └────────────── 俳優 1001〜1091

使い方:

    ./.venv/bin/python features/build_manifest_extra.py
    ./.venv/bin/python features/build_manifest_extra.py --check   # 作らずに件数だけ見る
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import common

RAV_ROOT = Path("/mnt/d/Databases/RAVDESS")
CRE_ROOT = Path("/mnt/d/Databases/CREMA-D")

# 感情コードと label_4（0=中立 1=喜び 2=悲しみ 3=怒り）の対応
RAV_EMO = {"01": 0, "03": 1, "04": 2, "05": 3}
CRE_EMO = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3}

# 元の感情名（公式の規定）
RAV_NAMES = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
             "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
RAV_INTENSITY = {"01": "normal", "02": "strong"}
CRE_NAMES = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "anger",
             "DIS": "disgust", "FEA": "fear"}

# RAVDESS で読む 2 つの文（公式の規定）
RAV_TEXT = {"01": "Kids are talking by the door.",
            "02": "Dogs are sitting by the door."}

# CREMA-D で読む 12 の文（公式 README の対応表）
CRE_TEXT = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}


def build_ravdess() -> pd.DataFrame:
    """音声は必ず，映像はあれば拾う。"""
    audio = {}
    for f in sorted((RAV_ROOT / "audio_speech").rglob("*.wav")):
        p = f.stem.split("-")
        if len(p) != 7 or p[0] != "03" or p[1] != "01":   # 音声のみ・話し声のみ
            continue
        audio["-".join(p[1:])] = f

    video = {}
    vdir = RAV_ROOT / "video_speech"
    if vdir.is_dir():
        for f in sorted(vdir.rglob("*.mp4")):
            p = f.stem.split("-")
            if len(p) != 7 or p[1] != "01":
                continue
            video["-".join(p[1:])] = f

    rows = []
    for sid, af in audio.items():
        p = sid.split("-")
        l4 = RAV_EMO.get(p[1])
        rows.append({
            "sample_id": sid, "corpus": "RAVDESS", "split": "all",
            "speaker_id": f"RAV_{p[5]}", "label_4": l4,
            "label_4_name": common.LABELS[l4] if l4 is not None else "",
            "emotion_raw": RAV_NAMES[p[1]],
            "emotion_intensity": RAV_INTENSITY[p[2]],
            "dialog_type": "", "marker_side": "",
            "path_video": str(video[sid]) if sid in video else "",
            "path_audio": str(af),
            "transcript": RAV_TEXT.get(p[3], ""),
            "exclude": "false",
        })
    return _finish(rows)


def build_cremad() -> pd.DataFrame:
    """音声（wav）と映像（flv）が揃っている。"""
    video = {f.stem: f for f in sorted((CRE_ROOT / "VideoFlash").glob("*.flv"))}
    rows = []
    for f in sorted((CRE_ROOT / "AudioWAV").glob("*.wav")):
        p = f.stem.split("_")
        if len(p) < 4 or p[2] not in CRE_NAMES:
            raise SystemExit(f"CREMA-D の名前が規則に合わない: {f.name}")
        l4 = CRE_EMO.get(p[2])
        rows.append({
            "sample_id": f.stem, "corpus": "CREMA-D", "split": "all",
            "speaker_id": f"CRE_{p[0]}", "label_4": l4,
            "label_4_name": common.LABELS[l4] if l4 is not None else "",
            "emotion_raw": CRE_NAMES[p[2]],
            "emotion_intensity": p[3],
            "dialog_type": "", "marker_side": "",
            "path_video": str(video[f.stem]) if f.stem in video else "",
            "path_audio": str(f),
            "transcript": CRE_TEXT.get(p[1], ""),
            "exclude": "false",
        })
    return _finish(rows)


def _finish(rows: list[dict]) -> pd.DataFrame:
    """`label_4` を空を許す整数にする。float にすると 0 が 0.0 と書かれる。"""
    df = pd.DataFrame(rows)
    if not df.empty:
        df["label_4"] = df["label_4"].astype("Int64")
    return df


def report(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"{name}: 0 件（データが見つからない）")
        return
    n_v = int((df["path_video"].astype(str).str.len() > 0).sum())
    n_a = int((df["path_audio"].astype(str).str.len() > 0).sum())
    print(f"\n{name}: {len(df)} 発話 / 話者 {df['speaker_id'].nunique()} 人 "
          f"/ 映像あり {n_v} / 音声あり {n_a}")
    per = df.groupby("speaker_id").size()
    print("  元の感情:", dict(df["emotion_raw"].value_counts().sort_index()))
    print("  4 クラス:", dict(df.loc[df["label_4"].notna(), "label_4_name"]
                              .value_counts().sort_index()))
    print(f"  話者あたり {per.min()}〜{per.max()} 発話")
    print(f"  読む文の種類: {df['transcript'].nunique()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="書き出さずに件数だけ表示する")
    args = ap.parse_args()

    for name, fn in (("RAVDESS", build_ravdess), ("CREMA-D", build_cremad)):
        df = fn()
        report(name, df)
        if not args.check and not df.empty:
            out = common.META_DIR / f"{name.lower()}_utterance_manifest.csv"
            df.to_csv(out, index=False)
            print(f"  書いた: {out}")


if __name__ == "__main__":
    main()
