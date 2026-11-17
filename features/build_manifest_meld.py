"""MELD の発話一覧（マニフェスト）を生データだけから作る。

**前身の CSV は参照しない。**`/mnt/d/Databases/MELD/Labels/*.csv` だけを読む。

## 前身のマニフェストにあった誤り

**`sample_id` が 2,576 件重複していた。**MELD の動画名 `dia0_utt0` は
train・dev・test のそれぞれに存在するのに，`sample_id` に分割名が入っていなかった。
`build_video_clip.py` は `sample_id` で顔の矩形をまとめるので，**別の発話の顔が
混ざる。**本コードは `train_dia0_utt0` のように分割名を前に付けて一意にする。

## ラベルの決め方

`Emotion` 列をそのまま使う（注釈者の多数決で決まっている）。

    neutral → Neutral    joy → Happy    sadness → Sad    anger → Angry

**surprise・disgust・fear は採らない。**4 クラスに入らない。

## 話者について

`Speaker` 列は登場人物の名前である。**Friends では登場人物と俳優が 1 対 1 に
対応する**ので話者の代わりに使えるが，2 つ注意がある（`measure/split.py`）。

* `Ross_and_Rachel` のような**合同発話**がある。片方を取り置いても他方から漏れる
* 上位 6 人（Ross・Joey・Rachel・Phoebe・Monica・Chandler）で **82.6%** を占める

## 書き起こしの文字化け

`Utterance` 列の 28.1% に Windows-1252 由来の文字化けがある（`data/README.md` 4 節「文字化けの修復（L1）」）。
**ここでは直さず，生のまま保存する。**修復は `features/build_text.py` が行う。
マニフェストは生データの写しであるべきで，加工はそのつど記録できる場所でやる。

使い方:

    ./.venv/bin/python features/build_manifest_meld.py --check
    ./.venv/bin/python features/build_manifest_meld.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import common

ROOT = Path("/mnt/d/Databases/MELD")

# Labels のファイル名と，本リポジトリで使う分割名・動画の置き場の対応。
# **動画の置き場は dev だが，分割名は val である**（前身の表記に合わせる）
SPLITS = {"train": ("train", "train"), "dev": ("val", "dev"), "test": ("test", "test")}

EMO4 = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3}


def build() -> pd.DataFrame:
    rows = []
    for stem, (split, vdir) in SPLITS.items():
        src = ROOT / "Labels" / f"{stem}_sent_emo.csv"
        df = pd.read_csv(src)
        for r in df.itertuples(index=False):
            base = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
            mp4 = ROOT / "Raw" / vdir / f"{base}.mp4"
            emo = str(r.Emotion).strip().lower()
            rows.append({
                # **分割名を前に付けて一意にする**（前身は付けておらず 2,576 件重複した）
                "sample_id": f"{split}_{base}",
                "corpus": "MELD",
                "split": split,
                "speaker_id": str(r.Speaker).strip().replace(" ", "_"),
                "label_4": EMO4.get(emo),
                "label_4_name": common.LABELS[EMO4[emo]] if emo in EMO4 else "",
                "emotion_raw": emo,
                "sentiment_raw": str(r.Sentiment).strip().lower(),
                "dialog_type": "",
                "dialogue_id": f"S{r.Season}_E{r.Episode}_dia{r.Dialogue_ID}",
                "utterance_id": int(r.Utterance_ID),
                "season": int(r.Season),
                "episode": int(r.Episode),
                "start_time": str(r.StartTime),
                "end_time": str(r.EndTime),
                "marker_side": "",
                "path_video": str(mp4) if mp4.is_file() else "",
                "path_audio": "",
                "transcript": str(r.Utterance),
                "exclude": "false",
            })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print(f"\n=== MELD: {len(df)} 発話 / 話者 {df['speaker_id'].nunique()} 人 ===")
    print(f"sample_id の重複: {int(df['sample_id'].duplicated().sum())} 件")

    print("\n生の感情 × 分割:")
    print(pd.crosstab(df["emotion_raw"], df["split"],
                      margins=True, margins_name="計").to_string())

    use = df[df["label_4"].notna()]
    print(f"\n4 クラスに入る発話: {len(use)}")
    print(pd.crosstab(use["label_4_name"], use["split"],
                      margins=True, margins_name="計").to_string())

    n_v = int((df["path_video"].str.len() > 0).sum())
    print(f"\n映像あり {n_v} / 全 {len(df)}")
    if n_v < len(df):
        miss = df[df["path_video"] == ""]
        print(f"**動画が見つからない発話: {len(miss)}**")
        print(miss[["sample_id", "emotion_raw"]].head(10).to_string(index=False))

    print("\n上位 10 話者:")
    print(df["speaker_id"].value_counts().head(10).to_string())
    joint = df[df["speaker_id"].str.contains("_and_", regex=False)]
    print(f"\n合同発話（`X_and_Y`）: {len(joint)} 件 → measure/split.py で除外する")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not ROOT.is_dir():
        raise SystemExit(f"生データが無い: {ROOT}")
    df = build()
    report(df)
    if not args.check:
        out = common.META_DIR / "meld_utterance_manifest.csv"
        df.to_csv(out, index=False)
        print(f"\n書いた: {out}")


if __name__ == "__main__":
    main()
