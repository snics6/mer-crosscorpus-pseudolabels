"""CMU-MOSEI の発話一覧（マニフェスト）を生データだけから作る。

**前身の CSV は参照しない。**次の 2 つだけを読む。

    label/label.csv                                  発話の一覧・書き起こし・公式の分割
    label/CMU_MOSEI_Labels_Sorted_with_ClipID.csd    感情強度（HDF5）

## ラベルの決め方（前身の規則を実測で特定し，そのまま採用した）

`.csd` は 1 セグメントにつき 7 つの数を持つ。

    [sentiment, happy, sad, anger, surprise, disgust, fear]

sentiment は −3〜+3，感情の 6 つは 0〜3 である。**強度が 0.5 以上なら
その感情が立っているとみなす。**

    立っている感情が 0 個 → Neutral
                    1 個 → その感情
                    2 個以上 → multi_label（4 クラスに入れない）

しきい値を 0.0 / 0.5 / 1.0 / 1.5 / 2.0 で総当たりして前身の内訳と突き合わせた結果，
**0.5 以上**でのみ一致した（差の合計 6 件＝読めないセグメント数と同じ）。
前身のラベルは再現可能であり，IEMOCAP のようには壊れていない。

## **重要：MOSEI の Neutral は他のコーパスと意味が違う**

| コーパス | Neutral の正体 |
|---|---|
| IEMOCAP・MELD | 注釈者が「中立」と判定した |
| AFEW | 「Neutral」フォルダに入っている |
| **MOSEI** | **どの感情も 0.5 未満という消去法。中立という注釈は存在しない** |

**ラベルの定義そのものがコーパス間で違う。**本研究が測るのはコーパス間の
分布のずれなので，この違いは測定結果に直接効く。論文では限界として書く。

## 話者について ── **再現できない部分がある**

MOSEI は YouTube の動画で，生データに話者の注釈が無い。同じ人物が複数の動画に
出るので，動画 ID を話者にすると**同じ人が学習側と評価側の両方に現れる。**

前身は YouTube から取得したチャンネル ID を使っていた。取得の処理は再現できない
（動画が消えている可能性もある）ので，**対応表を `data/metadata/mosei_channel_map.csv`
として切り出して保存した。**由来は `data/README.md` 3 節に書く。

**限界：チャンネルが分かるのは 3,225 本中 1,122 本（35%）だけである。**残りは
動画 ID を話者の代わりにするので，**MOSEI の話者分離は不完全である。**論文に書く。

使い方:

    ./.venv/bin/python features/build_manifest_mosei.py --check
    ./.venv/bin/python features/build_manifest_mosei.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common

ROOT = Path("/mnt/d/Databases/CMU-MOSEI")
CSD = ROOT / "label" / "CMU_MOSEI_Labels_Sorted_with_ClipID.csd"
CHANNEL_MAP = common.META_DIR / "mosei_channel_map.csv"

THRESHOLD = 0.5                      # これ以上でその感情が立っているとみなす
EMO_NAMES = ("happy", "sad", "anger", "surprise", "disgust", "fear")
EMO4 = {"happy": 1, "sad": 2, "anger": 3}      # neutral は別に付ける
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def read_emotions() -> dict[tuple[str, str], np.ndarray]:
    """`.csd` から (動画 ID, clip ID) → 7 次元の感情ベクトルを作る。

    **71 本の動画は `clip_ids` が空で，どのセグメントがどの clip かを決められない。**
    そのような動画は飛ばす（該当は label.csv の 6 行だけ）。
    """
    import h5py

    out: dict[tuple[str, str], np.ndarray] = {}
    with h5py.File(CSD, "r") as f:
        data = f["All Labels"]["data"]
        for vid in data:
            g = data[vid]
            cids, feats = g["clip_ids"][:], g["features"][:]
            if len(cids) != len(feats):
                continue
            for c, v in zip(cids, feats):
                cid = c.decode() if isinstance(c, bytes) else str(c)
                out[(str(vid), cid)] = np.asarray(v, dtype=np.float32)
    return out


def classify(vec: np.ndarray | None) -> tuple[str, int | None]:
    """感情ベクトルから (生のラベル名, label_4) を決める。"""
    if vec is None:
        return "", None
    on = [n for n, v in zip(EMO_NAMES, vec[1:]) if v >= THRESHOLD]
    if not on:
        return "neutral", 0
    if len(on) > 1:
        return "multi_label", None
    return on[0], EMO4.get(on[0])


def build() -> pd.DataFrame:
    lab = pd.read_csv(ROOT / "label" / "label.csv")
    emo = read_emotions()
    chan: dict[str, str] = {}
    if CHANNEL_MAP.is_file():
        cm = pd.read_csv(CHANNEL_MAP)
        chan = dict(zip(cm["video_id"].astype(str), cm["channel_id"].astype(str)))

    rows = []
    for r in lab.itertuples(index=False):
        vid, cid = str(r.video_id), str(r.clip_id)
        vec = emo.get((vid, cid))
        raw, l4 = classify(vec)
        mp4 = ROOT / "Raw" / vid / f"{cid}.mp4"
        ch = chan.get(vid, "")
        rows.append({
            "sample_id": f"{vid}_{cid}",
            "corpus": "MOSEI",
            "split": SPLIT_MAP.get(str(r.mode), str(r.mode)),
            "speaker_id": ch if ch else vid,
            "speaker_source": "youtube_channel" if ch else "video_id_fallback",
            "label_4": l4,
            "label_4_name": common.LABELS[l4] if l4 is not None else "",
            "emotion_raw": raw,
            "dialog_type": "",
            "video_id": vid,
            "clip_id": cid,
            "youtube_channel_id": ch,
            "sentiment": float(vec[0]) if vec is not None else np.nan,
            **{f"int_{n}": (float(vec[i + 1]) if vec is not None else np.nan)
               for i, n in enumerate(EMO_NAMES)},
            "marker_side": "",
            "path_video": str(mp4) if mp4.is_file() else "",
            "path_audio": "",
            "transcript": str(r.text),
            "exclude": "false",
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print(f"\n=== MOSEI: {len(df)} 発話 / 動画 {df['video_id'].nunique()} 本 ===")
    print(f"sample_id の重複: {int(df['sample_id'].duplicated().sum())} 件")

    print("\n生のラベル:")
    print(df["emotion_raw"].value_counts(dropna=False).to_string())

    use = df[df["label_4"].notna()]
    print(f"\n4 クラスに入る発話: {len(use)}")
    print(pd.crosstab(use["label_4_name"], use["split"],
                      margins=True, margins_name="計").to_string())

    n_v = int((df["path_video"].str.len() > 0).sum())
    n_t = int((df["transcript"].str.strip().str.len() > 0).sum())
    print(f"\n映像あり {n_v} / 書き起こしあり {n_t}（全 {len(df)}）")

    print("\n話者の出所:")
    print(df["speaker_source"].value_counts().to_string())
    print(f"話者 {df['speaker_id'].nunique()} 人")
    ch = df[df["speaker_source"] == "youtube_channel"]
    if len(ch):
        per = ch.groupby("speaker_id")["video_id"].nunique()
        print(f"**チャンネルが分かった分では，1 人あたり最大 {int(per.max())} 本の動画**"
              f"（動画 ID を話者にすると，この人が {int(per.max())} 人に分裂して漏れる）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not CSD.is_file():
        raise SystemExit(f"感情ラベルが無い: {CSD}")
    df = build()
    report(df)
    if not args.check:
        out = common.META_DIR / "mosei_utterance_manifest.csv"
        df.to_csv(out, index=False)
        print(f"\n書いた: {out}")


if __name__ == "__main__":
    main()
