"""IEMOCAP の発話一覧（マニフェスト）を生データだけから作る。

**前身の CSV は参照しない。**`/mnt/d/Databases/IEMOCAP_full_release/` の
注釈ファイルだけを読む。前身のマニフェストには 2 つの誤りがあった
（`data/README.md` 3 節「前身のマニフェストにあった誤り」）。

1. **台本セッションのラベルが芝居の番号で付けられていた。**script01 の発話は
   中身に関係なく全部 ang，script02 は全部 exc，script03 は全部 hap になっていた。
   **生の注釈は存在するのに使われていなかった**（台本側の neu 609 件・sad 476 件が
   丸ごと捨てられていた）
2. **fru（frustration・苛立ち）を Angry に潰していた。**前身の Angry 2,952 件のうち
   1,849 件（63%）は本当は frustration だった

## ラベルの決め方（2026-09-20 決定）

注釈ファイルの 1 行目の感情タグ（多数決で決まった代表ラベル）を使う。

    neu → Neutral      hap → Happy      sad → Sad      ang → Angry
                       exc → Happy

**exc（excited）は Happy に入れる。**IEMOCAP の 4 クラス設定として広く使われている
まとめ方で，hap と exc は覚醒度が違うだけで価数はどちらも正である。

**fru は採らない。**「苛立ち」と「怒り」は別の感情であり，混ぜると本研究が測る
分布のずれが濁る。ang のみを Angry とする。

**xxx（注釈者が一致しなかったもの）・sur・fea・dis・oth も採らない。**

この規則で 4 クラスに入るのは **5,531 発話**で，これは IEMOCAP の 4 クラス設定として
文献で広く使われている数と一致する。

## ファイルの構造

    Session{1..5}/dialog/EmoEvaluation/<dialog>.txt   感情の注釈
    Session{1..5}/dialog/transcriptions/<dialog>.txt  書き起こし
    Session{1..5}/sentences/wav/<dialog>/<turn>.wav   発話ごとの音声
    /mnt/d/Databases/IEMOCAP_sentences/<turn>.mp4     発話ごとの映像

注釈の行はこの形である。

    [6.2100 - 9.3200]	Ses01F_script01_1_F000	fru	[2.0000, 2.3333, 2.3333]
    └ 開始 - 終了 ┘   └ 発話の名前 ┘            └感情┘ └ 価数・覚醒度・支配度 ┘

## 発話の名前から読めること

    Ses01F_script01_1_F000
    │  │ │ └ 対話の種類と番号      └ 話している人（F か M）と通し番号
    │  │ └ **マーカーを付けている人**（映像の左側に座っている）
    │  └ セッション番号 01〜05
    └ 接頭辞

**マーカーの人と話している人は別々である。**`Ses01F_..._M000` はマーカーが女性で
話者が男性の発話であり，このとき話者は画面の右側にいる（`common.speaker_side`）。

使い方:

    ./.venv/bin/python features/build_manifest_iemocap.py --check   # 作らずに集計だけ
    ./.venv/bin/python features/build_manifest_iemocap.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import common

ROOT = Path("/mnt/d/Databases/IEMOCAP_full_release")
SENT_VIDEO = Path("/mnt/d/Databases/IEMOCAP_sentences")

# 感情タグと label_4 の対応。ここに無いタグは 4 クラスに入れない
EMO4 = {"neu": 0, "hap": 1, "exc": 1, "sad": 2, "ang": 3}

# 注釈の行。[開始 - 終了] 発話名 感情 [価数, 覚醒度, 支配度]
LINE = re.compile(
    r"^\[(?P<start>[\d.]+) - (?P<end>[\d.]+)\]\s+(?P<turn>\S+)\s+(?P<emo>\w+)\s+"
    r"\[(?P<v>[\d.]+), (?P<a>[\d.]+), (?P<d>[\d.]+)\]")

# 書き起こしの行。発話名 [開始-終了]: 本文
TRANS = re.compile(r"^(?P<turn>\S+)\s+\[[\d.]+-[\d.]+\]:\s*(?P<text>.*)$")


def read_transcripts(session: int) -> dict[str, str]:
    """1 セッションぶんの書き起こしを発話名で引けるようにして返す。"""
    out: dict[str, str] = {}
    d = ROOT / f"Session{session}" / "dialog" / "transcriptions"
    for f in sorted(d.glob("*.txt")):
        for line in f.read_text(errors="replace").splitlines():
            m = TRANS.match(line.strip())
            if m:
                out[m.group("turn")] = m.group("text").strip()
    return out


def read_session(session: int) -> list[dict]:
    """1 セッションぶんの注釈を読む。**1 行 1 発話に対応させる。**"""
    trans = read_transcripts(session)
    rows = []
    d = ROOT / f"Session{session}" / "dialog" / "EmoEvaluation"
    for f in sorted(d.glob("*.txt")):
        dialog = f.stem
        for line in f.read_text(errors="replace").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            turn = m.group("turn")
            emo = m.group("emo")
            # Ses01F_script01_1_F000 → 話者は末尾の F/M，マーカーは 6 文字目
            spk = turn.rsplit("_", 1)[-1][:1]
            marker = turn[5]
            wav = ROOT / f"Session{session}" / "sentences" / "wav" / dialog / f"{turn}.wav"
            mp4 = SENT_VIDEO / f"{turn}.mp4"
            rows.append({
                "sample_id": turn,
                "corpus": "IEMOCAP",
                "split": "all",
                "speaker_id": f"Ses{session:02d}_{spk}",
                "label_4": EMO4.get(emo),
                "label_4_name": common.LABELS[EMO4[emo]] if emo in EMO4 else "",
                "emotion_raw": emo,
                "dialog_type": "script" if "script" in dialog else "impro",
                "dialog_id": dialog,
                "session": session,
                "marker_side": marker,
                "utterance_speaker": spk,
                "start_sec": float(m.group("start")),
                "end_sec": float(m.group("end")),
                "valence": float(m.group("v")),
                "arousal": float(m.group("a")),
                "dominance": float(m.group("d")),
                "path_video": str(mp4) if mp4.is_file() else "",
                "path_audio": str(wav) if wav.is_file() else "",
                "transcript": trans.get(turn, ""),
                "exclude": "false",
            })
    return rows


def report(df: pd.DataFrame) -> None:
    print(f"\n=== IEMOCAP: {len(df)} 発話 / 話者 {df['speaker_id'].nunique()} 人 ===")

    print("\n生の感情タグ × 対話の種類:")
    print(pd.crosstab(df["emotion_raw"], df["dialog_type"],
                      margins=True, margins_name="計").to_string())

    use = df[df["label_4"].notna()]
    print(f"\n4 クラスに入る発話: {len(use)}")
    print(pd.crosstab(use["label_4_name"], use["dialog_type"],
                      margins=True, margins_name="計").to_string())

    print("\n話者 × ラベル:")
    print(pd.crosstab(use["speaker_id"], use["label_4_name"]).to_string())

    n_v = int((df["path_video"].str.len() > 0).sum())
    n_a = int((df["path_audio"].str.len() > 0).sum())
    n_t = int((df["transcript"].str.len() > 0).sum())
    print(f"\n映像あり {n_v} / 音声あり {n_a} / 書き起こしあり {n_t}（全 {len(df)}）")
    if n_v < len(df) or n_a < len(df) or n_t < len(df):
        miss = df[(df["path_video"] == "") | (df["path_audio"] == "")
                  | (df["transcript"] == "")]
        print(f"**どれかが欠けている発話: {len(miss)}**")
        print(miss[["sample_id", "emotion_raw"]].head(10).to_string(index=False))

    # マーカーと話者の関係（映像の左右判定の前提）
    print("\nマーカー × 話者（映像の左右判定に使う）:")
    print(pd.crosstab(df["marker_side"], df["utterance_speaker"]).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="書き出さずに集計だけ表示する")
    args = ap.parse_args()

    if not ROOT.is_dir():
        raise SystemExit(f"生データが無い: {ROOT}")

    rows = []
    for s in range(1, 6):
        rows += read_session(s)
    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    report(df)

    if not args.check:
        out = common.META_DIR / "iemocap_utterance_manifest.csv"
        df.to_csv(out, index=False)
        print(f"\n書いた: {out}")


if __name__ == "__main__":
    main()
