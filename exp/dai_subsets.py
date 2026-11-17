"""DAI の感度の解析に使う発話の部分集合を，評価の回の前に一覧にして固定する（実験 1 の設計 9.4 節．成績は使わない）。

  AFEW の書き起こしの疑い（9.4 節の 5，感情認識の査読者の a10）：圧縮率が 2.4 を超える（Whisper 自身が失敗とみなす基準），
    または英字だけにした小文字の書き起こしが「thank」と「you」の語だけからなる発話．
  MELD の顔が競る発話（9.4 節の 6，a11，m61）：確信度 0.5 以上の顔の検出だけを使う．フレームで 2 番目に大きい顔の面積が
    最も大きい顔の面積の 0.5 倍以上なら「競るフレーム」．競るフレームが，確信度 0.5 以上の顔が 1 つ以上あるフレームの半分を
    超える発話を「競る」とする．確信度 0.5 以上の顔が 1 つも無い発話は「判定できない」として，競らない部分集合に入れない．
  IEMOCAP 即興のマーカーの側（9.4 節の 2，A2）：発話者の性別（sample_id の話者の F/M）が `marker_side` と同じなら「付けた側」。

出力：`frozen/subsets.csv`（sample_id，corpus，subset，value）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai_data as DD  # noqa: E402

ROOT = DD.ROOT


def afew_asr() -> pd.DataFrame:
    sp = pd.read_csv(ROOT / "splits" / "AFEW.csv", keep_default_na=False)
    t = pd.read_csv(ROOT / "data" / "metadata" / "afew_transcript_whisper.csv", keep_default_na=False)
    t = t[t.sample_id.isin(sp.sample_id)].copy()
    rep = t.compression_ratio.astype(float) > 2.4
    words = t.transcript_asr.str.lower().str.replace(r"[^a-z ]", " ", regex=True).str.split()
    ty = words.apply(lambda w: len(w) > 0 and set(w) <= {"thank", "you"})
    out = t[rep | ty][["sample_id"]].copy()
    out["value"] = np.where(rep[rep | ty], "repeat", "thankyou")
    return out.assign(corpus="AFEW", subset="asr_suspect")


def meld_faces() -> pd.DataFrame:
    sp = pd.read_csv(ROOT / "splits" / "MELD.csv", keep_default_na=False)
    f = pd.read_csv(ROOT / "data" / "cache" / "faces_MELD.csv.gz")
    f = f[(f.score >= 0.5) & f.sample_id.isin(sp.sample_id)].copy()
    f["area"] = (f.x2 - f.x1).clip(lower=0) * (f.y2 - f.y1).clip(lower=0)
    g = f.sort_values("area", ascending=False).groupby(["sample_id", "frame_idx"])["area"]
    fr = pd.DataFrame({"a1": g.apply(lambda a: a.iloc[0]), "a2": g.apply(lambda a: a.iloc[1] if len(a) > 1 else 0.0)})
    fr["compete"] = fr.a2 >= 0.5 * fr.a1
    share = fr.groupby(level=0)["compete"].mean()
    out = pd.DataFrame({"sample_id": sp.sample_id})
    out["value"] = out.sample_id.map(lambda s: "undetermined" if s not in share.index else ("compete" if share[s] > 0.5 else "single"))
    return out.assign(corpus="MELD", subset="face_compete")


def iemocap_marker() -> pd.DataFrame:
    sp = pd.read_csv(ROOT / "splits" / "IEMOCAP.csv", keep_default_na=False)
    man = pd.read_csv(ROOT / "data" / "metadata" / "iemocap_utterance_manifest.csv", dtype=str, keep_default_na=False)
    man = man[man.sample_id.isin(sp.sample_id)]
    spk = man.sample_id.str.split("_").str[-1].str[0]                  # 例：Ses01F_impro01_F000 → F
    out = man[["sample_id"]].copy()
    out["value"] = np.where(spk.to_numpy() == man.marker_side.str[:1].to_numpy(), "marker", "nomarker")
    return out.assign(corpus="IEMOCAP", subset="marker")


if __name__ == "__main__":
    df = pd.concat([afew_asr(), meld_faces(), iemocap_marker()])[["sample_id", "corpus", "subset", "value"]]
    p = ROOT / "frozen" / "subsets.csv"
    df.to_csv(p, index=False)
    print(df.groupby(["corpus", "subset", "value"]).size())
