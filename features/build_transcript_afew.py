"""AFEW の書き起こしを Whisper large-v3 で作る（2026-11-04）。

AFEW には人が書いた書き起こしが無い。言葉の特徴を作るために，音声認識で書き起こす。
Ryumina ほか（2025，Pattern Recognition Letters 190）が公開した MER のリポジトリにも，
AFEW を Whisper で書き起こす台本（get_text_AFEW.py）がある（論文本文での記述は未確認）。

    動画 → ffmpeg で 16kHz モノラルの波形（`build_audio.decode` と同じ）
         → Whisper large-v3（英語に固定・書き起こし・ビーム幅 5・サンプリングなし）
         → data/metadata/afew_transcript_whisper.csv

**出力の表は git に入れ，これを正本にする。**`build_manifest_afew.py` がこの表を
`sample_id` で突き合わせ，発話一覧の `transcript` 列を埋める（MOSEI のチャンネル
対応表と同じ扱い）。Whisper を流し直すと GPU の丸めで文がまれに変わりうるので，
発話一覧を作り直すたびに書き起こしを作り直すことはしない。

設定の理由：

- **モデルは openai/whisper-large-v3**。Whisper の中で最も大きい版で，感情のデータで
  微調整していない（data/README.md 取り決め 9）。読み込んだ版のコミット番号を `asr_model`
  列に書く
- **言語を英語に固定する。**AFEW はハリウッド映画の切り抜きで，音楽や効果音の混ざる
  短い発話では言語の判定が外れうる
- **ビーム幅 5・サンプリングなし。**同じ入力から同じ文が出る（温度を上げてやり直す処理はしない）
- **波形を正規化しない。**Whisper は生の振幅を前提にする（`build_audio.normalize` は
  WavLM・HuBERT のための処理）
- **計算は fp32**（data/README.md 4 節「数値の約束」）
- **バッチは 2 本。**Whisper のデコーダは，ビームごとに音声の表現（1,500 フレーム × 32 層）の
  キーと値を複製する。fp32 では 1 本あたり約 0.5 GB × ビーム 5 で約 2.5 GB になり，最初の
  試走（バッチ 16）は 24 GB に貼り付いて止まった。WSL2 では上限に触ると落ちずに極端に遅く
  なるので，GPU メモリを 6 割に抑え（環境変数 `MERDA_GPU_FRACTION`），バッチごとに空きを返す
- **発話一覧の全行（4 クラスに入らない行を含む 1,156 行）を書き起こす。**ほかの
  コーパスの発話一覧も，4 クラスに入らない行に書き起こしを持つ

声が無い発話では，空の文や，英字も数字も含まない出力（点の並び・音符の記号）が出る。
表には出力をそのまま書き，`build_manifest_afew.py` が後者を空にする。空の文は `build_text.py` が
「文字起こしが無い」として `ok = False` にする。Whisper は声の無い区間で実際には話されていない文を作ることがあるが，
本研究では書き起こしをそのまま使う（2026-11-04 決定。docs/researchvision.md 決定事項）。
実行の最後に，同じ文が何回出たかの上位を表示する。

**同じ句のくり返し**（試走で，長さ 1.5 秒の発話に同じ文が 20 回並んだ）も直さずに残す。
言葉の特徴は単語方向の平均なので，同じ文のくり返しは 1 回ぶんの文とほぼ同じ特徴になる。
疑わしい行を後から数えられるように，文の圧縮率（zlib で縮めたときの元の長さ ÷ 縮めた長さ）を
`compression_ratio` 列に書く。Whisper 自身は 2.4 を超えたら失敗とみなして解き直す（Radford ほか 2023）。

使い方:

    ./.venv/bin/python features/build_transcript_afew.py --limit 16   # 試走（書き出さない）
    ./.venv/bin/python features/build_transcript_afew.py
"""

from __future__ import annotations

import argparse
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

import common
from build_audio import SR, decode

MODEL = "openai/whisper-large-v3"
OUT = common.META_DIR / "afew_transcript_whisper.csv"


def load_rows() -> pd.DataFrame:
    """発話一覧の全行（4 クラスに入らない行も含む）。"""
    df = pd.read_csv(common.META_DIR / "afew_utterance_manifest.csv", low_memory=False)
    df = df[df["path_video"].fillna("").astype(str).str.len() > 0]
    return df[["sample_id", "path_video"]].reset_index(drop=True)


def transcribe(model, proc, wavs: list[np.ndarray], device: str, batch: int,
               beams: int) -> list[str]:
    out: list[str] = []
    with torch.inference_mode():
        for s in range(0, len(wavs), batch):
            chunk = wavs[s : s + batch]
            feats = proc(chunk, sampling_rate=SR, return_tensors="pt").input_features
            ids = model.generate(feats.to(device), language="en", task="transcribe",
                                 num_beams=beams, do_sample=False, max_new_tokens=200)
            out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
            del feats, ids
            torch.cuda.empty_cache()
            if (s // batch) % 50 == 0 or s + batch >= len(wavs):
                print(f"  {min(s + batch, len(wavs))}/{len(wavs)}", flush=True)
    return out


def compression_ratio(t: str) -> float:
    """元の長さ ÷ zlib で縮めた長さ。同じ句のくり返しで大きくなる（Whisper の失敗の判定と同じ量）。"""
    b = t.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


def report(res: pd.DataFrame) -> None:
    t = res["transcript_asr"].fillna("")
    nw = t.str.split().str.len().fillna(0)
    print(f"\n{len(res)} 本 / 空の文 {int((t == '').sum())} 本")
    print("単語数の分位（0・5・25・50・75・95・100%）:",
          np.quantile(nw, [0, .05, .25, .5, .75, .95, 1]).round(1).tolist())
    print(f"圧縮率が 2.4 を超える（同じ句のくり返しの疑い）: {int((res['compression_ratio'] > 2.4).sum())} 本")
    vc = t[t != ""].value_counts()
    print(f"同じ文が 2 回以上出たもの: {int((vc >= 2).sum())} 種・のべ {int(vc[vc >= 2].sum())} 本")
    print(vc.head(15).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 本だけ流して表示する（書き出さない）")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = load_rows()
    if args.limit:
        rows = rows.head(args.limit)
    print(f"AFEW {len(rows)} 本を書き起こす", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        dec = list(ex.map(decode, rows["path_video"].tolist()))
    wavs = [w for w, _, _ in dec]
    dur = np.array([sec for _, sec, _ in dec], dtype=np.float32)
    print(f"波形の取り出し {time.perf_counter() - t0:.1f} 秒", flush=True)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("MERDA_GPU_FRACTION", "0.6")))

    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32).to(args.device).eval()
    rev = getattr(model.config, "_commit_hash", None) or "unknown"
    print(f"{MODEL}@{rev} を {args.device} に載せた（fp32）", flush=True)

    # 10 ミリ秒未満の波形は音が無いとみなす（build_audio.py と同じ基準）
    has = [i for i, w in enumerate(wavs) if len(w) >= SR // 100]
    texts = [""] * len(wavs)
    t1 = time.perf_counter()
    for i, t in zip(has, transcribe(model, proc, [wavs[i] for i in has],
                                    args.device, args.batch, args.beams)):
        texts[i] = t
    print(f"書き起こし {time.perf_counter() - t1:.1f} 秒（波形が無い {len(wavs) - len(has)} 本）",
          flush=True)

    res = pd.DataFrame({
        "sample_id": rows["sample_id"].astype(str),
        "transcript_asr": texts,
        "duration_sec": dur.round(3),
        "compression_ratio": [round(compression_ratio(t), 3) for t in texts],
        "asr_model": f"{MODEL}@{rev}",
    })
    report(res)
    if args.limit:
        print("\n試走なので書き出さない")
        for sid, t in zip(res["sample_id"], res["transcript_asr"]):
            print(f"  {sid}: {t!r}")
        return
    res.to_csv(OUT, index=False)
    print(f"\n書いた: {OUT}（{len(res)} 行）")


if __name__ == "__main__":
    main()
