"""音声特徴を作る。動画 → 波形 → 事前学習済みの音声モデル → 時間方向に平均。

    動画 → ffmpeg で 16kHz モノラルの波形にする（中間ファイルを作らない）
         → 先頭 60 秒で切る
         → 平均 0・標準偏差 1 に正規化（モデルがそれを前提にしている）
         → WavLM-large（または HuBERT-large）に通す
         → 時間方向に平均（詰め物の部分は除く）→ 1024 次元

**使うモデルはどれも本研究の 6 コーパスで微調整していない**（data/README.md「エンコーダ」）。
Ryumina のパイプラインが出力する音声 512 次元は**コーパス別に微調整されている**ので
使わない。素のモデルで取り直すのが本研究の中心的な工夫である。

**前身から変えた点**（`data/README.md` 4 節「前身から変えた点」）。

1. **計算を fp32 にした。**前身は fp16 で，実測すると特徴の値が平均 1.9% ずれた
   （値の平均の大きさ 0.0480 に対し平均差 0.00090）。速度の利得は無かった
2. **長さの上限を 20 秒から 60 秒にした。**20 秒だと 464 件が切られ，うち 435 件が
   MOSEI に集中していて，処理がコーパスに偏っていた。60 秒なら 9 件（0.02%）
3. **切り詰めたことを記録する**（`truncated`）。後から「切られたものを除いても
   結論が変わらない」と確かめられるようにするため
4. **時間方向は平均だけ**を取り，発話の長さは別に記録する（映像と揃える）

使い方:

    ./.venv/bin/python features/build_audio.py --corpus MELD --limit 100        # 試走
    ./.venv/bin/python features/build_audio.py --corpus all                     # 第 1
    ./.venv/bin/python features/build_audio.py --corpus all \\
        --model facebook/hubert-large-ll60k                                     # 第 2
    ./.venv/bin/python features/build_audio.py --corpus all --merge
"""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

import common

SR = 16000            # モデルが前提とする標本化周波数
MAX_SEC = 60.0        # 使う長さの上限（data/README.md 4 節「数値の約束」）


def probe_duration(path: str) -> float:
    """動画の長さ（秒）をメタデータから読む。復号しないので速い。"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return float("nan")


def decode(path: str) -> tuple[np.ndarray, float, bool]:
    """波形・元の長さ（秒）・切り詰めたかを返す。

    上限より少しだけ長く復号し，そこまで届いていたら**そのときだけ** ffprobe で
    正確な長さを調べる。全件で ffprobe を呼ぶと 4 万回のプロセス起動になるため。
    """
    if not path:
        return np.array([], dtype=np.float32), float("nan"), False
    cmd = ["ffmpeg", "-v", "error", "-t", str(MAX_SEC + 1.0), "-i", path,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=False).stdout
    w = np.frombuffer(raw, dtype=np.float32)
    sec = len(w) / SR
    if sec >= MAX_SEC + 0.5:
        return w[: int(MAX_SEC * SR)].copy(), probe_duration(path), True
    return w.copy(), sec, False


def normalize(w: np.ndarray) -> np.ndarray:
    """平均 0・標準偏差 1 にする。WavLM・HuBERT はこの前処理を前提にしている。"""
    return (w - w.mean()) / (w.std() + 1e-7)


def encode(model, wavs: list[np.ndarray], device: str, budget: int, max_size: int) -> np.ndarray:
    """波形の列を 1024 次元の列にする。詰め物の部分は平均から除く。"""
    out = np.zeros((len(wavs), model.config.hidden_size), dtype=np.float32)
    lengths = [len(w) for w in wavs]
    with torch.inference_mode():
        for idx in common.batches_by_budget(lengths, budget, max_size):
            n = max(lengths[i] for i in idx)
            x = torch.zeros(len(idx), n)
            for r, i in enumerate(idx):
                x[r, : lengths[i]] = torch.from_numpy(wavs[i])
            x = x.to(device)
            lens = torch.tensor([lengths[i] for i in idx], device=device)
            mask = (torch.arange(n, device=device)[None] < lens[:, None]).long()
            h = model(x, attention_mask=mask).last_hidden_state          # (B, T, D)
            # 波形の長さと特徴の長さは一致しないので，モデルに換算させる
            flen = model._get_feat_extract_output_lengths(lens).to(device)
            fmask = (torch.arange(h.shape[1], device=device)[None] < flen[:, None]).float()
            pooled = (h * fmask[..., None]).sum(1) / fmask.sum(1, keepdim=True)
            out[idx] = pooled.float().cpu().numpy()
    return out


def process_chunk(df: pd.DataFrame, model, args) -> dict:
    n = len(df)
    dim = model.config.hidden_size
    X = np.full((n, dim), np.nan, dtype=np.float32)
    dur = np.full(n, np.nan, dtype=np.float32)
    trunc = np.zeros(n, dtype=bool)
    ok = np.zeros(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)

    # IEMOCAP・RAVDESS・CREMA-D は wav を直接持っている。他は動画から取り出す
    paths = common.media_paths(df, prefer_audio=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        decoded = list(pool.map(decode, paths))

    keep, wavs = [], []
    for i, (w, sec, tr) in enumerate(decoded):
        dur[i], trunc[i] = sec, tr
        if not paths[i]:
            reason[i] = "音声ファイルが無い"
            continue
        if len(w) < SR // 100:          # 10 ミリ秒未満は音が無いとみなす
            reason[i] = "音声を取り出せない"
            continue
        keep.append(i)
        wavs.append(normalize(w))

    if keep:
        feats = encode(model, wavs, args.device, args.budget * SR, args.max_batch)
        X[keep] = feats
        ok[keep] = True

    return {
        "X": X, "ok": ok, "duration_sec": dur, "truncated": trunc,
        "fail_reason": np.asarray(reason, dtype=np.str_),
        **{c: df[c].astype(str).to_numpy(dtype=np.str_) for c in
           ("sample_id", "corpus", "split", "speaker_id", "dialog_type")},
        "label_4": df["label_4"].to_numpy(dtype=np.int8),
    }


def run_corpus(corpus: str, kind: str, model, args) -> None:
    df = common.load_manifest(corpus)
    if args.limit:
        df = df.head(args.limit)
    print(f"\n=== {corpus}: {len(df)} 発話 ===", flush=True)
    t0 = time.perf_counter()
    for ci, start in enumerate(range(0, len(df), args.chunk)):
        part = common.part_path(kind, corpus, ci)
        if part.exists() and not args.overwrite:
            print(f"  [{ci}] 済み: {part.name}", flush=True)
            continue
        sub = df.iloc[start : start + args.chunk].reset_index(drop=True)
        t1 = time.perf_counter()
        payload = process_chunk(sub, model, args)
        np.savez(part, **payload)
        dt = time.perf_counter() - t1
        print(f"  [{ci}] {len(sub)} 件 / 成功 {int(payload['ok'].sum())} / "
              f"切り詰め {int(payload['truncated'].sum())} / {dt:.1f} 秒 "
              f"({dt / max(len(sub), 1):.3f} 秒/件) → {part.name}", flush=True)
    print(f"{corpus} 完了: {time.perf_counter() - t0:.1f} 秒", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="all", help="IEMOCAP / MOSEI / MELD / AFEW / all")
    ap.add_argument("--model", default="microsoft/wavlm-large",
                    choices=list(common.AUDIO_MODELS))
    ap.add_argument("--limit", type=int, default=0, help="先頭から何件だけ処理するか（試走用）")
    ap.add_argument("--chunk", type=int, default=2000, help="途中保存の単位")
    ap.add_argument("--budget", type=int, default=320,
                    help="1 バッチに入れる音声の合計秒数の目安（長さ × 件数）")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12, help="復号を並べる数")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    kind = f"audio_{common.AUDIO_MODELS[args.model]}"
    targets = list(common.CORPORA_ALL) if args.corpus == "all" else [args.corpus]

    if args.merge:
        for c in targets:
            common.merge_npz(kind, c)
        return

    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    print(f"{args.model} を {args.device} に載せた（fp32・凍結・微調整なし）"
          f" 次元 {model.config.hidden_size}", flush=True)
    for c in targets:
        run_corpus(c, kind, model, args)


if __name__ == "__main__":
    main()
