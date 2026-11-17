"""映像の第 2 エンコーダ。保存した顔座標を読み，CLIP の視覚エンコーダに通す。

    faces_<corpus>.csv.gz（build_video.py が書いた顔の矩形）
         → 動画の同じフレームから**まったく同じ矩形**を切り出す
         → CLIP ViT-large の視覚エンコーダ → 1 枚 1024 次元
         → セグメントごとに平均（build_video.py と同じ切り方）
         → 全セグメントの平均 1024 と，セグメント間の標準偏差 1024

## なぜ第 2 のエンコーダが要るのか

査読では必ず**「その結果は，そのエンコーダの癖ではないのか」**と問われる。
別のエンコーダでも同じ結論が出れば，それはデータの性質だと言える。
音声（HuBERT）と言葉（DeBERTa）には第 2 が用意できるが，**映像だけは
前身で用意できなかった**。顔の切り出し結果を保存していなかったため，
取り直すと「検出器が違うせいで結果が変わった」可能性が混ざるからである。

**本リポジトリは顔の矩形を全件保存しているので，この問題が消える。**
ResNet-50 に渡したのと 1 ピクセルも違わない画像を CLIP に渡せる。

## なぜ CLIP なのか

MERBench（Lian et al., IEEE TPAMI 2026）が 12 種類の視覚エンコーダを
同じ条件で比べており，CLIP-large が平均 WAF 58.90 で 1 位だった
（ResNet-MSCeleb 48.16 / EmoNet 51.48 / VideoMAE-large 53.83 /
DINOv2-large 56.34 / CLIP-base 56.91 / MANet-RAFDB 57.24）。

順位だけが理由ではない。**第 1 のエンコーダと学習のしかたが独立している**ことが重要である。

| | 第 1 | 第 2（これ） |
|---|---|---|
| モデル | ResNet-50 | CLIP ViT-large |
| 学習データ | AffectNet（表情ラベル付きの顔写真） | 画像とその説明文の対 |
| 学習の目標 | 表情を当てる（教師あり） | 画像と文を対応づける（弱教師あり） |

**CLIP は表情ラベルを一度も見ていない**ので，表情認識モデル特有の癖を共有しない。
EmoNet や MANet を選ぶと第 1 と同じ系統（表情ラベルで学習）になり，独立性が落ちる。

## 取り出す表現について

視覚エンコーダの `pooler_output`（1024 次元）を使う。画像とテキストを対応づける
ための射影を通す前の表現で，視覚的な情報をより多く保っている。
射影後（`get_image_features`・768 次元）は言語と揃えた空間であり，本研究が見たい
「映像そのものの分布」からは一段遠い。

使い方:

    ./.venv/bin/python features/build_video_clip.py --corpus MELD --limit 100
    ./.venv/bin/python features/build_video_clip.py --corpus all
    ./.venv/bin/python features/build_video_clip.py --corpus all --merge
"""

from __future__ import annotations

import argparse
import traceback
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

import common
from build_video import summarize

MODEL = "openai/clip-vit-large-patch14"


def load_faces(corpus: str) -> dict[str, pd.DataFrame]:
    """build_video.py が保存した顔の矩形を，発話ごとに引けるようにして返す。"""
    merged = common.CACHE_DIR / f"faces_{corpus}.csv.gz"
    if merged.is_file():
        df = pd.read_csv(merged)
    else:
        parts = sorted(common.PARTS_DIR.glob(f"video_{corpus}_*.faces.csv.gz"))
        if not parts:
            raise SystemExit(
                f"{corpus} の顔の記録が無い。先に build_video.py を走らせること")
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df[df["chosen"] == 1]
    return {sid: g.sort_values("frame_idx") for sid, g in df.groupby("sample_id")}


def crop_faces(path: str, boxes: pd.DataFrame) -> list[np.ndarray]:
    """記録された矩形のとおりに切り出す。座標は元フレームの基準である。"""
    if not path:
        return []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    want = {int(r.frame_idx): r for r in boxes.itertuples(index=False)}
    crops, idx = [], 0
    while want:
        ok, fr = cap.read()
        if not ok:
            break
        r = want.pop(idx, None)
        if r is not None:
            sub = fr[int(r.y1):int(r.y2), int(r.x1):int(r.x2)]
            if sub.size:
                crops.append(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return crops


def encode(model, proc, crops: list[np.ndarray], device: str, batch: int) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for i in range(0, len(crops), batch):
            px = proc(images=crops[i : i + batch], return_tensors="pt")["pixel_values"]
            h = model(pixel_values=px.to(device)).pooler_output
            out.append(h.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def process_chunk(df: pd.DataFrame, faces: dict, model, proc, args) -> dict:
    """発話をまたいで画像をためてから符号化する。

    1 発話の顔は平均 10 枚ほどしかないので，発話ごとに符号化すると
    バッチが埋まらず GPU が遊ぶ。**一定枚数ためてから通す。**
    切り出し（動画の復号）は CPU 側なので，読み込みだけ並べる。
    """
    n = len(df)
    dim = model.config.hidden_size
    X_mean = np.full((n, dim), np.nan, dtype=np.float32)
    X_std = np.full((n, dim), np.nan, dtype=np.float32)
    nseg = np.zeros(n, dtype=np.int32)
    nface = np.zeros(n, dtype=np.int32)
    ok = np.zeros(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)

    paths = common.media_paths(df, prefer_audio=False)
    sids = df["sample_id"].tolist()

    # 切り出しを並べて行う（動画の復号は GPU を使わない）
    def _crop(i):
        b = faces.get(sids[i])
        if b is None or b.empty:
            return i, None
        return i, crop_faces(paths[i], b)

    per_utt: dict[int, list[np.ndarray]] = defaultdict(list)
    buf_imgs: list[np.ndarray] = []
    buf_owner: list[int] = []

    def flush():
        if not buf_imgs:
            return
        F = encode(model, proc, buf_imgs, args.device, args.batch)
        for owner, f in zip(buf_owner, F):
            per_utt[owner].append(f)
        buf_imgs.clear()
        buf_owner.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, crops in pool.map(_crop, range(n)):
            if crops is None:
                reason[i] = "第 1 の抽出で顔が取れていない"
                continue
            if not crops:
                reason[i] = "記録された矩形を切り出せない"
                continue
            nface[i] = len(crops)
            buf_imgs.extend(crops)
            buf_owner.extend([i] * len(crops))
            if len(buf_imgs) >= args.crop_budget:
                flush()
    flush()

    for i, fs in per_utt.items():
        X_mean[i], X_std[i], nseg[i] = summarize(np.stack(fs))
        ok[i] = True

    return {
        "X_mean": X_mean, "X_std": X_std, "ok": ok,
        "n_segments": nseg, "n_faces_frames": nface,
        "fail_reason": np.asarray(reason, dtype=np.str_),
        **{c: df[c].astype(str).to_numpy(dtype=np.str_) for c in
           ("sample_id", "corpus", "split", "speaker_id", "dialog_type")},
        "label_4": df["label_4"].to_numpy(dtype=np.int8),
    }


def run_corpus(corpus: str, kind: str, model, proc, args) -> None:
    df = common.load_manifest(corpus)
    if args.limit:
        df = df.head(args.limit)
    faces = load_faces(corpus)
    print(f"\n=== {corpus}: {len(df)} 発話 / 顔の記録がある発話 {len(faces)} ===", flush=True)
    t0 = time.perf_counter()
    for ci, start in enumerate(range(0, len(df), args.chunk)):
        part = common.part_path(kind, corpus, ci)
        if part.exists() and not args.overwrite:
            print(f"  [{ci}] 済み: {part.name}", flush=True)
            continue
        sub = df.iloc[start : start + args.chunk].reset_index(drop=True)
        t1 = time.perf_counter()
        payload = process_chunk(sub, faces, model, proc, args)
        np.savez(part, **payload)
        dt = time.perf_counter() - t1
        print(f"  [{ci}] {len(sub)} 件 / 成功 {int(payload['ok'].sum())} / {dt:.1f} 秒 "
              f"({dt / max(len(sub), 1):.2f} 秒/件) → {part.name}", flush=True)
    print(f"{corpus} 完了: {time.perf_counter() - t0:.1f} 秒", flush=True)


def run_all(targets: list[str], fn, what: str) -> None:
    """1 コーパスで落ちても次へ進む。

    **止めてはいけない理由．**RAVDESS には映像が無いので `faces_RAVDESS.csv.gz`
    が存在せず，`load_faces` が例外を投げる。ここで全体が止まると，順番が
    後ろのコーパス（CREMA-D）がそもそも処理されない。実際 2026-09-21 に
    これで CREMA-D の CLIP 特徴が欠けたまま測定を回してしまった。
    **落ちたことは最後にまとめて報せる**（黙って飛ばすと欠損に気づけない）。
    """
    failed = []
    for c in targets:
        try:
            fn(c)
        except Exception as e:
            failed.append(c)
            print(f"× {c} の{what}で落ちた（次へ進む）: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    if failed:
        print(f"\n**{what}が落ちたコーパス: {failed}**", flush=True)
    else:
        print(f"\n{what}はすべてのコーパスで通った: {targets}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64, help="一度に符号化する画像の枚数")
    ap.add_argument("--crop-budget", type=int, default=2048,
                    help="ためてから符号化する画像の枚数．大きいほど GPU が埋まるが記憶を使う")
    ap.add_argument("--workers", type=int, default=8, help="動画の復号を並べる数")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    kind = "video_clip"
    targets = list(common.CORPORA_ALL) if args.corpus == "all" else [args.corpus]

    if args.merge:
        run_all(targets, lambda c: common.merge_npz(kind, c), "まとめ")
        return

    from transformers import CLIPImageProcessor, CLIPVisionModel

    proc = CLIPImageProcessor.from_pretrained(MODEL)
    model = CLIPVisionModel.from_pretrained(MODEL).to(args.device).eval()
    print(f"{MODEL} の視覚エンコーダを {args.device} に載せた"
          f"（fp32・凍結・微調整なし）次元 {model.config.hidden_size}", flush=True)
    run_all(targets, lambda c: run_corpus(c, kind, model, proc, args), "抽出")


if __name__ == "__main__":
    main()
