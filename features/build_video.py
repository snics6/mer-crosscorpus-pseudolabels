"""映像特徴を作る。生の動画 → 顔検出 → ResNet-50 → セグメント要約。

    動画 → 1 秒に 5 枚になるように間引いて取り出す（25fps なら 5 枚に 1 枚，30fps なら 6 枚に 1 枚）
         → YOLOv8n-face で顔を検出し，**最も大きい顔を 1 つ**選ぶ
         → 224x224 に整えて ResNet-50（AffectNet 事前学習）へ → 1 枚 512 次元
         → セグメント（20 枚ぶんを 10 枚ずつずらして切る）ごとに平均
         → 全セグメントの平均 512 と，セグメント間の標準偏差 512

**前身からの変更点 4 つ**（`data/README.md` 4 節「前身から変えた点」・すべて実測に基づく）。

1. **顔検出の結果を全部保存する。**フレーム番号・矩形・確信度を `faces_<corpus>.csv.gz` に書く。
   後で別の表情モデルに差し替えても，まったく同じ切り出しを再現できる。
   前身では顔検出の結果が環境によって揺れ，MELD 10 件・MOSEI 5 件のすべてで
   旧キャッシュを再現できなかった（余弦類似度の中央値 0.999 だが最悪 0.61）。
2. **IEMOCAP は発話者側の半分だけを見る。**前身は判定が壊れていて常に右側を見ており，
   さらに切り出し幅がフレームごとに半分になり続けていた。詳しくは `common.speaker_side`。
3. **複数の顔を平均しない。**前身は検出した顔画像を全部足して個数で割っていたため，
   相手や背景の第三者の顔が混ざっていた（実測で 1 発話から最大 5 個の顔）。
4. **セグメント数を記録し，平均と標準偏差を分けて保存する。**前身は 1024 次元に
   連結していたため，「セグメントが 1 つしか取れない短い発話」で後半 512 次元が
   丸ごと 0 になり，その割合がコーパスで 5 倍違った（AFEW 89.8% 対 MOSEI 17.7%）。

使い方:

    ./.venv/bin/python features/build_video.py --corpus MELD --limit 100   # 試走
    ./.venv/bin/python features/build_video.py --corpus all                # 本番
    ./.venv/bin/python features/build_video.py --corpus MELD --merge       # 途中保存をまとめる
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import common
from resnet50_affectnet import ResNet50, img_processing

# セグメントの切り方。前身（Ryumina 実装）と同じ値を使う。
STEP, WINDOW, NEED_FRAMES = 2, 4, 5
WINDOW_V = WINDOW * NEED_FRAMES   # 20 枚ぶんを 1 セグメントとする
STEP_V = STEP * NEED_FRAMES       # 10 枚ずつずらす


def load_models(device: str, conf_weights: Path, yolo_weights: Path):
    """表情エンコーダと顔検出器を読む。どちらも学習しない（凍結）。"""
    from ultralytics import YOLO

    model = ResNet50(num_classes=7, channels=3)
    model.load_state_dict(torch.load(conf_weights, map_location=device))
    model.to(device).eval()
    det = YOLO(str(yolo_weights))
    return model, det


def sample_frames(path: str, side: str | None, max_frames: int):
    """動画からフレームを間引いて取り出す。IEMOCAP なら発話者側の半分にする。

    間引き幅は前身と同じ `int(fps * 5 / 25)`（25fps なら 5 枚に 1 枚，30fps なら 6 枚に 1 枚）。
    """
    if not path:
        return [], {}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], {}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(fps * 5 / 25), 1) if fps > 0 else 1

    half = w // 2
    frames: list[tuple[int, np.ndarray]] = []
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if side == "left":
                fr = fr[:, :half]
            elif side == "right":
                fr = fr[:, half:]
            frames.append((idx, fr))
        idx += 1
    cap.release()

    # 長すぎる発話で記憶が足りなくなるのを防ぐ。等間隔に間引き，間引いたことを記録する。
    thinned = False
    if len(frames) > max_frames:
        pick = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in pick]
        thinned = True

    meta = {
        "width": w, "height": h, "fps": fps, "n_frames_total": n_total,
        "frame_step": step, "x_offset": half if side == "right" else 0,
        "thinned": thinned,
    }
    return frames, meta


def detect_faces(det, frames, conf: float, device: str, batch: int):
    """フレームごとに顔を検出し，**最も大きい顔**を選ぶ。

    戻り値は (選んだ顔の画像, 記録用の行)。記録用の行には検出した顔を
    すべて残す（選ばなかったものも）。後から選び方を変えられるようにするため。
    """
    chosen_imgs: list[torch.Tensor] = []
    rows: list[dict] = []

    for i in range(0, len(frames), batch):
        chunk = frames[i : i + batch]
        rgbs = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for _, fr in chunk]
        results = det.predict(rgbs, imgsz=640, conf=conf, iou=0.5,
                              augment=False, device=device, verbose=False)
        for (fidx, _), rgb, res in zip(chunk, rgbs, results):
            H, W = rgb.shape[:2]
            boxes = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy.int().cpu().tolist()[0]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((x1, y1, x2, y2, float(b.conf.item())))
            if not boxes:
                continue
            best = max(range(len(boxes)), key=lambda k: (boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]))
            for k, (x1, y1, x2, y2, sc) in enumerate(boxes):
                rows.append({"frame_idx": fidx, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "score": round(sc, 4), "chosen": int(k == best)})
            x1, y1, x2, y2, _ = boxes[best]
            chosen_imgs.append(img_processing(Image.fromarray(rgb[y1:y2, x1:x2])))

    return chosen_imgs, rows


def encode(model, imgs: list[torch.Tensor], device: str, batch: int) -> np.ndarray:
    """顔画像を 512 次元にする。"""
    out = []
    with torch.inference_mode():
        for i in range(0, len(imgs), batch):
            x = torch.cat(imgs[i : i + batch]).to(device)
            out.append(model.extract_features(x).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def summarize(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """セグメントごとに平均し，その平均と標準偏差を返す。切り方は前身と同じ。"""
    segments = []
    n = feats.shape[0]
    for start in range(0, n + 1, STEP_V):
        end = min(start + WINDOW_V, n)
        if end - start < STEP_V and start != 0:
            break
        if end <= start:
            continue
        segments.append(feats[start:end].mean(axis=0))
    seg = np.asarray(segments, dtype=np.float32)
    return seg.mean(axis=0), seg.std(axis=0), len(seg)


def process_chunk(df: pd.DataFrame, corpus: str, model, det, args) -> tuple[dict, pd.DataFrame]:
    """発話をまとめて処理する。失敗した発話は NaN を入れ，`ok=False` にする。"""
    n = len(df)
    X_mean = np.full((n, 512), np.nan, dtype=np.float32)
    X_std = np.full((n, 512), np.nan, dtype=np.float32)
    cols = {k: np.zeros(n, dtype=np.int32) for k in
            ("n_segments", "n_faces_frames", "n_frames_sampled", "n_frames_total", "frame_step", "width", "height")}
    fps_arr = np.zeros(n, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    side_arr = np.array([""] * n, dtype=object)
    reason = np.array([""] * n, dtype=object)
    face_frames = []

    for i, row in enumerate(df.itertuples(index=False)):
        side = common.speaker_side(row.sample_id, row.marker_side) if corpus == "IEMOCAP" else None
        side_arr[i] = side or ""
        path = "" if pd.isna(row.path_video) else str(row.path_video)
        frames, meta = sample_frames(path, side, args.max_frames)
        if not frames:
            reason[i] = "映像ファイルが無い" if not path else "動画を開けない／フレームが無い"
            continue
        for k in ("n_frames_total", "frame_step", "width", "height"):
            cols[k][i] = meta[k]
        fps_arr[i] = meta["fps"]
        cols["n_frames_sampled"][i] = len(frames)

        imgs, rows = detect_faces(det, frames, args.conf, args.device, args.det_batch)
        if rows:
            fr = pd.DataFrame(rows)
            fr.insert(0, "sample_id", row.sample_id)
            fr["x1"] += meta["x_offset"]
            fr["x2"] += meta["x_offset"]
            face_frames.append(fr)
        if not imgs:
            reason[i] = "顔が 1 つも検出されない"
            continue
        cols["n_faces_frames"][i] = len(imgs)

        feats = encode(model, imgs, args.device, args.enc_batch)
        m, s, ns = summarize(feats)
        X_mean[i], X_std[i], cols["n_segments"][i], ok[i] = m, s, ns, True

        if args.verbose and (i + 1) % 200 == 0:
            print(f"    {i+1}/{n}", flush=True)

    payload = {
        "X_mean": X_mean, "X_std": X_std, "ok": ok, "fps": fps_arr,
        "side": np.asarray(side_arr, dtype=np.str_), "fail_reason": np.asarray(reason, dtype=np.str_),
        **{k: v for k, v in cols.items()},
        **{c: df[c].astype(str).to_numpy(dtype=np.str_) for c in
           ("sample_id", "corpus", "split", "speaker_id", "dialog_type", "marker_side")},
        "label_4": df["label_4"].to_numpy(dtype=np.int8),
    }
    faces = pd.concat(face_frames, ignore_index=True) if face_frames else pd.DataFrame(
        columns=["sample_id", "frame_idx", "x1", "y1", "x2", "y2", "score", "chosen"])
    return payload, faces


def run_corpus(corpus: str, model, det, args) -> None:
    df = common.load_manifest(corpus)
    if args.limit:
        df = df.head(args.limit)
    print(f"\n=== {corpus}: {len(df)} 発話 ===", flush=True)

    t0 = time.perf_counter()
    for ci, start in enumerate(range(0, len(df), args.chunk)):
        part = common.part_path("video", corpus, ci)
        face_part = part.with_suffix("").with_suffix(".faces.csv.gz")
        if part.exists() and not args.overwrite:
            print(f"  [{ci}] 済み: {part.name}", flush=True)
            continue
        sub = df.iloc[start : start + args.chunk].reset_index(drop=True)
        t1 = time.perf_counter()
        payload, faces = process_chunk(sub, corpus, model, det, args)
        np.savez(part, **payload)
        faces.to_csv(face_part, index=False, compression="gzip")
        dt = time.perf_counter() - t1
        n_ok = int(payload["ok"].sum())
        print(f"  [{ci}] {len(sub)} 件 / 成功 {n_ok} / {dt:.1f} 秒 "
              f"({dt/max(len(sub),1):.2f} 秒/件) → {part.name}", flush=True)

    print(f"{corpus} 完了: {time.perf_counter()-t0:.1f} 秒", flush=True)


def merge(corpus: str) -> None:
    """途中保存をまとめて 1 つの npz と 1 つの csv.gz にする。

    **途中保存が無ければ飛ばす（落とさない）。**RAVDESS には映像が無いので
    `--corpus all` では必ず該当する。ここで落とすと後ろのコーパスが結合されない。
    """
    parts = sorted(common.PARTS_DIR.glob(f"video_{corpus}_*.npz"))
    if not parts:
        print(f"  {corpus}: 途中保存が無いので飛ばす")
        return
    zs = [np.load(p, allow_pickle=False) for p in parts]
    keys = list(zs[0].keys())
    out = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    dest = common.out_path("video", corpus)
    np.savez(dest, **out)

    fparts = sorted(common.PARTS_DIR.glob(f"video_{corpus}_*.faces.csv.gz"))
    faces = pd.concat([pd.read_csv(p) for p in fparts], ignore_index=True)
    fdest = common.CACHE_DIR / f"faces_{corpus}.csv.gz"
    faces.to_csv(fdest, index=False, compression="gzip")

    ok = out["ok"]
    print(f"{corpus}: {len(ok)} 件（成功 {int(ok.sum())} / 失敗 {int((~ok).sum())}）"
          f" → {dest.name} {dest.stat().st_size/1e6:.1f} MB")
    print(f"  顔の記録 {len(faces)} 行 → {fdest.name} {fdest.stat().st_size/1e6:.1f} MB")
    if (~ok).any():
        print("  失敗の内訳:")
        print(pd.Series(out["fail_reason"][~ok]).value_counts().to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="all", help="IEMOCAP / MOSEI / MELD / AFEW / all")
    ap.add_argument("--limit", type=int, default=0, help="先頭から何件だけ処理するか（試走用）")
    ap.add_argument("--conf", type=float, default=0.25, help="顔検出の確信度のしきい値")
    ap.add_argument("--chunk", type=int, default=2000, help="途中保存の単位")
    ap.add_argument("--max-frames", type=int, default=400, help="1 発話で使うフレーム数の上限")
    ap.add_argument("--det-batch", type=int, default=16)
    ap.add_argument("--enc-batch", type=int, default=70)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge", action="store_true", help="抽出はせず，途中保存をまとめるだけ")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    targets = list(common.CORPORA_ALL) if args.corpus == "all" else [args.corpus]

    if args.merge:
        for c in targets:
            merge(c)
        return

    model, det = load_models(args.device,
                             common.WEIGHTS_DIR / "FER_static_ResNet50_AffectNet.pt",
                             common.WEIGHTS_DIR / "yolov8n-face.pt")
    print(f"表情エンコーダと顔検出器を {args.device} に載せた（どちらも凍結・微調整なし）", flush=True)
    for c in targets:
        run_corpus(c, model, det, args)


if __name__ == "__main__":
    main()
