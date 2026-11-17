"""言葉を聞き取れなくした音声から WavLM の特徴を作る（実験 23．設計 `docs/設計/実験23_設計.md` 2 節）。

    動画・wav → ffmpeg で 16kHz モノラル（`build_audio.decode` と同じ）
             → 操作：--op lp は 4 次の Butterworth の低域通過を前後 2 回（零位相．fc で −6 dB，約 48 dB/oct），
                     --op rev は時間の向きの反転（逆再生）
             → 平均 0・標準偏差 1 に正規化（`build_audio.normalize` と同じ）
             → WavLM-large の最後の層を時間方向に平均（`build_audio.encode` と同じ）→ 1024 次元

操作の 1 段を足した以外は `build_audio.py` と同じである。出力の名前は `audio_wavlm-lp<fc>_<コーパス>.npz` と
`audio_wavlm-rev_<コーパス>.npz`（既存の `audio_wavlm_*` は書き換えない）。--op lp の fc は results/dai23/fc.json の値だけを
本番の名前で書ける（ほかの fc は --limit つきの試しの名前でだけ書く）。

使い方:

    ./.venv/bin/python features/build_audio_lowpass.py --op lp --fc 500 --limit 64 --check-identity   # 流れの照合（書き出さない）
    ./.venv/bin/python features/build_audio_lowpass.py --op lp            # fc は fc.json から
    ./.venv/bin/python features/build_audio_lowpass.py --op rev
    ./.venv/bin/python features/build_audio_lowpass.py --op lp --merge ; ... --op rev --merge

`--check-identity` は，操作を外した同じ流れで作った特徴が既存の `audio_wavlm_<コーパス>.npz` の X と
一致するか（バッチの組による fp32 の丸めの範囲）を先頭 `--limit` 件で確かめ，書き出さない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

import common
from build_audio import SR, decode, encode, normalize

MODEL = "microsoft/wavlm-large"
ORDER = 4                  # 前後 2 回かけるので振幅は 4 次の 2 乗（fc で −6 dB，約 48 dB/oct．Lolli ほか 2015 と同じ傾き）
FC_JSON = Path(__file__).resolve().parents[1] / "results" / "dai23" / "fc.json"
FC_ALLOWED = (300, 400, 500)
TARGETS = ("IEMOCAP", "MOSEI", "MELD")
TOL_IDENTITY = 2e-3        # 既存の特徴との差の上限（実験 13 の照合の最大 9.16e-04 の約 2 倍）
LIMIT_MAX = 256            # 試しの本数の上限（規則で選んでいない fc の特徴を全発話ぶん作らないため）
FILTER = "butter4_sosfiltfilt_zero_phase"
import subprocess as _sp
CODE_COMMIT = _sp.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True).stdout.strip()
CODE_DIRTY = bool(_sp.run(["git", "status", "--porcelain", "features"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True).stdout.strip())


def lowpass(w: np.ndarray, fc: int | None) -> np.ndarray:
    """零位相の Butterworth の低域通過。fc が None なら何もしない（`--check-identity` 用）。"""
    if fc is None:
        return w
    sos = signal.butter(ORDER, fc, btype="low", fs=SR, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(w) <= padlen:                                  # sosfiltfilt の端の詰め物より短い波形は止める
        raise SystemExit(f"波形が短すぎて濾波できない（{len(w)} 標本）")
    return signal.sosfiltfilt(sos, w.astype(np.float64)).astype(np.float32)


def apply_op(w: np.ndarray, op: str | None, fc: int | None) -> np.ndarray:
    """操作：lp は低域通過，rev は逆再生，None は何もしない。"""
    if op is None:
        return w
    if op == "rev":
        return w[::-1].copy()
    return lowpass(w, fc)


def process_chunk(df: pd.DataFrame, model, args, op: str | None, fc: int | None) -> dict:
    """`build_audio.process_chunk` に操作の 1 段を足したもの。"""
    n = len(df)
    dim = model.config.hidden_size
    X = np.full((n, dim), np.nan, dtype=np.float32)
    dur = np.full(n, np.nan, dtype=np.float32)
    trunc = np.zeros(n, dtype=bool)
    ok = np.zeros(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)
    paths = common.media_paths(df, prefer_audio=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        decoded = list(pool.map(decode, paths))
    keep, wavs = [], []
    for i, (w, sec, tr) in enumerate(decoded):
        dur[i], trunc[i] = sec, tr
        if not paths[i]:
            reason[i] = "音声ファイルが無い"
            continue
        if len(w) < SR // 100:
            reason[i] = "音声を取り出せない"
            continue
        keep.append(i)
        wavs.append(normalize(apply_op(w, op, fc)))
    if keep:
        X[keep] = encode(model, wavs, args.device, args.budget * SR, args.max_batch)
        ok[keep] = True
        torch.cuda.empty_cache()
    return {
        "X": X, "ok": ok, "duration_sec": dur, "truncated": trunc,
        "fail_reason": np.asarray(reason, dtype=np.str_),
        **{c: df[c].astype(str).to_numpy(dtype=np.str_) for c in
           ("sample_id", "corpus", "split", "speaker_id", "dialog_type")},
        "label_4": df["label_4"].to_numpy(dtype=np.int8),
        # 操作の記録は発話ごとの列にする（途中保存をつなぐ common.merge_npz と load_cache が 0 次元の配列をつなげないため）
        "op": np.full(n, op or "", dtype="<U8"), "lowpass_fc": np.full(n, -1 if fc is None else fc, dtype=np.int16),
        "lowpass_order": np.full(n, ORDER, dtype=np.int8), "filter": np.full(n, FILTER if op == "lp" else "", dtype="<U40"),
        "model_revision": np.full(n, getattr(model.config, "_commit_hash", "") or "", dtype="<U64"),
        "code_commit": np.full(n, CODE_COMMIT, dtype="<U40"), "code_dirty": np.full(n, CODE_DIRTY, dtype=bool),
    }


def check_identity(corpus: str, model, args) -> None:
    """操作を外した流れが既存の特徴を作り直すか（sample_id の sha256 の順の limit 件）。書き出さない。"""
    import hashlib
    df = common.load_manifest(corpus)
    df = df.assign(h=df["sample_id"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())).sort_values("h").head(args.limit)
    df = df.drop(columns="h").reset_index(drop=True)
    pay = process_chunk(df, model, args, None, None)
    old = common.load_cache("audio_wavlm", corpus)
    pos = pd.Series(np.arange(len(old["sample_id"])), index=old["sample_id"].astype(str))
    idx = pos.reindex(pay["sample_id"]).to_numpy()
    if np.isnan(idx.astype(float)).any():
        raise SystemExit(f"{corpus}：既存の特徴に無い sample_id がある")
    idx = idx.astype(int)
    both = pay["ok"] & old["ok"][idx]
    d = np.abs(pay["X"][both] - old["X"][idx][both])
    print(f"{corpus}：{int(both.sum())} 件で照合．最大の差 {d.max():.2e}，平均 {d.mean():.2e}，ok の一致 "
          f"{int((pay['ok'] == old['ok'][idx]).sum())}/{len(idx)}．先頭の sample_id {list(pay['sample_id'][:3])}", flush=True)
    if d.max() > TOL_IDENTITY:
        raise SystemExit(f"{corpus}：濾波を外した流れが既存の特徴と {d.max():.2e} 違う（上限 {TOL_IDENTITY}）")


def run_corpus(corpus: str, kind: str, model, args) -> None:
    df = common.load_manifest(corpus)
    if args.limit:
        df = df.head(args.limit)
    print(f"\n=== {corpus}: {len(df)} 発話（{args.op}，fc {args.fc}）===", flush=True)
    t0 = time.perf_counter()
    for ci, start in enumerate(range(0, len(df), args.chunk)):
        part = common.part_path(kind, corpus, ci)
        if part.exists() and not args.overwrite:
            print(f"  [{ci}] 済み: {part.name}", flush=True)
            continue
        sub = df.iloc[start: start + args.chunk].reset_index(drop=True)
        t1 = time.perf_counter()
        payload = process_chunk(sub, model, args, args.op, args.fc)
        np.savez(part, **payload)
        dt = time.perf_counter() - t1
        print(f"  [{ci}] {len(sub)} 件 / 成功 {int(payload['ok'].sum())} / 切り詰め {int(payload['truncated'].sum())} / "
              f"{dt:.1f} 秒 ({dt / max(len(sub), 1):.3f} 秒/件) → {part.name}", flush=True)
    print(f"{corpus} 完了: {time.perf_counter() - t0:.1f} 秒", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", required=True, choices=("lp", "rev"))
    ap.add_argument("--fc", type=int, default=None, choices=FC_ALLOWED, help="--op lp の試しと照合だけ．本番は fc.json の値")
    ap.add_argument("--corpus", nargs="+", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=320)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--check-identity", action="store_true")
    args = ap.parse_args()
    if args.op == "rev":
        args.fc = None
        kind = "audio_wavlm-rev"
    else:
        if FC_JSON.exists():
            fc_rule = int(json.load(open(FC_JSON))["fc"])
            if args.fc is None:
                args.fc = fc_rule
            elif args.fc != fc_rule and not args.limit:
                raise SystemExit(f"fc.json の fc は {fc_rule}．ほかの fc は --limit つきの試しでだけ作る")
        elif args.fc is None or not (args.limit or args.check_identity or args.merge):
            raise SystemExit("fc.json が無い（先に features/check_lowpass_asr.py）．試しは --fc と --limit を付ける")
        kind = f"audio_wavlm-lp{args.fc}"
    if args.limit > LIMIT_MAX:
        raise SystemExit(f"--limit は {LIMIT_MAX} 以下")
    if args.limit and not args.check_identity:
        kind += "_trial"          # 試しの途中保存が本番の途中保存として「済み」と読まれないように名前を分ける
    if args.merge:
        for c in args.corpus:
            common.merge_npz(kind, c)
        return
    if args.check_identity and not args.limit:
        raise SystemExit("--check-identity には --limit を付ける")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MERDA_GPU_FRACTION", "0.6")))
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL).to(args.device).eval()
    print(f"{MODEL} を {args.device} に載せた（fp32・凍結）．操作 {args.op}（低域通過は {ORDER} 次の前後 2 回，fc {args.fc}）", flush=True)
    for c in args.corpus:
        if args.check_identity:
            check_identity(c, model, args)
        else:
            run_corpus(c, kind, model, args)


if __name__ == "__main__":
    main()
