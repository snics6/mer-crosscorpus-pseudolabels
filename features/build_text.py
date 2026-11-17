"""言葉の特徴を作る。文字起こし → 事前学習済みの言語モデル → 単語方向に平均。

    マニフェストの transcript（人が書き起こした文．AFEW だけ Whisper の書き起こし）
         → トークンに分ける（上限 256）
         → RoBERTa-base（または DeBERTa-v3-base）に通す
         → 単語方向に平均（詰め物の部分は除く）→ 768 次元

**IEMOCAP・MOSEI・MELD の文字起こしは人が書いたもの**である。出所は
IEMOCAP が公式の `transcriptions/`，MELD が公式 CSV の `Utterance` 列，
MOSEI が CMU 配布の `label.csv` の `text` 列である。

**AFEW には人の文字起こしが無いので，Whisper large-v3 で書き起こした文を使う**
（`features/build_transcript_afew.py`．2026-11-04 から．発話一覧の `transcript_source` 列が
`whisper-large-v3`）。ほかのコーパスと同じく，そのまま特徴にする。

**文字起こしが空の発話は特徴を作らず `ok = False` とする**（Whisper が声を拾えなかった
AFEW の発話など）。**前身は空文字をトークンに変換して，開始記号と終了記号だけの
平均を特徴として保存していた**が，それは言葉の情報を何も含まないので改めた。

**RAVDESS と CREMA-D は，この抽出の対象にしない。**俳優が同じ文を感情だけ変えて
読むコーパスで，読む文は 2 種類・12 種類しかない。文字起こしからは感情の手がかりが
得られないため，測る意味がない（`data/README.md` 4 節「言葉の正規化の段階」）。

**前身から変えた点**。

1. 計算を fp32 にした（実測で fp16 は平均 0.12% ずれる。速度の利得が無いので揃える）
2. 切り詰めたことを記録する
3. **文字化けを直す**（`common.fix_mojibake`）。MELD の転写の 28.1%（3,191 件）に
   Windows-1252 の文字が壊れて混入していた。前身はこれに気づいていない

使い方:

    ./.venv/bin/python features/build_text.py --corpus MELD --limit 100
    ./.venv/bin/python features/build_text.py --corpus all
    ./.venv/bin/python features/build_text.py --corpus all --model microsoft/deberta-v3-base
    ./.venv/bin/python features/build_text.py --corpus all --merge
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

import common

MAX_TOKENS = 256      # 実測で 256 を超えるのは MOSEI の 5 件のみ


def encode(model, tok, texts: list[str], device: str, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """文の列を 768 次元の列にする。詰め物の部分は平均から除く。"""
    dim = model.config.hidden_size
    X = np.zeros((len(texts), dim), dtype=np.float32)
    ntok = np.zeros(len(texts), dtype=np.int32)
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            chunk = texts[s : s + batch]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state                  # (B, T, D)
            m = enc["attention_mask"].float()
            X[s : s + len(chunk)] = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
                                     ).float().cpu().numpy()
            ntok[s : s + len(chunk)] = m.sum(1).cpu().numpy()
    return X, ntok


def process_chunk(df: pd.DataFrame, model, tok, args) -> dict:
    n = len(df)
    dim = model.config.hidden_size
    X = np.full((n, dim), np.nan, dtype=np.float32)
    ntok = np.zeros(n, dtype=np.int32)
    trunc = np.zeros(n, dtype=bool)
    ok = np.zeros(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)

    raw = df["transcript"].fillna("").astype(str)
    n_fixed = int(raw.map(lambda s: s != common.fix_mojibake(s)).sum())
    # **正規化の段階**（data/README.md 4 節）。既定は L1（文字化けの修復のみ）
    texts = raw.map(lambda s: common.normalize_text(s, args.level)).tolist()
    keep = [i for i, t in enumerate(texts) if len(t) > 0]
    for i, t in enumerate(texts):
        if len(t) == 0:
            reason[i] = "文字起こしが無い"

    if keep:
        kept = [texts[i] for i in keep]
        # 切り詰めの有無を知るため，先に切らずにトークン数だけ数える
        raw_len = [len(tok(t, truncation=False)["input_ids"]) for t in kept]
        feats, nt = encode(model, tok, kept, args.device, args.batch)
        X[keep] = feats
        ntok[keep] = nt
        for r, i in enumerate(keep):
            trunc[i] = raw_len[r] > MAX_TOKENS
        ok[keep] = True

    return {
        "X": X, "ok": ok, "n_token": ntok, "truncated": trunc,
        "fail_reason": np.asarray(reason, dtype=np.str_),
        "n_mojibake_fixed": np.full(n, n_fixed, dtype=np.int32),
        **{c: df[c].astype(str).to_numpy(dtype=np.str_) for c in
           ("sample_id", "corpus", "split", "speaker_id", "dialog_type")},
        "label_4": df["label_4"].to_numpy(dtype=np.int8),
    }


def run_corpus(corpus: str, kind: str, model, tok, args) -> None:
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
        payload = process_chunk(sub, model, tok, args)
        np.savez(part, **payload)
        print(f"  [{ci}] {len(sub)} 件 / 成功 {int(payload['ok'].sum())} / "
              f"切り詰め {int(payload['truncated'].sum())} / "
              f"{time.perf_counter() - t1:.1f} 秒 → {part.name}", flush=True)
    print(f"{corpus} 完了: {time.perf_counter() - t0:.1f} 秒", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="all", help="IEMOCAP / MOSEI / MELD / AFEW / all")
    ap.add_argument("--model", default="roberta-base", choices=list(common.TEXT_MODELS))
    ap.add_argument("--level", type=int, default=1, choices=(0, 1, 2, 3, 4),
                    help="正規化の段階 L0〜L4（data/README.md 4 節）．既定は L1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    kind = f"text_{common.TEXT_MODELS[args.model]}" + (
        "" if args.level == 1 else f"_L{args.level}")
    # 言葉を作るのは 4 つ（AFEW は Whisper の書き起こし・RAVDESS と CREMA-D は固定文なので作らない）
    targets = list(common.CORPORA_TEXT) if args.corpus == "all" else [args.corpus]

    if args.merge:
        for c in targets:
            common.merge_npz(kind, c)
        return

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    print(f"{args.model} を {args.device} に載せた（fp32・凍結・微調整なし）"
          f" 次元 {model.config.hidden_size}", flush=True)
    for c in targets:
        run_corpus(c, kind, model, tok, args)


if __name__ == "__main__":
    main()
