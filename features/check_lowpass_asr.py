"""操作の確かめ：言葉を聞き取れなくなったかを Whisper の書き起こしの一致率で測り，低域通過の切れ目 fc を決める（実験 23．設計 2 節）。

    標本：各コーパス（IEMOCAP，MOSEI，MELD）の解析に使う発話（`dai_data.load_corpus` の 3 モダリティがそろう発話．IEMOCAP は
    即興の部，MELD は複数の話者の発話を除いた後）のうち，人の書き起こしが 3 語以上で長さが 30 秒以下の発話を，sample_id の
    sha256 の順に 100 本ずつ選ぶ。テストの分割の発話も入るが，使うのは書き起こしと波形だけで，感情のラベルの値は使わない。
    条件：元の音声，逆再生（rev），fc = 500，400，300 Hz の低域通過（lp<fc>）。
    Whisper large-v3（`build_transcript_afew.py` と同じ設定：英語に固定，書き起こし，ビーム幅 5，サンプリングなし，波形は正規化しない）。
    一致率 H/N：人の書き起こしと出力を同じ規則で単語に分け，最長共通部分列の長さ H を参照の単語の数 N で割る。参考に単語誤り率 WER も出す。
    偶然の水準：同じ条件の出力を，同じコーパスの別の発話の参照と組ませたすべての組の H/N の平均（出力が長いだけで H/N は上がるため）。
    超過 = H/N の平均 − 偶然の水準。
    規則（計算の前に固定）：3 コーパスのすべてで超過が 0.10 以下になる最も高い fc を使う。どれも満たさないときは
    300 Hz を使い，fc.json の rule_met を false にする。逆再生の超過も同じ基準で記録する（rev_ok）。
    F0：同じ標本の元の音声で，YIN（de Cheveigné & Kawahara 2002）の推定で有声の区間のうち F0 が 0.8·fc を超える割合を出す（仮定の表）。

出力：results/dai23/asr_check.csv（発話ごと），asr_check_summary.csv，f0_check.csv，fc.json。

使い方:

    ./.venv/bin/python features/check_lowpass_asr.py --n 4 --out-dir <一時の置き場>   # 試し
    ./.venv/bin/python features/check_lowpass_asr.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
from build_audio import SR, decode
from build_audio_lowpass import apply_op

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import dai_data as DD  # noqa: E402

MODEL = "openai/whisper-large-v3"
TARGETS = ("IEMOCAP", "MOSEI", "MELD")
FCS = (500, 400, 300)                 # 高い順（規則は最も高い fc から確かめる）
EXCESS_MAX = 0.10                   # 超過の上限（慣用の値で，根拠は無い．設計 6 節）
MIN_WORDS = 3
MAX_SEC = 30.0
F0_RANGE = (60.0, 500.0)
F0_FRAME, F0_HOP, F0_THRESH, F0_GATE_DB = 0.04, 0.01, 0.15, -40.0
_PICKED: dict = {}


QUOTES = {"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'", "\u201c": '"', "\u201d": '"'}


def norm_words(t) -> list[str]:
    """`common.fix_mojibake` で文字化け（MELD の \\x92 など）を直し，引用符を ' と " にそろえ，小文字にし，
    角括弧・丸括弧・波括弧の注記を除き，アポストロフィ以外の記号を空白にして単語に分ける（参照と出力の両方に同じ規則を当てる）。"""
    t = common.fix_mojibake(str(t if isinstance(t, str) else ""))
    for k, v in QUOTES.items():
        t = t.replace(k, v)
    t = t.lower()
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}", " ", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return [w.strip("'") for w in t.split() if w.strip("'")]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def f0_yin(w: np.ndarray) -> np.ndarray:
    """YIN の F0（Hz）．有声と判定した区間だけを返す（無声と小さい音の区間は除く）。"""
    n, hop = int(F0_FRAME * SR), int(F0_HOP * SR)
    tmin, tmax = int(SR / F0_RANGE[1]), int(SR / F0_RANGE[0])
    W = n - tmax
    if len(w) < n:
        return np.array([])
    idx = np.arange(0, len(w) - n + 1, hop)
    X = np.stack([w[i: i + n] for i in idx]).astype(np.float64)
    rms = np.sqrt((X ** 2).mean(1))
    loud = rms > rms.max() * 10 ** (F0_GATE_DB / 20)
    d = np.stack([((X[:, :W] - X[:, t: t + W]) ** 2).sum(1) for t in range(1, tmax + 1)], 1)   # (区間, τ = 1..tmax)
    cm = d * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d, 1), 1e-12)
    out = []
    for k in np.flatnonzero(loud):
        c = cm[k, tmin - 1:]
        below = np.flatnonzero(c < F0_THRESH)
        if not len(below):
            continue
        j = below[0]
        while j + 1 < len(c) and c[j + 1] < c[j]:
            j += 1
        out.append(SR / (j + tmin))
    return np.array(out)


_SCOPE: list = []


def pick(corpus: str, n: int) -> tuple[pd.DataFrame, list]:
    ids = set(DD.load_corpus(corpus)[0]["sample_id"])
    df = common.load_manifest(corpus)
    df = df[df["sample_id"].isin(ids)]
    nw = df["transcript"].map(lambda t: len(norm_words(t)))
    _SCOPE.append(dict(corpus=corpus, n_analysis=len(df), share_lt_min_words=float((nw < MIN_WORDS).mean()),
                       share_0_words=float((nw == 0).mean()), min_words=MIN_WORDS))
    df = df[df["transcript"].map(lambda t: len(norm_words(t))) >= MIN_WORDS].copy()
    df["h"] = df["sample_id"].map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    df = df.sort_values("h").reset_index(drop=True)
    keep, dec = [], []
    for s in range(0, len(df), 2 * n):                       # 30 秒以下の発話が n 本そろうまで sha256 の順に読む
        sub = df.iloc[s: s + 2 * n]
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(decode, common.media_paths(sub, prefer_audio=True)))
        for i, (w, sec, tr) in zip(sub.index, got):
            if len(keep) < n and 0.01 <= sec <= MAX_SEC and not tr and len(w) >= SR // 100:
                keep.append(i); dec.append(w)
        if len(keep) >= n:
            break
    return df.loc[keep].reset_index(drop=True), dec


def transcribe(model, proc, wavs, device, batch=2, beams=5):
    out = []
    with torch.inference_mode():
        for s in range(0, len(wavs), batch):
            feats = proc(wavs[s: s + batch], sampling_rate=SR, return_tensors="pt").input_features
            ids = model.generate(feats.to(device), language="en", task="transcribe",
                                 num_beams=beams, do_sample=False, max_new_tokens=200)
            out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
            del feats, ids
            torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai23"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.resolve() == (ROOT / "results" / "dai23").resolve() and args.n != 100:
        raise SystemExit("本番の置き場では --n は 100（設計 2 節）")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fc.json").exists():
        raise SystemExit(f"{out / 'fc.json'} が既にある（fc を決め直さない）")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MERDA_GPU_FRACTION", "0.6")))
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).to(args.device).eval()
    rows, f0rows = [], []
    t0 = time.perf_counter()
    conds = ["orig", "rev"] + [f"lp{fc}" for fc in FCS]
    for c in TARGETS:
        df, dec = pick(c, args.n)
        if len(df) < args.n:
            raise SystemExit(f"{c}：条件を満たす発話が {len(df)} 本しか無い")
        _PICKED[c] = (df["sample_id"].tolist(), df["transcript"].tolist())
        f0 = [f0_yin(w) for w in dec]
        if c == "IEMOCAP":                                         # 性別ごとの F0（性別の注釈があるのは IEMOCAP だけ．実験 14 の corpus_meta と同じ列）
            man = pd.read_csv(common.META_DIR / "iemocap_utterance_manifest.csv", keep_default_na=False).set_index("sample_id")
            sex = man["utterance_speaker"].reindex(df["sample_id"]).astype(str).to_numpy()
            for sx in sorted(set(sex)):
                fx = [x for x, s_ in zip(f0, sex) if s_ == sx and len(x)]
                ax = np.concatenate(fx) if fx else np.array([])
                f0rows.append(dict(corpus=c, sex=sx, n_utt=int((sex == sx).sum()), n_voiced=len(ax),
                                   f0_median=float(np.median(ax)) if len(ax) else np.nan,
                                   **{f"share_above_0.8x{fc}": float((ax > 0.8 * fc).mean()) if len(ax) else np.nan for fc in FCS}))
        allf = np.concatenate([x for x in f0 if len(x)]) if any(len(x) for x in f0) else np.array([])
        f0rows.append(dict(corpus=c, sex="all", n_utt=len(dec), n_voiced=len(allf), f0_median=float(np.median(allf)) if len(allf) else np.nan,
                           **{f"share_above_0.8x{fc}": float((allf > 0.8 * fc).mean()) if len(allf) else np.nan for fc in FCS},
                           **{f"utt_share_any_above_0.8x{fc}": float(np.mean([(x > 0.8 * fc).any() for x in f0 if len(x)])) for fc in FCS}))
        for cond in conds:
            op, fc = (None, None) if cond == "orig" else (("rev", None) if cond == "rev" else ("lp", int(cond[2:])))
            hyp = transcribe(model, proc, [apply_op(w, op, fc) for w in dec], args.device)
            for sid, ref, h in zip(df["sample_id"], df["transcript"], hyp):
                r, hw = norm_words(ref), norm_words(h)
                rows.append(dict(corpus=c, sample_id=sid, condition=cond, n_ref=len(r), n_hyp=len(hw),
                                 hit=lcs(r, hw) / len(r), wer=edit_distance(r, hw) / len(r), hyp=h))
            print(f"{c} {cond}：{len(df)} 本 {time.perf_counter() - t0:.0f} 秒", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "asr_check.csv", index=False)
    refs = {(c, sid): norm_words(t) for c in TARGETS for sid, t in zip(*_PICKED[c])}
    chance = []
    for (c, cond), g in res.groupby(["corpus", "condition"]):       # 出力を別の発話の参照と組ませたすべての組の H/N の平均
        hyps = [norm_words(h) for h in g["hyp"]]
        sids = g["sample_id"].tolist()
        v = [lcs(refs[(c, sids[i])], hyps[j]) / len(refs[(c, sids[i])]) for i in range(len(sids)) for j in range(len(sids)) if i != j]
        chance.append(dict(corpus=c, condition=cond, hit_chance=float(np.mean(v))))
    summ = res.groupby(["corpus", "condition"]).agg(hit_mean=("hit", "mean"), hit_median=("hit", "median"),
                                                    wer_mean=("wer", "mean"), n_hyp_mean=("n_hyp", "mean"),
                                                    n=("hit", "size")).reset_index()
    summ = summ.merge(pd.DataFrame(chance), on=["corpus", "condition"])
    summ["excess"] = summ["hit_mean"] - summ["hit_chance"]
    summ.to_csv(out / "asr_check_summary.csv", index=False)
    pd.DataFrame(f0rows).to_csv(out / "f0_check.csv", index=False)
    pd.DataFrame(_SCOPE).to_csv(out / "sample_scope.csv", index=False)
    chosen, met = 300, False
    for fc in FCS:
        m = summ[summ.condition == f"lp{fc}"].set_index("corpus")["excess"]
        if len(m) == len(TARGETS) and (m <= EXCESS_MAX).all():
            chosen, met = fc, True
            break
    rv = summ[summ.condition == "rev"].set_index("corpus")["excess"]
    import scipy
    import transformers
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "features", "exp"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    asr_sha = hashlib.sha256(open(out / "asr_check.csv", "rb").read()).hexdigest()
    json.dump(dict(fc=chosen, rule_met=met, code_commit=commit, code_dirty=dirty, asr_check_sha256=asr_sha, rev_ok=bool(len(rv) == len(TARGETS) and (rv <= EXCESS_MAX).all()),
                   excess_max=EXCESS_MAX, n_per_corpus=args.n, max_sec=MAX_SEC, min_words=MIN_WORDS, order="sha256(sample_id)",
                   norm_rule="fix_mojibake; quotes to ' and \"; lower; drop [..] (..) {..}; non [a-z0-9'] -> space; strip apostrophes (both ref and hyp)", hit="LCS / n_ref",
                   chance="mean over all mismatched (ref i, hyp j), i != j, same corpus and condition",
                   model=MODEL, model_revision=getattr(model.config, "_commit_hash", ""), transformers=transformers.__version__,
                   scipy=scipy.__version__, filter="butter4 sosfiltfilt zero phase",
                   decoding=dict(language="en", task="transcribe", num_beams=5, do_sample=False, max_new_tokens=200),
                   candidates=list(FCS)), open(out / "fc.json", "w"), indent=1)
    print(summ.to_string(index=False))
    print(f"選んだ fc：{chosen} Hz（規則を満たしたか：{met}）")


if __name__ == "__main__":
    main()
