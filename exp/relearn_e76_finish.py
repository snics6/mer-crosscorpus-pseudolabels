"""E76 の残り（A(HuBERT) の AFEW・RAVDESS・CREMA-D）を，止める規則だけを替えて計算する（2026-11-02 研究者の承認）。

E76（`exp/relearn_e76.py`）は 2026-11-01 21:06 に A(HuBERT) の MELD を書いた後，h9（AUGMENT）の照合で止まった
（180 条件のうち 2 条件が幅 tol を超え，数の基準「1% 以下」に落ちた．差 ÷ tol の最大は 2.33 で，大きさの基準 2.5 は満たした．
`logs/e76.log` 807〜808 行）．超えた 2 条件はソース RAVDESS・K=1・C=10（最も弱い罰則）で，相手の E31 は別の GPU（RunPod）で
解いた値である．ほかの 7 種類では h9・h10 の幅を超えた条件は 0 だった．

この台本は `relearn_e76.verdict` だけを差し替え，`relearn_e76.main` をそのまま呼ぶ．計算・保存・照合の表は E76 と同じである．
差し替えた判定：h9 と h10 では大きさの基準（どの条件も tol の 2.5 倍以下）を残し，数の基準（tol を超える条件が 1% 以下）は
記録だけにする．①・②（tgt_only・src_only）では，C の 4 つの値すべてのテストの UAR が地図と tol の中で一致し，選ばれた C の
UAR の基準も満たすときに限り，C と規則の違いを記録だけにする（2026-11-02 に RAVDESS の ① で 60 条件のうち 1 条件の C が地図と
違って止まった後の研究者の承認．取り違えなら 4 つの C のテストの UAR がそろってずれるので，その場合は止める）．③・h2〜h8 の基準は
E76 のまま（そこで止まったら研究者に報告する．E76 の 10 節）．

計算の前に，この台本を除いた読み込むコードのハッシュが E76 の本番（A(HuBERT) の MELD.json）と同じことを確かめる．
引数は `--kinds`・`--targets`・`--gpu-fraction`・`--out`・`--allow-dirty` だけを受け付ける（設定を E76 と同じに保つ）．
計算の後に `<out>/check_rule_finish.json` へ，結果，コードのハッシュ，ターゲットごとに元の規則なら落ちていた内容（waived），
照合に落ちたまま使う既存のファイル（accepted_existing：A(HuBERT) の MELD の h9）を書く（関門 gate-code の [I12]・[R]．
`checks/E76/code_finish.md`）．

使い方（研究室 PC の tmux mer-da．台本をコミットしてから）：
    ./.venv/bin/python exp/relearn_e76_finish.py --kinds audio_hubert --targets AFEW RAVDESS CREMA-D --gpu-fraction 0.3
    （2026-11-02 の 2 回目：AFEW と RAVDESS は 1 回目で書いたので --targets CREMA-D だけ）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relearn_e76 as R  # noqa: E402

COUNT_WAIVED = ("h9", "h10")      # 数の基準を記録だけにする手（E31 の O2・O1 と照合する手）
C_WAIVED = ("tgt_only", "src_only")  # C と規則の違いを，テストの UAR が C の 4 つすべてで地図と合うときだけ記録にする（①・②）
ALLOWED_KINDS = ("audio_hubert",)
ALLOWED_TARGETS = ("AFEW", "RAVDESS", "CREMA-D")
ALLOWED_OPTS = ("--kinds", "--targets", "--gpu-fraction", "--out", "--allow-dirty")
SELF = "exp/relearn_e76_finish.py"
REF_JSON = R.ROOT / "results" / "e76" / "audio_hubert" / "MELD.json"   # E76 の本番の最後のファイル

_orig_verdict = R.verdict


def c_waivable(e: dict) -> bool:
    """①・② で C か規則が地図と違っても，C の 4 つの値すべてのテストの UAR が地図と tol の中で一致し，
    選ばれた C の UAR の基準（tol の 2.5 倍以下，tol を超える条件が 1% 以下）も満たすなら真（2026-11-02 研究者の承認）。"""
    n = e.get("n_ref", 0)
    return bool(n) and (e.get("max_allC_over_tol") is not None and e["max_allC_over_tol"] <= 1.0) \
        and (e.get("max_diff_over_tol") or 0) <= 2.5 and e.get("n_over_tol", 0) <= 0.01 * n


def verdict_finish(summ: dict) -> list[str]:
    """E76 の verdict と同じ．ただし h9・h10 は大きさの基準（tol の 2.5 倍）だけで止める．
    ①・②（tgt_only・src_only）は，C と規則の違いを c_waivable のときだけ記録にする。"""
    summ = dict(summ)
    for m in C_WAIVED:
        e = summ.get(m)
        if e and (e.get("C_mismatch") or e.get("rule_mismatch")) and c_waivable(e):
            summ[m] = {**e, "C_mismatch": 0, "rule_mismatch": 0}
    bad = _orig_verdict({m: e for m, e in summ.items() if m not in COUNT_WAIVED})
    for m in COUNT_WAIVED:
        e = summ.get(m, {})
        if e.get("n_ref") and (e.get("max_diff_over_tol") or 0) > 2.5:
            bad.append(f"{m}：前の値との差が tol の 2.5 倍を超える {e}")
    return bad


def parse_args(argv: list[str]) -> dict[str, list[str]]:
    """引数を ALLOWED_OPTS に限る．`--opt=値` の書き方は止める。"""
    opts: dict[str, list[str]] = {}
    cur = None
    for a in argv:
        if a.startswith("--"):
            if "=" in a:
                raise SystemExit(f"`{a}`：`--opt=値` の書き方は使わない（空白で区切る）")
            if a not in ALLOWED_OPTS:
                raise SystemExit(f"`{a}` は受け付けない（{ALLOWED_OPTS} だけ．設定を E76 と同じに保つ）")
            if a in opts:
                raise SystemExit(f"`{a}` が 2 回ある")
            opts[a] = []
            cur = a
        elif cur is None:
            raise SystemExit(f"`{a}`：位置引数は受け付けない")
        else:
            opts[cur].append(a)
    return opts


def main() -> None:
    argv = sys.argv[1:]
    opts = parse_args(argv)
    kinds, targets = opts.get("--kinds", []), opts.get("--targets", [])
    if not kinds or any(k not in ALLOWED_KINDS for k in kinds):
        raise SystemExit(f"--kinds は {ALLOWED_KINDS} だけ（承認した範囲）")
    if not targets or any(t not in ALLOWED_TARGETS for t in targets):
        raise SystemExit(f"--targets は {ALLOWED_TARGETS} から選ぶ（承認した範囲．MELD は E76 の出力をそのまま使う）")
    out = Path(opts["--out"][0]) if opts.get("--out") else R.ROOT / "results" / "e76"

    # 包み台本を除いた読み込むコードが E76 の本番と同じか（関門 gate-code の [I12]）
    ref = json.loads(REF_JSON.read_text())
    files = [f for f in R.code_files() if f != SELF]
    h = R.code_hash(files)
    if files != ref["code_files"] or h != ref["code_hash"]:
        raise SystemExit(f"包み台本を除いたコードが E76 の本番と違う：ハッシュ {h[:12]}（本番 {ref['code_hash'][:12]}），"
                         f"ファイル {sorted(set(files) ^ set(ref['code_files']))}")

    rec = {"script": SELF, "argv": argv, "commit": R.git("rev-parse", "HEAD"),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "kinds": kinds, "targets": targets,
           "code_hash_without_wrapper": h, "code_hash_e76": ref["code_hash"],
           "change": "h9 と h10 の照合で，数の基準（tol を超える条件が 1% 以下）を記録だけにし，"
                     "大きさの基準（どの条件も tol の 2.5 倍以下）を残す．ほかの照合は E76 のまま",
           "reason": "E76 は A(HuBERT) の MELD の h9 で 180 条件のうち 2 条件が tol を超えて止まった（差 ÷ tol の最大 2.33）．"
                     "超えた条件はソース RAVDESS・K=1・C=10．相手の E31 は別の GPU で解いた値．"
                     "ほかの 7 種類では h9・h10 の超過は 0",
           "approved": "2026-11-02 研究者",
           "change_c": "①・② で C か規則が地図と違っても，C の 4 つの値すべてのテストの UAR が地図と tol の中で一致し，"
                       "選ばれた C の UAR の基準も満たすなら記録だけにする（2026-11-02 研究者の承認．RAVDESS の ① の後）",
           "accepted_existing": {"audio_hubert/MELD": ref.get("check_failed"),
                                 "audio_hubert/RAVDESS": json.loads((out / "audio_hubert" / "RAVDESS.json").read_text())
                                 .get("check_failed") if (out / "audio_hubert" / "RAVDESS.json").exists() else None}}
    R.verdict = verdict_finish
    result = "error"
    try:
        R.main()
        result = "ok"
    except SystemExit as e:
        result = f"SystemExit: {e}"
        raise
    finally:
        rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rec["result"] = result
        waived, hashes = {}, {}
        for k in kinds:
            for t in targets:
                p = out / k / f"{t}.json"
                if not p.exists():
                    continue
                m = json.loads(p.read_text())
                if m.get("started", "") < rec["started"]:
                    continue                                   # この実行で書いていないファイル
                hashes[f"{k}/{t}"] = m.get("code_hash")
                waived[f"{k}/{t}"] = _orig_verdict(m.get("check", {}))   # 元の規則なら落ちていた内容
        rec["written"] = hashes
        rec["waived"] = waived
        p = out / "check_rule_finish.json"
        prev = json.loads(p.read_text()) if p.exists() else {"runs": []}
        prev["runs"].append(rec)
        p.write_text(json.dumps(prev, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
