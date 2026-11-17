"""地図の実験（`docs/methods.md` 2 節）。

1 条件＝（ターゲット，ソース，K，反復 r）。作り方 3 つで学習し，ターゲットのテストで UAR を測る。

    tgt_only  ① ターゲットの K 本だけ（ソースに依らないので source の列は空）
    src_only  ② ソースだけ（K に依らないので K の列は空）
    merged    ③ ソース全件＋ターゲットの K 本

手順の決定は `docs/researchvision.md` 6.2。要点：

- **テスト**：ターゲットの話者を 5 袋に分け（`speaker_folds`，種 r），r mod 5 番の袋を使う
- **K 本**：残りから感情ごとに K 本。**K をまたいで入れ子**にする（K=5 の 20 本は
  K=10 の 40 本に含まれる）。K の違い以外を揃えるため。乱数はターゲットと r だけで決まり，
  ソースに依らない（どのソースと組んでも同じ K 本）
- **同じコーパスの群どうし**（IEMOCAP-impro と -script は同じ 10 人）では，テストの人物を
  ソースからも外す。人物は「コーパス:話者群」で突き合わせる
- **C**：学習データを話者で 3 分割した交差検証で選ぶ。成績は**保留側のうちターゲットの
  発話だけ**で測り（② はソースの発話），3 回ぶんの予測をまとめて UAR を 1 回計算する。
  学習側に欠ける感情が出る袋があれば C=1.0 に固定（`fixed`）。同点なら 1.0 に近い方，
  なお同点なら小さい方（`tie`）
- **オラクル**：候補の C すべてでテストの UAR も測って残す（`docs/methods.md` 1.2。分析専用）
- ② は K にも r にも依らないので，交差検証の種を固定してソースごとに 1 回だけ学習する
  （IEMOCAP 同士の組は，テストの人物を外すので r ごとに学習し直す）

## 学習器と温め直し

学習器は `logreg.py`（GPU。sklearn と同じ目的関数）。1 条件の中の問（交差検証の袋と全体 ×
候補の C）は `fit_batch` で束ねて一度に解く。③ は ② の解から解き始める（温め直し）。
③ の交差検証では**ソース部分の 3 分割を ② と同じ種（`SRC_CV_SEED`）で作る**ので，
袋 j の学習データのソース部分は ② の袋 j と同じになり，② の袋 j の解から始められる
（`init` の列が `folds`）。IEMOCAP 同士の組では同じ人物がターゲット側の袋に従うので
分け方がずれ，② の最終の解から始める（`final`）。`--cold` で温め直しを切る（`none`）。

## 結果

`results/map_<kind>/<ターゲット>.csv` に 1 行 1 条件で書く。**反復 1 回が終わるたびに，
その回の全ソース・全 K の行を 1 回の書き込みで足す**ので，途中でも読める（`summarize_map.py`）。
止まったら `--resume` で続きから回す（終わった反復は飛ばす）。
実行ごとに `#` に続けて設定と git のコミット番号（JSON）を 1 行書き，
実行したコードの写しを `results/map_<kind>/code/<開始時刻>/` に置く。
読むときは `pd.read_csv(path, comment="#", dtype={"K": str, "source": str})`
（K を文字列にしないと "1" が 1.0 に，"all" の無いファイルでは数値の列になる）。

    ./.venv/bin/python exp/map.py --target AFEW --source RAVDESS --reps 0 --K 1 5 \
        --out-dir /tmp/map_try
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "measure"))
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "exp"))
import common  # noqa: E402
import logreg  # noqa: E402
import shift  # noqa: E402
from data import GROUP_SPEC, MODALITIES, groups_for, load_any  # noqa: E402
from split import check_disjoint, speaker_folds  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0)      # `docs/methods.md` 2.3
C_DEFAULT = 1.0                      # 交差検証ができないとき・同点のときに寄せる値
N_TEST_FOLDS = 5
N_CV_FOLDS = 3
K_GRID = ("1", "5", "10", "25", "50", "all")   # クラスあたり。"all" は学習プール全件
N_REPS = 30
N_CLASSES = shift.N_CLASSES
SRC_CV_SEED = 0                      # ② と，③ のソース部分の交差検証の種（r に依らせない）

COLUMNS = (
    ["target", "source", "method", "K", "rep", "kind",
     "n_src", "n_tgt", "n_test", "C", "C_rule", "uar", "C_oracle", "uar_oracle"]
    + [f"cv_uar_C{c:g}" for c in C_GRID]
    + [f"test_uar_C{c:g}" for c in C_GRID]
    + ["sec", "n_iter", "n_loop", "init", "cached"]
)


# ---------------------------------------------------------------- データ

@dataclass
class Group:
    name: str
    X: torch.Tensor          # 特徴（GPU 上）
    y: np.ndarray
    speaker: pd.Series       # 話者群（`speaker_folds` に渡す）
    person: np.ndarray       # 「コーパス:話者群」。群をまたいで同じ人物を突き合わせる


def load(name: str, kind: str) -> Group:
    """kind は単一のエンコーダ名（例 audio_wavlm）か融合名（例 fuse-audio-video）。"""
    df, X = load_any(name, kind)
    sp = df["speaker_group"].astype(str).reset_index(drop=True)
    person = (GROUP_SPEC[name][0] + ":" + sp).to_numpy()
    return Group(name, logreg.to_device(X), df["label_4"].to_numpy(), sp, person)


def rows(T: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    """numpy の行番号・真偽の配列で GPU 上の行を取る。"""
    return T[torch.as_tensor(np.array(idx), device=T.device)]   # 複製：読み取り専用の配列を避ける


def labels(y: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.array(y), dtype=torch.long, device=logreg.DEVICE)


# ---------------------------------------------------------------- 分割と K 本

def test_mask(g: Group, r: int) -> np.ndarray:
    """反復 r のテスト：話者 5 袋のうち r mod 5 番の袋（`docs/researchvision.md` 6.2）。"""
    f = speaker_folds(g.speaker, k=N_TEST_FOLDS, seed=r)
    check_disjoint(g.speaker, f)
    return (f == r % N_TEST_FOLDS).to_numpy()


def kshot_order(g: Group, pool: np.ndarray, r: int) -> list[np.ndarray]:
    """感情ごとに，学習プールの行番号を乱数で並べた列。**先頭の K 本が K 本の抽出**になる
    ので，K をまたいで入れ子になる。乱数はターゲットと r だけで決まる。"""
    rng = common.rng_for(f"map-kshot:{g.name}", r)
    return [rng.permutation(pool[g.y[pool] == c]) for c in range(N_CLASSES)]


def take_k(order: list[np.ndarray], K: str) -> np.ndarray:
    if K == "all":
        return np.sort(np.concatenate(order))
    k = int(K)
    if any(len(o) < k for o in order):
        raise ValueError(f"K={k} に対して学習プールが足りない: {[len(o) for o in order]}")
    return np.sort(np.concatenate([o[:k] for o in order]))


def cv_folds(person: np.ndarray, first: np.ndarray, seed: int,
             other_seed: int | None = None) -> np.ndarray:
    """交差検証の話者 3 分割の袋番号。**first の行（成績を測る行）の人物から先に配る**（種 seed）。

    ほかの行は，同じ人物が first にいればその袋に入れ（IEMOCAP 同士の組で，同じ人の
    即興と台本が学習側と保留側に割れないように），いなければ残りの行だけで配る（種 other_seed。
    省けば seed）。
    """
    folds = np.full(len(person), -1)
    f1 = speaker_folds(pd.Series(person[first]), k=N_CV_FOLDS, seed=seed).to_numpy()
    folds[first] = f1
    of = dict(zip(person[first], f1))
    rest = ~first
    shared = rest & np.isin(person, list(of))
    folds[shared] = [of[p] for p in person[shared]]
    other = rest & ~shared
    if other.any():
        folds[other] = speaker_folds(pd.Series(person[other]), k=N_CV_FOLDS,
                                     seed=seed if other_seed is None else other_seed
                                     ).to_numpy()
    check_disjoint(pd.Series(person), pd.Series(folds))
    return folds


# ---------------------------------------------------------------- 学習と C の選択

def choose_c(scores: dict[float, float]) -> tuple[float, bool]:
    """最良の C と，同点だったか。同点なら C_DEFAULT に近い方（10 倍刻み），なお同点なら小さい方。"""
    best = max(scores.values())
    tied = [c for c in C_GRID if np.isclose(scores[c], best, rtol=0.0, atol=1e-12)]
    tied.sort(key=lambda c: (round(abs(float(np.log10(c / C_DEFAULT))), 9), c))
    return tied[0], len(tied) > 1


@dataclass
class Trained:
    final: logreg.Fitted                   # 学習データ全体で，候補の C すべてを解いたもの
    C: float
    rule: str                              # cv / tie / fixed
    cv: dict[float, float] | None
    folds: np.ndarray                      # 交差検証の袋番号（行ごと）
    fold_fits: dict[int, logreg.Fitted] = field(default_factory=dict)
    sec: float = 0.0
    n_iter: int = 0                        # すべての学習の反復回数の合計（計算量の記録）
    n_loop: int = 0                        # 束ねた計算の回数（最も遅い問の反復回数）
    init: str = "none"                     # 温め直しの出発点：none / folds / final


def train(X: torch.Tensor, y: np.ndarray, person: np.ndarray, first: np.ndarray,
          seed: int, other_seed: int | None = None, init: Trained | None = None,
          anchor: Trained | None = None) -> Trained:
    """交差検証で C を選び（成績は first の行で測る），全体で候補の C すべてを解く。

    init を渡すと温め直す。ソース部分（first でない行）の袋の分け方が init と同じなら
    袋ごとに init の袋の解から，違えば init の最終の解から始める。
    anchor を渡すと，係数を anchor（選ばれた C の最終の解）へ引き寄せる罰則で解く（M5）。
    """
    t0 = time.perf_counter()
    yt = labels(y)
    folds = cv_folds(person, first, seed, other_seed)
    kind = "none"
    if init is not None:
        kind = "folds" if np.array_equal(folds[~first], init.folds) else "final"

    def start(j: int) -> logreg.Fitted | None:
        if kind == "folds" and j in init.fold_fits:
            return init.fold_fits[j]
        return None if init is None else init.final

    used = [j for j in range(N_CV_FOLDS) if (first & (folds == j)).any()]
    if not all(len(np.unique(y[folds != j])) == N_CLASSES for j in used):
        used = []                                      # 交差検証ができない → C を固定
    # 交差検証の袋と全体を，候補の C すべてについて束ねて一度に解く（logreg.fit_batch）
    fits = logreg.fit_batch(X, yt, [folds != j for j in used] + [np.ones(len(y), bool)],
                            C_GRID, [start(j) for j in used]
                            + [None if init is None else init.final],
                            anchor=None if anchor is None else anchor.final,
                            anchor_c=None if anchor is None else anchor.C)
    final, fold_fits = fits[-1], dict(zip(used, fits[:-1]))
    cv = None
    if used:
        pred = {c: np.full(len(y), -1) for c in C_GRID}
        for j in used:
            va = first & (folds == j)
            Xva = rows(X, va)
            for c in C_GRID:
                pred[c][va] = fold_fits[j].predict(Xva, c)
        cv = {c: shift.uar(y[first], pred[c][first]) for c in C_GRID}
        c, tie = choose_c(cv)
        rule = "tie" if tie else "cv"
    else:
        c, rule = C_DEFAULT, "fixed"
    n_iter = sum(sum(f.n_iter.values()) for f in fits)
    n_loop = max(max(f.n_iter.values()) for f in fits)
    return Trained(final, c, rule, cv, folds, fold_fits,
                   time.perf_counter() - t0, n_iter, n_loop, kind)


def score(t: Trained, X_test: torch.Tensor, y_test: np.ndarray, cached: bool) -> dict:
    """テストの UAR。選んだ C の値と，候補ごとの値（オラクル用）を返す。"""
    test = {c: shift.uar(y_test, t.final.predict(X_test, c)) for c in C_GRID}
    c_or, _ = choose_c(test)
    row = {"C": t.C, "C_rule": t.rule, "uar": test[t.C],
           "C_oracle": c_or, "uar_oracle": test[c_or],
           "sec": 0.0 if cached else t.sec, "n_iter": 0 if cached else t.n_iter,
           "n_loop": 0 if cached else t.n_loop,
           "init": t.init, "cached": cached}
    for c in C_GRID:
        row[f"cv_uar_C{c:g}"] = np.nan if t.cv is None else t.cv[c]
        row[f"test_uar_C{c:g}"] = test[c]
    return row


# ---------------------------------------------------------------- 1 ターゲット × 1 反復

def run_rep(tgt: Group, sources: list[Group], r: int, Ks: list[str], kind: str,
            src_cache: dict, warm: bool = True) -> list[dict]:
    te = test_mask(tgt, r)
    pool = np.flatnonzero(~te)
    order = kshot_order(tgt, pool, r)
    X_test, y_test = rows(tgt.X, te), tgt.y[te]
    test_persons = np.unique(tgt.person[te])
    base = {"target": tgt.name, "rep": r, "kind": kind, "n_test": int(te.sum())}

    rows_out, src_fit, src_X = [], {}, {}
    for s in sources:                                        # ② ソースのみ
        k = ~np.isin(s.person, test_persons)
        key = (s.name, tuple(np.intersect1d(test_persons, s.person)))
        cached = key in src_cache
        src_X[s.name] = (rows(s.X, k), s.y[k], s.person[k])
        if not cached:
            Xs, ys, ps = src_X[s.name]
            src_cache[key] = train(Xs, ys, ps, np.ones(len(ys), bool), SRC_CV_SEED)
        src_fit[s.name] = src_cache[key]
        rows_out.append({**base, "source": s.name, "method": "src_only", "K": "",
                         "n_src": int(k.sum()), "n_tgt": 0,
                         **score(src_fit[s.name], X_test, y_test, cached)})

    for K in Ks:
        idx = take_k(order, K)
        Xk, yk, pk = rows(tgt.X, idx), tgt.y[idx], tgt.person[idx]
        n = len(idx)
        t = train(Xk, yk, pk, np.ones(n, bool), r)           # ① ターゲットのみ
        rows_out.append({**base, "source": "", "method": "tgt_only", "K": K,
                         "n_src": 0, "n_tgt": n, **score(t, X_test, y_test, False)})
        for s in sources:                                    # ③ 合併
            Xs, ys, ps = src_X[s.name]
            first = np.r_[np.ones(n, bool), np.zeros(len(ys), bool)]
            t = train(torch.cat([Xk, Xs]), np.r_[yk, ys], np.r_[pk, ps], first,
                      r, SRC_CV_SEED, src_fit[s.name] if warm else None)
            rows_out.append({**base, "source": s.name, "method": "merged", "K": K,
                             "n_src": len(ys), "n_tgt": n,
                             **score(t, X_test, y_test, False)})
    return rows_out


# ---------------------------------------------------------------- 記録

def provenance(args: argparse.Namespace) -> dict:
    """結果ファイルの 1 行目に書く設定（git で追えない分の追跡。`docs/methods.md` 6.3）。"""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    return {
        "script": "exp/map.py", "argv": sys.argv[1:],
        "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": args.kind, "targets": args.target, "sources": args.source,
        "reps": args.reps, "K": args.K, "warm": not args.cold,
        "C_grid": list(C_GRID), "C_default": C_DEFAULT,
        "n_test_folds": N_TEST_FOLDS, "n_cv_folds": N_CV_FOLDS, "src_cv_seed": SRC_CV_SEED,
        "solver": {"tol": logreg.TOL, "max_iter": logreg.MAX_ITER, "history": logreg.HISTORY,
                   "dtype": str(logreg.DTYPE), "device": str(logreg.DEVICE),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="地図の実験（docs/methods.md 2 節）")
    ap.add_argument("--kind", default=MODALITIES["audio"][0])
    ap.add_argument("--target", nargs="+", help="既定：その kind で使える全群")
    ap.add_argument("--source", nargs="+", help="既定：その kind で使える全群（ターゲット自身を除く）")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(N_REPS)))
    ap.add_argument("--K", nargs="+", default=list(K_GRID), choices=K_GRID)
    ap.add_argument("--cold", action="store_true", help="③ の温め直しを切る")
    ap.add_argument("--out-dir", type=Path,
                    help="既定：results/map_<kind>/。ターゲットごとに <群>.csv を作る")
    ap.add_argument("--resume", action="store_true",
                    help="既にあるファイルに続きを足す（終わった反復は飛ばす）")
    args = ap.parse_args()

    avail = groups_for(args.kind)
    args.target = args.target or avail
    args.source = args.source or avail
    bad = [g for g in args.target + args.source if g not in avail]
    if bad:
        raise SystemExit(f"{args.kind} で使えない群: {bad}（使えるのは {avail}）")
    out_dir = args.out_dir or ROOT / "results" / f"map_{args.kind}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 終わった反復を調べる。**反復 1 回ぶんの行は 1 回の書き込みで足す**ので，
    #      行の数が合わない反復があれば途中で壊れたとみなして止める（黙って混ぜない）
    done: dict[str, set[int]] = {}
    for tname in args.target:
        path = out_dir / f"{tname}.csv"
        n_src = len([s for s in args.source if s != tname])
        want = n_src + len(args.K) * (1 + n_src)
        done[tname] = set()
        if path.exists():
            if not args.resume:
                raise SystemExit(f"{path} がすでにある。続きを足すなら --resume")
            prev = pd.read_csv(path, comment="#", dtype={"K": str, "source": str})
            cnt = prev.groupby("rep").size()
            if (cnt != want).any():
                raise SystemExit(f"{path}：行の数が {want} でない反復がある {cnt[cnt != want].to_dict()}")
            done[tname] = set(int(r) for r in cnt.index)

    prov = provenance(args)
    code = snapshot_code(out_dir.resolve(), prov["started"])
    prov["code"] = str(code.relative_to(ROOT) if code.is_relative_to(ROOT) else code)
    data = {g: load(g, args.kind) for g in dict.fromkeys(args.target + args.source)}
    for tname in args.target:
        with open(out_dir / f"{tname}.csv", "a") as f:     # 設定は実行ごとに 1 行足す
            f.write("# " + json.dumps(prov, ensure_ascii=False) + "\n")
            if not done[tname]:
                pd.DataFrame(columns=list(COLUMNS)).to_csv(f, index=False)

    # **反復を外側に回す**：反復 1 回ごとに全ターゲットを一巡するので，途中で見ても
    # 全部の組が同じくらいの回数ずつそろっている（ターゲットを外側にすると，
    # 朝の時点で 1〜2 ターゲットしか見られない）
    src_cache: dict = {}
    t_all = time.perf_counter()
    for r in args.reps:
        for tname in args.target:
            if r in done[tname]:
                continue
            sources = [data[s] for s in args.source if s != tname]
            t0 = time.perf_counter()
            out_rows = run_rep(data[tname], sources, r, args.K, args.kind, src_cache,
                               warm=not args.cold)
            pd.DataFrame(out_rows, columns=list(COLUMNS)).to_csv(
                out_dir / f"{tname}.csv", mode="a", header=False, index=False)
            print(f"{tname} r={r}: {len(out_rows)} 行 {time.perf_counter() - t0:.1f} 秒"
                  f"（累計 {time.perf_counter() - t_all:.0f} 秒）", flush=True)
    print(f"終わり: {out_dir}", flush=True)


def snapshot_code(out_dir: Path, stamp: str) -> Path:
    """実行したコードの写しを結果の横に置く（コミットしていない変更も追えるように）。"""
    dest = out_dir / "code" / stamp.replace(":", "")
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ("exp", "measure", "features"):
        (dest / sub).mkdir(exist_ok=True)
        for p in (ROOT / sub).glob("*.py"):
            shutil.copy2(p, dest / sub / p.name)
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    (dest / "git_diff_HEAD.patch").write_text(diff.stdout)
    return dest


if __name__ == "__main__":
    main()
