"""【探索的】E41（GPU）：teach が選ぶ前にラベルを使う 3 か所 (a)(b)(c) を，1 か所ずつその時点の予測に置き換える（ラベルを付ける前に選ぶ設定 1 への橋）。

規則は `docs/exploratory-analyses.md` E41（設計と予言 A1〜A3 は計算の前に書いた）。評価・候補・中心・下書き・学習器は E37 と同じ（P4 の既定）。
1 本選ぶごとにラベルが付く手順とし，選んだ発話の正解は選んだ直後に見える（模型の点数の更新は正解の残差で行う）。置き換えるのは選ぶ前の 3 か所：
  (a) 感情の番 → その感情を予測した候補から選ぶ（無ければその感情の確率が最大の 1 本）
  (b) 試しに教える発話の残差 → 4 つの感情それぞれの場合の目的を，その時点の予測の確率で平均する
  (c) 目的の d を測る候補の感情 → その時点の予測の確率で 4 つの感情に分ける
方針の名前は (a)(b)(c) の順に L（正解）か P（予測）を並べる（LLL が E37 の teach，PPP が 3 か所すべてを置き換えた版）。
比べる相手は random_true（E31 の P4 の K 本．ラベルを知っている）と random_pool（ラベル無しで候補から 4K 本．K で入れ子）。

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/teach_ablate.py --kind audio_wavlm [--target …] [--reps …]
    ./.venv/bin/python exp/teach_ablate.py --analyze

出力：`results/teach_ablate_<kind>/<ターゲット>.csv`（policy, rep, K, uar），`<ターゲット>_sel.csv`（選んだ発話の感情ごとの本数，
番の感情との一致，補った回数），`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e41_targets.csv`・`e41_tests.csv`・`e41_sel.csv`
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
from data import GROUP_SPEC, groups_for  # noqa: E402
from pair_table import map_dir  # noqa: E402
from select_labelfree import ht_block_present  # noqa: E402
from select_teach import CS, PAIRS, pname  # noqa: E402

LABS = ["".join(t) for t in itertools.product("LP", repeat=3)]       # LLL, LLP, …, PPP
SINGLE = {"a": "PLL", "b": "LPL", "c": "LLP"}
MAIN = ["audio_wavlm", "audio_hubert", "video", "fuse-audio-video"]
OUT_REL = ROOT / "results" / "explore_relation"


def ablate_order(Km: torch.Tensor, y: np.ndarray, L0: torch.Tensor, c: float, n_steps: int, lab: str, belief=None,
                 belief_order: bool = False, turn: str = "cycle", init=None):
    """lab の 3 か所を置き換えた teach の順（候補の中の位置），各段の番の感情，補ったかどうか。LLL は `select_teach.teach_order` と同じ計算。

    belief(L, P) を渡すと，P の 3 か所にその時点の予測の代わりに感情の見込み（(n, 4)）を入れる（E43）。残差は e_y − P のまま。
    belief_order が真なら belief(L, P, それまでに選んだ順) と呼ぶ（E48．付いたラベルで見込みを直す）。
    turn が "count" なら，番の感情を 0→3 の順ではなく，付いた本当のラベルが最も少ない感情にする（同数なら番の順．E52）。
    init（候補の中の位置の列）を渡すと，最初にそれらを順に選んだことにして（正解の残差で模型を更新し）から続きを選ぶ（E57．見本から始める）。"""
    use_a, use_b, use_c = (ch == "L" for ch in lab)
    K2 = Km * Km
    yt = torch.as_tensor(np.array(y), device=Km.device)
    onehot = torch.nn.functional.one_hot(yt, 4).float()
    eye = torch.eye(4, device=Km.device)
    L = L0.float().clone()
    avail = torch.ones(len(y), dtype=torch.bool, device=Km.device)
    order, turns, fills = [], [], []
    y_np = np.asarray(y)
    for i in (init or []):
        R0 = onehot - torch.softmax(L, 1)
        order.append(int(i))
        turns.append(int(y_np[i]))
        fills.append(False)
        avail[int(i)] = False
        L = L + c * Km[:, int(i):int(i) + 1] * R0[int(i):int(i) + 1, :]
    for step in range(len(order), n_steps):
        if turn == "count":
            cnt = np.bincount(y_np[order], minlength=4) if order else np.zeros(4, int)
            cls = int(np.lexsort(((np.arange(4) - step) % 4, cnt))[0])
        else:
            cls = step % 4
        P = torch.softmax(L, 1)
        B = P if belief is None else (belief(L, P, order) if belief_order else belief(L, P))
        R = onehot - P
        Z = torch.stack([L[:, k] - L[:, l] for k, l in PAIRS], 1)                    # (n, 6)
        if use_c:
            Ic = torch.stack([((yt == k) & avail).float() for k in range(4)], 1)
            nc = Ic.sum(0).clamp(min=1.0)
        else:                                                                        # (c)：予測の確率で 4 つの感情に分ける
            Ic = B * avail[:, None].float()
            nc = Ic.sum(0).clamp(min=1e-12)
        W = Ic / nc
        Bcols = torch.stack([W[:, cc] * Z[:, j] for j, (k, l) in enumerate(PAIRS) for cc in (k, l)], 1)   # (n, 12)
        G = Km @ torch.cat([W, Bcols], 1)                                            # (n, 16)
        Q = K2 @ W                                                                   # (n, 4)
        m = (W.T @ Z)                                                                # (4, 6)
        s = (W.T @ (Z * Z))                                                          # (4, 6)

        def objective(Rm):
            obj = torch.zeros(len(y), device=Km.device)
            for j, (k, l) in enumerate(PAIRS):
                dl = Rm[:, k] - Rm[:, l]
                mk = m[k, j] + c * dl * G[:, k]
                ml = m[l, j] + c * dl * G[:, l]
                sk = s[k, j] + 2 * c * dl * G[:, 4 + 2 * j] + (c * dl) ** 2 * Q[:, k]
                sl = s[l, j] + 2 * c * dl * G[:, 4 + 2 * j + 1] + (c * dl) ** 2 * Q[:, l]
                obj = obj + (mk - ml) / torch.sqrt(((sk - mk ** 2) + (sl - ml ** 2)).clamp(min=1e-12) / 2)
            return obj / len(PAIRS)

        if use_b:
            obj = objective(R)
        else:                                                                        # (b)：4 つの感情の場合を予測の確率で平均
            obj = sum(B[:, yy] * objective(eye[yy][None, :] - P) for yy in range(4))
        if use_a:
            cand = (yt == cls) & avail
            filled = False
        else:                                                                        # (a)：その感情を予測した候補
            cand = (B.argmax(1) == cls) & avail
            filled = not bool(cand.any())
            if filled:
                cand = torch.zeros_like(avail)
                cand[int(torch.argmax(torch.where(avail, B[:, cls], torch.full_like(B[:, cls], -1.0))))] = True
        if not bool(cand.any()):
            raise ValueError(f"感情 {cls} の候補が尽きた（{step} 本目，{lab}）")
        obj[~cand] = -float("inf")
        i = int(torch.argmax(obj))
        order.append(i)
        turns.append(cls)
        fills.append(filled)
        avail[i] = False
        L = L + c * Km[:, i:i + 1] * R[i:i + 1, :]                                   # 選んだ直後に正解が見えるので，正解の残差で更新する
    return order, turns, fills


def evaluate(kind, ev, src, C, reps, ks, rows, srows):
    ctx = CB.Ctx(ev, reps, ks, center="ind")
    kmax = max(int(K) for K in ks)
    idx = {f"{lab}_{pname(c)}": {} for lab in LABS for c in CS}
    idx["random_true"], idx["random_pool"] = {}, {}
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        yq = ev.y[pool]
        Xc = M.rows(ev.X, pool) - ctx.ct[r]
        L0 = src.logits(Xc, C)
        X = ((Xc - src.mu) / src.sd).float()
        Km = X @ X.T
        Km.fill_diagonal_(0.0)
        for lab in LABS:
            for c in CS:
                order, turns, fills = ablate_order(Km, yq, L0, c, 4 * kmax, lab)
                pn = f"{lab}_{pname(c)}"
                for K in ks:
                    n = 4 * int(K)
                    pp = np.array(order[:n])
                    if len(np.unique(pp)) != n:
                        raise ValueError(f"選んだ本数がおかしい: {pn} r{r} K{K}")
                    if lab[0] == "L" and (np.bincount(yq[pp], minlength=4) != int(K)).any():
                        raise ValueError(f"感情ごとの本数がおかしい: {pn} r{r} K{K}")
                    idx[pn][(r, K)] = pool[pp]
                    cnt = np.bincount(yq[pp], minlength=4)
                    srows.append({"target": ev.name, "policy": pn, "rep": r, "K": K, "missing": int((cnt == 0).sum()),
                                  "min_count": int(cnt.min()), "max_count": int(cnt.max()),
                                  "turn_match": float((yq[pp] == np.array(turns[:n])).mean()), "filled": int(sum(fills[:n])),
                                  **{f"n_{k}": int(cnt[k]) for k in range(4)}})
        del Km
        perm = np.random.default_rng(zlib.crc32(f"e41pool|{ev.name}|{r}".encode())).permutation(len(pool))
        for K in ks:
            idx["random_pool"][(r, K)] = pool[perm[:4 * int(K)]]
            idx["random_true"][(r, K)] = ctx.idx[(r, K)]
    for pn in idx:
        sub = CB.Ctx.__new__(CB.Ctx)
        sub.__dict__.update(ctx.__dict__)
        sub.idx = idx[pn]

        def add(r, K, c, u, cv, L, cen=0, cb="", pn=pn):
            rows.append({"target": ev.name, "policy": pn, "rep": r, "K": K, "uar": u})
        ht_block_present(sub, reps, src, C, add)


def run_target(kind, T, data, reps, ks, out):
    corp = lambda g: GROUP_SPEC[g][0]  # noqa: E731
    pool_T = [s for s in data if corp(s) != corp(T)]
    C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
    Xp = torch.cat([data[s].X for s in pool_T])
    yp = M.labels(np.concatenate([data[s].y for s in pool_T]))
    src = logreg.fit(Xp - Xp.mean(0), yp, (C_T,))
    del Xp
    rows, srows = [], []
    evaluate(kind, data[T], src, C_T, reps, ks, rows, srows)
    pd.DataFrame(srows).to_csv(out / f"{T}_sel.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(kinds=MAIN) -> int:
    star = pd.read_csv(OUT_REL / "e37_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"]).policy_star
    tabs, sels = [], []
    for kind in kinds:
        d = ROOT / "results" / f"teach_ablate_{kind}"
        fs = [f for f in sorted(d.glob("*.csv")) if not f.stem.endswith("_sel")]
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in fs])
        a = a[a.rep.isin(EC.REPS)]
        reps = sorted(a.rep.unique())                                                 # E37・E34 も同じ反復で平均する
        u = a.groupby(["target", "policy", "K"]).uar.mean() * 100
        sl = pd.concat([pd.read_csv(f, dtype={"K": str}) for f in sorted(d.glob("*_sel.csv"))])
        sl = sl[sl.rep.isin(EC.REPS)]
        for T in sorted(a.target.unique()):
            ps = star.loc[(kind, T)]                                                  # E37 がメタ学習で選んだ歩幅
            e37 = pd.read_csv(ROOT / "results" / f"select_teach_{kind}" / f"{T}.csv", dtype={"K": str})
            e37 = e37[(e37.role == "target") & e37.rep.isin(reps)].groupby(["policy", "K"]).uar.mean() * 100
            flf = ROOT / "results" / f"select_lf_{kind}" / f"{T}.csv"
            lf = pd.read_csv(flf, dtype={"K": str}) if flf.exists() else None
            for K in ("1", "5", "10", "25"):
                row = {"kind": kind, "target": T, "K": K, "step": ps,
                       "random_true": float(u.loc[(T, "random_true", K)]), "random_pool": float(u.loc[(T, "random_pool", K)])}
                for lab in LABS:
                    vals = {c: float(u.loc[(T, f"{lab}_{pname(c)}", K)]) for c in CS}
                    row[lab] = vals[float(ps.split("_c")[1])]
                    row[f"{lab}_oracle"] = max(vals.values())
                for c in CS:                                                          # 照合：LLL は E37 の teach と同じ計算
                    row[f"chk_LLL_c{c:g}"] = float(u.loc[(T, f"LLL_{pname(c)}", K)]) - float(e37.loc[(pname(c), K)])
                row["chk_random_true"] = row["random_true"] - float(e37.loc[("random", K)])
                for st in ("herding_pred", "random_pred"):
                    row[f"e34_{st}"] = np.nan if lf is None else float(lf[(lf.source == st) & (lf.K == K) & lf.rep.isin(reps)].uar.mean() * 100)
                tabs.append(row)
        sg = sl.groupby(["target", "policy", "K"])[["missing", "min_count", "max_count", "turn_match", "filled"]].mean().reset_index()
        sg.insert(0, "kind", kind)
        sels.append(sg)
    D = pd.DataFrame(tabs)
    tests = []
    for (kind, K), g in D.groupby(["kind", "K"], sort=False):
        gain = float((g.LLL - g.random_true).mean())
        row = {"kind": kind, "K": K, "targets": len(g), "random_true": float(g.random_true.mean()),
               "random_pool": float(g.random_pool.mean()), "teach_gain": gain}
        for lab in LABS:
            row[lab] = float(g[lab].mean())
            row[f"share_{lab}"] = float((g[lab] - g.random_true).mean()) / gain if gain else np.nan
        for part, pos in (("a", 0), ("b", 1), ("c", 2)):                             # 2×2×2 の主効果（置き換えた版 − 正解の版）
            row[f"main_{part}"] = float(np.mean([row[x] for x in LABS if x[pos] == "P"]) - np.mean([row[x] for x in LABS if x[pos] == "L"]))
        row["PPP-random_pool"] = float((g.PPP - g.random_pool).mean())
        row["PPP>random_pool"] = int((g.PPP > g.random_pool).sum())
        row["PPP-random_true"] = float((g.PPP - g.random_true).mean())
        row["PPP-e34_herding_pred"] = float((g.PPP - g.e34_herding_pred).mean())
        row["PPP_oracle-PPP"] = float((g.PPP_oracle - g.PPP).mean())
        row["chk_LLL_max"] = float(g[[f"chk_LLL_c{c:g}" for c in CS]].abs().max().max())
        row["chk_random_true_max"] = float(g.chk_random_true.abs().max())
        tests.append(row)
    R = pd.DataFrame(tests)
    for kind in ("audio_wavlm", "audio_hubert"):
        k5 = R[(R.kind == kind) & (R.K == "5")]
        if len(k5):
            drops = {p: float(k5.LLL.iloc[0] - k5[SINGLE[p]].iloc[0]) for p in SINGLE}
            R.loc[R.kind == kind, "A1"] = max(drops, key=drops.get) == "b"
            R.loc[R.kind == kind, "A2"] = bool(k5.share_PLL.iloc[0] >= 0.5)
            R.loc[R.kind == kind, "A3"] = bool(k5.share_LPL.iloc[0] < 0.5)
    D.to_csv(OUT_REL / "e41_targets.csv", index=False)
    R.to_csv(OUT_REL / "e41_tests.csv", index=False)
    S = pd.concat(sels) if sels else pd.DataFrame()
    S.to_csv(OUT_REL / "e41_sel.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["kind", "K", "random_true", "random_pool", "teach_gain"] + LABS + [f"share_{x}" for x in ("PLL", "LPL", "LLP", "PPP")] + \
        ["main_a", "main_b", "main_c", "PPP-random_pool", "PPP>random_pool", "PPP-e34_herding_pred", "chk_LLL_max", "chk_random_true_max"]
    print(R[cols].round(2).to_string(index=False))
    if "A1" in R:
        print(R.drop_duplicates("kind")[["kind", "A1", "A2", "A3"]].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--ks", nargs="+", default=list(CB.KS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=MAIN)
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.kinds)
    out = args.out_dir or ROOT / "results" / f"teach_ablate_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(args.kind)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{args.kind}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/teach_ablate.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(args.kind)), "K": args.ks, "reps": args.reps, "labs": LABS, "steps": list(CS),
            "cp": CB.DEF_CP, "cb": CB.DEF_CB, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    data = {g: M.load(g, args.kind) for g in groups}
    for t in todo:
        t0 = time.perf_counter()
        run_target(args.kind, t, data, args.reps, args.ks, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
