"""【探索的】E60（GPU）：モダリティをまたぐ蒸留．運用時は 1 つのモダリティ（E60-1）と運用時も複数モダリティ（E60-2）の 2 つの設定を，
AV・VT・AT・AVT の 4 つの世界で測る（生徒は E39 の学習器）。

規則は `docs/exploratory-analyses.md` E60（設計と予言は計算の前に書いた）。特徴は融合の種類の特徴をモダリティごとのブロックに切ったもの．
出発点はモダリティごとと融合の E39（設定は各種類の E39／E40 の値．合併の C は 0.01）．弱いラベルの付け方・総重み m_eff・K 本との合わせ方は E59 と同じ．
  E60-1：モダリティ m の生徒に，その世界の融合の出発点 pF，モダリティの出発点の確率の積 poe，モダリティの票が全部一致した発話だけ重い agree（不一致は 0.2 倍）を蒸留する．
         対照 p（出発点自身＝釣り合い直し）．m_eff ∈ {100, 500, 2000}
  E60-2：融合の生徒に，pF で鍛えたモダリティごとの生徒（m_in ∈ {100, 500}．E60-1 の pF の生徒そのもの）の確率の積 ref を蒸留する（m_eff ∈ {100, 500}）．
         鍛えた生徒の積をそのまま使う版 refdirect，元の出発点の積 poe0，対照 p も測る
  (見込み, m_eff) は K ∈ {0, 1, 5, 10, 25} ごとに，世界のソースを 1 つずつ仮のターゲットにして選ぶ（反復 0〜4．「蒸留しない」も候補）

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/multimodal_distill.py --world fuse-audio-video [--target …] [--reps …]
    ./.venv/bin/python exp/multimodal_distill.py --analyze

出力：`results/multimodal_distill_<world>/<ターゲット>.csv`（student, belief, meff, rep, K, uar），`<ターゲット>_pseudo.csv`，`<ターゲット>_meta.csv`，
`provenance.json`，`provenance_log.jsonl`。判定は `results/explore_relation/e60_targets.csv`・`e60_tests.csv`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import explore_compare as EC  # noqa: E402
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import solve_all, uar_of  # noqa: E402
from geom_e0 import settings  # noqa: E402
from pair_table import map_dir  # noqa: E402
from teach_student39 import Student  # noqa: E402

WORLDS1 = {"fuse-audio-video": ["audio_wavlm", "video"], "fuse-video-text": ["video", "text_roberta"],
           "fuse-audio-text": ["audio_wavlm", "text_roberta"], "fuse-audio-video-text": ["audio_wavlm", "video", "text_roberta"]}
WORLDS2 = {"fuse2-audio-video-text": ["audio_hubert", "video_clip", "text_deberta"]}   # 第 2 エンコーダの AVT（E62．判定は --suffix _2nd で別のファイルに）
WORLDS = {**WORLDS1, **WORLDS2}
DIMS = {"audio_wavlm": 1024, "video": 512, "text_roberta": 768, "audio_hubert": 1024, "video_clip": 1024, "text_deberta": 768}
KS = ("0", "1", "5", "10", "25")
PSEUDO_REPS = [0, 1, 2, 3, 4]
C_FIT = 0.01
BEL1, MEFF1 = ["p", "pF", "poe", "agree"], (100, 500, 2000)
MIN2 = (100, 500)
BEL2, MEFF2 = ["p", "poe0", "ref100", "ref500"], (100, 500)
AGREE_LOW = 0.2
OUT_REL = ROOT / "results" / "explore_relation"


def blocks(world):
    out, a = [], 0
    for m in WORLDS[world]:
        out.append((m, a, a + DIMS[m]))
        a += DIMS[m]
    return out


def sub_group(g, a, b):
    return M.Group(g.name, g.X[:, a:b], g.y, g.speaker, g.person)


def lg(p):
    return np.log(np.clip(p, 1e-12, None))


def make_students(world, T, data_f, data_m, S, pool, te, sets, info):
    """各モダリティと融合の生徒（E39）を候補の上に作る．戻りは kind → dict(st, rho_m, cb, cp, Z, Zte, P)。"""
    out = {}
    for k in WORLDS[world] + [world]:
        dat = data_m[k][T] if k != world else data_f[T]
        mu = M.rows(dat.X, pool).mean(0)
        Xc = M.rows(dat.X, pool) - mu
        gamma, rho_s, rho_m, cb, cp, _ = sets[k]
        src, lin, n_of = info[k]
        st = Student(dat, Xc, src, lin, n_of, S, C_FIT, gamma, rho_s, [rho_m])
        Am = st.mats[rho_m][0]
        out[k] = {"st": st, "rho_m": rho_m, "cb": cb, "cp": cp, "Z": st.Xstd @ Am,
                  "Zte": ((M.rows(dat.X, te) - mu - st.src.mu) / st.src.sd) @ Am, "P": torch.softmax(st.L0, 1).cpu().numpy()}
    return out


def beliefs1(stu, world):
    mods = WORLDS[world]
    pF = stu[world]["P"]
    poe = CM.softmax_np(sum(lg(stu[m]["P"]) for m in mods))
    am = np.stack([stu[m]["P"].argmax(1) for m in mods], 1)
    agree = pF * np.where((am == am[:, :1]).all(1), 1.0, AGREE_LOW)[:, None]
    return {"pF": pF, "poe": poe, "agree": agree}


def solve_world(world, stu, yq, yte, label_sets, rows, tag):
    """E60-1 と E60-2 の全部の問題を解き，テストの UAR を rows に足す（tag はターゲットか仮のターゲットの名前）。"""
    mods = WORLDS[world]
    b1 = beliefs1(stu, world)
    ref = {m_in: {} for m_in in MIN2}
    direct = {m_in: {} for m_in in MIN2}
    for m in mods:
        s = stu[m]
        bel = {"p": s["P"], **b1}
        combos = [("p", 0)] + [(b, e) for b in BEL1 for e in MEFF1]
        sol = solve_all(s["Z"], yq, bel, label_sets, combos, s["st"], s["rho_m"], C_FIT, s["cp"], s["cb"])
        rows.append({**tag, "student": m, "belief": "none", "meff": 0, "K": "0", "uar": uar_of(s["st"].W_std @ s["st"].mats[s["rho_m"]][1], s["st"].b_std, s["Zte"], yte)})
        for (b, e, K), (W, bb) in sol.items():
            rows.append({**tag, "student": m, "belief": "none" if e == 0 else b, "meff": e, "K": K, "uar": uar_of(W, bb, s["Zte"], yte)})
            if b == "pF" and e in MIN2:
                ref[e].setdefault(K, []).append(torch.softmax(s["Z"] @ W.T + bb, 1).cpu().numpy())
                direct[e].setdefault(K, []).append(torch.softmax(s["Zte"] @ W.T + bb, 1).cpu().numpy())
    f = stu[world]
    rows.append({**tag, "student": world, "belief": "none", "meff": 0, "K": "0", "uar": uar_of(f["st"].W_std @ f["st"].mats[f["rho_m"]][1], f["st"].b_std, f["Zte"], yte)})
    for K in label_sets:
        bel = {"p": f["P"], "poe0": b1["poe"], **{f"ref{m_in}": CM.softmax_np(sum(lg(p) for p in ref[m_in][K])) for m_in in MIN2}}
        combos = [("p", 0)] + [(b, e) for b in BEL2 for e in MEFF2]
        sol = solve_all(f["Z"], yq, bel, {K: label_sets[K]}, combos, f["st"], f["rho_m"], C_FIT, f["cp"], f["cb"])
        for (b, e, K_), (W, bb) in sol.items():
            rows.append({**tag, "student": world, "belief": "none" if e == 0 else b, "meff": e, "K": K_, "uar": uar_of(W, bb, f["Zte"], yte)})
        for m_in in MIN2:
            pd_ = CM.softmax_np(sum(lg(p) for p in direct[m_in][K]))
            rows.append({**tag, "student": world, "belief": f"refdirect{m_in}", "meff": 0, "K": K, "uar": CB.uar(yte, pd_.argmax(1))})


def choose(pdf, student):
    """仮のターゲットの値から，K ごとに (見込み, m_eff) を選ぶ（候補：蒸留しない，生徒に応じた見込み × m_eff，融合なら refdirect も）。"""
    cand_b = {"pF", "poe", "agree"} if student in DIMS else {"poe0", "ref100", "ref500", "refdirect100", "refdirect500"}
    out, mrows = {}, []
    for K in KS:
        sub = pdf[(pdf.student == student) & (pdf.K == K)]
        sc = sub.groupby(["belief", "meff"]).uar.mean()
        sc = sc[[k for k in sc.index if k[0] == "none" or k[0] in cand_b]]
        out[K] = sc.idxmax()
        mrows += [{"student": student, "K": K, "belief": b, "meff": e, "uar_pseudo": u, "chosen": (b, e) == out[K]} for (b, e), u in sc.items()]
    return out, mrows


def run_target(world, T, data_f, data_m, reps, out, cache):
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in WORLDS[world] + [world]}
    prows = []
    for c in S:                                                                  # 仮のターゲット
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c:
            continue
        info_c = {k: M5.pool_info(data_m[k] if k != world else data_f, pool_c, C_FIT, cache[k]) for k in sets}
        evc = data_f[c]
        ctxc = CB.Ctx(evc, PSEUDO_REPS, ["1", "5", "10", "25"], center="ind")
        for r in PSEUDO_REPS:
            pool = np.flatnonzero(~ctxc.te[r])
            te = np.flatnonzero(ctxc.te[r])
            pos_of = {int(i): j for j, i in enumerate(pool)}
            label_sets = {"0": np.array([], int), **{K: np.array([pos_of[int(i)] for i in ctxc.idx[(r, K)]]) for K in ("1", "5", "10", "25")}}
            stu = make_students(world, c, data_f, data_m, pool_c, pool, te, sets, info_c)
            solve_world(world, stu, evc.y[pool], evc.y[te], label_sets, prows, {"pseudo": c, "rep": r})
    pdf = pd.DataFrame(prows)
    pdf.to_csv(out / f"{T}_pseudo.csv", index=False)
    star, mrows = {}, []
    for student in WORLDS[world] + [world]:
        star[student], mr = choose(pdf, student)
        mrows += [{"target": T, **x} for x in mr]
    pd.DataFrame(mrows).to_csv(out / f"{T}_meta.csv", index=False)
    info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, C_FIT, cache[k]) for k in sets}
    ev = data_f[T]
    ctx = CB.Ctx(ev, reps, ["1", "5", "10", "25"], center="ind")
    rows = []
    for r in reps:
        pool = np.flatnonzero(~ctx.te[r])
        te = np.flatnonzero(ctx.te[r])
        pos_of = {int(i): j for j, i in enumerate(pool)}
        label_sets = {"0": np.array([], int), **{K: np.array([pos_of[int(i)] for i in ctx.idx[(r, K)]]) for K in ("1", "5", "10", "25")}}
        stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
        solve_world(world, stu, ev.y[pool], ev.y[te], label_sets, rows, {"target": T, "rep": r})
    pd.DataFrame(rows).to_csv(out / f"{T}.csv", index=False)


# ---------------------------------------------------------------- 判定（CPU）

def analyze(worlds=None, suffix="", results=None) -> int:
    """既定は第 1 エンコーダの 4 世界（e60_*.csv）．第 2 エンコーダの世界は --worlds fuse2-… --suffix _2nd で e60_2nd_*.csv に書く。"""
    worlds = tuple(worlds or WORLDS1)
    results = results or ROOT / "results"
    tabs = []
    for world in worlds:
        d = results / f"multimodal_distill_{world}"
        for x in sorted(f_ for f_ in d.glob("*.csv") if not f_.stem.endswith(("_pseudo", "_meta"))):
            T = x.stem
            a = pd.read_csv(x, dtype={"K": str})
            a = a[a.rep.isin(EC.REPS)]
            meta = pd.read_csv(d / f"{T}_meta.csv", dtype={"K": str})
            u = a.groupby(["student", "belief", "meff", "K"]).uar.mean() * 100
            for student in WORLDS[world] + [world]:
                ch = {str(r_.K): (r_.belief, int(r_.meff)) for r_ in meta[(meta.chosen) & (meta.student == student)].itertuples()}
                for K in KS:
                    b, e = ch[K]
                    row = {"world": world, "target": T, "student": student, "K": K, "b_star": b, "m_star": e,
                           "e39": float(u.loc[(student, "none", 0, K)]), "meta": float(u.loc[(student, b, e, K)]),
                           "p_ctrl": float(u.loc[(student, "p", e if e in (100, 500, 2000) else 100, K)])}
                    cands = [(b_, e_) for (s_, b_, e_, K_) in u.index if s_ == student and K_ == K and b_ not in ("none", "p")]
                    row["oracle"] = float(max(u.loc[(student, b_, e_, K)] for b_, e_ in cands))
                    for (s_, b_, e_, K_), v in u.items():
                        if s_ == student and K_ == K and b_ not in ("none",):
                            row[f"{b_}{e_}"] = float(v)
                    tabs.append(row)
    D = pd.DataFrame(tabs)
    tests = []
    for (world, student, K), g in D.groupby(["world", "student", "K"], sort=False):
        row = {"world": world, "student": student, "K": K, "targets": len(g), "e39": float(g.e39.mean()), "meta": float(g.meta.mean()),
               "p_ctrl": float(g.p_ctrl.mean()), "oracle": float(g.oracle.mean()),
               "meta-e39": float((g.meta - g.e39).mean()), "meta>e39": int((g.meta > g.e39).sum()),
               "p_ctrl-e39": float((g.p_ctrl - g.e39).mean()), "meta-p_ctrl": float((g.meta - g.p_ctrl).mean()), "oracle-e39": float((g.oracle - g.e39).mean())}
        for c in [c for c in g.columns if c not in ("world", "target", "student", "K", "b_star", "m_star", "e39", "meta", "p_ctrl", "oracle")]:
            if g[c].notna().all():
                row[f"{c}-e39"] = float((g[c] - g.e39).mean())
        tests.append(row)
    R = pd.DataFrame(tests)
    r = R.set_index(["world", "student", "K"])
    for c in ("M1", "M2", "M3"):
        R[c] = pd.Series(np.nan, index=R.index, dtype=object)                    # 真偽と NaN を混ぜるので object にしておく
    for world in worlds:
        for student in WORLDS[world] + [world]:
            if (world, student, "5") not in r.index:
                continue
            m = R[(R.world == world) & (R.student == student)]
            d05 = {K: r.loc[(world, student, K), "meta-e39"] for K in KS}
            R.loc[m.index, "M1"] = bool(d05["0"] > 0 and d05["5"] > 0) if student in DIMS else np.nan
            R.loc[m.index, "M2"] = bool(d05["0"] >= 0.5 and d05["5"] >= 0.5) if student not in DIMS else np.nan
            R.loc[m.index, "M3"] = bool(all(d05[K] >= -0.1 for K in KS))
    D.to_csv(OUT_REL / f"e60{suffix}_targets.csv", index=False)
    R.to_csv(OUT_REL / f"e60{suffix}_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(R[["world", "student", "K", "e39", "meta", "p_ctrl", "oracle", "meta-e39", "meta>e39", "p_ctrl-e39", "meta-p_ctrl"]].round(2).to_string(index=False))
    print("\n== 選ばれた (見込み, m_eff)（世界・生徒・K ごとのターゲットの数）")
    D["pair"] = D.b_star + D.m_star.astype(str)
    print(D.groupby(["world", "student", "K"]).pair.value_counts().unstack(fill_value=0).to_string())
    vc = ["M1", "M2", "M3"]
    print(R.drop_duplicates(["world", "student"])[["world", "student"] + vc].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world")
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--worlds", nargs="+", default=list(WORLDS1), help="--analyze の世界（既定は第 1 エンコーダの 4 世界）")
    ap.add_argument("--suffix", default="", help="--analyze の出力名の接尾辞（第 2 エンコーダは _2nd）")
    ap.add_argument("--results-dir", type=Path, help="試走用．--analyze が multimodal_distill_* を読む場所")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, help="試走用．既定は 0〜4")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.worlds, args.suffix, args.results_dir)
    global PSEUDO_REPS
    if args.pseudo_reps is not None:
        PSEUDO_REPS = args.pseudo_reps
    world = args.world
    out = args.out_dir or ROOT / "results" / f"multimodal_distill_{world}"
    out.mkdir(parents=True, exist_ok=True)
    groups = groups_for(world)
    todo = [t for t in args.target or groups if not (out / f"{t}.csv").exists()]
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
    prov = {"script": "exp/multimodal_distill.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": todo,
            "map_dir": str(map_dir(world)), "reps": args.reps, "world": world, "mods": WORLDS[world], "ks": KS, "pseudo_reps": PSEUDO_REPS,
            "c_fit": C_FIT, "bel1": BEL1, "meff1": MEFF1, "min2": MIN2, "bel2": BEL2, "meff2": MEFF2, "agree_low": AGREE_LOW,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    cache = {k: {} for k in WORLDS[world] + [world]}
    for t in todo:
        t0 = time.perf_counter()
        run_target(world, t, data_f, data_m, args.reps, out, cache)
        print(f"{world} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
