"""【本番の追加】E80 段 c（GPU）：kept.csv で残した教師のうち，E77 に行が無い教師の dist・kd・kdh・teacher の行を，E77 と同じ形で足す。

設計の正本は `docs/exploratory-analyses.md` の E80 の節の「関門（gate-probe…）を受けた書き直し」（段 c の作り方，P5-1 の情報をそろえる）と
「E80 の段 c の設計」．関門の求めは `checks/E79/compare.md` の追記の指摘 4，`checks/E80/code.md`・`checks/E80/probe.md` の段 c の項。

対象：kept.csv（--kept）で keep が真の（世界, T, 生徒, 教師, K）のうち，
  dist ：E77 の dist の教師（`compare_dai.world_unit` の run_c．モダリティの生徒 pF・pFK・ref・ownK，融合の生徒 poe0・ref・ref0・ownK）に無い教師
  kd・kdh：E77 の KD の教師（モダリティの生徒 pF・pFK・ref，融合の生徒 poe0・ref）に無い教師（P5-1 の C1：KD にも同じ教師を渡す）
  teacher：上のどちらかで足した教師のうち，E77 の teacher の行（pF・pFK・poe0・ref・ref0）に無いもの（教師そのもののテストの予測．生徒の列は世界）
K は keep が真の K だけ．IEMOCAP-script の kept の行が無ければ即興の行を使う（E80 の段 a は台本を即興の単位で測る．E79 の load_kept と同じ）。

行の形は E77 と同じ：dist（重みの付け方 bal・raw・hard0.5・0.7・0.9 × 総重み 100・500・2000・n．param は「教師:重み」），
kd・kdh（温度 2・4 × 総重み．param は「教師T温度」），teacher（param は教師，meff 0）．このほかに E77 と同じ形の none の行を書く
（仮のターゲットは UAR，評価のターゲットは対数の確率と W・b）．none の行は E79 が E77 の none と照合するためだけのもので（仮のターゲットは UAR の差 1e−9，
評価は対数の確率の差 1e−5），E79 は段 c の dist・kd・kdh・teacher の行だけを手法の候補にする．own などほかの手法の行は作らない。

単位と作り方：
  仮のターゲット c の単位（R4 の選択に要る）：E77 と同じ（c ∈ S，c と同じコーパスでないソース，反復 --pseudo-reps）．教師は段 a と同じ
    （`e80_teacher_j.unit_pass1`．poolC・mlp は同じ T の仮のターゲットのうち c と同じコーパスでない群で学ぶ．λ・種も段 a と同じ）．
    1 回目で全単位の教師を作り，poolC・mlp を学び，2 回目で生徒を作り直して（出発点と none が 1 回目とビット単位で同じことを確かめる）none・dist・KD を解く。
    IEMOCAP-script は同じ出力の場所の IEMOCAP-impro の仮のターゲットの結果を写す（E77 の --copy-pseudo と同じ考え．設定・kept の集合・引数の一致と，
    写し元の none・teachers の npz の鍵と teachers の配列の sha256 を確かめる）
  評価のターゲット T の単位（P5 に要る）：生徒・none・K 本は E77 と同じ（`make_students`，S の全群，`unit_pass1`）．poolC・mlp は
    T の仮のターゲットの全単位（S の全群の候補．全反復）で学び，λ は S のコーパスを 1 つずつ外して選ぶ（1 つのコーパスなら話者の 2 つの袋．段 a の choose_lam）。
    ds は T の候補の票から（ラベル無し）．agree・other は段 a と同じ
教師の配列：段 a の `teachers_for` で取り出した float32 の確率（段 a で J・UAR の差を測った配列そのもの）を float64 にして dist と KD の両方に渡す。

出力（--out-dir．既定 results/e80c_<世界>．E77 と同じ形）：
  `<T>.csv`（target, rep, row_id, student, method, param, meff, K, uar, commit），`<T>_r<r>_logp.npy`・`_pred.npy`・`_rowids.npy`・`_te.npy`・
  `_W_<生徒>.npy`・`_b_<生徒>.npy`・`_wrows_<生徒>.npy`，`<T>_r<r>_teachers.npz`（足した教師の候補の上の確率．鍵は「教師__K<K>」），
  `<T>_pseudo.csv`（pseudo, rep, row_id, …），`<T>_ps_<c>_r<r>_teachers.npz`，`<T>_conv.csv`・`<T>_convp.csv`，`<T>_samples.csv`，`<T>_settings.json`
  照合と記録：`<T>_ps_<c>_r<r>_none.npz`（E77 と同じ形の仮のターゲットの none），
  `<T>_teach_hash.csv`（dist・kd・kdh に渡した配列と保存した配列の sha256），`<T>_pool.csv`（λ の選び，MLP の種と損失），
  `<T>_ds.csv`（Dawid–Skene の反復の回数と最後の 1 回の変化），`<T>_plan.csv`（計算した（生徒, K, 教師, 手法））．provenance.json・provenance_log.jsonl
  照合（--check．GPU を使わない）：`check.csv`（none の E77 との一致，同じ配列，E77 と鍵が重ならないこと，E77 の教師の集合，段 a の教師の配列との一致）．
  不一致・状態が ok でない項目・許容幅（none の対数の確率 1e−5，仮のターゲットの none の UAR 1e−9，段 a の教師の配列 0）を超える項目が 1 つでもあれば終了の値 1

    ./.venv/bin/python exp/e80_stage_c.py --world fuse-audio-video-text --kept results/e80/kept.csv [--target …] [--reps …] [--pseudo-reps …] [--ks 0 5 25]
    ./.venv/bin/python exp/e80_stage_c.py --world fuse-audio-video-text --check --check-e77 results/e77_fuse-audio-video-text --check-e80a results/e80/avt
    ./.venv/bin/python exp/e80_stage_c.py --world fuse-audio-video --check --check-e77 results/e77_fuse-audio-video --check-e80a results/e80/av
  H100 の本番は既定の --gpu-fraction 0.3（本物の kept では 1 回に束ねる問が最大 480．A5000 の 0.25 では落ちる）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
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
import baselines_distill as BD  # noqa: E402
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import compare_dai as CD  # noqa: E402
import e80_teacher_j as EA  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import hard_rows, mass, temp_rows, tens  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import weak_rows  # noqa: E402
from geom_e0 import settings  # noqa: E402
from multimodal_distill import AGREE_LOW, WORLDS, blocks, make_students, sub_group  # noqa: E402

DT, DEV = EA.DT, EA.DEV
C_FIT = CD.C_FIT
MEFFS, WGTS, KD_TEMPS = CD.MEFFS, CD.WGTS, CD.KD_TEMPS
E77_DIST = {"mod": ("pF", "pFK", "ref", "ownK"), "fused": ("poe0", "ref", "ref0", "ownK")}   # compare_dai.py の world_unit の run_c（675・676 行）
E77_KD = {"mod": ("pF", "pFK", "ref"), "fused": ("poe0", "ref")}
E77_TEACHER_ROWS = ("pF", "pFK", "poe0", "ref", "ref0")                    # compare_dai.py の world_unit の teacher の行
SCRIPT_SRC = dict(CD.COPY_SRC)                                               # {"IEMOCAP-script": "IEMOCAP-impro"}
AGG = ("poolC", "mlp")
TOL_LOGP, TOL_UAR, TOL_TEACH = 1e-5, 1e-9, 0.0                  # --check の許容幅（E79 の none の照合と同じ．段 a の教師の配列は同じ値のはず）
GRID = {"meffs": list(MEFFS), "wgts": list(WGTS), "kd_temps": list(KD_TEMPS), "e77_dist": E77_DIST, "e77_kd": E77_KD,
        "e77_teacher_rows": list(E77_TEACHER_ROWS), "eval_mlp_seed": "crc32(world|T|eval)", "pseudo_mlp_seed": "crc32(world|T|c)（段 a と同じ）",
        "e80a": EA.GRID}


# ---------------------------------------------------------------- 部品

def sha(x):
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def ds_apply(votes, pi, Cs):
    """`exp/check_e60_refuse.py` の ds_apply と同じ（あの台本は読み込むと全部を計算するので import できない）。"""
    logp = np.log(pi)[None, :].repeat(len(votes), 0)
    for j in range(len(Cs)):
        logp += np.log(Cs[j][:, votes[:, j]].T)
    post = np.exp(logp - logp.max(1, keepdims=True))
    return post / post.sum(1, keepdims=True)


def ds_diag(stu, mods):
    """unit_pass1 の Dawid–Skene と同じ票で，反復の回数と，もう 1 回まわしたときの事後確率の変化の最大を返す（収束の記録）。"""
    votes = np.stack([stu[m]["P"].argmax(1) for m in mods], 1)
    p0, _, _ = EA.dawid_skene(votes, n_iter=EA.DS_ITERS)
    p1, _, _ = EA.dawid_skene(votes, n_iter=EA.DS_ITERS + 1)
    return {"ds_iters": EA.DS_ITERS, "ds_last_change": float(np.abs(p1 - p0).max())}


def rows_of(P, wgt, meff, n):
    """`compare_dai.run_c` の rows_of と同じ。"""
    if wgt == "bal":
        return weak_rows(P, mass(meff, n))
    if wgt == "raw":
        return temp_rows(P, 1, meff)
    return hard_rows(P, wgt[4:], meff)


def role(world, k):
    return "fused" if k == world else "mod"


def r1_names(world, k):
    """R1 の教師の名前（段 a の teachers_for の鍵．順も同じ）。"""
    mods = WORLDS[world]
    dummy = {"poe0": 0, "agree": 0, "ds": 0, "P": {m: 0 for m in mods + [world]}, "none": {world: {"0": 0}}, "ref": {"0": 0}, "ref0": {"0": 0}}
    return list(EA.teachers_for(world, mods, dummy, k, "0", 0, 0))


# ---------------------------------------------------------------- kept.csv から計算する組

def read_kept(path, world):
    k = pd.read_csv(path, dtype={"K": str, "world": str, "target": str, "student": str, "teacher": str})
    need = {"world", "target", "student", "teacher", "K", "keep"}
    if not need <= set(k.columns):
        raise SystemExit(f"{path} に列 {sorted(need - set(k.columns))} が無い")
    if k.duplicated(["world", "target", "student", "teacher", "K"]).any():
        raise SystemExit(f"{path} に（世界, 評価のターゲット, 生徒, 教師, K）の重複がある")
    k["keep"] = k.keep.map(lambda v: v is True or str(v).strip().lower() in ("true", "1"))
    return k[k.world == world].reset_index(drop=True)


def make_plan(kw, world, T, ks, trial, notes):
    """（生徒 → {"dist": [(教師, K)], "kd": [(教師, K)]}），teacher の行の [(教師, K)] と，使った kept の行。"""
    rows = kw[kw.target == T]
    if rows.empty and T in SCRIPT_SRC:
        rows = kw[kw.target == SCRIPT_SRC[T]]
        notes.append(f"{T} の kept の行が無いので {SCRIPT_SRC[T]} の行を使った")
    if rows.empty:
        raise SystemExit(f"kept に {world} {T} の行が無い")
    bad = rows[rows.keep & ~rows.K.isin(ks)]
    if len(bad) and not trial:
        raise SystemExit(f"{world} {T}：keep が真の行の K {sorted(set(bad.K))} が --ks {list(ks)} に無い")
    plan, used = {}, []
    for k in [world] + WORLDS[world]:
        names = r1_names(world, k)
        plan[k] = {"dist": [], "kd": []}
        for K in ks:
            g = rows[(rows.student == k) & (rows.K == K)]
            if g.empty and not trial:
                raise SystemExit(f"kept に {world} {T} の生徒 {k}（K={K}）の行が無い")
            odd = sorted(set(g.teacher) - set(names))
            if odd:
                raise SystemExit(f"kept の教師 {odd} は生徒 {k} の R1 の候補に無い")
            kept = [n for n in names if n in trial] if trial else [n for n in names if n in set(g.teacher[g.keep])]
            for n in kept:
                meth = []
                if n not in E77_DIST[role(world, k)]:
                    plan[k]["dist"].append((n, K))
                    meth.append("dist")
                if n not in E77_KD[role(world, k)]:
                    plan[k]["kd"].append((n, K))
                    meth += ["kd", "kdh"]
                used.append({"student": k, "K": K, "teacher": n, "methods": " ".join(meth) or "-（E77 にある）"})
    order = {n: i for i, n in enumerate(dict.fromkeys(n for k in plan for n in r1_names(world, k)))}
    trows = sorted({(n, K) for k in plan for meth in ("dist", "kd") for n, K in plan[k][meth] if n not in E77_TEACHER_ROWS},
                   key=lambda x: (list(ks).index(x[1]), order[x[0]]))
    return plan, trows, used


def plan_names(plan, trows):
    return sorted({(n, K) for k in plan for meth in ("dist", "kd") for n, K in plan[k][meth]} | set(trows))


def teacher_array(world, u, n, K, qC, qM):
    """段 a の teachers_for から，n を候補に持つ生徒で取り出す（同じ名前は生徒によらず同じ配列）。"""
    mods = WORLDS[world]
    for k in [world] + mods:
        t = EA.teachers_for(world, mods, u, k, K, qC, qM)
        if n in t:
            return t[n]
    raise KeyError(n)


# ---------------------------------------------------------------- poolC・mlp

def fit_agg(world, keys, units, tag, seed):
    """段 a の run_target の 2 回目と同じ手順（choose_lam → pool_fit → stack_members → fit_mlp）。"""
    mods = WORLDS[world]
    t0 = time.perf_counter()
    lam, mode, sel = EA.choose_lam(mods, keys, units, tag)
    mem = [(EA.feat(units[k], mods), units[k]["y"]) for k in keys]
    poolC = EA.pool_fit(mods, mem, lam)
    Fp, yp, w = EA.stack_members(mem)
    mlp, mloss = EA.fit_mlp(Fp.reshape(len(Fp), -1), yp, w, seed)
    rec = {**tag, "what": "final", "mode": mode, "lam": lam, "members": " ".join(dict.fromkeys(c for c, _ in keys)), "n_units": len(keys),
           "n_rows": len(yp), "mlp_seed": seed, "mlp_loss": mloss, "sec": time.perf_counter() - t0}
    return (poolC, mlp), sel + [rec]


def agg_pred(agg, F):
    poolC, mlp = agg
    qC = EA.pool_pred(poolC, F).astype(np.float32)
    with torch.no_grad():
        Ft = torch.as_tensor(F, dtype=DT, device=DEV)
        qM = EA.f32(torch.softmax(mlp(Ft.reshape(len(Ft), -1)), 1))
    return qC, qM


# ---------------------------------------------------------------- 1 つの単位の dist・KD・teacher の行

def none_rows(world, prs, rec, u, ks):
    """E77 と同じ形の none の行（compare_dai.run_ab の 1 行目と同じ束ね方で解き，Rec.add で足す）．E79 が E77 の none と照合するためだけの行で，
    手法の候補にはならない．値が 1 回目（unit_pass1）の none とビット単位で同じことを確かめる（同じ生徒を作り直したことの照合も兼ねる）。"""
    for k in [world] + WORLDS[world]:
        sol = prs[k].solve([(K, None, K) for K in ks], what="none")
        for K in ks:
            W, b = sol[K]
            lp = EA.f32(torch.log_softmax(prs[k].s["Zte"] @ W.T + b, 1))
            if not np.array_equal(lp, u["none_te"][k][K]):
                raise SystemExit(f"{rec.tag} の生徒 {k}（K={K}）の none が 1 回目と違う")
            rec.add(k, "none", "-", "0", K, Wb=(W, b), Zte=prs[k].s["Zte"])


def solve_unit(world, prs, rec, plan, trows, u, agg, hrows, htag):
    """plan の dist・kd・kdh を解き，teacher の行（テストの上の教師の予測）を足す．戻りは保存する教師の配列 {"教師__K<K>": float64}。"""
    mods = WORLDS[world]
    qC = qM = None
    if agg is not None:
        qC, qM = agg_pred(agg, EA.feat(u, mods))
    arrs = {(n, K): tens(teacher_array(world, u, n, K, qC, qM)) for n, K in plan_names(plan, [])}
    hs = {key: sha(t) for key, t in arrs.items()}
    for k, pl in plan.items():
        pr, Zte = prs[k], prs[k].s["Zte"]
        n_ = pr.n
        if pl["kd"]:                                                             # compare_dai.run_c と同じ並び（教師 → 温度 → 総重み → K）
            names = list(dict.fromkeys(n for n, _ in pl["kd"]))
            kks = {n: [K for m, K in pl["kd"] if m == n] for n in names}
            items = [((n, T, meff, K), temp_rows(arrs[(n, K)], T, meff), K) for n in names for T in KD_TEMPS for meff in MEFFS for K in kks[n]]
            for (n, T, meff, K), Wb in pr.solve(items, what="kd").items():
                rec.add(k, "kd", f"{n}T{T}", meff, K, Wb=Wb, Zte=Zte)
            items = [((n, T, meff, K), arrs[(n, K)], T, meff, K) for n in names for T in KD_TEMPS for meff in MEFFS for K in kks[n]]
            for (n, T, meff, K), Wb in CD.kdh_solve(pr, items).items():
                rec.add(k, "kdh", f"{n}T{T}", meff, K, Wb=Wb, Zte=Zte)
            for n, K in pl["kd"]:
                hrows += [{**htag, "student": k, "teacher": n, "K": K, "method": m_, "sha256": sha(arrs[(n, K)])} for m_ in ("kd", "kdh")]
        if pl["dist"]:
            names = list(dict.fromkeys(n for n, _ in pl["dist"]))
            kks = {n: [K for m, K in pl["dist"] if m == n] for n in names}
            items = [((n, wgt, meff, K), rows_of(arrs[(n, K)], wgt, meff, n_), K) for n in names for wgt in WGTS for meff in MEFFS for K in kks[n]]
            for (n, wgt, meff, K), Wb in pr.solve(items, what="dist").items():
                rec.add(k, "dist", f"{n}:{wgt}", meff, K, Wb=Wb, Zte=Zte)
            hrows += [{**htag, "student": k, "teacher": n, "K": K, "method": "dist", "sha256": sha(arrs[(n, K)])} for n, K in pl["dist"]]
    if trows:                                                                    # 教師そのもののテストの予測（E77 の teacher の行と同じ形）
        Lte = {m: torch.log_softmax(prs[m].s["Zte"] @ prs[m].W0.T + prs[m].b0, 1) for m in mods}
        P32 = {m: EA.f32(Lte[m].exp()) for m in mods}
        Fte = np.log(np.clip(np.stack([P32[m] for m in mods], 1).astype(np.float64), 1e-12, None))   # 段 a の feat と同じ丸め
        votes = np.stack([Lte[m].argmax(1).cpu().numpy() for m in mods], 1)                  # check_e60_refuse.py の votes_te と同じ（float64 の argmax）
        pF_te = torch.softmax(prs[world].s["Zte"] @ prs[world].W0.T + prs[world].b0, 1).cpu().numpy()
        aggte = agg_pred(agg, Fte) if agg is not None else (None, None)
        for n, K in trows:
            if n.startswith("other_"):
                m = n[len("other_"):]
                rec.add(world, "teacher", n, "0", K, logits=prs[m].s["Zte"] @ prs[m].W0.T + prs[m].b0)
            elif n in AGG:
                rec.add(world, "teacher", n, "0", K, probs=aggte[AGG.index(n)])
            elif n == "ds":
                rec.add(world, "teacher", n, "0", K, probs=ds_apply(votes, u["ds_pi"], list(u["ds_C"])))
            elif n == "agree":
                rec.add(world, "teacher", n, "0", K, probs=pF_te * np.where((votes == votes[:, :1]).all(1), 1.0, AGREE_LOW)[:, None])
            else:
                raise SystemExit(f"teacher の行の作り方が無い教師 {n}")
    return {f"{n}__K{K}": t.detach().cpu().numpy() for (n, K), t in arrs.items()}, hs


def check_saved(f, saved, hs, hrows, htag):
    """保存した教師の配列を読み直し，dist・KD に渡した配列と sha256 が同じことを確かめる（C1・C6）。"""
    z = np.load(f, allow_pickle=False)
    for key in saved:
        n, K = key.rsplit("__K", 1)
        h = sha(z[key])
        if h != hs[(n, K)]:
            raise SystemExit(f"{f} の {key} が dist・KD に渡した配列と違う")
        hrows.append({**htag, "student": "*", "teacher": n, "K": K, "method": "saved", "sha256": h})


# ---------------------------------------------------------------- 仮のターゲットの単位

def pseudo_pass_a(world, T, data_fT, data_mT, S, sets, pseudo_reps, ks, cache, conv, convp, dsr):
    """段 a の run_target の 1 回目と同じ．戻りは units（(c, r) → 段 a の u に pool・te・person を足したもの）と c ごとの (ctx, pool_c, info)。"""
    mods = WORLDS[world]
    units, ctxs = {}, {}
    for c in S:
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c:
            continue
        assert CM.corp(c) != CM.corp(T) and all(CM.corp(s) != CM.corp(T) for s in pool_c)
        info_c = {k: M5.pool_info(data_mT[k] if k != world else data_fT, pool_c, C_FIT, cache[k]) for k in sets}
        evc = data_fT[c]
        ctxc = CB.Ctx(evc, pseudo_reps, list(EA.KS_LAB_ALL), center="ind")
        ctxs[c] = (ctxc, pool_c, info_c)
        for r in pseudo_reps:
            t0 = time.perf_counter()
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            stu = make_students(world, c, data_fT, data_mT, pool_c, pool, te, sets, info_c)
            u, prs = EA.unit_pass1(world, stu, evc.y[pool], evc.y[te], BD.label_sets_of(ctxc, r, pool), ks)
            u.update({"person": np.asarray(evc.person)[pool], "pool": pool, "te": te})
            tag = {"pseudo": c, "rep": r}
            dsr.append({**tag, **ds_diag(stu, mods)})
            for k, pr in prs.items():
                conv += [{**tag, "student": k, "pass": "a", **s_} for s_ in pr.stats]
                convp += [{**tag, "student": k, "pass": "a", **s_} for s_ in pr.per]
            units[(c, r)] = u
            print(f"  {world} {T} 仮 {c} r{r}：1 回目　候補 {len(pool)}　{time.perf_counter() - t0:.0f} 秒", flush=True)
            del stu, prs
            torch.cuda.empty_cache()
    return units, ctxs


def pseudo_pass_b(world, T, data_fT, data_mT, sets, units, ctxs, aggs, plan, trows, out, conv, convp, hrows, ks):
    """生徒を作り直し（出発点と none が 1 回目と同じことを確かめる），none（照合用）・dist・KD・teacher の行を解く。"""
    mods = WORLDS[world]
    prows = []
    for (c, r), u in units.items():
        t0 = time.perf_counter()
        ctxc, pool_c, info_c = ctxs[c]
        evc = data_fT[c]
        pool, te = u["pool"], u["te"]
        stu = make_students(world, c, data_fT, data_mT, pool_c, pool, te, sets, info_c)
        for k in mods + [world]:
            if not np.array_equal(EA.f32(stu[k]["P"]), u["P"][k]):
                raise SystemExit(f"{T} 仮 {c} r{r}：作り直した生徒 {k} の出発点が 1 回目と違う")
        prs = {k: CD.Prob2(stu[k], evc.y[pool], BD.label_sets_of(ctxc, r, pool)) for k in mods + [world]}
        rec = CD.Rec({"pseudo": c, "rep": r}, evc.y[te], keep=False)
        rec.conv, rec.convp = conv, convp
        htag = {"pseudo": c, "rep": r}
        none_rows(world, prs, rec, u, ks)                                        # E77 と同じ none の行（rec.none_lp にも入る）
        saved, hs = solve_unit(world, prs, rec, plan, trows, u, aggs.get(c), hrows, htag)
        f = out / f"{T}_ps_{c}_r{r}_teachers.npz"
        np.savez(f, **saved)
        check_saved(f, saved, hs, hrows, htag)
        rec.save_pseudo(out, T, c, r, te)
        for k, pr in prs.items():
            conv += [{**rec.tag, "student": k, "pass": "b", **s_} for s_ in pr.stats]
            convp += [{**rec.tag, "student": k, "pass": "b", **s_} for s_ in pr.per]
        prows += rec.rows
        print(f"  {world} {T} 仮 {c} r{r}：2 回目　{len(rec.rows)} 行　{time.perf_counter() - t0:.0f} 秒", flush=True)
        del stu, prs, rec
        torch.cuda.empty_cache()
    return prows


def copy_pseudo_c(out, T, commit, pairs, args_now, expect_none, expect_teach):
    """IEMOCAP-script の仮のターゲットを，同じ出力の場所の即興の結果から写す（E77 の copy_pseudo と同じ考え）．
    止める条件：写し元の `<src>.csv` が無いか今と違うコミット，`_settings.json`（設定・ソース・kept の集合・計算する組）の不一致，
    写し元を回したときの引数の不一致，（仮のターゲット, 反復）の組の不一致，写し元の none・teachers の npz が無いか鍵が期待
    （expect_none：「生徒|K」，expect_teach：「教師__K<K>」）とそろわない，teachers の配列の sha256 が写し元の `_teach_hash.csv` の saved の行と違う。"""
    src = SCRIPT_SRC[T]

    def stop(m):
        raise SystemExit(f"{T} の写し：{m}")
    f0 = out / f"{src}.csv"
    if not f0.exists():
        stop(f"写し元 {f0} が無い（写し元を先に回す）")
    if set(pd.read_csv(f0, usecols=["commit"]).commit.astype(str)) != {commit}:
        stop("写し元のコミットが今と違う")
    a_, b_ = (json.loads((out / f"{g}_settings.json").read_text()) for g in (src, T))
    if a_ != b_:
        stop("設定・ソース・kept の集合・計算する組のどれかが違う")
    ent = [json.loads(x) for x in (out / "provenance_log.jsonl").read_text().splitlines() if x.strip()]
    ent = [e for e in ent if e.get("commit") == commit and src in e.get("targets", [])]
    if not ent:
        stop("写し元を回したときの記録（provenance_log.jsonl）が無い")
    for k, v in args_now.items():
        if json.dumps(ent[-1].get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
            stop(f"写し元の引数 {k}={ent[-1].get(k)} が今 {v} と違う")
    d = pd.read_csv(out / f"{src}_pseudo.csv", dtype={"K": str, "param": str, "meff": str})
    if set(zip(d.pseudo.astype(str), d.rep.astype(int))) != set(pairs):
        stop("（仮のターゲット, 反復）の組がそろわない")
    d = d.rename(columns={"commit": "src_commit"}).assign(copied_from=src)
    extra = {}
    for name in ("conv", "convp", "pool", "ds", "teach_hash"):
        e = pd.read_csv(out / f"{src}_{name}.csv", dtype={"K": str})
        e = e[e.pseudo.notna() & (e.pseudo.astype(str) != "*eval*")] if "pseudo" in e.columns else e.iloc[:0]
        extra[name] = e.rename(columns={"commit": "src_commit"}).assign(copied_from=src).to_dict("records")
    hsrc = pd.read_csv(out / f"{src}_teach_hash.csv", dtype={"K": str, "pseudo": str})
    hsrc = hsrc[(hsrc.method == "saved") & hsrc.pseudo.notna()]
    for c, r in pairs:
        gs = {suf: out / f"{src}_ps_{c}_r{r}_{suf}.npz" for suf in ("none", "teachers")}
        for g in gs.values():
            if not g.exists():
                stop(f"写し元の {g.name} が無い")
        zn = np.load(gs["none"], allow_pickle=False)
        if set(map(str, zn["keys"])) != set(expect_none):
            stop(f"写し元の {gs['none'].name} の鍵 {sorted(map(str, zn['keys']))} が期待 {sorted(expect_none)} とそろわない")
        zt = np.load(gs["teachers"], allow_pickle=False)
        if set(zt.files) != set(expect_teach):
            stop(f"写し元の {gs['teachers'].name} の鍵 {sorted(zt.files)} が期待 {sorted(expect_teach)} とそろわない")
        for key in zt.files:
            n, K = key.rsplit("__K", 1)
            hh = hsrc[(hsrc.pseudo == c) & (hsrc.rep == r) & (hsrc.teacher == n) & (hsrc.K == K)].sha256.unique()
            if len(hh) != 1 or hh[0] != sha(zt[key]):
                stop(f"写し元の {gs['teachers'].name} の {key} が写し元の記録の sha256 と違う")
        for g in gs.values():
            shutil.copyfile(g, out / g.name.replace(f"{src}_ps_", f"{T}_ps_", 1))
    print(f"{T}：仮のターゲットを {src} から写した（{len(d)} 行，組 {len(pairs)}）", flush=True)
    return d.to_dict("records"), extra


# ---------------------------------------------------------------- 評価のターゲットごと

def run_target(world, T, data_f, data_m, args, ks, out, cache, commit, kw, kept_sha, args_now, unit_cache, prov):
    mods = WORLDS[world]
    notes = []
    plan, trows, used = make_plan(kw, world, T, ks, args.trial_teachers, notes)
    if not any(plan[k][m] for k in plan for m in ("dist", "kd")):
        prov["skipped"][T] = "E77 に無い残した教師が無い"
        print(f"{world} {T}：計算する組が無い（飛ばす）", flush=True)
        return
    S = [s for s in data_f if CM.corp(s) != CM.corp(T)]
    sets = {k: settings(k, T) for k in mods + [world]}
    CD.save_settings(out, T, sets, {"c_fit": C_FIT, "sources": S, "kept_sha256": kept_sha, "trial_teachers": args.trial_teachers,
                                    "plan": {k: {m: [list(x) for x in v] for m, v in pl.items()} for k, pl in plan.items()},
                                    "teacher_rows": [list(x) for x in trows]})
    CD.samples_table(world, T).to_csv(out / f"{T}_samples.csv", index_label="row")
    pd.DataFrame(used).to_csv(out / f"{T}_plan.csv", index=False)
    needs_agg = any(n in AGG for n, _ in plan_names(plan, trows))
    data_fT = {g: v for g, v in data_f.items() if CM.corp(g) != CM.corp(T)}     # 仮のターゲットの単位では T のコーパスの群に触れない
    data_mT = {k: {g: v for g, v in d.items() if CM.corp(g) != CM.corp(T)} for k, d in data_m.items()}
    assert T not in data_fT and all(CM.corp(g) != CM.corp(T) for g in data_fT)
    assert all(CM.corp(g) != CM.corp(T) for d in data_mT.values() for g in d)
    pairs = CD.pseudo_pairs(S, None, args.pseudo_reps)
    conv, convp, poolr, dsr, hrows = [], [], [], [], []
    ukey = (tuple(S), json.dumps(sets, sort_keys=True, default=str), tuple(args.pseudo_reps), tuple(ks))
    units = None
    if T in SCRIPT_SRC and not args.no_copy_pseudo:
        prows, extra = copy_pseudo_c(out, T, commit, pairs, args_now, [f"{k}|{K}" for k in [world] + mods for K in ks],
                                     [f"{n}__K{K}" for n, K in plan_names(plan, [])])
        conv, convp, poolr, dsr, hrows = extra["conv"], extra["convp"], extra["pool"], extra["ds"], extra["teach_hash"]
        if needs_agg:                                                            # 評価のターゲットの poolC・mlp を学ぶ単位（写し元と同じ）
            units = unit_cache.get(ukey)
            if units is None:
                units, _ = pseudo_pass_a(world, T, data_fT, data_mT, S, sets, args.pseudo_reps, ks, cache, [], [], [])
    else:
        units, ctxs = pseudo_pass_a(world, T, data_fT, data_mT, S, sets, args.pseudo_reps, ks, cache, conv, convp, dsr)
        if sorted(units) != sorted(pairs):
            raise SystemExit(f"{T}：仮のターゲットの単位 {sorted(units)} が期待 {sorted(pairs)} とそろわない")
        aggs = {}
        if needs_agg:                                                            # 段 a と同じ：c と同じコーパスでない群で学ぶ
            pcs = sorted({c for c, _ in units}, key=S.index)
            for c in pcs:
                others = [c2 for c2 in pcs if CM.corp(c2) != CM.corp(c)]
                assert all(CM.corp(c2) != CM.corp(T) for c2 in others)
                keys = [k for k in units if k[0] in others]
                if not keys:
                    raise SystemExit(f"{T} 仮 {c}：poolC・mlp を学ぶ群が無い")
                aggs[c], rr = fit_agg(world, keys, units, {"pseudo": c}, zlib.crc32(f"{world}|{T}|{c}".encode()) & 0x7FFFFFFF)
                poolr += rr
        prows = pseudo_pass_b(world, T, data_fT, data_mT, sets, units, ctxs, aggs, plan, trows, out, conv, convp, hrows, ks)
        if T in SCRIPT_SRC.values():
            unit_cache.clear()
            unit_cache[ukey] = units
    agg_T = None
    if needs_agg:                                                                # 段 c の作り方：S の全群の候補（T の仮のターゲットの全単位）で学ぶ
        keys = list(units)
        assert all(CM.corp(c) != CM.corp(T) for c, _ in keys)
        agg_T, rr = fit_agg(world, keys, units, {"pseudo": "*eval*"}, zlib.crc32(f"{world}|{T}|eval".encode()) & 0x7FFFFFFF)
        poolr += rr
    del units
    info = {k: M5.pool_info(data_m[k] if k != world else data_f, S, C_FIT, cache[k]) for k in sets}
    ev = data_f[T]
    ctx = CB.Ctx(ev, args.reps, list(EA.KS_LAB_ALL), center="ind")
    rows = []
    for r in args.reps:
        t0 = time.perf_counter()
        pool, te = np.flatnonzero(~ctx.te[r]), np.flatnonzero(ctx.te[r])
        stu = make_students(world, T, data_f, data_m, S, pool, te, sets, info)
        u, prs = EA.unit_pass1(world, stu, ev.y[pool], ev.y[te], BD.label_sets_of(ctx, r, pool), ks)
        rec = CD.Rec({"target": T, "rep": r}, ev.y[te], keep=True)
        rec.conv, rec.convp = conv, convp
        dsr.append({**rec.tag, **ds_diag(stu, mods)})
        for k in mods + [world]:                                                 # compare_dai.world_unit と同じ（中心は make_students と同じ候補の平均）
            dat = data_m[k][T] if k != world else ev
            st = stu[k]["st"]
            rec.xform[k] = (st.mats[stu[k]["rho_m"]][0], st.src.sd, M.rows(dat.X, pool).mean(0) + st.src.mu)
        htag = {"target": T, "rep": r}
        none_rows(world, prs, rec, u, ks)                                        # E77 と同じ none の行（対数の確率と W・b も E77 と同じ形で残る）
        saved, hs = solve_unit(world, prs, rec, plan, trows, u, agg_T, hrows, htag)
        rec.teachers = saved
        rec.save(out, T, r, te)
        check_saved(out / f"{T}_r{r}_teachers.npz", saved, hs, hrows, htag)
        CD.flush_stats(rec, prs)
        rows += rec.rows
        print(f"  {world} {T} r{r}：評価　{len(rec.rows)} 行　{time.perf_counter() - t0:.0f} 秒", flush=True)
        del stu, prs, rec, u
        torch.cuda.empty_cache()
    for name, rows_ in (("pseudo", prows), ("conv", conv), ("convp", convp), ("pool", poolr), ("ds", dsr), ("teach_hash", hrows)):
        pd.DataFrame(rows_).assign(commit=commit).to_csv(out / f"{T}_{name}.csv", index=False)
    d = pd.DataFrame(rows).assign(commit=commit)
    if d.duplicated(["rep", "student", "method", "param", "meff", "K"]).any():
        raise SystemExit(f"{T}：行の鍵が重なる")
    d.to_csv(out / f"{T}.csv", index=False)                                      # 最後に書く（再開の印）
    if notes:
        prov.setdefault("notes", {})[T] = notes


# ---------------------------------------------------------------- 照合（GPU を使わない）

def run_checks(out, world, e77, e80a):
    """出力を照合して check.csv を書く．不一致・状態が ok でない項目・許容幅を超える項目が 1 つでもあれば 1 を返す（終了の値）。"""
    rows = []
    mods = WORLDS[world]
    ks_run = [str(k) for k in json.loads((out / "provenance.json").read_text()).get("KS", [])]
    for fs in sorted(out.glob("*_settings.json")):
        T = fs.name[: -len("_settings.json")]
        if not (out / f"{T}.csv").exists():
            continue
        h = pd.read_csv(out / f"{T}_teach_hash.csv", dtype={"K": str, "pseudo": str, "target": str})
        h["unit"] = h.pseudo.fillna("*eval*") if "pseudo" in h.columns else "*eval*"
        g = h.groupby(["unit", "rep", "teacher", "K"]).sha256.nunique()
        has_saved = h[h.method == "saved"].groupby(["unit", "rep", "teacher", "K"]).size()
        bad = int((g != 1).sum()) + int(len(g) - len(has_saved))
        rows.append({"check": "same_array", "target": T, "n": len(g), "bad": bad, "ok": bad == 0})
        bad = 0
        for f in sorted(out.glob(f"{T}_ps_*_teachers.npz")) + sorted(out.glob(f"{T}_r*_teachers.npz")):
            mt = re.match(rf"^{re.escape(T)}_(?:ps_(.+)_)?r(\d+)_teachers\.npz$", f.name)
            unit, r = (mt.group(1) or "*eval*"), int(mt.group(2))
            z = np.load(f, allow_pickle=False)
            for key in z.files:
                n, K = key.rsplit("__K", 1)
                hh = h[(h.unit == unit) & (h.rep == r) & (h.teacher == n) & (h.K == K)].sha256.unique()
                bad += int(len(hh) != 1 or hh[0] != sha(z[key]))
        rows.append({"check": "saved_file_hash", "target": T, "bad": bad, "ok": bad == 0})
        own = {nm: pd.read_csv(out / nm, dtype={"K": str, "param": str, "meff": str}) for nm in (f"{T}.csv", f"{T}_pseudo.csv")}
        for nm, kk in ((f"{T}.csv", ["rep"]), (f"{T}_pseudo.csv", ["pseudo", "rep"])):   # none の行が全部の（単位, 生徒, K）にあるか
            x = own[nm]
            want = {(*u_, st_, K) for u_ in x[kk].drop_duplicates().itertuples(index=False) for st_ in [world] + mods for K in ks_run}
            have = set(x[x.method == "none"][kk + ["student", "K"]].itertuples(index=False, name=None))
            miss = len(want - have) + len(have - want)
            rows.append({"check": "none_rows_present", "target": T, "file": nm, "n": len(want), "bad": miss, "ok": miss == 0})
        if e77:
            for f in sorted(out.glob(f"{T}_ps_*_none.npz")):                    # 仮のターゲットの none の対数の確率（E77 と同じ形の npz）
                g77 = e77 / f.name
                if not g77.exists():
                    rows.append({"check": "none_pseudo_logp", "target": T, "file": f.name, "status": "E77 に無い", "ok": False})
                    continue
                a, b = np.load(f), np.load(g77)
                kb = {str(x): i for i, x in enumerate(b["keys"])}
                same_te = bool(np.array_equal(a["te"], b["te"]) and np.array_equal(a["y"], b["y"]))
                for i, key in enumerate(a["keys"]):
                    if str(key) not in kb or not same_te:
                        rows.append({"check": "none_pseudo_logp", "target": T, "file": f.name, "key": str(key),
                                     "status": "E77 に鍵が無い" if same_te else "te・y が違う", "ok": False})
                        continue
                    x, y = a[f"lp{i}"], b[f"lp{kb[str(key)]}"]
                    v = float(np.abs(x - y).max())
                    rows.append({"check": "none_pseudo_logp", "target": T, "file": f.name, "key": str(key), "status": "ok", "max_abs": v,
                                 "pred_agree": float((x.argmax(1) == y.argmax(1)).mean()), "n": len(x), "ok": v <= TOL_LOGP})
            p77 = e77 / f"{T}_pseudo.csv"                                        # 仮のターゲットの none の UAR（E79 の照合と同じ）
            if p77.exists():
                b = pd.read_csv(p77, dtype={"K": str, "param": str, "meff": str})
                kk = ["pseudo", "rep", "student", "K"]
                x = own[f"{T}_pseudo.csv"]
                j = x[x.method == "none"].set_index(kk).uar.rename("c").to_frame().join(
                    b[b.method == "none"].set_index(kk).uar.rename("e"), how="left")
                nmiss = int(j.e.isna().sum())
                v = float((j.c - j.e).abs().max()) if len(j) and nmiss < len(j) else np.nan
                rows.append({"check": "none_pseudo_uar", "target": T, "n": len(j), "bad": nmiss, "max_abs": v,
                             "ok": nmiss == 0 and len(j) > 0 and v <= TOL_UAR})
            else:
                rows.append({"check": "none_pseudo_uar", "target": T, "status": "E77 に無い", "ok": False})
            f77 = e77 / f"{T}.csv"                                               # 評価のターゲットの none の対数の確率（E79 の照合と同じ）
            x = own[f"{T}.csv"]
            d77 = pd.read_csv(f77, dtype={"K": str, "param": str, "meff": str}) if f77.exists() else None
            for r in sorted(set(x.rep)):
                if d77 is None or not (e77 / f"{T}_r{r}_logp.npy").exists():
                    rows.append({"check": "none_eval_logp", "target": T, "rep": r, "status": "E77 に無い", "ok": False})
                    continue
                if not np.array_equal(np.load(out / f"{T}_r{r}_te.npy"), np.load(e77 / f"{T}_r{r}_te.npy")):
                    rows.append({"check": "none_eval_logp", "target": T, "rep": r, "status": "te が違う", "ok": False})
                    continue
                L7 = np.load(e77 / f"{T}_r{r}_logp.npy", mmap_mode="r")
                L8 = np.load(out / f"{T}_r{r}_logp.npy", mmap_mode="r")
                a_ = d77[(d77.rep == r) & (d77.method == "none")].set_index(["student", "K"]).row_id
                b_ = x[(x.rep == r) & (x.method == "none")].set_index(["student", "K"]).row_id
                for key in b_.index:
                    if key not in a_.index:
                        rows.append({"check": "none_eval_logp", "target": T, "rep": r, "key": "|".join(key), "status": "E77 に行が無い", "ok": False})
                        continue
                    p_, q_ = np.asarray(L8[int(b_[key])]), np.asarray(L7[int(a_[key])])
                    v = float(np.abs(p_ - q_).max())
                    rows.append({"check": "none_eval_logp", "target": T, "rep": r, "key": "|".join(key), "status": "ok", "max_abs": v,
                                 "pred_agree": float((p_.argmax(1) == q_.argmax(1)).mean()), "n": len(p_), "ok": v <= TOL_LOGP})
            for nm, kk in ((f"{T}.csv", ["rep", "student", "method", "param", "meff", "K"]),
                           (f"{T}_pseudo.csv", ["pseudo", "rep", "student", "method", "param", "meff", "K"])):
                if (e77 / nm).exists():                                          # E79 が手法の候補にする行（none を除く）が E77 と重ならないか
                    a = own[nm]
                    a = a[a.method.isin(("dist", "kd", "kdh", "teacher"))][kk].astype(str)
                    odd = int((~own[nm].method.isin(("dist", "kd", "kdh", "teacher", "none"))).sum())
                    b = pd.read_csv(e77 / nm, dtype={"K": str, "param": str, "meff": str})[kk].astype(str)
                    nb = len(a.merge(b, on=kk)) + odd
                    rows.append({"check": "key_overlap", "target": T, "file": nm, "bad": nb, "ok": nb == 0})
            if p77.exists():
                b = pd.read_csv(p77, dtype={"K": str, "param": str, "meff": str})
                for st_, g_ in b.groupby("student"):
                    rl = role(world, st_)
                    dd = set(g_[g_.method == "dist"].param.str.split(":").str[0])
                    kd = set(g_[g_.method == "kd"].param.str.replace(r"T\d+$", "", regex=True))
                    nb = int(dd != set(E77_DIST[rl])) + int(kd != set(E77_KD[rl]))
                    rows.append({"check": "e77_teacher_sets", "target": T, "key": st_, "bad": nb, "ok": nb == 0})
        if e80a:
            ta = SCRIPT_SRC.get(T, T)
            for f in sorted(out.glob(f"{T}_ps_*_teachers.npz")):
                mt = re.match(rf"^{re.escape(T)}_ps_(.+)_r(\d+)_teachers\.npz$", f.name)
                c, r = mt.group(1), int(mt.group(2))
                ua = e80a / "units" / world / ta / f"{c}_r{r}.npz"
                if not ua.exists():
                    rows.append({"check": "teacher_vs_e80a", "target": T, "file": f.name, "status": "段 a に無い", "ok": False})
                    continue
                z = np.load(f, allow_pickle=False)
                ks_ = sorted({key.rsplit("__K", 1)[1] for key in z.files}, key=list(BD.KS).index)
                u, qC, qM = EA.load_unit(ua, world, ks_)
                for key in z.files:
                    n, K = key.rsplit("__K", 1)
                    try:
                        y = np.asarray(teacher_array(world, u, n, K, qC, qM), np.float64)
                    except KeyError:
                        rows.append({"check": "teacher_vs_e80a", "target": T, "file": f.name, "key": key, "status": "段 a に教師が無い", "ok": False})
                        continue
                    v = float(np.abs(z[key] - y).max())
                    rows.append({"check": "teacher_vs_e80a", "target": T, "file": f.name, "key": key, "status": "ok", "max_abs": v, "n": len(y),
                                 "ok": v <= TOL_TEACH})
    d = pd.DataFrame(rows)
    if d.empty:
        print("照合する出力が無い", flush=True)
        return 1
    d.to_csv(out / "check.csv", index=False)
    for c_, g in d.groupby("check", sort=False):                                # 差と数だけを出す（UAR などの値は出さない）
        msg = f"{c_}：{len(g)} 項目，食い違い {int((~g.ok.astype(bool)).sum())}"
        if "bad" in g and g.bad.notna().any():
            msg += f"，不一致の数 {int(g.bad.fillna(0).sum())}"
        if "status" in g and g.status.notna().any():
            msg += f"，状態 {g.status.value_counts().to_dict()}"
        if "max_abs" in g and g.max_abs.notna().any():
            msg += f"，差の最大 {g.max_abs.max():.3g}"
        if "pred_agree" in g and g.pred_agree.notna().any():
            msg += f"，予測の一致の最小 {g.pred_agree.min():.6f}"
        print(msg, flush=True)
    nbad = int((~d.ok.astype(bool)).sum())
    print(f"照合：食い違い {nbad} 項目（{out / 'check.csv'}）", flush=True)
    return 1 if nbad else 0


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=EA.E80_WORLDS, required=True)
    ap.add_argument("--kept", type=Path, default=ROOT / "results" / "e80" / "kept.csv")
    ap.add_argument("--target", nargs="+", help="評価のターゲット（既定は世界の全群．IEMOCAP-script は IEMOCAP-impro の後に回し，仮のターゲットを写す）")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", nargs="+", default=["0", "5", "25"], help="E77 の本番と同じ（K 本の抜き方は全 K で作る）")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--gpu-fraction", type=float, default=0.3)
    ap.add_argument("--no-copy-pseudo", action="store_true", help="IEMOCAP-script の仮のターゲットも計算する（写さない）")
    ap.add_argument("--trial-teachers", nargs="+", default=[], help="試走用：kept の keep を使わず，この教師を全部の（生徒, K）で残したとみなす")
    ap.add_argument("--check", action="store_true", help="出力を照合する（GPU を使わない）")
    ap.add_argument("--check-e77", type=Path, help="照合：E77 の同じ世界の出力の場所（none の一致，鍵の重なり，E77 の教師の集合）")
    ap.add_argument("--check-e80a", type=Path, help="照合：段 a の出力の場所（仮のターゲットの教師の配列の一致）")
    args = ap.parse_args()
    world = args.world
    out = args.out_dir or ROOT / "results" / f"e80c_{world}"
    if args.check:
        return run_checks(out, world, args.check_e77, args.check_e80a)
    CD.set_ks(args.ks)
    ks = CD.KS
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    commit = git("rev-parse", "HEAD")
    groups = groups_for(world)
    targets = args.target or groups
    bad = [t for t in targets if t not in groups]
    if bad:
        ap.error(f"{world} に無い群：{bad}")
    targets = sorted(targets, key=lambda t: (t in SCRIPT_SRC, groups.index(t)))   # 台本は即興の後
    todo = [t for t in targets if not CD.check_resume(out, t, commit)]
    kept_sha = sha_file(args.kept)
    kw = read_kept(args.kept, world)
    args_now = {"KS": list(ks), "pseudo_reps": list(args.pseudo_reps), "kept_sha256": kept_sha, "trial_teachers": list(args.trial_teachers)}
    prov = {"script": "exp/e80_stage_c.py", "experiment": "E80c", "argv": sys.argv[1:], "commit": commit,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "script_sha256": sha_file(__file__),
            "script_tracked": bool(git("ls-files", "exp/e80_stage_c.py")), "e80a_script_sha256": sha_file(EA.__file__),
            "compare_dai_sha256": sha_file(CD.__file__), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "world": world, "targets": todo,
            "reps": args.reps, "kept": str(args.kept), "trial": bool(args.trial_teachers), **args_now, "gpu_fraction": args.gpu_fraction,
            "copy_pseudo": not args.no_copy_pseudo, **CD.versions(), **GRID, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "sec": {}, "skipped": {}}
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:                         # 写しの照合が読む（compare_dai と同じ）
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    t_all = time.perf_counter()
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
    cache = {k: {} for k in WORLDS[world] + [world]}
    unit_cache = {}
    for T in todo:
        t0 = time.perf_counter()
        run_target(world, T, data_f, data_m, args, ks, out, cache, commit, kw, kept_sha, args_now, unit_cache, prov)
        prov["sec"][T] = round(time.perf_counter() - t0, 1)
        (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
        mem = f"　GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"{world} {T}：{time.perf_counter() - t0:.0f} 秒{mem}", flush=True)
    prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["sec"]["total"] = round(time.perf_counter() - t_all, 1)
    prov["gpu_max_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
