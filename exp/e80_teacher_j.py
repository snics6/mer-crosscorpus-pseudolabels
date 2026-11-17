"""【下調べ】E80 段 a（GPU）：教師の候補ごとの純相補性 J を，仮のターゲットの単位で測る。

設計の正本は `docs/exploratory-analyses.md` の E80（規則 R1〜R3 と「計算」の段 a）。評価のターゲット T のデータ（テストの発話も
候補のラベルも）は使わない．使うのは T のソースの群（仮のターゲット c と，c のソース）だけである（T と同じコーパスの群は
読み込んだ後に辞書から外し，assert で確かめる）。

単位：世界（AVT・AV）× 評価のターゲット T × 仮のターゲット c × 反復 r．c は T と同じコーパスでないソースの群のうち，
c と同じコーパスでないソースが残るもの（E77 の仮のターゲットと同じ規則）．既定では T から IEMOCAP-script を除く（即興とソース・設定・分割が同じ）。
生徒と none_K は E77（`exp/compare_dai.py`）の仮のターゲットと同じ作り方（`make_students`，E39 の設定 `settings(·, T)`，合併の C = C_FIT，
`compare_dai.Prob2` の none を同じ束ね方で解く．ref・ref0 も `compare_dai.world_unit` と同じ束ね方）。

1 回目：(c, r) ごとに，各生徒の出発点と none_K（候補の上），pF，poe0，agree，pFK，ref，ref0，ds（Dawid–Skene）を作って保存する。
2 回目：c ごとに poolC（E44 の log-linear pool．`committee.Pool`）と mlp（小さな MLP）を，同じ T の仮のターゲットのうち
c と同じコーパスでない群の候補（ラベルあり．全反復）で学び，c に当てる．poolC の λ は，学ぶ群が 2 つ以上のコーパスにまたがれば
コーパスを 1 つずつ外して選び（E60 の形），1 つのコーパスだけなら話者で 2 つに分けて選ぶ。
教師の候補（R1）：融合の生徒は poe0，other_<m>（各モダリティの出発点），poolC，mlp，ds，ref，ref0．
モダリティの生徒 k は pF，poe0，agree，poolC，mlp，other_<m>（m ≠ k），ds，pFK，ref，ref0。

J（R2）＝ P(none_K の argmax が外れ ∧ 教師の argmax が正解) − P(none_K が当たり ∧ 教師が外れ)．c の候補から K 本を除いた発話の上で測る。

出力（--out-dir．既定 results/e80）：
  j_units.csv：world, target, pseudo, rep, student, teacher, K, J, acc_teacher, acc_none, n, commit（parts/j_*.csv をつないだもの）
  parts/j_<world>_<T>.csv，parts/pool_<world>_<T>.csv（λ の選びの UAR と MLP の損失），parts/conv_<world>_<T>.csv・convp_<world>_<T>.csv
  （束ねた解きごと・問ごとの反復の回数と勾配），parts/units_<world>_<T>.csv（単位ごとの発話の数と所要）
  units/<world>/<T>/<c>_r<r>.npz（1 回目の確率．float32．J はこの float32 の値から出す），<c>_r<r>_agg.npz（poolC・mlp），
  units/<world>/samples_<c>.csv（群の行番号 → sample_id・話者）
  provenance.json（世界ごと），provenance_log.jsonl
  kept.csv（--summarize DIR…．R3′・R3″：（世界, 評価のターゲット, 生徒, 教師, K）ごとに，コーパスごとの J の中央値の平均・単位の数・コーパスの数・正の割合・残す，記述の UAR の差）

    ./.venv/bin/python exp/e80_teacher_j.py --world fuse-audio-video-text [--target …] [--pseudo-reps …] [--ks 0 5 25] [--out-dir …] [--gpu-fraction …]
    ./.venv/bin/python exp/e80_teacher_j.py --summarize results/e80/avt results/e80/av [--kept-out results/e80/kept.csv]
"""

from __future__ import annotations

import argparse
import json
import os
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
import logreg  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from baselines_distill import tens  # noqa: E402
from data import groups_for  # noqa: E402
from distill_belief import weak_rows  # noqa: E402
from geom_e0 import settings  # noqa: E402
from multimodal_distill import AGREE_LOW, WORLDS, beliefs1, blocks, make_students, sub_group  # noqa: E402

DT, DEV = logreg.DTYPE, logreg.DEVICE
C_FIT = CD.C_FIT                             # 0.01（E77 と同じ）
REF_MIN = CD.REF_MIN                         # 500（E77 と同じ）
KS_LAB_ALL = CD.KS_LAB_ALL                   # 話者の分割と K 本の抜き方は常に全 K で作る（E77 と同じ）
LAMS = CM.LAMS                               # poolC の λ の候補（E44・E60 と同じ）
E80_WORLDS = ("fuse-audio-video-text", "fuse-audio-video")
SKIP_T = ("IEMOCAP-script",)                 # 既定で回さない評価のターゲット
MLP_HID, MLP_WD, MLP_ITERS, MLP_LR = 16, 1e-3, 300, 0.5     # check_e60_mm.py と同じ
DS_ITERS, DS_SMOOTH = 30, 1.0                # check_e60_refuse.py と同じ
CV_FOLDS, CV_SEED = 2, 0                     # 学ぶ群が 1 つのコーパスだけのときの λ の選び（話者で 2 つに分ける）
KEY = ["world", "target", "pseudo", "rep", "student", "teacher", "K"]
GRID = {"c_fit": C_FIT, "ref_min": REF_MIN, "ks_lab_all": list(KS_LAB_ALL), "lams": list(LAMS), "agree_low": AGREE_LOW,
        "mlp": {"hidden": MLP_HID, "wd": MLP_WD, "iters": MLP_ITERS, "lr": MLP_LR, "seed": "crc32(world|T|c)"},
        "ds": {"iters": DS_ITERS, "smooth": DS_SMOOTH},
        "pool_members": "同じ T の仮のターゲットのうち c と同じコーパスでない群の候補（全反復）．重みは単位（群, 反復）と感情ごとにそろえる",
        "lam_rule": "学ぶ群が 2 つ以上のコーパス：コーパスを 1 つずつ外して単位ごとの UAR の平均が最大の λ．1 つのコーパス：話者で 2 つに分けた袋（split.speaker_folds, seed 0）で同じ．同点は小さい λ",
        "skip_targets": list(SKIP_T)}


# ---------------------------------------------------------------- E60 の部品の写し（あの台本は読み込むと全部を計算するので import できない）

def dawid_skene(votes, n_iter=DS_ITERS, smooth=DS_SMOOTH):
    """`exp/check_e60_refuse.py` の dawid_skene と同じ．votes: (n, M) の言い切りの票．ラベル無しで事前確率と各票の混同の表を推定し，
    (事後確率, π, C) を返す。"""
    n, Mn = votes.shape
    post = np.zeros((n, 4))
    for j in range(Mn):
        post[np.arange(n), votes[:, j]] += 1
    post = post / post.sum(1, keepdims=True)
    for _ in range(n_iter):
        pi = post.mean(0)
        Cs = []
        for j in range(Mn):
            Cj = np.full((4, 4), smooth)
            for lab in range(4):
                Cj[:, lab] += post[votes[:, j] == lab].sum(0)
            Cs.append(Cj / Cj.sum(1, keepdims=True))
        logp = np.log(pi)[None, :].repeat(n, 0)
        for j in range(Mn):
            logp += np.log(Cs[j][:, votes[:, j]].T)
        post = np.exp(logp - logp.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)
    return post, pi, Cs


class MLP(torch.nn.Module):
    """`exp/check_e60_mm.py` の MLP と同じ（入力 → 16 → 4，tanh）。"""

    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, MLP_HID), torch.nn.Tanh(), torch.nn.Linear(MLP_HID, 4))

    def forward(self, x):
        return self.net(x)


def fit_mlp(F, y, w, seed, wd=MLP_WD, iters=MLP_ITERS):
    """`exp/check_e60_mm.py` の fit_mlp と同じ問．初期値だけ種で固定する（E60 は種を固定していない）。戻りは (MLP, 止まったときの目的関数)。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = MLP(F.shape[1])
    m = m.to(DEV).double()
    opt = torch.optim.LBFGS(m.parameters(), lr=MLP_LR, max_iter=iters, line_search_fn="strong_wolfe")

    def loss_fn():
        return (w * torch.nn.functional.cross_entropy(m(F), y, reduction="none")).sum() + wd * sum((p ** 2).sum() for p in m.parameters())

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return m, float(loss_fn())


# ---------------------------------------------------------------- poolC と mlp

def feat(u, mods):
    """単位 u の候補の上の，モダリティの出発点の log 確率（n × M × 4．保存した float32 から作る）。"""
    return np.log(np.clip(np.stack([u["P"][m] for m in mods], 1).astype(np.float64), 1e-12, None))


def stack_members(mem):
    """mem = [(F, y)]．E60 の pool_fit と同じ重み：単位ごとに総重みをそろえ，単位の中では感情ごとにそろえる。"""
    Fp = torch.as_tensor(np.concatenate([F for F, _ in mem]), dtype=DT, device=DEV)
    yp = torch.as_tensor(np.concatenate([y for _, y in mem]), dtype=torch.long, device=DEV)
    w = torch.cat([torch.as_tensor(1.0 / (len(mem) * 4 * np.bincount(y, minlength=4)[y]), dtype=DT, device=DEV) for _, y in mem])
    return Fp, yp, w


def pool_fit(mods, mem, lam):
    Fp, yp, w = stack_members(mem)
    return CM.Pool(mods).fit(Fp, torch.ones(len(yp), len(mods), dtype=DT, device=DEV), yp, w, lam)


def pool_pred(pool, F):
    Ft = torch.as_tensor(F, dtype=DT, device=DEV)
    return pool.predict(Ft, torch.ones(len(Ft), Ft.shape[1], dtype=DT, device=DEV))


def choose_lam(mods, keys, units, tag):
    """poolC の λ を，学ぶ単位 keys（c 以外の群）だけで選ぶ．戻りは (λ, 選び方, 記録の行)。"""
    corps = sorted({CM.corp(c2) for c2, _ in keys})
    splits = []
    if len(corps) >= 2:                                                          # E60 の形：群（ここではコーパス）を 1 つずつ外す
        mode = "loo_corpus"
        for g in corps:
            tr = [(feat(units[k], mods), units[k]["y"]) for k in keys if CM.corp(k[0]) != g]
            ev = [(feat(units[k], mods), units[k]["y"], k) for k in keys if CM.corp(k[0]) == g]
            splits.append((g, tr, ev))
    else:                                                                        # 1 つのコーパスだけ：話者で 2 つに分ける（同じ話者は同じ袋）
        mode = "cv_speaker2"
        persons = pd.Series(np.concatenate([units[k]["person"] for k in keys]))
        fold_of = dict(zip(persons.to_numpy(), M.speaker_folds(persons, k=CV_FOLDS, seed=CV_SEED).to_numpy()))
        for j in range(CV_FOLDS):
            tr, ev = [], []
            for k in keys:
                u = units[k]
                f = np.array([fold_of[p] for p in u["person"]])
                F = feat(u, mods)
                if (f != j).any():
                    tr.append((F[f != j], u["y"][f != j]))
                if (f == j).any():
                    ev.append((F[f == j], u["y"][f == j], k))
            splits.append((f"fold{j}", tr, ev))
    sc, rows = {}, []
    for lam in LAMS:
        for name, tr, ev in splits:
            pool = pool_fit(mods, tr, lam)
            for F, y, k in ev:
                v = CM.uar(y, pool_pred(pool, F))
                sc.setdefault(lam, []).append(v)
                rows.append({**tag, "what": "lam_sel", "mode": mode, "lam": lam, "split": name, "eval_pseudo": k[0], "eval_rep": k[1],
                             "n_eval": len(y), "uar": v})
    lam = max(sc, key=lambda x: float(np.mean(sc[x])))                           # 同点は先（小さい λ．E60 と同じ）
    return lam, mode, rows


# ---------------------------------------------------------------- 1 回目（単位ごと）

def f32(x):
    return (x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)).astype(np.float32)


def unit_pass1(world, stu, yq, yte, label_sets, ks):
    """none_K と，学ばずに作れる教師（pF・poe0・agree・other・ds・pFK・ref・ref0）。ref・ref0 は `compare_dai.world_unit` と同じ作り方。"""
    mods = WORLDS[world]
    prs = {k: CD.Prob2(stu[k], yq, label_sets) for k in mods + [world]}
    none = {k: prs[k].solve([(K, None, K) for K in ks], what="none") for k in [world] + mods}   # run_ab の 1 行目と同じ束ね方
    P0 = {k: {K: prs[k].probs(none[k][K], stu[k]["Z"]) for K in ks} for k in none}
    P0_f = P0[world]                                                             # pFK
    pF = tens(stu[world]["P"])
    b1 = beliefs1(stu, world)                                                    # poe（= poe0）と agree（票が全部一致した発話は pF，不一致は pF × 0.2）
    lg, lg0 = {K: 0 for K in ks}, {K: 0 for K in ks}
    for m in mods:                                                               # compare_dai.world_unit の 2 段目の参照と同じ
        sol = prs[m].solve([(("ref", K), weak_rows(P0_f[K], REF_MIN), K) for K in ks] + [(("ref0", K), weak_rows(pF, REF_MIN), K) for K in ks],
                           what="ref")
        for K in ks:
            lg[K] = lg[K] + prs[m].probs(sol[("ref", K)], prs[m].s["Z"]).clamp(min=1e-12).log()
            lg0[K] = lg0[K] + prs[m].probs(sol[("ref0", K)], prs[m].s["Z"]).clamp(min=1e-12).log()
    votes = np.stack([stu[m]["P"].argmax(1) for m in mods], 1)
    post, pi, Cs = dawid_skene(votes)
    u = {"y": np.asarray(yq).astype(np.int64), "yte": np.asarray(yte).astype(np.int64), "lab": {K: np.asarray(label_sets[K], int) for K in ks},
         "P": {k: f32(stu[k]["P"]) for k in mods + [world]},
         "none": {k: {K: f32(P0[k][K]) for K in ks} for k in P0},
         "none_te": {k: {K: f32(torch.log_softmax(stu[k]["Zte"] @ none[k][K][0].T + none[k][K][1], 1)) for K in ks} for k in none},
         "poe0": f32(b1["poe"]), "agree": f32(b1["agree"]), "ds": f32(post), "ds_pi": pi, "ds_C": np.stack(Cs),
         "ref": {K: f32(torch.softmax(lg[K], 1)) for K in ks}, "ref0": {K: f32(torch.softmax(lg0[K], 1)) for K in ks}}
    return u, prs


def save_unit(path, u, cand, te, ks):
    arrs = {"cand": np.asarray(cand), "te": np.asarray(te), "y": u["y"], "yte": u["yte"], "poe0": u["poe0"], "agree": u["agree"],
            "ds": u["ds"], "ds_pi": u["ds_pi"], "ds_C": u["ds_C"]}
    for K in ks:
        arrs[f"lab__{K}"] = u["lab"][K]
        arrs[f"ref__{K}"] = u["ref"][K]
        arrs[f"ref0__{K}"] = u["ref0"][K]
    for k, P in u["P"].items():
        arrs[f"P__{k}"] = P
    for k in u["none"]:
        for K in ks:
            arrs[f"none__{k}__{K}"] = u["none"][k][K]
            arrs[f"none_te__{k}__{K}"] = u["none_te"][k][K]
    np.savez(path, **arrs)


# ---------------------------------------------------------------- J

def teachers_for(world, mods, u, k, K, qC, qM):
    """R1 の教師の候補（候補の上の確率．順は設計の節の並び）。poolC・mlp は学ぶ群が無ければ None で，候補から外す。"""
    agg = {"poolC": qC, "mlp": qM} if qC is not None else {}
    if k == world:
        return {"poe0": u["poe0"], **{f"other_{m}": u["P"][m] for m in mods}, **agg, "ds": u["ds"], "ref": u["ref"][K], "ref0": u["ref0"][K]}
    return {"pF": u["P"][world], "poe0": u["poe0"], "agree": u["agree"], **agg, **{f"other_{m}": u["P"][m] for m in mods if m != k},
            "ds": u["ds"], "pFK": u["none"][world][K], "ref": u["ref"][K], "ref0": u["ref0"][K]}


def j_stats(y, base, teach, mask):
    """mask の発話で J ＝ P(基準が外れ ∧ 教師が正解) − P(基準が当たり ∧ 教師が外れ)，教師と基準の正解率，発話の数。"""
    yy = y[mask]
    a = base[mask].argmax(1) == yy
    b = teach[mask].argmax(1) == yy
    return {"J": float((~a & b).mean() - (a & ~b).mean()), "acc_teacher": float(b.mean()), "acc_none": float(a.mean()), "n": int(mask.sum())}


def j_rows(world, T, c, r, u, ks, qC, qM):
    mods = WORLDS[world]
    rows = []
    for K in ks:
        mask = np.ones(len(u["y"]), bool)
        mask[u["lab"][K]] = False                                                # K 本を除く
        for k in [world] + mods:
            for name, q in teachers_for(world, mods, u, k, K, qC, qM).items():
                rows.append({"world": world, "target": T, "pseudo": c, "rep": r, "student": k, "teacher": name, "K": K,
                             **j_stats(u["y"], u["none"][k][K], np.asarray(q, np.float32), mask)})
    return rows


# ---------------------------------------------------------------- 評価のターゲットごと

def write_atomic(df, path):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False, float_format="%.17g")                            # 読み直して保存した確率から出した J と一致させる（round_trip で読む）
    os.replace(tmp, path)


def check_resume(parts, world, T, commit):
    f = parts / f"j_{world}_{T}.csv"
    if not f.exists():
        return False
    c = set(pd.read_csv(f, usecols=["commit"]).commit.astype(str))
    if c != {commit}:
        raise SystemExit(f"{f} は別のコミット {c} で作られている（今は {commit}）．出力の場所を変えるか，消してから流し直す")
    return True


def run_target(world, T, data_f, data_m, pseudo_reps, ks, out, cache, commit):
    mods = WORLDS[world]
    data_fT = {g: v for g, v in data_f.items() if CM.corp(g) != CM.corp(T)}     # 評価のターゲット T のコーパスの群を外す（触れたら KeyError）
    data_mT = {k: {g: v for g, v in d.items() if CM.corp(g) != CM.corp(T)} for k, d in data_m.items()}
    assert T not in data_fT and all(CM.corp(g) != CM.corp(T) for g in data_fT)
    assert all(CM.corp(g) != CM.corp(T) for d in data_mT.values() for g in d)
    S = [s for s in data_fT]                                                     # = E77 の S（T と同じコーパスでない群）
    sets = {k: settings(k, T) for k in mods + [world]}                           # E39 の設定（E77 と同じ．T の値ではなく T のために選んだ設定）
    udir = out / "units" / world / T
    udir.mkdir(parents=True, exist_ok=True)
    units, conv, convp, urows = {}, [], [], []
    for c in S:                                                                  # 1 回目（compare_dai.run_world の仮のターゲットの繰り返しと同じ）
        pool_c = [s for s in S if CM.corp(s) != CM.corp(c)]
        if not pool_c:
            continue
        assert CM.corp(c) != CM.corp(T) and all(CM.corp(s) != CM.corp(T) for s in pool_c)
        f = out / "units" / world / f"samples_{c}.csv"
        if not f.exists():
            CD.samples_table(world, c).to_csv(f, index_label="row")
        info_c = {k: M5.pool_info(data_mT[k] if k != world else data_fT, pool_c, C_FIT, cache[k]) for k in sets}
        evc = data_fT[c]
        ctxc = CB.Ctx(evc, pseudo_reps, list(KS_LAB_ALL), center="ind")
        for r in pseudo_reps:
            t0 = time.perf_counter()
            pool, te = np.flatnonzero(~ctxc.te[r]), np.flatnonzero(ctxc.te[r])
            stu = make_students(world, c, data_fT, data_mT, pool_c, pool, te, sets, info_c)
            u, prs = unit_pass1(world, stu, evc.y[pool], evc.y[te], BD.label_sets_of(ctxc, r, pool), ks)
            u["person"] = np.asarray(evc.person)[pool]
            assert len(u["y"]) == len(pool) and pool.max() < len(evc.y)
            save_unit(udir / f"{c}_r{r}.npz", u, pool, te, ks)
            tag = {"world": world, "target": T, "pseudo": c, "rep": r}
            for k, pr in prs.items():
                conv += [{**tag, "student": k, **s_} for s_ in pr.stats]
                convp += [{**tag, "student": k, **s_} for s_ in pr.per]
            units[(c, r)] = u
            urows.append({**tag, "n_cand": len(pool), "n_te": len(te), "sec_pass1": time.perf_counter() - t0})
            print(f"  {world} {T} 仮 {c} r{r}：候補 {len(pool)}　{time.perf_counter() - t0:.0f} 秒", flush=True)
            del stu, prs
            torch.cuda.empty_cache()
    pcs = sorted({c for c, _ in units}, key=S.index)
    jrows, prow = [], []
    for c in pcs:                                                                # 2 回目：c 以外の群（c と同じコーパスでない）で poolC・mlp を学ぶ
        others = [c2 for c2 in pcs if CM.corp(c2) != CM.corp(c)]
        assert all(CM.corp(c2) != CM.corp(T) for c2 in others)
        keys = [k for k in units if k[0] in others]
        tag = {"world": world, "target": T, "pseudo": c}
        qC = qM = None
        if keys:
            t0 = time.perf_counter()
            lam, mode, sel = choose_lam(mods, keys, units, tag)
            mem = [(feat(units[k], mods), units[k]["y"]) for k in keys]
            poolC = pool_fit(mods, mem, lam)
            Fp, yp, w = stack_members(mem)
            seed = zlib.crc32(f"{world}|{T}|{c}".encode()) & 0x7FFFFFFF
            mlp, mloss = fit_mlp(Fp.reshape(len(Fp), -1), yp, w, seed)
            prow += sel + [{**tag, "what": "final", "mode": mode, "lam": lam, "members": " ".join(others), "n_units": len(keys),
                            "n_rows": len(yp), "mlp_seed": seed, "mlp_loss": mloss, "sec": time.perf_counter() - t0}]
        else:
            print(f"  {world} {T} 仮 {c}：poolC・mlp を学ぶ群が無い（候補から外す）", flush=True)
            prow.append({**tag, "what": "final", "mode": "none", "members": ""})
        for r in pseudo_reps:
            if (c, r) not in units:
                continue
            u = units[(c, r)]
            if keys:
                F = feat(u, mods)
                qC = pool_pred(poolC, F).astype(np.float32)
                with torch.no_grad():
                    Ft = torch.as_tensor(F, dtype=DT, device=DEV)
                    qM = f32(torch.softmax(mlp(Ft.reshape(len(Ft), -1)), 1))
                np.savez(udir / f"{c}_r{r}_agg.npz", poolC=qC, mlp=qM, lam=lam)
            jrows += j_rows(world, T, c, r, u, ks, qC, qM)
    parts = out / "parts"
    for name, rows_ in (("pool", prow), ("conv", conv), ("convp", convp), ("units", urows)):
        pd.DataFrame(rows_).assign(commit=commit).to_csv(parts / f"{name}_{world}_{T}.csv", index=False)
    d = pd.DataFrame(jrows).assign(commit=commit)
    assert not d.duplicated(KEY).any()
    write_atomic(d, parts / f"j_{world}_{T}.csv")                               # 最後に書く（再開の印）


def rebuild(out):
    fs = sorted((out / "parts").glob("j_*.csv"))
    if fs:
        write_atomic(pd.concat([pd.read_csv(f, dtype={"K": str}, float_precision="round_trip") for f in fs], ignore_index=True), out / "j_units.csv")


# ---------------------------------------------------------------- R3

def load_unit(path, world, ks):
    """save_unit の逆（要約で UAR の差を出し直すため）。"""
    z = np.load(path, allow_pickle=False)
    mods = WORLDS[world]
    u = {"y": z["y"], "poe0": z["poe0"], "agree": z["agree"], "ds": z["ds"],
         "lab": {K: z[f"lab__{K}"] for K in ks}, "ref": {K: z[f"ref__{K}"] for K in ks}, "ref0": {K: z[f"ref0__{K}"] for K in ks},
         "P": {k: z[f"P__{k}"] for k in mods + [world]}, "none": {k: {K: z[f"none__{k}__{K}"] for K in ks} for k in mods + [world]}}
    a = path.with_name(path.stem + "_agg.npz")
    qC = qM = None
    if a.exists():
        za = np.load(a, allow_pickle=False)
        qC, qM = za["poolC"], za["mlp"]
    return u, qC, qM


def uar_np(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def uar_rows(out, d):
    """j_units の各行について，K 本を除いた候補の上の UAR（教師 − none_K）を記述の列として出し直す（R3″ の記述の列．判定には使わない）。"""
    rows = []
    for (world, T, c, r), g in d.groupby(["world", "target", "pseudo", "rep"], sort=False):
        ks = sorted(set(g.K), key=list(BD.KS).index)
        u, qC, qM = load_unit(out / "units" / world / T / f"{c}_r{r}.npz", world, ks)
        mods = WORLDS[world]
        for K in ks:
            mask = np.ones(len(u["y"]), bool)
            mask[u["lab"][K]] = False
            yy = u["y"][mask]
            for k in [world] + mods:
                ub = uar_np(yy, u["none"][k][K][mask].argmax(1))
                for name, q in teachers_for(world, mods, u, k, K, qC, qM).items():
                    rows.append({"world": world, "target": T, "pseudo": c, "rep": int(r), "student": k, "teacher": name, "K": K,
                                 "uar_diff": uar_np(yy, np.asarray(q)[mask].argmax(1)) - ub})
    return pd.DataFrame(rows)


def summarize(outs, dest) -> int:
    """R3′・R3″・R3⁗（E80 の節の「関門を受けた書き直し」）：j_units.csv から（世界, 評価のターゲット T, 生徒, 教師, K）ごとに，
    T の単位だけを使い，仮のターゲットをコーパスでまとめてコーパスごとの UAR の差（教師 − none_K）の中央値を出し，その平均
    uar_diff_median が 0 より大きい教師を残す（正解率の差の J は記述の列）．
    単位の欠け・重複・コミットの混ざりは止める．kept.csv の列は E79 が読む形（world, target, student, teacher, K, J_median, n_units,
    n_corpora, frac_pos, keep）に，記述の列（uar_diff_median：コーパスごとの UAR の差の中央値の平均）を足す。"""
    ds_ = []
    for out in outs:
        rebuild(out)
        d = pd.read_csv(out / "j_units.csv", dtype={"K": str}, float_precision="round_trip")
        u = uar_rows(out, d)
        d = d.merge(u, on=KEY, how="left", validate="one_to_one")
        if d.uar_diff.isna().any():
            raise SystemExit(f"{out}：UAR の差を出せない行が {int(d.uar_diff.isna().sum())}")
        ds_.append(d)
    d = pd.concat(ds_, ignore_index=True)
    if int(d.duplicated(KEY).sum()):
        raise SystemExit(f"j_units に重複が {int(d.duplicated(KEY).sum())} 行ある")
    cms = sorted(d.commit.astype(str).unique())
    if len(cms) > 1:                                                             # 計算の台本が同じなら許す（exp/・measure/ に差が無いこと）
        for c_ in cms[1:]:
            r_ = subprocess.run(["git", "diff", "--quiet", cms[0], c_, "--", "exp/", "measure/"], cwd=ROOT)
            if r_.returncode != 0:
                raise SystemExit(f"コミットが混ざっていて，exp/・measure/ に差がある {cms}")
    for (w, T), g in d.groupby(["world", "target"]):                             # 単位の欠け：T ごとの（仮のターゲット, 反復）の組が全部の（生徒, 教師, K）にそろうか
        n = len(g[["pseudo", "rep"]].drop_duplicates())
        cnt = g.groupby(["student", "teacher", "K"]).size()
        if (cnt != n).any():
            raise SystemExit(f"{w} {T}：単位が {n} にそろわない行 {cnt[cnt != n].to_dict()}")
    d["corpus"] = d.pseudo.map(CM.corp)
    g1 = d.groupby(["world", "target", "student", "teacher", "K", "corpus"], sort=False).agg(
        mJ=("J", "median"), mU=("uar_diff", "median"), n=("J", "size")).reset_index()
    k = g1.groupby(["world", "target", "student", "teacher", "K"], sort=False).agg(
        J_median=("mJ", "mean"), uar_diff_median=("mU", "mean"), n_units=("n", "sum"), n_corpora=("corpus", "nunique")).reset_index()
    kk = ["world", "target", "student", "teacher", "K"]
    fp = d.groupby(kk, sort=False).uar_diff.apply(lambda s: float((s > 0).mean())).rename("frac_pos")
    fj = d.groupby(kk, sort=False).J.apply(lambda s: float((s > 0).mean())).rename("frac_pos_J")
    k = k.merge(fp.reset_index(), on=kk, validate="one_to_one").merge(fj.reset_index(), on=kk, validate="one_to_one")
    k["keep"] = k.uar_diff_median > 0                                            # R3⁗：UAR の差（コーパスごとの中央値の平均）が 0 より大きい．ちょうど 0 は残さない
    k = k[["world", "target", "student", "teacher", "K", "uar_diff_median", "n_units", "n_corpora", "frac_pos", "keep", "J_median", "frac_pos_J"]]
    write_atomic(k, dest)
    import hashlib
    sh = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()  # noqa: E731
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    prov = {"script": "exp/e80_teacher_j.py", "script_sha256": sh(Path(__file__)), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "rule": "R3⁗：uar_diff_median > 0（コーパスごとの中央値のコーパスの平均）",
            "inputs": {str(o / "j_units.csv"): sh(o / "j_units.csv") for o in outs}, "unit_commits": cms,
            "kept_sha256": sh(dest), "made": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    dest.with_name(dest.stem + "_provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    print(f"{dest}：{len(k)} 行（残す行の数は表に書いた．値は画面に出さない）", flush=True)
    return 0


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=E80_WORLDS)
    ap.add_argument("--target", nargs="+", help=f"評価のターゲット（既定は世界の全群から {list(SKIP_T)} を除いたもの）")
    ap.add_argument("--pseudo-reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", nargs="+", default=["0", "5", "25"], help="J を測る K（none_K と pFK・ref・ref0 もこの K だけ解く）")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "e80")
    ap.add_argument("--gpu-fraction", type=float, default=0.3)
    ap.add_argument("--summarize", nargs="+", type=Path, metavar="DIR", help="段 a の出力の場所（世界ごと）から kept.csv を作る（R3′・R3″．GPU は使わない）")
    ap.add_argument("--kept-out", type=Path, default=ROOT / "results" / "e80" / "kept.csv")
    args = ap.parse_args()
    out = args.out_dir
    if args.summarize:
        return summarize(args.summarize, args.kept_out)
    if not args.world:
        ap.error("--world か --summarize を指定する")
    CD.set_ks(args.ks)                                                           # 検査と順（BD.KS の順）．K 本の抜き方は変わらない
    ks = CD.KS
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    world = args.world
    (out / "parts").mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    commit = git("rev-parse", "HEAD")
    groups = groups_for(world)
    targets = args.target or [g for g in groups if g not in SKIP_T]
    bad = [t for t in targets if t not in groups]
    if bad:
        ap.error(f"{world} に無い群：{bad}")
    todo = [t for t in targets if not check_resume(out / "parts", world, t, commit)]
    prov = {"script": "exp/e80_teacher_j.py", "experiment": "E80a", "argv": sys.argv[1:], "commit": commit,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "script_tracked": bool(git("ls-files", "exp/e80_teacher_j.py")), "script_dirty": bool(git("status", "--porcelain", "exp/e80_teacher_j.py")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "world": world, "targets": todo, "pseudo_reps": args.pseudo_reps, "KS": list(ks),
            "gpu_fraction": args.gpu_fraction, **CD.versions(), **GRID, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "sec": {}}

    def save_prov():
        f = out / "provenance.json"
        allp = json.loads(f.read_text()) if f.exists() else {}
        allp[world] = prov
        tmp = f.with_name(f.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(allp, ensure_ascii=False, indent=1))
        os.replace(tmp, f)
    save_prov()
    if not todo:
        print(f"{world}：全ターゲット済み", flush=True)
        return 0
    t_all = time.perf_counter()
    data_f = {g: M.load(g, world) for g in groups}
    data_m = {m: {g: sub_group(data_f[g], a, b) for g in groups} for m, a, b in blocks(world)}
    cache = {k: {} for k in WORLDS[world] + [world]}
    for T in todo:
        t0 = time.perf_counter()
        run_target(world, T, data_f, data_m, args.pseudo_reps, ks, out, cache, commit)
        rebuild(out)
        prov["sec"][T] = round(time.perf_counter() - t0, 1)
        save_prov()
        mem = f"　GPU 最大 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"{world} {T}：{time.perf_counter() - t0:.0f} 秒{mem}", flush=True)
    prov["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["sec"]["total"] = round(time.perf_counter() - t_all, 1)
    prov["gpu_max_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    save_prov()
    with open(out / "provenance_log.jsonl", "a") as fh:
        fh.write(json.dumps(prov, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
