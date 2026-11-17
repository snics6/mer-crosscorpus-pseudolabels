"""【探索的】E51（GPU）：条件 1 の教える選び方（LLL．E39 の生徒）が選ぶ発話を，ラベル無しの手がかりでどれだけ見分けられるか。

規則は `docs/exploratory-analyses.md` E51（設計と予言は計算の前に書いた）。A と A(HuBERT) の 14 ターゲット，反復 0〜9。
LLL の順は E49 と同じ（E40 の teach の組の歩幅と ρ_M）。正例は LLL が K 本で選んだ発話，負例は同じ感情の残りの候補で，
感情ごとに AUC を出して平均する（ラベル無しで選ぶときは感情がわからないので，感情の違いで見分けた分を除く）。
出発点（E39 の下書き）が外している発話についても同じく感情ごとの AUC を出す。手がかり（どれもラベルを使わない．大きいほど選ぶ向き）：
  unc     ：1 −（下書きの確率の 1 位 − 2 位）
  q_kl    ：KL(q ‖ 下書き)．q は E44 の C
  q_dis   ：q と下書きの予測が違えば 1
  vote_ent：ソースごとの分類器の言い切りの票の割合のエントロピー
  mi      ：ソースの確率の平均のエントロピー − 各ソースのエントロピーの平均
  typ     ：近い 20 本とのコサイン距離の平均の逆数（E46 の typiclust と同じ）
  o_kl    ：KL(q_o ‖ 下書き)．q_o は別のエンコーダ（A なら HuBERT）の特徴で E44 の C と同じく作った見込み
  o_dis   ：q_o と下書きの予測が違えば 1

    MERDA_MAP_ROOT=results/fixc_map ./.venv/bin/python exp/check_lll_cues.py
    ./.venv/bin/python exp/check_lll_cues.py --analyze

出力：`results/explore_relation/e51_auc.csv`（kind, target, rep, task, cue, emotion, auc），`e51_rates.csv`，`e51_tests.csv`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import committee as CM  # noqa: E402
import compare_b1 as CB  # noqa: E402
import map as M  # noqa: E402
import meta5 as M5  # noqa: E402
from belief_check import knn  # noqa: E402
from data import groups_for  # noqa: E402
from select_baselines import K_TYP  # noqa: E402
from teach_ablate import ablate_order  # noqa: E402
from teach_student39 import Student  # noqa: E402

PAIR = {"audio_wavlm": "audio_hubert", "audio_hubert": "audio_wavlm"}
CUES = ["unc", "q_kl", "q_dis", "vote_ent", "mi", "typ", "o_kl", "o_dis"]
TASKS = {"lll5": 20, "lll25": 100}
REPS = list(range(10))
OUT_REL = ROOT / "results" / "explore_relation"


def committee_q(kind, T, data, f, Cs, P, Y, Xc):
    """E44 の C の見込みと，ソースごとの確率（候補の上）。"""
    S = [s for s in data if CM.corp(s) != CM.corp(T)]
    e44 = pd.read_csv(ROOT / "results" / f"committee_{kind}" / f"{T}.csv")
    pool_C = CM.Pool(S).fit(*CM.pool_data(P, Y, S, S), float(e44[e44.variant == "C"].lam.iloc[0]))
    Ps = [torch.softmax(f[s].logits(Xc, Cs[s]), 1).cpu().numpy() for s in S]
    F = torch.as_tensor(np.stack([np.log(np.clip(p, 1e-12, None)) for p in Ps], 1), dtype=torch.float64, device=Xc.device)
    return pool_C.predict(F, torch.ones(len(Xc), len(S), dtype=torch.float64, device=Xc.device)), np.stack(Ps, 1)


def kl(a, b):
    a, b = np.clip(a, 1e-12, None), np.clip(b, 1e-12, None)
    return (a * np.log(a / b)).sum(1)


def ent(p):
    p = np.clip(p, 1e-12, None)
    return -(p * np.log(p)).sum(-1)


def run(kind, rows, rrows):
    other = PAIR[kind]
    groups = groups_for(kind)
    data = {g: M.load(g, kind) for g in groups}
    data_o = {g: M.load(g, other) for g in groups}
    f, Cs, P, Y = CM.prepare(kind, data)
    f_o, Cs_o, P_o, Y_o = CM.prepare(other, data_o)
    sel = pd.read_csv(OUT_REL / "e40_targets.csv", dtype={"K": str}).drop_duplicates(["kind", "target"]).set_index(["kind", "target"])
    cache = {}
    for T in groups:
        S = [s for s in data if CM.corp(s) != CM.corp(T)]
        s_ = sel.loc[(kind, T)]
        C_T = float(pd.read_csv(ROOT / "results" / f"multisource_scores_{kind}" / f"{T}.csv", dtype={"K": str}).C.iloc[0])
        src_pc, lin, n_of = M5.pool_info(data, S, C_T, cache)
        ev, ev_o = data[T], data_o[T]
        ctx = CB.Ctx(ev, REPS, ["5"], center="ind")
        ctx_o = CB.Ctx(ev_o, REPS, ["5"], center="ind")
        for r in REPS:
            pool = np.flatnonzero(~ctx.te[r])
            if not (np.flatnonzero(~ctx_o.te[r]) == pool).all():
                raise ValueError(f"候補が 2 つのエンコーダで違う: {T} r{r}")
            yq = ev.y[pool]
            Xc = M.rows(ev.X, pool) - ctx.ct[r]
            Xc_o = M.rows(ev_o.X, pool) - ctx_o.ct[r]
            st = Student(ev, Xc, src_pc, lin, n_of, S, C_T, float(s_.gamma), float(s_.rho_s), [float(s_.t_rho_M)])
            Xt = (st.Xstd @ st.mats[float(s_.t_rho_M)][0]).float()
            Km = Xt @ Xt.T
            Km.fill_diagonal_(0.0)
            order = np.array(ablate_order(Km, yq, st.L0, float(s_.t_c), max(TASKS.values()), "LLL")[0])
            del Km
            p = torch.softmax(st.L0, 1).cpu().numpy()
            q, Ps = committee_q(kind, T, data, f, Cs, P, Y, Xc)
            q_o, _ = committee_q(other, T, data_o, f_o, Cs_o, P_o, Y_o, Xc_o)
            ps = np.sort(p, 1)
            votes = np.stack([np.bincount(row, minlength=4) for row in Ps.argmax(2)]) / Ps.shape[1]
            X = st.Xstd.float()
            Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
            nb = knn(X, K_TYP)
            typ = (1.0 / (1.0 - (Xn[:, None, :] * Xn[nb]).sum(2)).mean(1).clamp(min=1e-6)).cpu().numpy()
            cue = {"unc": 1 - (ps[:, -1] - ps[:, -2]), "q_kl": kl(q, p), "q_dis": (q.argmax(1) != p.argmax(1)).astype(float),
                   "vote_ent": ent(votes), "mi": ent(Ps.mean(1)) - ent(Ps).mean(1), "typ": typ,
                   "o_kl": kl(q_o, p), "o_dis": (q_o.argmax(1) != p.argmax(1)).astype(float)}
            err = p.argmax(1) != yq
            labs = {t: np.isin(np.arange(len(yq)), order[:n]) for t, n in TASKS.items()}
            labs["err"] = err
            for t, lab in labs.items():
                for k in range(4):
                    m = yq == k
                    if lab[m].all() or not lab[m].any():
                        continue
                    for c, v in cue.items():
                        rows.append({"kind": kind, "target": T, "rep": r, "task": t, "cue": c, "emotion": k,
                                     "auc": roc_auc_score(lab[m], v[m]), "n_pos": int(lab[m].sum())})
            rrows.append({"kind": kind, "target": T, "rep": r, "err_all": float(err.mean()),
                          **{f"err_{t}": float(err[order[:n]].mean()) for t, n in TASKS.items()},
                          "uar_p": CB.uar(yq, p.argmax(1)), "uar_q": CB.uar(yq, q.argmax(1)), "uar_q_o": CB.uar(yq, q_o.argmax(1)),
                          "dis_q_p": float(cue["q_dis"].mean()), "dis_qo_p": float(cue["o_dis"].mean()),
                          "dis_qo_q": float((q_o.argmax(1) != q.argmax(1)).mean())})
        print(f"{kind} {T} 済み", flush=True)


def analyze() -> int:
    a = pd.read_csv(OUT_REL / "e51_auc.csv")
    rt = pd.read_csv(OUT_REL / "e51_rates.csv")
    g = a.groupby(["kind", "target", "task", "cue"]).auc.mean().groupby(["kind", "task", "cue"]).mean().unstack("cue")[CUES]
    pd.set_option("display.width", 250)
    print("== 感情ごとの AUC（感情・反復・ターゲットの平均．0.5 なら見分けられない）")
    print(g.round(3).to_string())
    print("\n== 割合と UAR（ターゲットの平均．％）")
    print((rt.groupby(["kind", "target"]).mean(numeric_only=True).drop(columns="rep").groupby("kind").mean() * 100).round(2).T.to_string())
    tests = g.reset_index()
    for kind in PAIR:
        r = rt[rt.kind == kind].groupby("target").mean(numeric_only=True)
        e = g.loc[(kind, "err")]
        tests.loc[tests.kind == kind, "R1"] = bool((r.err_lll5 > r.err_all).mean() > 0.5 and r.err_lll5.mean() > r.err_all.mean())
        tests.loc[tests.kind == kind, "R2"] = bool(e["o_kl"] > e["q_kl"])
        tests.loc[tests.kind == kind, "max_lll5"] = float((g.loc[(kind, "lll5")] - 0.5).abs().max() + 0.5)
    tests.to_csv(OUT_REL / "e51_tests.csv", index=False)
    print(tests.drop_duplicates("kind")[["kind", "R1", "R2", "max_lll5"]].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=list(PAIR))
    args = ap.parse_args()
    if args.analyze:
        return analyze()
    rows, rrows = [], []
    for kind in args.kinds:
        run(kind, rows, rrows)
    pd.DataFrame(rows).to_csv(OUT_REL / "e51_auc.csv", index=False)
    pd.DataFrame(rrows).to_csv(OUT_REL / "e51_rates.csv", index=False)
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
