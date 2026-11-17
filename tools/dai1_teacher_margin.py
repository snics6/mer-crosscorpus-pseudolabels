"""疑似ラベルの「先生の当たり方 − 自分の当たり方」と，学び直しの差の関係（計算の後の解析．2026-11-16．学び直しはしない）．

研究者の依頼（2026-11-16）：声だけでなくほかのモダリティでなぜ上がらないか（支配的なモダリティでは上がらない，など）を，読む人が
面白いと思う形で分析する．

実験 1 の評価の回（results/dai1/eval．コミット ba98974）は，学び直す分類器（生徒）ごとに，疑似ラベルを作る側（先生）の種類を替えた
学び直しを計算済みである（設計 docs/設計/実験1_設計.md 7 節の「予測を付ける側の種類」．tt の行．損失は主の版と同じ joint，ρ = 1，a = 1）．
その差（学び直し − K 本の基準の分類器．テストの発話の上）と区間は results/dai1/stats/desc/teacher_types.csv（B = none）にある．
先生の当たり方は，単位ごとの _teacher_stats の行の uar_pool（K 本を除いた候補の上の，先生の判定の UAR）にある．
自分（K 本の基準の分類器）の当たり方は，実験 14（声・顔・言葉）と実験 22（AVT）の単位の保存の none__K<K> の候補の上の判定から，
同じ K 本を除いた候補の上で作る．

量（単位 (分割, 種) ごとに作り，コーパス・生徒・先生・K ごとに単位の平均を取る）：
  margin = 先生の UAR − 自分の UAR（K 本を除いた候補の上．点）
  gain   = teacher_types.csv の effect（B = none．学び直し − K 本の基準．テストの上．点）と区間
主の先生（3 つの学び直した分類器の確率の積．teacher = one）の行も足す（gain は results/dai1/stats/families_flagged.csv の族 4 の
既定の設定 default − none ではなく，族 1 の prop − none を使う．声・顔・言葉・AVT の 48 条件のうち 44 で prop と default は同じ識別器）．

K は 5 と 25（自分の候補の上の判定が実験 14 と 22 の保存にある K）．記述の解析で，予言は置かない（値は計算の後に見る）．
区間は gain にだけ付く（teacher_types.csv の区間）．margin には区間を付けない．点どうしは同じ候補とテストを共有するので，
数え上げと順位相関は検定ではない．

出力：results/dai1/stats/teacher_margin.csv（点ごと），teacher_margin_summary.csv（K・生徒ごとの数え）．
使い方：.venv/bin/python tools/dai1_teacher_margin.py [--out-dir 一時の置き場]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
import dai1  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

OUT1 = ROOT / "results" / "dai1"
OUT14 = ROOT / "results" / "dai14"
OUT22 = ROOT / "results" / "dai22"
KS = (5, 25)
STUDENTS = ("A", "V", "T", "AVT")


def baseline_pool_uar(T: str, f: int, r: int, s: str, K: int, sid_pool: np.ndarray) -> float:
    """自分（K 本の基準の分類器）の，K 本を除いた候補の上の UAR（%）．"""
    if s == "AVT":
        a = np.load(OUT22 / "units" / f"{T}_f{f}_r{r}.npz", allow_pickle=False)
        pred = a[f"none__K{K}"]
    else:
        a = np.load(OUT14 / "units" / f"{T}_f{f}_r{r}.npz", allow_pickle=False)
        pred = a[f"none__K{K}__{s}"]
    if not (a["sid"].astype(str) == sid_pool).all():
        raise SystemExit(f"{T}_f{f}_r{r}：実験 14／22 の候補の並びが実験 1 と違う")
    nk = a[f"nk__K{K}"].astype(bool)
    y = a["y"].astype(int)
    return 100.0 * S.uar(y[nk], pred[nk].astype(int))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT1 / "stats"))
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    args = ap.parse_args()
    sdir = Path(args.out_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    for name in ("teacher_margin.csv", "teacher_margin_summary.csv"):
        if (sdir / name).exists():
            raise SystemExit(f"{sdir / name} が既にある（上書きしない）")
    import json
    frozen = json.load(open(OUT1 / "select" / "frozen.json"))
    tt = pd.read_csv(OUT1 / "stats" / "desc" / "teacher_types.csv", comment="#")
    tt = tt[(tt.B == "none") & tt.K.isin(KS)]
    fam = pd.read_csv(OUT1 / "stats" / "families_flagged.csv", comment="#")
    fam = fam[(fam.family == 1) & fam.K.isin(KS)]
    rows = []
    for T in args.targets:
        R = int(frozen[T]["n_seeds_eval"])
        per = {}
        for f in range(DD.N_FOLDS):
            for r in range(R):
                base = OUT1 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
                p3 = pd.read_csv(f"{base}_p3.csv", low_memory=False)
                p1 = np.load(f"{base}_p1.npz", allow_pickle=False)
                sid_pool = p1["sample_pool"].astype(str)
                ts = p3[(p3.method == "_teacher_stats") & p3.K.isin(KS)]
                for s in STUDENTS:
                    for K in KS:
                        b = baseline_pool_uar(T, f, r, s, K, sid_pool)
                        x = ts[(ts.K == K) & ((ts.student == s) | ts.student.isna()) & ts.teacher.astype(str).str.match(r"^(tt:|one$)")]
                        for _, row in x.iterrows():
                            name = str(row.teacher)
                            name = "main_q" if name == "one" else name[3:]
                            per.setdefault((s, K, name), []).append((float(row.uar_pool), b))
        for (s, K, name), v in per.items():
            v = np.array(v)
            if len(v) != DD.N_FOLDS * R:
                raise SystemExit(f"{T}/{s}/K{K}/{name}：単位が {len(v)} 個（見込み {DD.N_FOLDS * R}）")
            if name == "main_q":
                g = fam[(fam["T"] == T) & (fam.student == s) & (fam.K == K)]
            else:
                g = tt[(tt["T"] == T) & (tt.student == s) & (tt.K == K) & (tt.teacher == name)]
            if len(g) != 1:
                continue                                              # この生徒に当てていない先生（teacher_types.csv に行が無い）
            g = g.iloc[0]
            rows.append(dict(T=T, student=s, K=K, teacher=name, teacher_uar_pool=float(v[:, 0].mean()), self_uar_pool=float(v[:, 1].mean()),
                             margin=float((v[:, 0] - v[:, 1]).mean()), n_units=len(v),
                             no_own_modality=(bool(g.no_own_modality) if name != "main_q" else False),
                             gain=float(g.effect), gain_lo=float(g.lo), gain_hi=float(g.hi)))
        print(f"{T}：{sum(1 for x in rows if x['T'] == T)} 点", flush=True)
    D = pd.DataFrame(rows)
    # 除く点（台本の検査 checks/dai1_teacher_margin/code.md の 1・2）：
    #  (a) AVT の生徒の fused_AVT は自分（none_AVT）そのもので margin が定義から 0 になる．
    #  (b) 選んだ指数が (1,1,1) のコーパスと K では noret_sel と poe_none が同じ先生になる（先生の UAR と gain が同じ）．noret_sel を除く．
    selfp = (D.student == "AVT") & (D.teacher == "fused_AVT")
    key = ["T", "student", "K"]
    pn = D[D.teacher == "poe_none"].set_index(key)[["teacher_uar_pool", "gain"]]
    nr = D[D.teacher == "noret_sel"].join(pn, on=key, rsuffix="_poe")
    dup_idx = nr[(nr.teacher_uar_pool - nr.teacher_uar_pool_poe).abs().lt(1e-9) & (nr.gain - nr.gain_poe).abs().lt(1e-9)].index
    D["excluded"] = ""
    D.loc[selfp, "excluded"] = "self_teacher"
    D.loc[dup_idx, "excluded"] = "same_as_poe_none"
    cm, dt = dai1.git_commit()
    D.assign(commit=cm, dirty=dt, units_commit="ba98974").to_csv(sdir / "teacher_margin.csv", index=False)
    E = D[D.excluded == ""]
    summ = []
    groups = ([(("all", K, s), x) for (K, s), x in E.groupby(["K", "student"])] + [(("all", K, "all"), E[E.K == K]) for K in KS]
              + [((T, K, s), x) for (T, K, s), x in E.groupby(["T", "K", "student"])])            # (T, 生徒, K) の中の数えも出す（検査の 3）
    for (T, K, s), x in groups:
        rho = st.spearmanr(x.margin, x.gain).statistic if len(x) > 2 else np.nan
        summ.append(dict(T=T, K=K, student=s, n=len(x), spearman=float(rho),
                         sign_agree=int(((x.margin > 0) & (x.gain > 0)).sum() + ((x.margin < 0) & (x.gain < 0)).sum()),
                         margin_pos_gain_ci_pos=int(((x.margin > 0) & (x.gain_lo > 0)).sum()),
                         margin_neg_gain_ci_neg=int(((x.margin < 0) & (x.gain_hi < 0)).sum()),
                         margin_neg_gain_ci_pos=int(((x.margin < 0) & (x.gain_lo > 0)).sum()),
                         margin_pos_gain_ci_neg=int(((x.margin > 0) & (x.gain_hi < 0)).sum()),
                         gain_gt_margin=int((x.gain > x.margin).sum())))
    pd.DataFrame(summ).assign(commit=cm, dirty=dt).to_csv(sdir / "teacher_margin_summary.csv", index=False)
    print(f"書いた：{sdir / 'teacher_margin.csv'}（{len(D)} 点），{sdir / 'teacher_margin_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
