"""提案の水準：K 本のラベルで測った積を作る分類器の強さは，最も強い分類器を当てたか（計算の後の解析．2026-11-16．学び直しはしない）．

範囲（台本の検査 checks/dai20_strongest/code.md を受けて限った）：この台本の「強さ」は，積を作る声・顔・言葉の分類器（融合の分類器の
疑似ラベルで学び直した ret_A，ret_V，ret_T）の感情ごとの d′ の 4 感情の平均で，提案（論文 7 節）が K 本で測る水準の材料である．
論文 6.7 節の「最も強い単独のモダリティ」（K 本の基準の分類器の候補の上の UAR）とは別の量で，6.7 節の意味で K 本から当てられるかは
この台本では答えない（K 本の基準の分類器を K 本の外で学んだ予測が要り，どの保存にも無い）．

動機：計算の後の解析（`docs/結果/計算の後の解析_先生と自分の差とK10_2026-11-16.md` の要点 5・8）では，疑似ラベルが学び直す分類器より
当たらないと UAR が上がらず，そのコーパスで最も強い単独のモダリティは自分を含まない疑似ラベルで UAR が下がった．この知見を新しい
コーパスで使うには，手元の K 本のラベルで，どのモダリティが最も強いかを当てる必要がある．

量：実験 20 の評価の回（results/dai20_eval/units/．単位を作ったコミット 341c62b）は，提案の水準を作るために，積を作る声・顔・言葉の分類器
（ret_A，ret_V，ret_T）の感情ごとの分離の度合い d′ を，(i) K 本のラベルの上で交差に作った予測から（D_M_K_K<K>．提案が使う量），
(ii) K 本を除いた候補の全ラベルから（D_M_orc_K<K>．新しいコーパスでは手元に無い量）作って保存している（exp/dai17_probe.py 429〜431 行と
同じ作り）．各単位と K について，モダリティごとに 4 感情の平均を取り，(i) と (ii) で最も大きいモダリティが一致するか，3 つの順位の
Kendall の τ を数える．

出力：results/dai20_eval/posthoc/strongest_from_k.csv（単位 × K），strongest_from_k_summary.csv（コーパス × K の一致の割合など）．
記述の解析で予言は置かない．単位どうしは同じテストと候補を共有し，K は入れ子に抜くので，割合は検定ではない．
使い方：.venv/bin/python tools/dai20_strongest_from_k.py [--out-dir 一時の置き場]
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

UNITS = ROOT / "results" / "dai20_eval" / "units"
MODS = ("A", "V", "T")
KS = (5, 10, 25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dai20_eval" / "posthoc"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("strongest_from_k.csv", "strongest_from_k_summary.csv"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} が既にある（上書きしない）")
    rows = []
    for T in DD.CORPORA:
        files = sorted(UNITS.glob(f"{T}_ALL_f*_r*.npz"))
        if len(files) != 50:
            raise SystemExit(f"{T}：単位が {len(files)} 個（見込み 50）")
        for p in files:
            f, r = (int(x[1:]) for x in p.stem.split("_")[2:4])
            z = np.load(p, allow_pickle=True)
            for K in KS:
                dk = z[f"D_M_K_K{K}"].astype(float)          # (3 モダリティ, 4 感情)
                do = z[f"D_M_orc_K{K}"].astype(float)
                if dk.shape != (3, 4) or do.shape != (3, 4) or not (np.isfinite(dk).all() and np.isfinite(do).all()):
                    raise SystemExit(f"{p.name} K{K}：形か値がおかしい")
                de = z["SR_M_ext"].astype(float)              # ソースのコーパスだけで決めた強さ（提案の比べる相手の材料）
                mk, mo, me = dk.mean(1), do.mean(1), de.mean(1)
                so = np.sort(mo)[::-1]
                ik, io, ie = int(np.argmax(mk)), int(np.argmax(mo)), int(np.argmax(me))
                rows.append(dict(T=T, fold=f, seed=r, K=K, top_k=MODS[ik], top_orc=MODS[io], top_ext=MODS[ie],
                                 agree=bool(ik == io), agree_ext=bool(ie == io),
                                 loss_k=float(mo[io] - mo[ik]), loss_ext=float(mo[io] - mo[ie]),
                                 tau=float(st.kendalltau(mk, mo).statistic), tau_ext=float(st.kendalltau(me, mo).statistic),
                                 orc_gap12=float(so[0] - so[1]), **{f"dk_{m}": float(mk[i]) for i, m in enumerate(MODS)},
                                 **{f"do_{m}": float(mo[i]) for i, m in enumerate(MODS)}))
    D = pd.DataFrame(rows)
    cm, dt = dai1.git_commit()
    D.assign(commit=cm, dirty=dt, units_commit="341c62b").to_csv(out / "strongest_from_k.csv", index=False)
    summ = []
    for (T, K), x in list(D.groupby(["T", "K"])) + [(("all", K), D[D.K == K]) for K in KS]:
        fold_rates = x.groupby("fold").agree.mean()
        summ.append(dict(T=T, K=K, n=len(x), agree_rate=float(x.agree.mean()), agree_ext_rate=float(x.agree_ext.mean()),
                         agree_rate_fold_min=float(fold_rates.min()), agree_rate_fold_max=float(fold_rates.max()),
                         loss_k_mean=float(x.loss_k.mean()), loss_ext_mean=float(x.loss_ext.mean()),
                         tau_mean=float(x.tau.mean()), tau_ext_mean=float(x.tau_ext.mean()),
                         top_orc_mode=x.top_orc.mode().iloc[0], top_orc_share=float((x.top_orc == x.top_orc.mode().iloc[0]).mean()),
                         orc_gap12_median=float(x.orc_gap12.median())))
    pd.DataFrame(summ).assign(commit=cm, dirty=dt).to_csv(out / "strongest_from_k_summary.csv", index=False)
    print(f"書いた：{out / 'strongest_from_k.csv'}（{len(D)} 行）と要約", flush=True)


if __name__ == "__main__":
    main()
