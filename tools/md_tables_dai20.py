"""実験 20（提案の評価の回．4 コーパス）の集計を表に書き出す（結果ファイルを読むだけ）．
計算の後の点検（gate-code）が通ってから流す．
使い方：python md_tables_dai20.py [--summary <summary の置き場>] [--out <書き出す md>]
既定の出力：docs/結果/実験20_結果の表_2026-11-15.md"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
C4 = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
KS = (5, 10, 25)
A_LAB = [("prod", "積（今の提案の疑似ラベルの作り方）"), ("psel", "別のコーパスで選んだ指数の積"), ("avg", "確率の平均"), ("conf", "確信度で重みづける平均"),
         ("M_F2_K", "水準だけの重み（K 本で測る）"), ("M_F6_K", "K 本だけで測った表"), ("cf_D_l05", "取り違えの表で読み替えて掛ける"),
         ("tcal", "分類器ごとの温度の較正"), ("temp", "K 本で合わせたモダリティごとの重み"), ("tempb", "同上（切片も合わせる）"),
         ("wf10", "重みの表を乱数で探す（候補 10）"), ("wf100", "重みの表を乱数で探す（候補 100）"), ("wf1000", "重みの表を乱数で探す（候補 1,000）")]
ABL_LAB = [("prop_noprof", "相性を外す（同じ n0・戻し先）"), ("prop_noprof_sel", "相性を外す（設定も選び直す）"), ("prop_noint", "分類器 × 感情の成分を消す"),
           ("prop_n0zero", "水準を戻さない（n0 = 0）"), ("prop_home", "残り方を掛けない"), ("prop_levelfree", "分類器どうしの水準の違いを外す")]
B_LAB = [("none_AVT", "K 本だけの融合"), ("joint_AVT", "今の提案（積を疑似ラベルにして学び直した融合）"), ("b_kd", "温度つき蒸留"),
         ("b_agree", "一致した発話だけの疑似ラベル"), ("b_mix", "ラベルと疑似ラベルを混ぜる方法"), ("b_cotrain", "共訓練"), ("b_coreg", "共正則化")]
REF_LAB = [("M_F6_orc", "全ラベルで測った表（参照．実際には使えない）"), ("M_F6_ext", "ソースだけで測った表（参考）"), ("M_F8", "ソースどうしで学んだ重み（参考）"),
           ("prop_loto", "ほかの 3 コーパスの評価の回で選んだ点（記述）"), ("oracle_grid", "自分のテストで最良の点（参照．上に偏る）")]
SINGLE_LAB = [("ret_A", "微調整した声の分類器"), ("ret_V", "微調整した顔の分類器"), ("ret_T", "微調整した言葉の分類器")]
C_LAB = {"own": "自分の答えを疑似ラベルにする学び直し", "self": "自己学習", "entmin": "エントロピー最小化", "im": "情報最大化", "t3a": "T3A", "t3a_b": "T3A（切片つき．記述）",
         "lp": "ラベル伝播", "st": "ソースの発話と K 本を合わせて学ぶ", "saerens": "感情の割合の補正（Saerens の EM．付録）", "coral": "CORAL", "dann": "DANN", "mme": "MME",
         "mapctl": "勾配法の対照（DA の項なし）", "xmuda": "xMUDA 型"}
NAME = dict(A_LAB + ABL_LAB + B_LAB + REF_LAB + SINGLE_LAB + [("prop", "提案の手法")])


def fest(x) -> str:
    """差：小数 2 桁．丸めると ±1.00 になって 1.0 点の境目のどちら側か読めないときは桁を足す．"""
    s = f"{x:+.2f}"
    if abs(float(s)) == 1.0 and abs(x) != 1.0:
        for nd in (3, 4, 5):
            s = f"{x:+.{nd}f}"
            if abs(float(s)) != 1.0:
                break
    return s


def fend(x) -> str:
    """区間の端：小数 1 桁．0.05 より 0 に近いときは符号が読めるよう桁を足す．"""
    s = f"{x:+.1f}"
    if abs(x) < 0.05 and x != 0:
        for nd in (2, 3, 4):
            s = f"{x:+.{nd}f}"
            if float(s) != 0.0:
                break
    return s


def nm(v: str) -> str:
    b = v.split("@")[0]
    return NAME.get(b, b) + ("（切片をずらす前）" if v.endswith("@raw") else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "dai20_eval" / "summary"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "結果" / "実験20_結果の表_2026-11-15.md"))
    args = ap.parse_args()
    R = Path(args.summary)
    lv = pd.read_csv(R / "levels.csv", low_memory=False)
    ef = pd.read_csv(R / "effects.csv", low_memory=False)
    ot = pd.read_csv(R / "others.csv", low_memory=False)
    gl = pd.read_csv(R / "grid_levels.csv", low_memory=False)
    lv["same_as"] = lv["same_as"].fillna("")
    ef["mark"] = ef["mark"].fillna("")
    conv = json.load(open(R / "convergence.json")) if (R / "convergence.json").exists() else {}
    lva = lv[lv.seeds == "all"]
    L: list[str] = []
    w = L.append
    notes: list[str] = []

    def level(v: str, c: str, K: int, ci: bool = False) -> str:
        q = lva[(lva.version == v) & (lva.c == c) & (lva.K == K)]
        if not len(q):
            return "—"
        q = q.iloc[0]
        s = f"{q.uar:.2f}" + (f" [{q.lo:.1f}, {q.hi:.1f}]" if ci else "")
        if q.same_as and q.same_as != v:
            s += "＝"
            notes.append((C4.index(c), K, f"{c}，K = {K}：{nm(v)} は {nm(q.same_as)} と，塊ごと・感情ごとの正解の数が全部の単位で同じ"))
        return s

    def diff(a: str, b: str, c: str, K: int, seeds: str = "all") -> str:
        q = ef[(ef.a == a) & (ef.b == b) & (ef.c == c) & (ef.K == K) & (ef.seeds == seeds)]
        if not len(q):
            return "—"
        q = q.iloc[0]
        if bool(q.same_counts):
            return "0（同じ数）"
        return f"{fest(q.est)} [{fend(q.lo)}, {fend(q.hi)}]{q.mark}"

    def has(a: str, b: str) -> bool:
        return bool(((ef.a == a) & (ef.b == b)).any())

    def level_table(title: str, rows: list[tuple[str, str]], ci: bool = False) -> None:
        w(f"### {title}")
        w("")
        for K in KS:
            w(f"**K = {K}**")
            w("")
            w("| 版 | " + " | ".join(C4) + " |")
            w("|---|" + "---|" * len(C4))
            for v, lab in rows:
                if not (lva.version == v).any():
                    continue
                w(f"| {lab} | " + " | ".join(level(v, c, K, ci) for c in C4) + " |")
            w("")

    def diff_table(title: str, rows: list[tuple[str, str, str]], lead: str = "") -> None:
        rows = [r for r in rows if has(r[0], r[1])]
        if not rows:
            return
        w(f"### {title}")
        w("")
        if lead:
            w(lead)
            w("")
        w("| 差 | K | " + " | ".join(C4) + " |")
        w("|---|---|" + "---|" * len(C4))
        for a, b, lab in rows:
            for K in KS:
                w(f"| {lab} | {K} | " + " | ".join(diff(a, b, c, K) for c in C4) + " |")
        w("")

    w("# 実験 20（提案の評価の回．4 コーパス）の結果の表（2026-11-15）")
    w("")
    w(f"`{R.relative_to(ROOT) if R.is_relative_to(ROOT) else R}/` から書き出した（結果ファイルを読むだけ．書き出しの台本は `tools/md_tables_dai20.py`）．照合（gate-claim）の前の表である．")
    w("")
    w(f"- 設計は `docs/設計/実験20_設計.md`（第 5 版）．単位は，コーパスごとに分割 5 × K 本の引き方 10 ＝ 50（集計に入った単位の数は {conv.get('n_units', '不明')}）．K = 5・10・25 を同じ扱いで並べる．主の K と判定の規則は置いていない．")
    w("- 数の桁：差は小数 2 桁，区間の端は小数 1 桁．区間の端が 0.05 より 0 に近いときと，差が丸めると ±1.00 になるときは，符号と 1.0 点の境目が読めるように桁を足す．")
    w("- 値は UAR（%）．差は UAR の点の差で，括弧は 95% 区間（補正していない）．↑・↓ は読みやすさのための印（区間が 0 を含まず，差が 1.0 点以上）で，判定ではない．印の無い差は「同じ水準」を意味しない．")
    w("- 「切片をずらした答え」は，ラベルを使わずに切片をずらした答えである．ずらす量は，ラベル無しの候補の発話の上で，その版の確率の平均が，積の確率の平均と同じになるように決める（積はずらす前と後で同じ答えになる）．「切片をずらす前」はそのままの答えである．")
    w("- 「＝」は，別の版と，塊（区間を作る単位．話者など）ごと・感情ごとの正解の数が全部の単位で同じ版（表の下の注）．「0（同じ数）」は，差の両側で，塊ごと・感情ごとの正解の数が全部の単位で同じことを表す．")
    w("- テストの発話は，設定を選んだ下調べ（実験 16 と K = 10 の追補）で同じコーパスを仮のターゲットにした単位のテストと同じ発話である（設計 0 節）．")
    if conv:
        w(f"- 検算の量（`convergence.json`）：単位 {conv.get('n_units')}，微調整した分類器の再現のずれの最大 {conv.get('teacher_dev_max'):.2e}，勾配の最大 {conv.get('grad_max'):.1e}，"
          f"止まり {conv.get('stalled')}，作り直した提案の判定の一致の最小 {conv.get('recheck_agree_min'):.4f}，実験 1 の集計から写した水準との差の最大 {conv.get('others_B_dev_max'):.1e}．")
    w("")
    w("## 1. 提案の手法と積")
    w("")
    diff_table("提案の手法 − 積", [("prop", "prod", "どちらも切片をずらした答え"), ("prop@raw", "prod@raw", "どちらも切片をずらす前"), ("prop", "prop@raw", "提案の手法：切片をずらした答え − ずらす前")])
    w("### K 本の引き方で分けた値（提案の手法 − 積．どちらも切片をずらした答え）")
    w("")
    w("引き方 0・1 は下調べと同じ K 本，引き方 2〜9 は評価の回で初めて使う K 本である．テストの発話はどちらも同じである．")
    w("")
    w("| 引き方 | K | " + " | ".join(C4) + " |")
    w("|---|---|" + "---|" * len(C4))
    for sn, lab in (("0-1", "0・1"), ("2-9", "2〜9")):
        for K in KS:
            w(f"| {lab} | {K} | " + " | ".join(diff("prop", "prod", c, K, sn) for c in C4) + " |")
    w("")
    w("## 2. 同じ 3 つの分類器の答えの別のまとめ方（比べる相手 A）")
    w("")
    level_table("UAR（切片をずらした答え）", [("prop", "**提案の手法**")] + A_LAB)
    level_table("UAR（切片をずらす前）", [("prop@raw", "**提案の手法**")] + [(f"{v}@raw", lab) for v, lab in A_LAB])
    diff_table("提案の手法 − 相手（どちらも切片をずらした答え）", [("prop", v, f"− {lab}") for v, lab in A_LAB])
    diff_table("提案の手法 − 相手（どちらも切片をずらす前）", [("prop@raw", f"{v}@raw", f"− {lab}") for v, lab in A_LAB])
    diff_table("提案の手法（切片をずらした答え）− 相手（切片をずらす前）", [("prop", f"{v}@raw", f"− {lab}") for v, lab in A_LAB[1:]],
               "相手に切片のずらしを当てない判定との差である．")
    diff_table("相手 − 積（どちらも切片をずらした答え）", [(v, "prod", f"{lab} − 積") for v, lab in A_LAB[1:]])
    w("## 3. 提案の部品を 1 つ外した版")
    w("")
    level_table("UAR（切片をずらした答え）", [("prop", "**提案の手法**")] + ABL_LAB)
    diff_table("提案の手法 − 部品を外した版", [("prop", v, f"− {lab}") for v, lab in ABL_LAB])
    diff_table("部品を外した版どうし・積との差", [("prop_noprof", "prod", "相性を外す − 積"), ("prop_noprof_sel", "prod", "相性を外す（設定も選び直す）− 積"),
                                    ("prop_levelfree", "prod", "水準の違いを外す − 積"), ("prop_noint", "prop_noprof", "分類器 × 感情の成分を消す − 相性を外す")])
    w("## 4. 学び直した 1 つの分類器（比べる相手 B）")
    w("")
    w("B の行は実験 1 の評価の回の保存の判定で，提案の手法と同じ単位の上の値である．提案の手法の行は推定時の答え（学び直す前）である．提案の手法の答えを疑似ラベルにした学び直しは実験 21 で測る．区間はこの実験の集計で作り直した値で，塊の重みを引き直すコーパス（MOSEI）では，実験 1 の報告の区間と端が少し違いうる．")
    w("")
    level_table("UAR", [("prop", "**提案の手法**（推定時の答え）")] + B_LAB + SINGLE_LAB, ci=True)
    diff_table("提案の手法（推定時の答え）− 相手", [("prop", v, f"− {lab}") for v, lab in B_LAB])
    w("## 5. 標準の手法（比べる相手 C．実験 2 の評価の回の集計から写した値）")
    w("")
    w("UAR を並べるだけである．「今の提案 − 手法」の差は実験 2 が出した値（同じ単位の上の差）で，提案の手法との差ではない．勾配法の手法（DANN・MME とその対照）は K 本の引き方 0〜2 の 15 単位で測ってあるので，表を単位の数で分け，提案の手法・積・今の提案の値も同じ単位の上の値を並べる．単位の数が違う表どうしで値を見比べない．")
    w("")
    oc = ot[ot.group == "C"]
    lv02 = lv[lv.seeds == "0-2"]
    if len(oc) and not set(oc.n_units.astype(int)) <= {50, 15}:
        raise SystemExit(f"標準の手法の単位の数に 50・15 以外がある：{sorted(set(oc.n_units.astype(int)))}")
    if len(oc):
        for K in KS:
            for nu, lab_u in ((50, "50 単位（K 本の引き方 0〜9）で測った手法"), (15, "15 単位（K 本の引き方 0〜2）で測った手法")):
                ocu = oc[(oc.K == K) & (oc.n_units == nu)]
                if not len(ocu):
                    continue
                w(f"**K = {K}：{lab_u}**")
                w("")
                w("| 手法 | " + " | ".join(f"{c}：UAR | {c}：今の提案 − 手法" for c in C4) + " |")
                w("|---|" + "---|---|" * len(C4))
                src = lva if nu == 50 else lv02                              # 提案の手法と積を，同じ単位の上の値で並べる
                for v, lab in (("prop", "**提案の手法**（推定時の答え．同じ単位）"), ("prod", "積（同じ単位）")):
                    cells = []
                    for c in C4:
                        q = src[(src.version == v) & (src.c == c) & (src.K == K)]
                        cells += [f"{q.iloc[0].uar:.2f}" if len(q) else "—", "—"]
                    w(f"| {lab} | " + " | ".join(cells) + " |")
                cells = []
                for c in C4:                                                 # 今の提案の，同じ単位の上の値（実験 2 の集計が持つ．手法によらず同じはずなので 1 つ目の行から取る）
                    q = ocu[ocu.c == c]
                    vals = sorted(set(round(float(x), 6) for x in q.uar_exp1prop_same_units.dropna())) if "uar_exp1prop_same_units" in q.columns else []
                    cells += ["／".join(f"{x:.2f}" for x in vals) if vals else "—", "—"]
                w("| 今の提案（学び直した融合．同じ単位） | " + " | ".join(cells) + " |")
                for m in [m for m in C_LAB if (ocu.method == m).any()] + sorted(set(ocu.method) - set(C_LAB)):
                    cells = []
                    for c in C4:
                        q = ocu[(ocu.method == m) & (ocu.c == c)]
                        if not len(q):
                            cells += ["—", "—"]
                            continue
                        q = q.iloc[0]
                        cells += [f"{q.uar:.2f}", f"{fest(q.exp1prop_minus_method)} [{fend(q.exp1prop_minus_method_lo)}, {fend(q.exp1prop_minus_method_hi)}]"]
                    w(f"| {C_LAB.get(m, m)} | " + " | ".join(cells) + " |")
                w("")
    w("## 6. 参照と参考（実際には使えない版，ソースのラベルだけで作る版，設定の選び方）")
    w("")
    level_table("UAR（切片をずらした答え）", [("prop", "**提案の手法**"), ("prod", "積")] + REF_LAB)
    diff_table("差", [("M_F6_orc", "prod", "全ラベルで測った表 − 積"), ("M_F6_ext", "prod", "ソースだけで測った表 − 積"),
                     ("prop", "M_F6_ext", "提案の手法 − ソースだけで測った表"), ("M_F8", "prod", "ソースどうしで学んだ重み − 積"), ("prop", "M_F8", "提案の手法 − ソースどうしで学んだ重み"),
                     ("prop_loto", "prop", "ほかの 3 コーパスで選んだ点 − 固定の点"), ("oracle_grid", "prop", "自分のテストで最良の点 − 固定の点")])
    w("### 提案の手法の設定の点（格子 16 点のうち）")
    w("")
    w("n0 は K 本で測った水準を戻す強さ（ラベルの本数），β は相性を使うか（1 は使う），戻し先は eq（3 つの平均）か ext（ソースで見込んだ水準）．")
    w("")
    w("| コーパス | K | 固定の点（下調べが選んだ） | ほかの 3 コーパスで選んだ点 | 自分のテストで最良の点 |")
    w("|---|---|---|---|---|")
    for c in C4:
        for K in KS:
            g = gl[(gl.c == c) & (gl.K == K)]
            if not len(g):
                continue
            def pt(col: str) -> str:
                q = g[g[col].astype(bool)]
                if not len(q):
                    return "—"
                q = q.iloc[0]
                return f"点 {int(q.grid)}（n0 = {q.n0:g}，β = {q.beta:g}，{q.prior}）：{q.uar_adj:.2f}"
            w(f"| {c} | {K} | {pt('frozen')} | {pt('loto')} | {pt('oracle')} |")
    w("")
    if notes:
        w("## 注：塊ごと・感情ごとの正解の数が全部の単位で同じ版（「＝」の印）")
        w("")
        for _, _, n_ in sorted(set(notes)):
            w(f"- {n_}")
        w("")
        w("この印は正解の数で付けている．判定そのものが同じとは限らない（誤りの行き先が違っても付く）．")
        w("")
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"{args.out}：{len(L)} 行")


if __name__ == "__main__":
    main()
