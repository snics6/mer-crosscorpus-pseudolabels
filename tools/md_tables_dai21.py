"""実験 21（学び直しを 5 コーパスに足し，CREMA-D を評価するコーパスに足す）の集計を表に書き出す（結果ファイルを読むだけ）．
実験 20 の集計と合わせた「提案の評価」のまとめの表（5 コーパス × K）も同じ文書の冒頭に書く．
計算の後の点検（gate-code）が通ってから流す．
使い方：python md_tables_dai21.py [--summary21 <置き場>] [--summary20 <置き場>] [--out <書き出す md>]
既定の出力：docs/結果/実験21_結果の表_2026-11-15.md"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
C4 = ["AFEW", "IEMOCAP", "MOSEI", "MELD"]
CD = "CREMA-D"
C5 = C4 + [CD]
KS = (5, 10, 25)
EXTRA = [("noprof", "相性を外した表"), ("noint", "分類器 × 感情の成分を消した表"), ("levelfree", "分類器どうしの水準の違いを外した表"), ("konly", "K 本だけで測った表"),
         ("tcal", "分類器ごとの温度の較正")]
EX1 = [("kd", "温度つき蒸留"), ("agree", "一致した発話だけの疑似ラベル"), ("mix", "ラベルと疑似ラベルを混ぜる方法")]
B_ONLY = [("cotrain", "共訓練"), ("coreg", "共正則化")]
A2 = [("prod", "積"), ("avg", "確率の平均"), ("conf", "確信度で重みづける平均"), ("M_F2_K", "水準だけの重み（K 本で測る）"), ("M_F6_K", "K 本だけで測った表"),
      ("cf_D_l05", "取り違えの表で読み替えて掛ける"), ("tcal", "分類器ごとの温度の較正"), ("temp", "K 本で合わせたモダリティごとの重み"), ("tempb", "同上（切片も合わせる）"),
      ("wf10", "重みの表を乱数で探す（候補 10）"), ("wf100", "重みの表を乱数で探す（候補 100）"), ("wf1000", "重みの表を乱数で探す（候補 1,000）")]
ABL = [("prop_noprof", "相性を外す（同じ n0・戻し先）"), ("prop_noprof_sel", "相性を外す（設定も選び直す）"), ("prop_noint", "分類器 × 感情の成分を消す"),
       ("prop_n0zero", "水準を戻さない（n0 = 0）"), ("prop_home", "残り方を掛けない"), ("prop_levelfree", "分類器どうしの水準の違いを外す")]


def fest(x) -> str:
    """差：小数 2 桁．丸めると ±1.00 になって 1.0 点の境目のどちら側か読めないときは桁を足す．"""
    s = f"{x:+.2f}"
    if abs(float(s)) == 1.0 and abs(x) != 1.0:
        for nd in (3, 4, 5):
            s = f"{x:+.{nd}f}"
            if abs(float(s)) != 1.0:
                break
    if float(s) == 0.0 and x != 0:                                          # 0.00 に丸まる差は，符号が読めるところまで桁を足す
        for nd in (3, 4, 5, 6):
            s = f"{x:+.{nd}f}"
            if float(s) != 0.0:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary21", default=str(ROOT / "results" / "dai21_eval5" / "summary"))
    ap.add_argument("--summary20", default=str(ROOT / "results" / "dai20_eval" / "summary"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "結果" / "実験21_結果の表_2026-11-15.md"))
    args = ap.parse_args()
    R, R20 = Path(args.summary21), Path(args.summary20)
    lv = pd.read_csv(R / "levels.csv", low_memory=False)
    ef = pd.read_csv(R / "effects.csv", low_memory=False)
    sg = pd.read_csv(R / "student_grid.csv", low_memory=False) if (R / "student_grid.csv").exists() else pd.DataFrame()
    conv = json.load(open(R / "convergence.json")) if (R / "convergence.json").exists() else {}
    for d in (lv, ef):
        for col in ("same_as", "same_pred", "mark", "a_same_as", "b_same_as"):
            if col in d.columns and d[col].dtype != bool:                    # 空の欄は NaN で読まれる（差の表の same_pred は真偽なので触らない）
                d[col] = d[col].astype(object).where(d[col].notna(), "")
    lva = lv[lv.seeds == "all"]
    lv20 = pd.read_csv(R20 / "levels.csv", low_memory=False) if (R20 / "levels.csv").exists() else pd.DataFrame(columns=["c", "K", "version", "seeds", "uar"])
    ef20 = pd.read_csv(R20 / "effects.csv", low_memory=False) if (R20 / "effects.csv").exists() else pd.DataFrame(columns=["a", "b", "c", "K", "seeds", "est", "lo", "hi", "mark"])
    if len(ef20):
        ef20["mark"] = ef20["mark"].fillna("")
    fus = {c: "none_AVT" for c in C4} | {CD: "none_AV"}
    if len(lv20):                                                           # 実験 21 の集計の 4 コーパスの prop・prod は実験 20 の単位の判定から数えた値．実験 20 の集計と同じはず
        m_ = lva[lva.c.isin(C4) & lva.version.isin(["prop", "prod"])].merge(
            lv20[lv20.seeds == "all"][["c", "K", "version", "uar"]].rename(columns={"uar": "uar20"}), on=["c", "K", "version"])
        if len(m_) != len(C4) * len(KS) * 2 or not ((m_.uar - m_.uar20).abs().max() < 1e-9):
            raise SystemExit(f"実験 20 の集計と実験 21 の集計で，提案の手法と積の水準が合わない（{len(m_)} 行）")
    L: list[str] = []
    w = L.append
    skipped: list[str] = []                                                 # 全部の欄が「—」で表に出さなかった行（集計に無い版や対．末尾に一覧を書く）

    def level(v: str, c: str, K: int, src=None) -> str:
        d = lva if src is None else src[src.seeds == "all"]
        q = d[(d.version == v) & (d.c == c) & (d.K == K)]
        return f"{q.iloc[0].uar:.2f}" if len(q) else "—"

    def diff(a: str, b: str, c: str, K: int, seeds: str = "all", src=None) -> str:
        d = ef if src is None else src
        q = d[(d.a == a) & (d.b == b) & (d.c == c) & (d.K == K) & (d.seeds == seeds)]
        if not len(q):
            return "—"
        q = q.iloc[0]
        same = bool(q.same_pred) if "same_pred" in q.index and src is None else False
        zero = (q.est == 0 and q.lo == 0 and q.hi == 0)                     # 全部の単位で両側の UAR が同じ（正解の数が同じ）
        note = "（同じ判定）" if same else ("（正解の数は同じ．判定は違う）" if zero and src is None else "")
        return f"{fest(q.est)} [{fend(q.lo)}, {fend(q.hi)}]{q.mark}" + note

    def by_k(title: str, rows: list, corp: list, lead: str = "") -> None:
        """rows：(a, b, 行の名前)．b は文字列か，コーパスごとの相手を返す関数．"""
        w(f"### {title}")
        w("")
        if lead:
            w(lead)
            w("")
        w("| 差 | K | " + " | ".join(corp) + " |")
        w("|---|---|" + "---|" * len(corp))
        for a, b, lab in rows:
            for K in KS:
                cells = [diff(a(c) if callable(a) else a, b(c) if callable(b) else b, c, K) for c in corp]
                if all(x == "—" for x in cells):
                    skipped.append(f"{title}：{lab}（K = {K}）")
                    continue
                w(f"| {lab} | {K} | " + " | ".join(cells) + " |")
        w("")

    def levels_by_k(title: str, rows: list, corp: list) -> None:
        w(f"### {title}")
        w("")
        for K in KS:
            w(f"**K = {K}**")
            w("")
            w("| 版 | " + " | ".join(corp) + " |")
            w("|---|" + "---|" * len(corp))
            for v, lab in rows:
                cells = [level(v(c) if callable(v) else v, c, K) for c in corp]
                if all(x == "—" for x in cells):
                    skipped.append(f"{title}：{lab}（K = {K}）")
                    continue
                w(f"| {lab} | " + " | ".join(cells) + " |")
            w("")

    w("# 実験 21 の結果の表：学び直し（5 コーパス）と CREMA-D の評価の回（2026-11-15）")
    w("")
    w("`results/dai21_eval5/summary/` と `results/dai20_eval/summary/` から書き出した（結果ファイルを読むだけ．書き出しの台本は `tools/md_tables_dai21.py`）．照合（gate-claim）の前の表である．")
    w("")
    n4, ncd = conv.get("students4", {}).get("n_units", "不明"), conv.get("cremad", {}).get("n_units", "不明")
    w(f"- 設計は `docs/設計/実験21_設計.md`（第 3 版）．単位は，コーパスごとに分割 5 × K 本の引き方 10 ＝ 50（集計に入った単位の数は，元の 4 コーパスが {n4}，CREMA-D が {ncd}）．元の 4 コーパスは声・顔・言葉，CREMA-D は声・顔（言葉は無い）．")
    w("- 数の桁：差は小数 2 桁，区間の端は小数 1 桁．区間の端が 0.05 より 0 に近いとき，差が丸めると ±1.00 になるとき，差が 0.00 に丸まるときは，符号と 1.0 点の境目が読めるように桁を足す．差がちょうど 0 で区間の両端も 0 の欄は，全部の単位で両側の正解の数が同じ条件である．判定まで同じなら「（同じ判定）」，判定が違えば「（正解の数は同じ．判定は違う）」と書く．")
    w("- 値は UAR（%）．差は UAR の点の差で，括弧は 95% 区間（補正していない）．↑・↓ は読みやすさのための印（区間が 0 を含まず，差が 1.0 点以上）で，判定ではない．印の無い差は「同じ水準」を意味しない．")
    w("- 「提案の手法」は，感情ごとの信頼度の表で分類器の答えをまとめる手法である．「推定時の答え」はまとめた答えそのもの，「学び直した融合」は，その答えを疑似ラベルにして融合の分類器を学び直したものである．")
    w("- 「今の提案」は，分類器の確率の積を疑似ラベルにして融合の分類器を学び直す手法（実験 1）である．")
    w("- 「切片をずらした答え」は，ラベルを使わずに切片をずらした答えである．ずらす量は，ラベル無しの候補の発話の上で，その版の確率の平均が，積の確率の平均と同じになるように決める（積はずらす前と後で同じ答えになる）．「切片をずらす前」はそのままの答えである．")
    w("- 学び直しの設定（疑似ラベルの重みと K 本の倍率）は，積の疑似ラベルのために選ばれた点を，どの疑似ラベルにも使った（主の点）．")
    w("- テストの発話は，設定を選んだ下調べ（元の 4 コーパスは実験 16 と K = 10 の追補，CREMA-D は実験 18）のテストと同じ発話である．5 コーパスの結果は互いに独立ではない（設計 0 節）．")
    if conv:
        s4, cd_ = conv.get("students4", {}), conv.get("cremad", {})
        if s4:
            w(f"- 検算の量（4 コーパス．`convergence.json`）：単位 {s4.get('n_units')}，勾配の最大 {s4.get('grad_max'):.1e}，止まり {s4.get('stalled')}，今の提案の再現（実験 1 の保存との判定の一致）の最小 "
              f"{s4.get('joint_repro_min'):.4f}（基準を下回る単位 {s4.get('n_joint_repro_below')}），疑似ラベルの作り直しの一致の最小 {s4.get('agree_min'):.4f}，"
              + "，".join(f"{lab}の再現の最小 {s4.get(m + '_repro_min'):.4f}（下回る単位 {s4.get('n_' + m + '_repro_below')}）" for m, lab in EX1) + "．")
        if cd_:
            w(f"- 検算の量（CREMA-D）：単位 {cd_.get('n_units')}，勾配の最大 {cd_.get('grad_max'):.1e}，止まり {cd_.get('stalled')}，切片のずらしの残りの最大 {cd_.get('adj_resid_max'):.1e}．")
    w("")
    # ---------------------------------------------------------------- まとめ
    w("## 1. まとめの表（実験 20 と実験 21 を合わせた「提案の評価」．5 コーパス）")
    w("")
    w("この表の値は，手法と設定を選んだときに見たテストと同じテストの発話の上の値で，探索の結果として読む（設計 0 節）．CREMA-D は，元の 4 コーパスと並ぶ独立な確かめとして数えない．")
    w("")
    w("元の 4 コーパスの「推定時の答え」の行は実験 20 の集計から引いた（実験 21 の集計にある同じ版の値と一致することを，書き出しの台本が確かめる）．それ以外は実験 21 の値である．")
    w("")
    for K in KS:
        w(f"**K = {K}：UAR**")
        w("")
        w("| 版 | " + " | ".join(C5) + " |")
        w("|---|" + "---|" * len(C5))
        w("| K 本だけの融合 | " + " | ".join(level(fus[c], c, K) for c in C5) + " |")
        w("| 積（推定時の答え） | " + " | ".join((level("prod", c, K, lv20) if c in C4 and len(lv20) else level("prod", c, K)) for c in C5) + " |")
        w("| 今の提案（積を疑似ラベルにして学び直した融合） | " + " | ".join(level("stud_prod", c, K) for c in C5) + " |")
        w("| **提案の手法（推定時の答え）** | " + " | ".join((level("prop", c, K, lv20) if c in C4 and len(lv20) else level("prop", c, K)) for c in C5) + " |")
        w("| **提案の手法（学び直した融合）** | " + " | ".join(level("stud_prop", c, K) for c in C5) + " |")
        w("")
    by_k("提案の手法との差（5 コーパス）", [
        ("prop", "prod", "推定時の答え：提案の手法 − 積"),
        ("stud_prop", "stud_prod", "学び直した融合：提案の手法 − 今の提案（この実験の中で疑似ラベルだけを替えた差）"),
        ("stud_prop", lambda c: fus[c], "提案の手法（学び直した融合）− K 本だけの融合"),
        ("stud_prop", "prop", "提案の手法：学び直した融合 − 推定時の答え"),
        ("stud_prop", "prod", "提案の手法（学び直した融合）− 積（推定時の答え）"),
        (lambda c: "stud_prod" if c == CD else "joint_AVT", lambda c: fus[c], "今の提案 − K 本だけの融合（元の 4 コーパスの今の提案は実験 1 の保存，CREMA-D はこの実験で解いた値）"),
        ("stud_prod", lambda c: "joint_AVT" if c in C4 else "__none__", "再現の確かめ：今の提案をこの実験で解いた値 − 実験 1 の保存（元の 4 コーパスだけ．性能の差ではない）")], C5,
        "「提案の手法 − 積」の元の 4 コーパスの値は，実験 21 の集計には無い（実験 20 の表を見る）．" if not ((ef.a == "prop") & (ef.b == "prod") & ef.c.isin(C4)).any() else "")
    if len(ef20):
        w("### 推定時の答え：提案の手法 − 積（元の 4 コーパスは実験 20，CREMA-D は実験 21）")
        w("")
        w("| K | " + " | ".join(C5) + " |")
        w("|---|" + "---|" * len(C5))
        for K in KS:
            w(f"| {K} | " + " | ".join((diff("prop", "prod", c, K, src=ef20) if c in C4 else diff("prop", "prod", c, K)) for c in C5) + " |")
        w("")
    w("### K 本の引き方で分けた値（学び直した融合：提案の手法 − 今の提案）")
    w("")
    w("引き方 0・1 は下調べと同じ K 本，引き方 2〜9 は評価の回で初めて使う K 本である（元の 4 コーパスは実験 20 と同じ単位，CREMA-D はこの実験の単位）．テストの発話はどちらも同じである．")
    w("")
    w("| 引き方 | K | " + " | ".join(C5) + " |")
    w("|---|---|" + "---|" * len(C5))
    for sn, lab in (("0-1", "0・1"), ("2-9", "2〜9")):
        for K in KS:
            w(f"| {lab} | {K} | " + " | ".join(diff("stud_prop", "stud_prod", c, K, sn) for c in C5) + " |")
    w("")
    # ---------------------------------------------------------------- 疑似ラベルの部品
    w("## 2. 疑似ラベルを替えた学び直し（提案の疑似ラベルのどの部品が差を生んだか）")
    w("")
    levels_by_k("UAR（学び直した融合．主の点）", [("stud_prod", "積の疑似ラベル（今の提案）"), ("stud_prop", "**提案の手法の疑似ラベル**")] + [(f"stud_{t}", f"{lab}の疑似ラベル") for t, lab in EXTRA], C5)
    by_k("提案の手法の疑似ラベル − 部品を替えた疑似ラベル", [("stud_prop", f"stud_{t}", f"− {lab}") for t, lab in EXTRA], C5)
    by_k("部品を替えた疑似ラベル − 積の疑似ラベル", [(f"stud_{t}", "stud_prod", f"{lab} − 積") for t, lab in EXTRA], C5)
    # ---------------------------------------------------------------- 実験 1 の手法
    w("## 3. 実験 1 の比較手法と並べる（元の 4 コーパス）")
    w("")
    w("温度つき蒸留・一致した発話だけの疑似ラベル・ラベルと疑似ラベルを混ぜる方法は，積の疑似ラベルを受け取る方法である．設定は実験 1 が積の疑似ラベルで選んだ点のままで，疑似ラベルだけを提案の手法の答えに替えた版も解いた（設定は提案の疑似ラベルのためには選んでいない）．共訓練と共正則化は積を受け取らない．")
    w("")
    levels_by_k("UAR", [("none_AVT", "K 本だけの融合"), ("joint_AVT", "今の提案（実験 1 の保存）"), ("stud_prop", "**提案の手法（学び直した融合）**")]
                + [(f"b_{m}", f"{lab}（実験 1 の保存）") for m, lab in EX1 + B_ONLY] + [(f"{m}_prod", f"{lab}：積の疑似ラベル（この実験で解いた）") for m, lab in EX1]
                + [(f"{m}_prop", f"{lab}：提案の手法の疑似ラベル") for m, lab in EX1], C4)
    by_k("提案の手法（学び直した融合）− 実験 1 の手法（保存）", [("stud_prop", "joint_AVT", "− 今の提案")] + [("stud_prop", f"b_{m}", f"− {lab}") for m, lab in EX1 + B_ONLY], C4)
    by_k("同じ手法の中で疑似ラベルだけを替えた差（提案の手法の疑似ラベル − 積の疑似ラベル．どちらもこの実験で解いた）", [(f"{m}_prop", f"{m}_prod", lab) for m, lab in EX1], C4)
    by_k("同じ疑似ラベルで学び直しの形を替えた差（提案の手法の学び直し − 同じ疑似ラベルを渡した手法）", [("stud_prop", f"{m}_prop", f"− {lab}") for m, lab in EX1], C4)
    by_k("実験 1 の保存との差", [(f"{m}_prop", f"b_{m}", f"{lab}：提案の疑似ラベルの版 − 保存") for m, lab in EX1] + [(f"{m}_prod", f"b_{m}", f"{lab}：積の疑似ラベルの版 − 保存（再現の確かめ）") for m, lab in EX1], C4)
    by_k("実験 1 の手法 − 今の提案（実験 1 の保存どうし）", [(f"b_{m}", "joint_AVT", lab) for m, lab in EX1 + B_ONLY] + [("joint_AVT", "none_AVT", "今の提案 − K 本だけの融合"), ("prop", "joint_AVT", "提案の手法（推定時の答え）− 今の提案")], C4)
    # ---------------------------------------------------------------- CREMA-D
    w("## 4. CREMA-D：推定時の答え（声と顔の 2 つの分類器）")
    w("")
    w("分類器が 2 つなので，K 本で量を合わせる相手は，自由に動かせる値の数が元の 4 コーパスより少ない．相手の位置を 4 コーパスと並べて同じ性質として読まない．")
    w("")
    w("### UAR")
    w("")
    w("部品を外した版と，全ラベルで測った表・ソースだけで測った表は，切片をずらした答えだけを集計している．「切片をずらす前」の欄の「—」は，その値を集計していないことを表す．")
    w("")
    w("| 版 | 切片をずらした答え：K = 5 | K = 10 | K = 25 | 切片をずらす前：K = 5 | K = 10 | K = 25 |")
    w("|---|---|---|---|---|---|---|")
    for v, lab in [("prop", "**提案の手法**")] + A2 + ABL + [("M_F6_orc", "全ラベルで測った表（参照）"), ("M_F6_ext", "ソースだけで測った表（参考）")]:
        w(f"| {lab} | " + " | ".join(level(v, CD, K) for K in KS) + " | " + " | ".join(level(f"{v}@raw", CD, K) for K in KS) + " |")
    w("")
    w("| 学び直した 1 つの分類器・参照 | K = 5 | K = 10 | K = 25 |")
    w("|---|---|---|---|")
    for v, lab in [("start_AV", "出発点（ソースの融合の分類器の平均．K = 0）"), ("none_AV", "K 本だけの融合"), ("none_A", "K 本だけの声"), ("none_V", "K 本だけの顔"), ("ret_A", "微調整した声"),
                   ("ret_V", "微調整した顔"), ("stud_prod", "今の提案（積を疑似ラベルにして学び直した融合）"), ("stud_prop", "**提案の手法（学び直した融合）**"),
                   ("full_AV", "上限（候補の全ラベルで学んだ融合．参照）"), ("full_A", "上限（声）"), ("full_V", "上限（顔）")]:
        w(f"| {lab} | " + " | ".join(level(v, CD, K) for K in KS) + " |")
    w("")
    w("### 提案の手法との差（CREMA-D）")
    w("")
    w("| 差 | K = 5 | K = 10 | K = 25 |")
    w("|---|---|---|---|")

    def cd_row(a: str, b: str, lab: str) -> None:
        cells = [diff(a, b, CD, K) for K in KS]
        if all(x == "—" for x in cells):
            skipped.append(f"提案の手法との差（CREMA-D）：{lab}")
            return
        w(f"| {lab} | " + " | ".join(cells) + " |")
    for v, lab in A2:
        cd_row("prop", v, f"提案の手法 − {lab}（どちらも切片をずらした答え）")
    for v, lab in A2:
        cd_row("prop@raw", f"{v}@raw", f"提案の手法 − {lab}（どちらも切片をずらす前）")
    for v, lab in A2[1:]:
        cd_row("prop", f"{v}@raw", f"提案の手法（切片をずらした答え）− {lab}（切片をずらす前）")
    for v, lab in ABL:
        cd_row("prop", v, f"提案の手法 − {lab}")
    for a, b, lab in [("prop", "none_AV", "提案の手法（推定時の答え）− K 本だけの融合"), ("prop", "stud_prod", "提案の手法（推定時の答え）− 今の提案"), ("prop", "prop@raw", "提案の手法：切片をずらした答え − ずらす前"),
                      ("prop_noprof", "prod", "相性を外す − 積"), ("prop_levelfree", "prod", "水準の違いを外す − 積"), ("prop_noint", "prop_noprof", "分類器 × 感情の成分を消す − 相性を外す"),
                      ("M_F6_orc", "prod", "全ラベルで測った表 − 積"), ("M_F6_ext", "prod", "ソースだけで測った表 − 積"), ("prop", "M_F6_ext", "提案の手法 − ソースだけで測った表"),
                      ("stud_prod", "none_AV", "今の提案 − K 本だけの融合"), ("stud_prod", "prod", "今の提案 − 積（推定時の答え）")]:
        cd_row(a, b, lab)
    for v, lab in A2[1:]:
        cd_row(v, "prod", f"{lab} − 積")
    w("")
    # ---------------------------------------------------------------- 記述
    w("## 5. 記述：設定を疑似ラベルごとに選んだ点")
    w("")
    w("実験 18 の規則で疑似ラベルごとに選んだ点での値である．提案の疑似ラベルの側の点の選択には，評価するコーパスのテストのラベルが間接に入る（設計 3.1 節）．記述だけに使う．")
    w("")
    by_k("疑似ラベルごとに選んだ点どうしの差（提案の手法の疑似ラベル − 積の疑似ラベル）", [("stud_prop_sel", lambda c: "stud_prod" if c == CD else "stud_prod_sel", "学び直した融合")], C5)
    if len(sg):
        w("### 設定の 12 点の UAR（学び直した融合．区間は付けない．テストで最良の点の値は 12 個の最大なので上に偏る）")
        w("")
        g = sg.rename(columns={c: c.replace("_mean", "") for c in sg.columns})
        g = g[g.version.isin(["stud_prop", "stud_prod"])]
        w("| コーパス | K | 疑似ラベル | 12 点の最小 | 12 点の中央 | 12 点の最大 |")
        w("|---|---|---|---|---|---|")
        for c in C5:
            for K in KS:
                for v, lab in (("stud_prod", "積"), ("stud_prop", "提案の手法")):
                    q = g[(g.c == c) & (g.K == K) & (g.version == v)].uar
                    if len(q):
                        w(f"| {c} | {K} | {lab} | {q.min():.2f} | {q.median():.2f} | {q.max():.2f} |")
        w("")
    sp = lva[(lva.same_pred != "") & (lva.same_pred != lva.version)] if "same_pred" in lva.columns else pd.DataFrame()
    if len(sp):
        w("## 注：判定が全部の単位で同じ版")
        w("")
        for r in sp.sort_values(["c", "K", "version"]).itertuples():
            w(f"- {r.c}，K = {r.K}：`{r.version}` は `{r.same_pred}` と同じ判定")
        w("")
    if skipped:
        w("## 注：集計に無いので表に出さなかった行")
        w("")
        for x in skipped:
            w(f"- {x}")
        w("")
        print(f"表に出さなかった行：{len(skipped)}")
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"{args.out}：{len(L)} 行")


if __name__ == "__main__":
    main()
