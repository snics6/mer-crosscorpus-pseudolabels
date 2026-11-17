"""DAI 実験 1 の族の外の数（記述の数）を，区間と単位の数つきで作る集計（設計 `docs/設計/実験1_設計.md` 7.2・9.1〜9.4・9.6・14 節）。

族 1〜4 の検定は `dai1_stats.py evaluate` が作る（`stats/families.csv`）．この台本はそれを作り直さず，設計が
「区間と単位の数を添え，p 値を付けない」と決めた数（9.3 節の S5）と，結果の表の注に要る数を作る．評価の値を見る前に
集計の仕方を固めるため，評価の回の前にコミットする（定義の 5 回目の D1，比較の 5 回目 2-I）。

使い方：
  python exp/dai1_desc.py eval   [置き場]   評価の回の出力 <置き場>/eval/units/<T>_ALL_f<f>_r<r>_p{1,2,3}.{npz,csv} から作る．
      動く条件：<置き場>/select/frozen.json と frozen/<置き場の名前>_frozen.json の sha256 が一致し，写しがコミット済みで，
      作業木（`dai1.git_commit` が見る exp・splits・measure・features・data/metadata．未追跡のファイルを含む）が汚れていない．
      さらに，全部の T で 5 分割 × 種 n_seeds_eval の単位の段 1〜3 がそろい，単位の frozen_sha256 が固定のファイルと一致し，
      単位のコミットが 1 つで，そのコミットの exp/dai1_desc.py と dai1_stats.py が今のファイルと同じであること（集計の仕方を
      評価の回の前に固定したことの確かめ．定義の 5 回目の D1）．出力は <置き場>/stats/desc/（すでに中身があれば止まる）．
      評価の回の数は評価の回の単位だけから作る．仮のターゲットの行を読むのは，設計がそれと比べると決めた 2 つの量だけ
      （regret.csv の rank_spearman と，9.4 節の 3 の選び直し）で，読む行の単位の一覧とコミットが固定のファイルの units と
      pseudo_commit に一致しなければ止まる。
  python exp/dai1_desc.py pseudo [置き場]   試走．仮のターゲットの回の出力 <置き場>/pseudo/units/<T>_<c>_f<f>_r<r>_*（c はソースの
      名前，種 0・1）に同じ集計を (T, c) ごとに当てる．単位のコミットが固定のファイルの pseudo_commit と違えば止まる．
      出力は <置き場>/stats/desc_pseudo/（--out-dir で替える）．仮のターゲットの回には引き戻しの曲線の問が無いので，
      tether_curve.csv と tether_sign.csv は書かれない（marker_only.csv などの部分集合の表も，そのコーパスが c に無ければ書かれない）。
  任意の引数：--out-dir DIR（出力の置き場を替える），--targets AFEW,MELD（試走だけ），--encoder 1|2（既定は置き場の名前が
      dai1_enc2 なら 2，ほかは 1），--jobs N（(T, c) を N 個のプロセスで並べる．値は N に依らない）。
  GPU は使わない（CUDA_VISIBLE_DEVICES を空にしてから読み込む）．画面には行の数，件数，照合の食い違いの数，所要だけを出す。
  所要：MOSEI のテストの区間は塊の引き直し（1 回の区間ごとに分割ごとの多項分布 2,000 回）なので，MOSEI をテストにする組が最も
      長い．--jobs を使うときは OMP_NUM_THREADS=1（OPENBLAS_NUM_THREADS，MKL_NUM_THREADS も）で BLAS のスレッドを抑える
      （値はスレッドの数に依らない．行列の積は整数の件数の和だけ）。

UAR の作り方（全部の表で共通．統計の検査 S3，m39）：
  保存したテストの発話ごとの判定（int8）と，`measure/data.py` の `load_group` を通したラベル（`dai_data.load_corpus` の label_4．
  npz の label_4 は使わない）から，単位（分割 f，種 r）ごとに作り直す．単位の値は `dai1_stats.uar_full`（テストに現れる感情の
  再現率の平均）と同じ定義．塊ごとの数は `dai1_stats.counts`（塊はコーパスの全部の塊の名前の順の番号で渡す．名前の順と番号の順は
  同じなので jackknife と引き直しの結果は名前で渡したときと同じ）．作り直した単位の UAR は，使った全部の (行，単位) について
  `rows.csv`（段 1・3 の csv）の uar と照合し，食い違いの数を check.csv に書く（設計 9.1 節の「個別の照合」．比較の基準，
  役割を分ける版，単一の生徒の共訓練，共正則化，切片だけ，ゼロから，出発点，ソースの統計で標準化した出発点を含む）。

区間（全部の表で共通）：`dai1_stats.effect_stats` をそのまま使う（9.3 節．G_eff ≥ 20 の MOSEI は塊の引き直し B = 2,000 回の分散と
  種の項，ほかは塊の jackknife と種の項，30 本未満か感情が欠ける分割を θ_(−g) から外し，分割ごと外れる分割の種の項は足さない）．
  p 値と t は出さない（S5）．標準誤差が 0 のときは se_zero を真にする．K = 0 の行の単位の数は分割の数（S9）．乱数は
  (20261104, T の番号, c の番号, 表の番号) で表ごとに作る（MOSEI の引き直しだけが使う）．
  - 差の区間：2 つの行の判定の (f, r) ごとの数を effect_stats に渡す．same_pred_all_units は全部の単位で 2 つの行の判定が
    1 本残らず同じこと（このとき区間は [0, 0] になる．選ぶ段で同じ設定になった行の印）。
  - 1 つの行の値（水準）の区間：比べる相手に「同じ件数で正解 0 の数」を渡す．単位の UAR の差がその行の UAR そのものになり，
    30 本の規則と感情が欠けるときの扱いもその行と同じ条件で当たる（effect_stats を書き直さないための形）。
  - 感情 k の再現率の区間：正解の数を感情 k の列だけ残した数を渡し，得た値と区間を 4 倍する（全部の分割のテストに 4 感情が
    そろうことを確かめてから．分散は 16 倍）．θ は分割ごとの再現率の分割の平均．あわせて 5 つのテストを合わせた再現率
    （各発話を 1 回ずつ数え，種で平均）を recall_pooled に書く（a8）．表と結果の文で使うのは θ で，recall_pooled は参考の列（gate-code の 5 回目の記録）。

表（出力の csv．行の鍵は T，c，生徒，K，行の名前．行の名前は台本の行の値 method|teacher|つまみ）：
  levels.csv（9.3 節 S5 の表の全部の数の水準）：(生徒, K) ごとに，比較の基準 none[K]（K = 0 は出発点），提案（主の版と主でない版．
      選んだ設定），役割を分ける版（両方の版），選択なしの提案（one，ρ = 1，a = 1），族 3 の相手，a = 1 に制限した提案（両方の版），
      切片だけ（kbias．選んだ C_b．K ≥ 1），ゼロから（tgtonly．選んだ C．K ≥ 1），出発点を使わない提案（joint_scratch．両方の版，
      選んだ C），温度 1 の蒸留（kd，one，T = 1，ρ = 1），族 3 の相手の候補 5 手法の選んだ設定（蒸留・一致・混ぜる手は両方の版），
      予測を付ける側の種類（tt の全部の行．ρ = 1，a = 1）．値は単位の UAR の θ と区間．
  outside_families.csv（S5 の「K = 1・10」と「役割を分ける版の K = 0・1・10」）：族 1・3・4 の差を K = 1，10 で，族 2 の差を
      K = 0，1，10 で（主の版．選んだ設定．族の定義は `dai1_stats.FAMS` と同じ行）。
  nonmain.csv（S5 の「主でない版」）：族 1〜4 の差を，提案と役割を分ける版を主でない版にして，全部の K（0，1，5，10，25）で．
      族 4 の主でない版は「joint|主でない版の選んだ設定 − 選択なし」．
  teacher_types.csv（S5 の「予測を付ける側の種類」，7 節の tt）：tt の各行 − 選択なしの提案（予測だけが違う），tt の各行 − none[K]．
      no_own_modality は 7 節の表の「生徒のモダリティを含まない予測の行」（single_*，pair_*，src_*_* のうち生徒のモダリティを含まないもの）。
  ablation.csv（7.2 節の全部の行）：a. none − tgtonly（K ≥ 1），b. none − kbias（K ≥ 1），c. joint(one, ρ=1, a=1) − joint_scratch(one,
      選んだ C)（主の版が sel の K では sel の版も並べる），d. joint(one, ρ=1, a=1) − kd(one, T=1, ρ=1)，e. joint(主の版) − joint_a1(主の版)，
      f. joint_a1(主の版) − 族 3 の相手，g. joint(sel) − joint(one)（どちらも選んだ設定），h. joint(one, ρ=1, a=1) − tt(poe_none)，
      i. 役割を分けること（= 族 2．K = 0，1，10 は outside_families.csv の行と同じ量，K = 5，25 は families.csv を見る印だけ）．
      全部の K．same_setting は選ぶ段で 2 つの行の設定が同じだったか（固定のファイルから．e・f・g だけ），same_pred_all_units は
      判定から．同じ設定の e・f・g の行には note に「選ぶ段で同じ設定になった」を書く（7 節の注．e は同じ設定どうしの差で効果の
      大きさとして読まない，f は族 3 と同じ量で重ねて数えない）。
  fam3_notes.csv（9.3 節の族 3 の表の注，m57，m58，m62，C10）：(生徒, K) ごとに提案の版，相手の手法・版・設定，版が違うか，
      版が違うセルでは「joint|相手の版の選んだ設定 − 相手」の区間，蒸留の T = 1 が選ばれたか，AVT の共訓練か，共正則化の ρ = 0 か．
      fam3_candidates.csv は族 3 の候補の手法ごとのつまみと点の数（7 節の D12）。
  oracle.csv（5 節の全ラベルの参照 W*．R10 の上限の参照）：4 生徒 × 引き戻しの格子 14 点の水準（K = -1．種に依らないので単位の数は
      分割の数）．どの強さを使うかは実験 3 が仮のターゲットの出力から決めるので，ここでは選ばずに全点を並べる。
  srcstd.csv（5 節，14 節の 3 行目）：K = 0，7 種類で，出発点，ソースの統計で標準化した出発点，その差（出発点 − 後者）。
  regret.csv（9.3 節 S8 の取りこぼし，14 節の 4 行目）：選ぶ段の全部の (生徒, K, 手法, 版) と比較の基準の引き戻しの強さについて，
      格子の各点のテストの θ を判定から作り，「テストで最良の点 − 選んだ点」の区間（最良の点が同点なら選んだ点，それも無ければ格子の
      並びで先の点．この区間は同じテストで選んだ点を固定して作るので，選ぶ上乗せを含む）．差が無いときの水準は，実験 3 の設計 1 節の
      2 組（`dai3.test_halves`：分割ごと・感情ごとに (塊の名前, sample_id) の順に並べ，層をまたいで A，B と交互に配る．同じ話者が
      両方の組に入る）で，組 A の θ が最大の点（同点は格子の並びで先）の組 B の θ − 選んだ点の組 B の θ と，A と B を入れ替えた値の
      平均（θ は分割の平均．どちらかの組が 30 本未満か感情が欠ける分割はその量から外し，数を書く．区間は付けない）．
      rank_spearman は格子の点の，仮のターゲットの 30 単位の rows.csv の UAR の平均（選ぶ段が使った値）とこの回の θ の Spearman の
      順位相関（eval の段でも仮のターゲットの行をこの量のためだけに読む）．ratio_range は 7 節の K 本と疑似ラベルの重みの比が
      仮のターゲットの範囲の外（AFEW，MOSEI）か中（IEMOCAP，MELD）か（14 節）。
  recall.csv，confusion.csv（a8，S5 の感情ごとの再現率）：族の行（none，提案の主の版，役割を分ける版の主の版，族 3 の相手，
      選択なし）の感情ごとの再現率の水準と，族 1〜4 の差の感情ごとの値．全部の K．混同行列は 5 つのテストを合わせた件数の種の平均。
  tether_curve.csv，tether_sign.csv（9.1 節の引き戻しの曲線．評価の回だけ．pseudo の段では保存に無いので作らない）：(生徒, K) ごとに
      引き戻しの格子 14 点で，提案の固定した点（joint_tether），族 3 の相手の点（<相手>_tether．共訓練と共正則化は解かない．9.6 節），
      その引き戻しの none[K]（K = 0 は出発点）の水準と，族 1 型（提案 − none）と族 3 型（提案 − 相手）の差．tether_sign は 14 点のうち
      θ の符号が選んだ引き戻し（K = 0 は K = 1 の値）の点の符号と同じ点の数．選んだ引き戻しの曲線の点の判定が主の行と同じかも数える。
  sensitivity.csv，sensitivity_folds.csv，marker_only.csv，iemocap_excluded_selection.csv（9.4 節）：
      1. MOSEI のチャンネルの塊のテストだけ（`dai3.subset_masks` の channel），全生徒，族 1〜4．
      2. IEMOCAP 即興のマーカーの層（marker，nomarker）ごと，全生徒，族 1・3・4．same_direction は両方の層の点の値の符号が
         全体の効果の符号と同じこと（a18）．区間はセッションの jackknife（自由度 3，V_seed = 0 は 9.3 節の規則から出る）．
         marker_only.csv はテストごとの「付けた側を Angry，付けていない側を Neutral と答えたときの UAR」（ラベルだけから）．
      3. IEMOCAP を除いた選択（T = AFEW，MOSEI，MELD）：仮のターゲットの回の IEMOCAP 以外の 20 単位で，引き戻しの強さ
         （`dai1.select_tether`），指数（`dai1.select_w`），選んだ設定・主の版・族 3 の相手（`dai1_select.select_T`．引き戻しと指数は
         固定の値のまま）を選び直し，固定の値と一致するかを数える（成績の値は書かない）．
      4. IEMOCAP の書き起こしに `[` の注記を含む発話（manifest の transcript．129 本）をテストから除く．生徒 T，AVT，族 1〜4．
      5. AFEW の書き起こしの疑い（subsets.csv の asr_suspect）をテストから除く．生徒 T，AVT，族 1〜4．
      6. MELD の顔が競らない発話（face_compete が single）だけ．生徒 V，AVT，族 1〜4．
      7. MELD の対話を塊にした jackknife（全生徒，族 1〜4）．自由度は主の区間と同じ ⌊G_eff⌋ − 1 = 7 にし（effect_stats に MELD の
         話者の G_eff を渡す），対話の G_eff から作る自由度の区間も lo_dlgdf，hi_dlgdf に並べる．論文と結果の文で使う区間は lo・hi で，
         lo_dlgdf・hi_dlgdf は付録の記録だけにし，結果の文に使わない（gate-code の 5 回目の 2．評価の値を見る前に決める）．
      部分集合の 1 回のテスト（分割）の件数が 30 本未満の分割は θ と区間から外し（実験 3 の設計 4 節），全部の分割が外れたセルは
      区間を付けずに点の値と件数だけを書く（appendix_only）．sensitivity_folds.csv は分割ごとの点の値と件数（区間なし．付録の表．
      9.4 節の 2 のとおり IEMOCAP のマーカーの層は分割ごとの値を出さない）。
  start_C_curve.csv（9.6 節）：K = 0，7 種類，段 source の C の 12 点のそれぞれで，全部のソースの同じ C の点の識別器（W_allC，b_allC）の
      平均を出発点にしてテストに当てた UAR の水準と，「実際の出発点（各ソースが選んだ C）− 同じ C の出発点」の差．単位は分割（5）．
      特徴は `load_group` で読み，段 1 に保存した候補の平均と標準偏差で標準化する．保存の W から作り直した出発点の判定と段 1 の
      保存の判定の食い違いの本数を recon_mismatch に書く。
  same_setting.csv（7 節の共正則化の注，定義の 5 回目の記録，統計の 5 回目の記録）：(生徒, K) ごとに，joint と joint_a1 が同じ設定か
      （版ごと），主の版の提案が選択なしと同じ識別器か（族 4），指数が (1,1,1) で sel と one の版が同じか，共正則化が ρ = 0 を選んだか，
      族 3 の相手が共正則化（ρ = 0）・蒸留の T = 1・AVT の共訓練か，族のセルが記述に回ったか・同じ水準を出しうるか（固定のファイルの
      detect）．設定の一致は固定のファイルから，判定の一致は保存した判定から。
  counts.csv（7 節の共正則化の注と C10，定義の 5 回目の記録，9.2 節の記述に回したセルと族 4 の「同じ識別器」）：T ごとと 4 つの合計
      （T = ALL）で，共正則化が ρ = 0 を選んだ (生徒, K)（4 つで 80），提案が a = 1 を選んだ (生徒, K, 版) と joint_a1 と同じ設定の
      (生徒, K, 版)（4 つで 160），族 4 で選んだ設定が選択なし（one，ρ = 1，a = 1）と同じ識別器のセル（族の K で 4 つで 48），
      単一の生徒で ρ = 0 の共正則化が族 3 の相手のセル，族 3 の相手の手法の内訳，蒸留の T = 1・AVT の共訓練・版の違うセル，
      族ごとの記述に回したセルと同じ水準を出しうるセル．scope は all_K（K = 0，1，5，10，25）と family_K（族の K）．
      source = frozen は固定のファイルの設定だけから (T, 生徒, K) ごとに 1 回数える．source = predictions は same_setting.csv の
      判定の一致の列を (T, c, 生徒, K) の行で数える。
  grid_edges.csv（2 節の段 source の C，5 節の C′，7 節の格子の端．比較の検査の 3 回目の 4，定義の 5 回目の記録 3）：
      格子の下端・上端を選んだ数をつまみごとに．3 点以上の格子だけで，joint_a1 の a は数えない（`dai1_select` の edge と同じ）。
  notes_weights.csv（6 節の D11，7 節の D1）：予測（sel，one，tt の各行，共訓練の最後の巡の疑似ラベル `p3c_*_pseudo`）の感情ごとの
      重みの有効な本数 (Σw)²/Σw² が 10 未満の感情を持つ単位の数（1 つでもあればそのセルを「重みが集まったセル」とする），
      K 本と疑似ラベルの重みの比 n/(4K)（ρ = 1，a = 1．n は候補の本数）。
  convergence.csv（4 節の m42）：(段, 手法) ごとの問の数，勾配の最大が 1e-5 を超えた問の数，行き詰まった問の数．族の行（none，提案，
      役割を分ける版，相手，選択なし）の問で 1 つでも超えたら fam_rows_over_tol に印を付ける。
  assumptions.csv（14 節の 1・2 行目）：(T, K) ごとに，学び直した A・V・T の誤りの印の φ 係数（候補の K 本を除いた発話，感情ごと，
      組 3 × 感情 4 × 単位．段 2 の候補の確率 float16 の argmax．定義できない値を除いた中央値と除いた数）と基準 0.2，(T, 種類) ごとに，
      ソースごとの識別器の候補の上の「感情の平均を引いた点数」の標準偏差の単位の中央値と，ソースの間の最大 ÷ 最小と基準 2，‖W‖．
      設計 14 節の φ は仮のターゲットの回の単位で測る量なので，pseudo の段の値がそれに当たる（eval の段の値は評価の回の単位での同じ量）。
  check.csv：作り直した UAR と rows.csv の照合（種類ごとの照合の数と食い違いの数）．manifest.json：コミット，dirty，固定のファイルの
      sha256，単位のコミット，単位の数，退化の検算（同じ行どうしの差の区間が [0, 0] か），表ごとの行の数，読み込む台本の sha256 と
      単位のコミットの版との一致，counts.csv の写し（same_setting_counts），所要．全部の csv の行に mode，commit，dirty，
      frozen_sha256，units_commit を書く。
  degenerate_check.csv：(T, c, 生徒, K) ごとに，提案の主の行どうしの差（判定が全部の単位で同じ）の効果・区間・標準誤差が 0 か。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                               # GPU を使わない（読み込みの前に）

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_select as SEL  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai3 as D3  # noqa: E402
import dai_data as DD  # noqa: E402

ROOT = DD.ROOT
KS_ALL = (0,) + DD.KS
SEED = 20261104
MIN_SUB = ST.MIN_TEST                                                 # 部分集合の 1 回のテストの件数の下限（9.4 節．30 本）
GRAD_TOL = 1e-5                                                       # 4 節の m42
NEFF_MIN = 10.0                                                       # 6 節の D11
RATIO_RANGE = {"AFEW": "out", "MOSEI": "out", "IEMOCAP": "in", "MELD": "in"}   # 7 節：n/(4K) が仮のターゲットの範囲の外か中か
TABLE_ID = {n: i for i, n in enumerate(
    ["levels", "outside_families", "nonmain", "teacher_types", "ablation", "fam3_notes", "srcstd", "regret", "recall",
     "tether_curve", "sensitivity", "start_C_curve", "degenerate", "oracle"])}
KEYCOLS = {"joint": ("teacher", "rho", "a"), "tt": ("teacher", "rho", "a"), "kd": ("teacher", "temp", "rho"),
           "agree": ("teacher", "tau", "f"), "mix": ("teacher", "alpha", "f"), "cotrain": ("teacher", "rho"),
           "coreg": ("teacher", "rho"), "rolesep": ("teacher", "rho", "a"), "joint_scratch": ("teacher", "rho", "a", "C"),
           "joint_tether": ("teacher", "rho", "a", "Cw", "Cb"), "kd_tether": ("teacher", "temp", "rho", "Cw", "Cb"),
           "agree_tether": ("teacher", "tau", "f", "Cw", "Cb"), "mix_tether": ("teacher", "alpha", "f", "Cw", "Cb")}
STAT_DROP = ("p", "t", "identical")
N = ST.norm


# ------------------------------------------------------------------ 行の指定（spec）

def P3(s, K, key="p3", **want):
    """段 3 の束（p3：格子と tt と AVT の共訓練，p3z：出発点を使わない提案，p3t：引き戻しの曲線）の 1 つの問。"""
    return ("p3", key, s, int(K), tuple(sorted((k, N(v)) for k, v in want.items())))


def NONE(s, K, cw, cb):
    return ("start", s) if int(K) == 0 else ("none", s, int(K), float(cw), float(cb))


def RS(s, K, teacher, rho, a):
    return ("rolesep", s, int(K), str(teacher), float(rho), int(float(a)))


def method_spec(s: str, K: int, method: str, ver: str, params: dict):
    """選ぶ段の (手法, 版, 設定) → 行の指定。"""
    p = {k: v for k, v in params.items() if k != "pseudo_uar"}
    if method in ("joint", "joint_a1"):
        return P3(s, K, method="joint", teacher=ver, rho=p["rho"], a=p["a"])
    if method in ("kd", "agree", "mix"):
        return P3(s, K, method=method, teacher=ver, **p)
    if method == "joint_scratch":
        return P3(s, K, "p3z", method="joint_scratch", teacher=ver, rho=1.0, a=1, C=p["C"])
    if method == "rolesep":
        return RS(s, K, ver, p["rho"], p["a"])
    if method == "cotrain":
        return P3(s, K, method="cotrain", teacher="ct", rho=p["rho"]) if s == "AVT" else ("ct1", s, int(K), float(p["rho"]))
    if method == "coreg":
        return ("coreg", s, int(K), float(p["rho"]))
    if method == "tgtonly":
        return ("tgtonly", s, int(K), float(p["C"]))
    if method == "kbias":
        return ("kbias", s, int(K), float(p["Cb"]))
    raise ValueError(method)


def spec_str(spec) -> str:
    if spec[0] == "p3":
        return f"{spec[1]}:" + ",".join(f"{k}={v}" for k, v in spec[4])
    return ":".join(str(x) for x in spec)


def csv_key(spec):
    """行の指定 → rows.csv の行の鍵（Unit.idx1・idx3 と同じ形）。"""
    t = spec[0]
    if t in ("start", "srcstd"):
        return ("p1", "start" if t == "start" else "start_srcstd", spec[1])
    if t == "oracle":
        return ("p1", "oracle", spec[1], N(spec[2]), N(spec[3]))
    if t == "none":
        return ("p1", "none", spec[1], spec[2], N(spec[3]), N(spec[4]))
    if t == "kbias":
        return ("p1", "kbias", spec[1], spec[2], N(spec[3]))
    if t == "tgtonly":
        return ("p1", "tgtonly", spec[1], spec[2], N(spec[3]))
    if t == "p3":
        want = dict(spec[4])
        m = want["method"]
        return ("p3", spec[2], spec[3], m, tuple((k, want[k]) for k in KEYCOLS[m]))
    if t == "rolesep":
        _, s, K, tv, rho, a = spec
        return ("p3", s, K, "rolesep", (("teacher", tv), ("rho", N(rho)), ("a", N(a))))
    if t == "ct1":
        return ("p3", spec[1], spec[2], "cotrain", (("teacher", "ct"), ("rho", N(spec[3]))))
    if t == "coreg":
        return ("p3", spec[1], spec[2], "coreg", (("teacher", "cr"), ("rho", N(spec[3]))))
    raise ValueError(spec)


def check_cat(spec) -> str:
    return f"p3:{dict(spec[4])['method']}" if spec[0] == "p3" else spec[0]


# ------------------------------------------------------------------ ラベルと塊

class Labels:
    """コーパスの発話の表（load_group を通したラベル，塊，対話）と特徴（float32．出発点の C の曲線と 14 節の点数に使う）。"""

    def __init__(self, c: str, enc: dict):
        df, feats = DD.load_corpus(c, enc)
        self.c, self.df, self.feats = c, df, feats
        sid = df["sample_id"].astype(str).to_numpy()
        self.pos = pd.Series(np.arange(len(df)), index=sid)
        self.yv = df["label_4"].to_numpy().astype(int)
        cl = df["cluster"].astype(str).to_numpy()
        self.cl_names = np.array(sorted(set(cl)))
        self.clv = cl
        self.clc = np.searchsorted(self.cl_names, cl)                 # 名前の順の番号
        dl = df["dialogue"].astype(str).to_numpy()
        self.dl_names = np.array(sorted(set(dl)))
        self.dlc = np.searchsorted(self.dl_names, dl)
        self.foldv = df["fold"].to_numpy()

    def ix(self, sid: np.ndarray) -> np.ndarray:
        return self.pos.loc[sid].to_numpy()

    def X(self, kind: str, rows: np.ndarray) -> np.ndarray:
        return np.hstack([self.feats[m][rows] for m in kind]).astype(np.float64)


def g_eff_from_counts(n) -> float:
    n = np.asarray(list(n), float)
    return float(n.sum() ** 2 / (n ** 2).sum())


# ------------------------------------------------------------------ 1 単位の保存

class Unit:
    def __init__(self, base: Path, f: int, r: int, lab: Labels, chk: Counter):
        self.base, self.f, self.r, self.chk = base, f, r, chk
        self.p1 = np.load(f"{base}_p1.npz")
        self.p3 = np.load(f"{base}_p3.npz")
        self._p2 = None
        self.sid = self.p1["sample_test"].astype(str)
        ix = lab.ix(self.sid)
        self.y, self.cl, self.cls, self.dl = lab.yv[ix], lab.clc[ix], lab.clv[ix], lab.dlc[ix]
        assert (lab.foldv[ix] == f).all(), base
        self.halves = D3.test_halves(self.y, self.cls, self.sid)
        self.r1 = pd.read_csv(f"{base}_p1.csv", low_memory=False)
        self.r3 = pd.read_csv(f"{base}_p3.csv", low_memory=False)
        self.idx = self._index()
        self.commit = str(self.p3["commit"])
        self.frozen_sha = str(self.p3["frozen_sha256"])
        self.items, self.cache, self.checked = {}, {}, set()

    @property
    def p2(self):
        if self._p2 is None:
            self._p2 = np.load(f"{self.base}_p2.npz")
        return self._p2

    def _index(self) -> dict:
        idx = {}
        d = self.r1[self.r1.stage == 1]
        for r in d.itertuples(index=False):
            m = r.method
            if m == "none":
                k = ("p1", "none", r.kind, int(r.K), N(r.Cw), N(r.Cb))
            elif m in ("start", "start_srcstd"):
                k = ("p1", m, r.kind)
            elif m == "oracle":
                k = ("p1", "oracle", r.kind, N(r.Cw), N(r.Cb))
            elif m == "kbias":
                k = ("p1", "kbias", r.kind, int(r.K), N(r.Cb))
            elif m == "tgtonly":
                k = ("p1", "tgtonly", r.kind, int(r.K), N(r.C))
            else:
                continue
            idx.setdefault(k, []).append((float(r.uar), getattr(r, "grad", np.nan), getattr(r, "stalled", np.nan)))
        d = self.r3[(self.r3.stage == 3) & self.r3.method.isin(list(KEYCOLS))]
        cols = d.columns
        for r in d.to_dict("records"):
            m = r["method"]
            k = ("p3", r["student"], int(r["K"]), m, tuple((c, N(r[c]) if c in cols else "nan") for c in KEYCOLS[m]))
            idx.setdefault(k, []).append((float(r["uar"]), r.get("grad", np.nan), r.get("stalled", np.nan)))
        return idx

    def row(self, spec):
        """rows.csv の (uar, grad, stalled)．無いか重なるなら None。"""
        v = self.idx.get(csv_key(spec))
        if not v or len({round(x[0], 9) for x in v}) != 1:
            return None
        return v[0]

    def _items(self, name: str):
        if name not in self.items:
            arr = self.p3[name] if name.startswith("p3") else self.p1[name]
            self.items[name] = [json.loads(x) for x in arr] if name.startswith("p3") else [str(x) for x in arr]
        return self.items[name]

    def pred(self, spec) -> np.ndarray:
        if spec in self.cache:
            return self.cache[spec]
        t = spec[0]
        if t == "start":
            p = self.p1[f"p1_{spec[1]}_start_pred"]
        elif t == "srcstd":
            p = self.p1[f"p1_{spec[1]}_start_srcstd_pred"]
        elif t == "oracle":
            _, s, cw, cb = spec
            p = self.p1[f"p1_{s}_pred"][self._items(f"p1_{s}_items").index(f"all|{cw}|{cb}")]
        elif t == "none":
            _, s, K, cw, cb = spec
            p = self.p1[f"p1_{s}_pred"][self._items(f"p1_{s}_items").index(f"{K}|{cw}|{cb}")]
        elif t == "kbias":
            _, s, K, cb = spec
            p = self.p1[f"p1_{s}_kbias_pred"][self._items(f"p1_{s}_kbias_items").index(f"{K}|{cb}")]
        elif t == "tgtonly":
            _, s, K, C = spec
            p = self.p1[f"p1_{s}_tgt_pred"][self._items(f"p1_{s}_tgt_items").index(f"{K}|{C}")]
        elif t == "p3":
            _, key, s, K, want = spec
            items = self._items(f"{key}_{s}_K{K}_items")
            hits = [j for j, it in enumerate(items) if all(N(it.get(k)) == v for k, v in want)]
            assert len(hits) == 1, (self.base.name, spec, len(hits))
            p = self.p3[f"{key}_{s}_K{K}_pred"][hits[0]]
        elif t == "rolesep":
            _, s, K, tv, rho, a = spec
            ja = [(v, r) for v in ("sel", "one") for r in dai1.RHO_GRID].index((tv, rho))
            p = (self.p3[f"p3r_{s}_K{K}_pred_a"][ja] if K == 0 else
                 self.p3[f"p3r_{s}_K{K}_pred_b"][ja * len(dai1.A_GRID) + list(dai1.A_GRID).index(a)])
        elif t == "ct1":
            p = self.p3[f"p3c_K{spec[2]}_{spec[1]}_pred"][list(dai1.RHO_GRID).index(spec[3])]
        elif t == "coreg":
            p = self.p3[f"p3g_K{spec[2]}_{spec[1]}_pred"][list(dai1.COREG_RHO).index(spec[3])]
        else:
            raise ValueError(spec)
        p = np.asarray(p).astype(np.int64)
        if spec not in self.checked:                                  # 作り直した UAR と rows.csv の照合（9.1 節）
            self.checked.add(spec)
            rw = self.row(spec)
            cat = check_cat(spec)
            self.chk[(cat, "checked")] += 1
            if rw is None:
                self.chk[(cat, "missing_row")] += 1
            elif abs(uar_fast(self.y, p) - rw[0]) > 1e-6:
                self.chk[(cat, "mismatch")] += 1
        self.cache[spec] = p
        return p

    def clear(self):
        self.cache = {}


def uar_fast(y: np.ndarray, p: np.ndarray) -> float:
    """`dai1_stats.uar_full(counts(y, p, cl))` と同じ値（テストに現れる感情の再現率の平均．%）。"""
    t = np.bincount(y, minlength=4).astype(float)
    c = np.bincount(y[p == y], minlength=4).astype(float)
    ok = t > 0
    return 100.0 * float((c[ok] / t[ok]).mean())


# ------------------------------------------------------------------ (T, c) ごとの集計の場

class Ctx:
    def __init__(self, mode, out, T, c, fz, units, lab, g_eff, tables, chk):
        self.mode, self.out, self.T, self.c, self.fz = mode, out, T, c, fz
        self.units, self.lab, self.g_eff, self.tables, self.chk = units, lab, g_eff, tables, chk
        self.cname = "ALL" if mode == "eval" else c
        self.cells = {}
        self.rngs = {}
        self.masks = {}

    def rng(self, table: str):
        if table not in self.rngs:
            self.rngs[table] = np.random.default_rng([SEED, DD.CORPUS_ID[self.T], DD.CORPUS_ID[self.c], TABLE_ID[table]])
        return self.rngs[table]

    def base(self, **kw) -> dict:
        return dict(T=self.T, c=self.cname, **kw)

    def clear(self):
        self.cells = {}
        for u in self.units:
            u.clear()

    # 部分集合の印（テストの発話ごと）
    def mask(self, u: Unit, variant: str) -> np.ndarray | None:
        if variant in ("all", "dialogue"):
            return None
        key = (u.f, u.r, variant)
        if key not in self.masks:
            if variant == "halfA":
                m = u.halves == 0
            elif variant == "halfB":
                m = u.halves == 1
            elif variant == "annot_excluded":
                m = ~np.isin(u.sid, list(iemocap_annotated()))
            else:
                m = D3.subset_masks(self.c, u.sid, u.cls)[variant]
            self.masks[key] = np.asarray(m, bool)
        return self.masks[key]

    def cell(self, spec, variant="all") -> dict:
        key = (spec, variant)
        if key not in self.cells:
            out = {}
            for u in self.units:
                p = u.pred(spec)
                m = self.mask(u, variant)
                cl = u.dl if variant == "dialogue" else u.cl
                out[(u.f, u.r)] = ST.counts(u.y, p, cl) if m is None else ST.counts(u.y[m], p[m], cl[m])
            self.cells[key] = out
        return self.cells[key]

    def theta_point(self, spec, variant="all", folds=None) -> float:
        d = defaultdict(list)
        for u in self.units:
            if folds is not None and u.f not in folds:
                continue
            p = u.pred(spec)
            m = self.mask(u, variant)
            d[u.f].append(uar_fast(u.y, p) if m is None else uar_fast(u.y[m], p[m]))
        return ST.theta(d) if d else float("nan")

    def same_pred(self, a, b) -> bool:
        return all(np.array_equal(u.pred(a), u.pred(b)) for u in self.units)


def _fmt(e: dict, K: int, scale: float = 1.0) -> dict:
    e = {("interval" if k == "method" else k): v for k, v in e.items() if k not in STAT_DROP} | {"se_zero": bool(e.get("se", 1.0) == 0)}
    if K == 0:
        e["n_units"] = e["n_folds"]                                  # K = 0 は種に依らない（S9）
    if scale != 1.0:
        for k in ("effect", "lo", "hi", "se", "boot_lo", "boot_hi"):
            if k in e and e[k] is not None:
                e[k] = float(e[k]) * scale
        for k in ("v_jk", "v_seed", "v_seed_all"):
            if k in e:
                e[k] = float(e[k]) * scale ** 2
    return e


def zero_of(cell: dict) -> dict:
    return {k: (n, np.zeros_like(c), t) for k, (n, c, t) in cell.items()}


def emo_of(cell: dict, k: int) -> dict:
    keep = np.zeros(4)
    keep[k] = 1.0
    return {key: (n, c * keep, t) for key, (n, c, t) in cell.items()}


def drop_folds(cell: dict, bad: set) -> dict:
    return {k: v for k, v in cell.items() if k[0] not in bad}


def level_stats(ctx: Ctx, spec, K: int, table: str, variant="all", bad=frozenset()) -> dict:
    cA = drop_folds(ctx.cell(spec, variant), bad)
    return _fmt(ST.effect_stats(cA, zero_of(cA), ctx.g_eff, ctx.rng(table)), K)


def diff_stats(ctx: Ctx, a, b, K: int, table: str, variant="all", bad=frozenset(), g_eff=None) -> dict:
    cA, cB = drop_folds(ctx.cell(a, variant), bad), drop_folds(ctx.cell(b, variant), bad)
    e = _fmt(ST.effect_stats(cA, cB, ctx.g_eff if g_eff is None else g_eff, ctx.rng(table)), K)
    if variant == "all" and not bad:
        e["same_pred_all_units"] = ctx.same_pred(a, b)
    return e


# ------------------------------------------------------------------ 固定のファイルから行を作る

def main_ver(fz, K) -> str:
    return fz["main"][str(K)]


def other_ver(v: str) -> str:
    return "sel" if v == "one" else "one"


def sel_params(fz, s, K, method, ver) -> dict:
    return {k: v for k, v in fz["select"][f"{s}|{K}|{method}|{ver}"].items() if k != "pseudo_uar"}


def fam_rows(fz, s, K, v=None) -> dict:
    """族の 5 つの行（`dai1_stats.unit_preds` と同じ選び方）。"""
    v = v or main_ver(fz, K)
    j = sel_params(fz, s, K, "joint", v)
    rs = sel_params(fz, s, K, "rolesep", v)
    return {"prop": method_spec(s, K, "joint", v, j),
            "none": NONE(s, K, *fz["tether"][f"{s}|{max(K, 1)}"]),
            "rolesep": RS(s, K, v, rs["rho"], rs["a"]),
            "default": P3(s, K, method="joint", teacher="one", rho=1.0, a=1),
            "partner": partner_spec(fz, s, K)}


def partner_spec(fz, s, K):
    part = fz["partner"][f"{s}|{K}"]
    return method_spec(s, K, part["method"], part["version"], sel_params(fz, s, K, part["method"], part["version"]))


def tt_names(u: Unit, s: str, K: int) -> list:
    return [it["teacher"] for it in u._items(f"p3_{s}_K{K}_items") if it["method"] == "tt"]


def teacher_mods(name: str) -> set:
    if name.startswith("single_"):
        return set(name.split("_")[1])
    if name.startswith("pair_"):
        return set(name.split("_")[1])
    if name.startswith("src_"):
        return set(name.split("_")[-1])
    return set("AVT")                                                  # 融合，積，学び直し，出発点の AVT はどの生徒のモダリティも含む（a5）


_ANN: dict = {}


def iemocap_annotated() -> set:
    """IEMOCAP の書き起こしに `[` で始まる注記（[LAUGHTER] など）を含む発話（9.4 節の 4，a2）。"""
    if "s" not in _ANN:
        man = pd.read_csv(ROOT / "data" / "metadata" / "iemocap_utterance_manifest.csv", dtype=str, keep_default_na=False)
        _ANN["s"] = set(man.loc[man.transcript.str.contains("[", regex=False), "sample_id"])
    return _ANN["s"]


# ------------------------------------------------------------------ 表

def t_levels(ctx: Ctx, s: str, K: int) -> None:
    fz = ctx.fz
    v, w = main_ver(fz, K), other_ver(main_ver(fz, K))
    rows = {"none": fam_rows(fz, s, K)["none"],
            f"joint|{v}|main": method_spec(s, K, "joint", v, sel_params(fz, s, K, "joint", v)),
            f"joint|{w}|nonmain": method_spec(s, K, "joint", w, sel_params(fz, s, K, "joint", w)),
            f"rolesep|{v}|main": method_spec(s, K, "rolesep", v, sel_params(fz, s, K, "rolesep", v)),
            f"rolesep|{w}|nonmain": method_spec(s, K, "rolesep", w, sel_params(fz, s, K, "rolesep", w)),
            "joint|one|rho=1|a=1 (default)": P3(s, K, method="joint", teacher="one", rho=1.0, a=1),
            "partner": partner_spec(fz, s, K),
            f"joint_a1|{v}|main": method_spec(s, K, "joint_a1", v, sel_params(fz, s, K, "joint_a1", v)),
            f"joint_a1|{w}|nonmain": method_spec(s, K, "joint_a1", w, sel_params(fz, s, K, "joint_a1", w)),
            "joint_scratch|one": method_spec(s, K, "joint_scratch", "one", sel_params(fz, s, K, "joint_scratch", "one")),
            "joint_scratch|sel": method_spec(s, K, "joint_scratch", "sel", sel_params(fz, s, K, "joint_scratch", "sel")),
            "kd|one|temp=1|rho=1": P3(s, K, method="kd", teacher="one", temp=1, rho=1.0)}
    if K > 0:
        rows["kbias"] = method_spec(s, K, "kbias", "-", sel_params(fz, s, K, "kbias", "-"))
        rows["tgtonly"] = method_spec(s, K, "tgtonly", "-", sel_params(fz, s, K, "tgtonly", "-"))
    for pm in SEL.PARTNERS:
        for pv in SEL.PARTNER_VERSIONS.get(pm, ["sel", "one"]):
            rows[f"cand:{pm}|{pv}"] = method_spec(s, K, pm, pv, sel_params(fz, s, K, pm, pv))
    for name in tt_names(ctx.units[0], s, K):
        rows[f"tt:{name}"] = P3(s, K, method="tt", teacher=name, rho=1.0, a=1)
    part = fz["partner"][f"{s}|{K}"]
    for name, spec in rows.items():
        e = level_stats(ctx, spec, K, "levels")
        ctx.tables["levels"].append(ctx.base(student=s, K=K, row=name, spec=spec_str(spec),
                                             partner=f"{part['method']}|{part['version']}" if name == "partner" else "", **e))


FAM_NAME = {1: ("prop", "none"), 2: ("rolesep", "prop"), 3: ("prop", "partner"), 4: ("prop", "default")}


def t_outside(ctx: Ctx, s: str, K: int) -> None:
    fams = [f for f in (1, 3, 4) if K in (1, 10)] + ([2] if K in (0, 1, 10) else [])
    R = fam_rows(ctx.fz, s, K)
    for fam in fams:
        a, b = FAM_NAME[fam]
        e = diff_stats(ctx, R[a], R[b], K, "outside_families")
        ctx.tables["outside_families"].append(ctx.base(family=fam, student=s, K=K, A=a, B=b, version=main_ver(ctx.fz, K),
                                                       spec_A=spec_str(R[a]), spec_B=spec_str(R[b]), **e))


def t_nonmain(ctx: Ctx, s: str, K: int) -> None:
    w = other_ver(main_ver(ctx.fz, K))
    R = fam_rows(ctx.fz, s, K, w)
    for fam in (1, 2, 3, 4):
        a, b = FAM_NAME[fam]
        e = diff_stats(ctx, R[a], R[b], K, "nonmain")
        ctx.tables["nonmain"].append(ctx.base(family=fam, student=s, K=K, A=a, B=b, version=w,
                                              spec_A=spec_str(R[a]), spec_B=spec_str(R[b]), **e))


def t_teacher_types(ctx: Ctx, s: str, K: int) -> None:
    R = fam_rows(ctx.fz, s, K)
    for name in tt_names(ctx.units[0], s, K):
        spec = P3(s, K, method="tt", teacher=name, rho=1.0, a=1)
        no_own = s != "AVT" and s not in teacher_mods(name) and not name.startswith(("fused", "poe", "noret", "ret", "start"))
        for bn in ("default", "none"):
            e = diff_stats(ctx, spec, R[bn], K, "teacher_types")
            ctx.tables["teacher_types"].append(ctx.base(student=s, K=K, teacher=name, B=bn, no_own_modality=no_own,
                                                        spec_B=spec_str(R[bn]), **e))


def same_setting(fz, s, K, m1, v1, m2, v2) -> bool:
    a, b = sel_params(fz, s, K, m1, v1), sel_params(fz, s, K, m2, v2)
    return {k: N(x) for k, x in a.items()} == {k: N(x) for k, x in b.items()}


def t_ablation(ctx: Ctx, s: str, K: int) -> None:
    fz = ctx.fz
    v = main_ver(fz, K)
    R = fam_rows(fz, s, K)
    d1 = P3(s, K, method="joint", teacher="one", rho=1.0, a=1)
    rows = []
    if K > 0:
        rows.append(("a", "ソースの資産（出発点）", R["none"], method_spec(s, K, "tgtonly", "-", sel_params(fz, s, K, "tgtonly", "-")), None))
        rows.append(("b", "K 本で重みを学ぶこと", R["none"], method_spec(s, K, "kbias", "-", sel_params(fz, s, K, "kbias", "-")), None))
    rows.append(("c", "出発点（初期値と引き戻しの中心）", d1,
                 method_spec(s, K, "joint_scratch", "one", sel_params(fz, s, K, "joint_scratch", "one")), None))
    if v == "sel":
        rows.append(("c-sel", "出発点（主の版 sel）", P3(s, K, method="joint", teacher="sel", rho=1.0, a=1),
                     method_spec(s, K, "joint_scratch", "sel", sel_params(fz, s, K, "joint_scratch", "sel")), None))
    rows.append(("d", "感情ごとの総量をそろえる操作", d1, P3(s, K, method="kd", teacher="one", temp=1, rho=1.0), None))
    rows.append(("e", "K 本の倍率 a", R["prop"], method_spec(s, K, "joint_a1", v, sel_params(fz, s, K, "joint_a1", v)),
                 same_setting(fz, s, K, "joint", v, "joint_a1", v)))
    part = fz["partner"][f"{s}|{K}"]
    rows.append(("f", "a だけを外して族 3 の相手と比べる", method_spec(s, K, "joint_a1", v, sel_params(fz, s, K, "joint_a1", v)),
                 R["partner"], same_setting(fz, s, K, "joint", v, "joint_a1", v)))
    rows.append(("g", "指数の選択", method_spec(s, K, "joint", "sel", sel_params(fz, s, K, "joint", "sel")),
                 method_spec(s, K, "joint", "one", sel_params(fz, s, K, "joint", "one")),
                 tuple(fz["w"][str(K)]) == (1.0, 1.0, 1.0) and same_setting(fz, s, K, "joint", "sel", "joint", "one")))
    rows.append(("h", "学び直し", d1, P3(s, K, method="tt", teacher="poe_none", rho=1.0, a=1), None))
    for rid, label, a, b, same in rows:
        e = diff_stats(ctx, a, b, K, "ablation")
        note = ""
        if same:                                                       # 7 節の共正則化の注，定義の 5 回目の記録
            note = {"e": "選ぶ段で joint と joint_a1 が同じ設定になった（同じ設定どうしの差．効果の大きさとして読まない）",
                    "f": "選ぶ段で joint と joint_a1 が同じ設定になった（族 3 と同じ量．重ねて数えない）",
                    "g": "指数が (1,1,1) で sel と one が同じ設定（同じ設定どうしの差）"}[rid]
        ctx.tables["ablation"].append(ctx.base(row=rid, removed=label, student=s, K=K, spec_A=spec_str(a), spec_B=spec_str(b),
                                               same_setting=same, note=note, partner=f"{part['method']}|{part['version']}", **e))
    ctx.tables["ablation"].append(ctx.base(row="i", removed="役割を分けること（族 2）", student=s, K=K,     # 同じ量を二度作らない
                                           spec_A=spec_str(R["rolesep"]), spec_B=spec_str(R["prop"]),
                                           see="stats/families.csv の族 2" if K in (5, 25) else "outside_families.csv の族 2"))


def t_fam3_notes(ctx: Ctx, s: str, K: int) -> None:
    fz = ctx.fz
    v = main_ver(fz, K)
    part = fz["partner"][f"{s}|{K}"]
    pm, pv = part["method"], part["version"]
    pp = sel_params(fz, s, K, pm, pv)
    row = ctx.base(student=s, K=K, in_family=K in ST.KS_MAIN, prop_version=v, partner_method=pm, partner_version=pv,
                   partner_setting=json.dumps({k: N(x) for k, x in pp.items()}, sort_keys=True),
                   versions_differ=(pm in ("kd", "agree", "mix") and pv != v),
                   kd_temp1=(pm == "kd" and float(pp.get("temp", 0)) == 1.0),
                   cotrain_avt=(pm == "cotrain" and s == "AVT"),
                   coreg=(pm == "coreg"), coreg_rho0_single=(pm == "coreg" and float(pp["rho"]) == 0.0 and s != "AVT"),
                   coreg_avt=(pm == "coreg" and s == "AVT"))
    if row["versions_differ"]:
        a = method_spec(s, K, "joint", pv, sel_params(fz, s, K, "joint", pv))
        e = diff_stats(ctx, a, partner_spec(fz, s, K), K, "fam3_notes")
        row |= {f"aligned_{k}": x for k, x in e.items()} | {"aligned_spec_A": spec_str(a)}
    ctx.tables["fam3_notes"].append(row)


def t_oracle(ctx: Ctx) -> None:
    """5 節の全ラベルの参照 W*（候補の全ラベルで出発点から学び直した識別器）の水準を，引き戻しの格子 14 点で．K 本を使わないので
    種に依らず，単位の数は分割の数．どの強さを説明に使うかは実験 3 が決める（ここでは選ばない）。"""
    for s in DD.STUDENTS:
        for cw, cb in dai1.TETHER:
            spec = ("oracle", s, float(cw), float(cb))
            ctx.tables["oracle"].append(ctx.base(student=s, K=-1, Cw=cw, Cb=cb, row="oracle", spec=spec_str(spec),
                                                 **level_stats(ctx, spec, 0, "oracle")))


def t_srcstd(ctx: Ctx) -> None:
    for kind in DD.KINDS:
        a, b = ("start", kind), ("srcstd", kind)
        for name, spec in (("start", a), ("start_srcstd", b)):
            ctx.tables["srcstd"].append(ctx.base(kind=kind, K=0, row=name, **level_stats(ctx, spec, 0, "srcstd")))
        ctx.tables["srcstd"].append(ctx.base(kind=kind, K=0, row="start − start_srcstd", **diff_stats(ctx, a, b, 0, "srcstd")))


def grid_of(method: str) -> list[dict]:
    gm = "joint" if method == "joint_a1" else method
    keys = [k for k, _ in SEL.GRIDS[gm]]
    pts = [dict(zip(keys, combo)) for combo in itertools.product(*[v for _, v in SEL.GRIDS[gm]])]
    return [p for p in pts if not (method == "joint_a1" and float(p["a"]) != 1.0)]


def half_ok_folds(ctx: Ctx) -> set:
    ok = set()
    for u in ctx.units:
        good = True
        for h in (0, 1):
            yy = u.y[u.halves == h]
            good &= len(yy) >= ST.MIN_TEST and (np.bincount(yy, minlength=4) > 0).all()
        if good:
            ok.add(u.f)
    return ok


def t_regret(ctx: Ctx, s: str, K: int, pseudo_mean) -> None:
    fz = ctx.fz
    okf = half_ok_folds(ctx)
    cells = []
    for key, ch in fz["select"].items():
        ss, KK, method, ver = key.split("|")
        if ss != s or int(KK) != K:
            continue
        pts = [method_spec(s, K, method, ver, p) for p in grid_of(method)]
        cells.append((method, ver, pts, method_spec(s, K, method, ver, ch)))
    if K > 0:
        pts = [NONE(s, K, cw, cb) for cw, cb in dai1.TETHER]
        cells.append(("none(tether)", "-", pts, NONE(s, K, *fz["tether"][f"{s}|{K}"])))
    for method, ver, pts, chosen in cells:
        uniq = list(dict.fromkeys(pts))
        th = np.array([ctx.theta_point(p) for p in uniq])
        isel = uniq.index(chosen)
        best = isel if th[isel] >= th.max() else int(np.flatnonzero(th == th.max())[0])
        e = diff_stats(ctx, uniq[best], chosen, K, "regret")
        nulls = []
        if okf:
            tA = np.array([ctx.theta_point(p, "halfA", okf) for p in uniq])
            tB = np.array([ctx.theta_point(p, "halfB", okf) for p in uniq])
            bA, bB = int(np.argmax(tA)), int(np.argmax(tB))           # 同点は格子の並びで先
            nulls = [tB[bA] - tB[isel], tA[bB] - tA[isel]]
        pm = np.array([pseudo_mean(p) for p in uniq])
        okr = np.isfinite(pm) & np.isfinite(th)
        rho = (float(st.spearmanr(pm[okr], th[okr]).statistic)
               if okr.sum() >= 3 and np.ptp(pm[okr]) > 0 and np.ptp(th[okr]) > 0 else np.nan)
        ctx.tables["regret"].append(ctx.base(student=s, K=K, method=method, version=ver, n_points=len(uniq),
                                             chosen=spec_str(chosen), best=spec_str(uniq[best]), best_is_chosen=best == isel,
                                             null_level=float(np.mean(nulls)) if nulls else np.nan,
                                             null_folds_used=len(okf), null_folds_dropped=DD.N_FOLDS - len(okf),
                                             rank_spearman=rho, rank_n_points=int(okr.sum()), ratio_range=RATIO_RANGE[ctx.T], **e))


def t_recall(ctx: Ctx, s: str, K: int) -> None:
    R = fam_rows(ctx.fz, s, K)
    for u in ctx.units:
        assert (np.bincount(u.y, minlength=4) > 0).all(), "感情が欠けるテストがある（4 倍の換算が成り立たない）"
    for name, spec in R.items():
        cA = ctx.cell(spec)
        for k in range(4):
            ck = emo_of(cA, k)
            e = _fmt(ST.effect_stats(ck, zero_of(ck), ctx.g_eff, ctx.rng("recall")), K, scale=4.0)
            pooled = []
            for r in sorted({u.r for u in ctx.units}):
                c_ = sum(cA[(f, r)][1][:, k].sum() for f in range(DD.N_FOLDS) if (f, r) in cA)
                t_ = sum(cA[(f, r)][2][:, k].sum() for f in range(DD.N_FOLDS) if (f, r) in cA)
                pooled.append(100.0 * c_ / t_)
            ctx.tables["recall"].append(ctx.base(kind="level", student=s, K=K, row=name, emotion=k, spec=spec_str(spec),
                                                 recall_pooled=float(np.mean(pooled)), **e))
        conf = np.zeros((4, 4))
        rs = sorted({u.r for u in ctx.units})
        for u in ctx.units:
            p = u.pred(spec)
            np.add.at(conf, (u.y, p), 1.0 / len(rs))
        for i in range(4):
            for j in range(4):
                ctx.tables["confusion"].append(ctx.base(student=s, K=K, row=name, true=i, pred=j, count_mean_over_seeds=conf[i, j],
                                                        spec=spec_str(spec)))
    for fam, (a, b) in FAM_NAME.items():
        cA, cB = ctx.cell(R[a]), ctx.cell(R[b])
        for k in range(4):
            e = _fmt(ST.effect_stats(emo_of(cA, k), emo_of(cB, k), ctx.g_eff, ctx.rng("recall")), K, scale=4.0)
            ctx.tables["recall"].append(ctx.base(kind=f"family{fam}", student=s, K=K, row=f"{a} − {b}", emotion=k, **e))


def t_tether(ctx: Ctx, s: str, K: int) -> None:
    u0 = ctx.units[0]
    name = f"p3t_{s}_K{K}_items"
    if name not in u0.p3:
        return                                                         # 仮のターゲットの回には曲線が無い（評価の回だけ．9.1 節）
    items = u0._items(name)
    fz = ctx.fz
    R = fam_rows(fz, s, K)
    sel_t = tuple(float(x) for x in fz["tether"][f"{s}|{max(K, 1)}"])
    curves = {"prop": [], "partner": []}
    for it in items:
        tag = it["tag"]
        want = {k: v for k, v in it.items() if k not in ("n_kept",)}
        curves["prop" if tag == "prop_sel" else "partner"].append((float(it["Cw"]), float(it["Cb"]), P3(s, K, "p3t", **want)))
    res = defaultdict(dict)
    for cw, cb in dai1.TETHER:
        nn = NONE(s, K, cw, cb)
        pp = [x[2] for x in curves["prop"] if (x[0], x[1]) == (cw, cb)]
        qq = [x[2] for x in curves["partner"] if (x[0], x[1]) == (cw, cb)]
        rowsL = [("none", nn)] + ([("prop", pp[0])] if pp else []) + ([("partner", qq[0])] if qq else [])
        for nm, spec in rowsL:
            ctx.tables["tether_curve"].append(ctx.base(student=s, K=K, Cw=cw, Cb=cb, kind="level", row=nm, spec=spec_str(spec),
                                                       selected_tether=(cw, cb) == sel_t, **level_stats(ctx, spec, K, "tether_curve")))
        for fam, other in ((1, nn), (3, qq[0] if qq else None)):
            if not pp or other is None:
                continue
            e = diff_stats(ctx, pp[0], other, K, "tether_curve")
            res[fam][(cw, cb)] = e["effect"]
            ctx.tables["tether_curve"].append(ctx.base(student=s, K=K, Cw=cw, Cb=cb, kind=f"family{fam}", row=f"prop − {'none' if fam == 1 else 'partner'}",
                                                       selected_tether=(cw, cb) == sel_t, **e))
        if (cw, cb) == sel_t:                                          # 選んだ引き戻しの曲線の点は主の行と同じ判定か
            if pp:
                res["check"]["prop"] = ctx.same_pred(pp[0], R["prop"])
            if qq:
                res["check"]["partner"] = ctx.same_pred(qq[0], R["partner"])
    for fam in (1, 3):
        if not res[fam]:
            continue
        ref = np.sign(res[fam].get(sel_t, np.nan))
        sg = np.sign(np.array(list(res[fam].values())))
        ctx.tables["tether_sign"].append(ctx.base(student=s, K=K, family=fam, n_points=len(sg), sign_at_selected=ref,
                                                  n_same_sign=int((sg == ref).sum()), all_same_sign=bool((sg == ref).all()),
                                                  n_pos=int((sg > 0).sum()), n_neg=int((sg < 0).sum()), n_zero=int((sg == 0).sum()),
                                                  selected_point_matches_main=res["check"].get("prop" if fam == 1 else "partner")))


SUBSETS = {  # 9.4 節：(コーパス, 印の名前, 生徒, 族)
    "MOSEI": [("channel", DD.STUDENTS, (1, 2, 3, 4))],
    "IEMOCAP": [("marker", DD.STUDENTS, (1, 3, 4)), ("nomarker", DD.STUDENTS, (1, 3, 4)), ("annot_excluded", ("T", "AVT"), (1, 2, 3, 4))],
    "AFEW": [("asr_ok", ("T", "AVT"), (1, 2, 3, 4))],
    "MELD": [("face_single", ("V", "AVT"), (1, 2, 3, 4)), ("dialogue", DD.STUDENTS, (1, 2, 3, 4))],
}


def t_sensitivity(ctx: Ctx, s: str, K: int) -> None:
    R = fam_rows(ctx.fz, s, K)
    for variant, studs, fams in SUBSETS.get(ctx.c, []):
        if s not in studs:
            continue
        nf = {}
        for u in ctx.units:
            m = ctx.mask(u, variant)
            nf[u.f] = int(len(u.y) if m is None else m.sum())
        bad = {f for f, n in nf.items() if n < MIN_SUB}
        for fam in fams:
            _, _, ks = ST.FAMS[fam]
            if K not in ks:
                continue
            a, b = FAM_NAME[fam]
            base = ctx.base(subset=variant, family=fam, student=s, K=K, A=a, B=b, n_test_by_fold=json.dumps(nf),
                            folds_under_30=",".join(map(str, sorted(bad))))
            overall = ctx.theta_point(R[a]) - ctx.theta_point(R[b])
            if len(bad) == len(nf):
                pt = ctx.theta_point(R[a], variant) - ctx.theta_point(R[b], variant)
                ctx.tables["sensitivity"].append(base | dict(appendix_only=True, effect=pt, overall_effect=overall))
            else:
                g_eff = ctx.g_eff
                extra = {}
                if variant == "dialogue":
                    dl = Counter(np.concatenate([u.dl for u in ctx.units if u.r == ctx.units[0].r]).tolist())
                    extra["g_eff_dialogue"] = g_eff_from_counts(dl.values())
                e = diff_stats(ctx, R[a], R[b], K, "sensitivity", variant, bad, g_eff=g_eff)
                if variant == "dialogue":
                    df_d = max(1, int(np.floor(extra["g_eff_dialogue"])) - 1)
                    q = float(st.t.ppf(0.975, df_d))
                    extra |= dict(df_dialogue=df_d, lo_dlgdf=e["effect"] - q * e["se"], hi_dlgdf=e["effect"] + q * e["se"])
                if variant in ("marker", "nomarker"):
                    extra["overall_sign"] = float(np.sign(overall))
                ctx.tables["sensitivity"].append(base | dict(appendix_only=False, overall_effect=overall, **extra, **e))
            if variant not in ("marker", "nomarker", "dialogue"):        # 9.4 節の 2：マーカーの層は分割ごとの値を出さない
                for f in sorted(nf):
                    pt = ctx.theta_point(R[a], variant, {f}) - ctx.theta_point(R[b], variant, {f})
                    ctx.tables["sensitivity_folds"].append(ctx.base(subset=variant, family=fam, student=s, K=K, fold=f,
                                                                    n_test=nf[f], under_30=nf[f] < MIN_SUB, effect=pt))


def marker_same_direction(tab: list) -> None:
    """9.4 節の 2 の「同じ向き」：両方の層の点の値の符号が全体の効果の符号と同じ（a18）。"""
    by = defaultdict(dict)
    for r in tab:
        if r.get("subset") in ("marker", "nomarker"):
            by[(r["T"], r["c"], r["family"], r["student"], r["K"])][r["subset"]] = r
    for _, d in by.items():
        if len(d) != 2:
            continue
        sgn = d["marker"].get("overall_sign", np.nan)
        same = bool(sgn != 0 and all(np.sign(x.get("effect", np.nan)) == sgn for x in d.values()))
        for x in d.values():
            x["same_direction"] = same


def t_marker_only(ctx: Ctx) -> None:
    if ctx.c != "IEMOCAP":
        return
    allv = []
    for u in [x for x in ctx.units if x.r == ctx.units[0].r]:
        m = ctx.mask(u, "marker")
        p = np.where(m, 3, 0)
        v = uar_fast(u.y, p)
        allv.append(v)
        ctx.tables["marker_only"].append(ctx.base(fold=u.f, n_test=len(u.y), uar_marker_only=v))
    ctx.tables["marker_only"].append(ctx.base(fold="mean", n_test=np.nan, uar_marker_only=float(np.mean(allv))))


def t_start_C_curve(ctx: Ctx, srcs: list) -> None:
    lab = ctx.lab
    r0 = ctx.units[0].r
    us = [u for u in ctx.units if u.r == r0]
    src = {sname: {k: np.load(ctx.out / "source" / f"{sname}_{k}.npz") for k in DD.KINDS} for sname in srcs}
    grid = list(src[srcs[0]]["A"]["C_grid"])
    for kind in DD.KINDS:
        selC = {sname: float(src[sname][kind]["C"]) for sname in srcs}
        preds = {j: {} for j in range(len(grid))}
        recon_bad = 0
        for u in us:
            Xt = (lab.X(kind, lab.ix(u.sid)) - u.p1[f"stat_{kind}_mu"]) / u.p1[f"stat_{kind}_sd"]
            W = np.mean([src[sname][kind]["W"] for sname in srcs], 0)
            b = np.mean([src[sname][kind]["b"] for sname in srcs], 0)
            recon_bad += int((np.argmax(Xt @ W.T + b, 1) != u.p1[f"p1_{kind}_start_pred"]).sum())
            for j in range(len(grid)):
                Wj = np.mean([src[sname][kind]["W_allC"][j].astype(np.float64) for sname in srcs], 0)
                bj = np.mean([src[sname][kind]["b_allC"][j].astype(np.float64) for sname in srcs], 0)
                preds[j][u.f] = np.argmax(Xt @ Wj.T + bj, 1)
        for j, C in enumerate(grid):
            cell = {(u.f, r0): ST.counts(u.y, preds[j][u.f], u.cl) for u in us}
            start = {(u.f, r0): ST.counts(u.y, u.p1[f"p1_{kind}_start_pred"].astype(np.int64), u.cl) for u in us}
            lv = _fmt(ST.effect_stats(cell, zero_of(cell), ctx.g_eff, ctx.rng("start_C_curve")), 0)
            df_ = _fmt(ST.effect_stats(start, cell, ctx.g_eff, ctx.rng("start_C_curve")), 0)
            ctx.tables["start_C_curve"].append(ctx.base(kind=kind, C=float(C), sources=",".join(srcs),
                                                        selected_C=json.dumps(selC), recon_mismatch=recon_bad,
                                                        all_sources_selected_this_C=all(np.isclose(v, C) for v in selC.values()),
                                                        **{f"level_{k}": v for k, v in lv.items()},
                                                        **{f"start_minus_{k}": v for k, v in df_.items()}))


def t_same_setting(ctx: Ctx, s: str, K: int) -> None:
    fz = ctx.fz
    v = main_ver(fz, K)
    part = fz["partner"][f"{s}|{K}"]
    pp = sel_params(fz, s, K, part["method"], part["version"])
    j = sel_params(fz, s, K, "joint", v)
    R = fam_rows(fz, s, K)
    det = fz.get("detect", {})
    row = ctx.base(student=s, K=K, in_family=K in ST.KS_MAIN, main=v, w=json.dumps(fz["w"][str(K)]),
                   joint_eq_a1_sel=same_setting(fz, s, K, "joint", "sel", "joint_a1", "sel"),
                   joint_eq_a1_one=same_setting(fz, s, K, "joint", "one", "joint_a1", "one"),
                   fam4_same_as_default=(v == "one" and float(j["rho"]) == 1.0 and int(float(j["a"])) == 1),
                   w_is_111=tuple(fz["w"][str(K)]) == (1.0, 1.0, 1.0),
                   coreg_selected_rho0=float(sel_params(fz, s, K, "coreg", "cr")["rho"]) == 0.0,
                   partner_method=part["method"], partner_version=part["version"],
                   partner_coreg_rho0=(part["method"] == "coreg" and float(pp["rho"]) == 0.0),
                   partner_kd_temp1=(part["method"] == "kd" and float(pp.get("temp", 0)) == 1.0),
                   partner_cotrain_avt=(part["method"] == "cotrain" and s == "AVT"))
    row["pred_joint_eq_a1_main"] = ctx.same_pred(R["prop"], method_spec(s, K, "joint_a1", v, sel_params(fz, s, K, "joint_a1", v)))
    row["pred_fam4_eq_default"] = ctx.same_pred(R["prop"], R["default"])
    row["pred_sel_eq_one"] = ctx.same_pred(method_spec(s, K, "joint", "sel", sel_params(fz, s, K, "joint", "sel")),
                                           method_spec(s, K, "joint", "one", sel_params(fz, s, K, "joint", "one")))
    row["pred_partner_eq_none"] = ctx.same_pred(R["partner"], R["none"])
    for fam, (_, _, ks) in ST.FAMS.items():
        if K in ks:
            key = f"{fam}|{s}|{K}"
            row[f"fam{fam}_descriptive"] = key in det.get("descriptive_cells", [])
            row[f"fam{fam}_same_level_possible"] = key in det.get("same_level_cells", [])
    ctx.tables["same_setting"].append(row)


def t_notes_weights(ctx: Ctx) -> None:
    acc = defaultdict(lambda: [0, 0, np.inf])                          # [単位の数, 10 未満の感情を持つ単位の数, 最小]
    for u in ctx.units:
        d = u.r3[(u.r3.method == "_teacher_stats")]
        for r in d.to_dict("records"):
            mn = min(float(r[f"neff_{k}"]) for k in range(4))
            stu = r.get("student")
            key = (int(r["K"]), str(r["teacher"]), "" if not isinstance(stu, str) else stu)
            a = acc[key]
            a[0] += 1; a[1] += int(mn < NEFF_MIN); a[2] = min(a[2], mn)
        for K in KS_ALL:
            for m in "AVT":
                P = u.p3[f"p3c_K{K}_{m}_pseudo"].astype(np.float64)    # (ρ, n, 4)
                neff = (P.sum(1) ** 2) / np.maximum((P ** 2).sum(1), 1e-30)
                for jr, rho in enumerate(dai1.RHO_GRID):
                    mn = float(neff[jr].min())
                    a = acc[(K, f"cotrain_pseudo|rho={N(rho)}", m)]
                    a[0] += 1; a[1] += int(mn < NEFF_MIN); a[2] = min(a[2], mn)
    for (K, teacher, stu), (n, nb, mn) in sorted(acc.items()):
        ctx.tables["notes_weights"].append(ctx.base(note="neff", K=K, teacher=teacher, student=stu, n_units=n,
                                                    n_units_neff_under_10=nb, min_neff=mn, concentrated=nb > 0))
    npool = {u.f: len(u.p1["sample_pool"]) for u in ctx.units}
    for K in DD.KS:
        r = np.array([n / (4 * K) for n in npool.values()])
        ctx.tables["notes_weights"].append(ctx.base(note="n_over_4K", K=K, ratio_mean=float(r.mean()), ratio_min=float(r.min()),
                                                    ratio_max=float(r.max()), ratio_range=RATIO_RANGE[ctx.T]))


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def t_convergence(ctx: Ctx) -> None:
    """問ごとの行（grad，stalled）と束の行（_conv*：grad_max，n_stalled）から数える．単一の生徒の共訓練は束の行だけにある。"""
    acc = defaultdict(lambda: [0, 0, 0, 0.0])
    for u in ctx.units:
        r2 = pd.read_csv(f"{u.base}_p2.csv", low_memory=False)
        for st_, d in ((1, u.r1), (2, r2), (3, u.r3)):
            for r in d.to_dict("records"):
                m = r["method"]
                g = _num(r.get("grad"))
                if g is None:
                    g = _num(r.get("grad_max"))
                if g is None or m == "_teacher_stats":
                    continue
                ns = str(r.get("stalled")) == "True" or (_num(r.get("n_stalled")) or 0) > 0
                a = acc[(st_, m)]
                a[0] += 1; a[1] += int(g > GRAD_TOL); a[2] += int(ns); a[3] = max(a[3], g)
    for (st_, m), (n, no, ns, gm) in sorted(acc.items()):
        ctx.tables["convergence"].append(ctx.base(stage=st_, method=m, n_rows=n, n_grad_over_tol=no, n_stalled=ns, grad_max=gm))
    # 族の行の問で 1 つでも 1e-5 を超えたか（4 節の m42）
    for s in DD.STUDENTS:
        for K in KS_ALL:
            R = fam_rows(ctx.fz, s, K)
            over = []
            for nm, spec in R.items():
                if spec[0] == "start":
                    continue
                for u in ctx.units:
                    if spec[0] == "ct1":
                        d = u.r3[(u.r3.method == "_conv_cotrain") & (u.r3.kind == s) & (u.r3.K == K)]
                        g = float(d.grad_max.max())
                    else:
                        rw = u.row(spec)
                        g = _num(rw[1]) if rw is not None else None
                    if g is not None and g > GRAD_TOL:
                        over.append(nm)
                        break
            ctx.tables["convergence"].append(ctx.base(stage="family_rows", method=f"{s}|K{K}", fam_rows_over_tol=",".join(over),
                                                      flag=bool(over)))


def t_assumptions(ctx: Ctx, srcs: list, phi_acc: dict) -> None:
    lab = ctx.lab
    # 14 節の 1 行目：学び直した A・V・T の誤りの φ（候補の K 本を除いた発話，感情ごと）
    for u in ctx.units:
        pool = u.p1["sample_pool"].astype(str)
        yp = lab.yv[lab.ix(pool)]
        for K in KS_ALL:
            notk = np.ones(len(pool), bool) if K == 0 else ~np.isin(pool, u.p1[f"sample_K{K}"].astype(str))
            pr = {m: np.argmax(u.p2[f"p2_K{K}_ret_{m}_pool"].astype(np.float32), 1) for m in "AVT"}
            for m1, m2 in itertools.combinations("AVT", 2):
                for k in range(4):
                    ix = notk & (yp == k)
                    e1, e2 = (pr[m1][ix] != k).astype(float), (pr[m2][ix] != k).astype(float)
                    v = float(np.corrcoef(e1, e2)[0, 1]) if e1.std() > 0 and e2.std() > 0 else np.nan
                    phi_acc.setdefault((ctx.T, K), []).append(v)
    # 14 節の 2 行目：ソースごとの識別器の点数の大きさ（候補の上，感情の平均を引いた点数の標準偏差）
    r0 = ctx.units[0].r
    for kind in DD.KINDS:
        sds = defaultdict(list)
        zs = {sname: np.load(ctx.out / "source" / f"{sname}_{kind}.npz") for sname in srcs}
        wn = {sname: float(np.linalg.norm(z["W"])) for sname, z in zs.items()}
        for u in [x for x in ctx.units if x.r == r0]:
            Xp = (lab.X(kind, lab.ix(u.p1["sample_pool"].astype(str))) - u.p1[f"stat_{kind}_mu"]) / u.p1[f"stat_{kind}_sd"]
            for sname, z in zs.items():
                Z = Xp @ z["W"].T + z["b"]
                sds[sname].append(float((Z - Z.mean(1, keepdims=True)).std()))
            del Xp
        med = {k: float(np.median(v)) for k, v in sds.items()}
        ratio = max(med.values()) / min(med.values())
        ctx.tables["assumptions"].append(ctx.base(assumption="出発点の等しい重み（ソースごとの点数の大きさが近い）", key=kind,
                                                  value=ratio, criterion="< 2", holds=ratio < 2,
                                                  detail=json.dumps({"median_sd": med, "W_norm": wn})))


# ------------------------------------------------------------------ 仮のターゲットの格子の平均（取りこぼしの順位と 9.4 節の 3）

def check_pseudo_rows(T: str, fz: dict, *frames) -> None:
    """読んだ仮のターゲットの行が，固定のファイルを作った回のもの（単位の一覧とコミット）であることを確かめる（I12）。"""
    for d in frames:
        if SEL.unit_names(d) != sorted(fz["units"]):
            sys.exit(f"{T}: 仮のターゲットの行の単位が固定のファイルの units と違う")
        if set(d.commit.astype(str)) != {str(fz["pseudo_commit"])}:
            sys.exit(f"{T}: 仮のターゲットの行のコミット {sorted(set(d.commit.astype(str)))} が固定のファイルの pseudo_commit と違う")


class PseudoMeans:
    """選ぶ段と同じ，仮のターゲットの 30 単位の rows.csv の UAR の格子の点ごとの平均。"""

    def __init__(self, out: Path, T: str, fz: dict):
        p1, p3 = SEL.load(out, T, 1), SEL.load(out, T, 3)
        check_pseudo_rows(T, fz, p1, p3)
        self.m = {}
        d = p1[p1.stage == 1]
        for m in ("none", "kbias", "tgtonly"):
            x = d[d.method == m]
            cols = {"none": ["Cw", "Cb"], "kbias": ["Cb"], "tgtonly": ["C"]}[m]
            for key, g in x.groupby(["kind", "K"] + cols):
                kind, K, *vals = key
                self.m[("p1", m, kind, int(K), *[N(v) for v in vals])] = float(g.uar.mean())
        d = p3[(p3.stage == 3) & p3.method.isin(list(KEYCOLS))]
        for m, x in d.groupby("method"):
            cols = list(KEYCOLS[m])
            for key, g in x.groupby(["student", "K"] + cols):
                s, K, *vals = key
                self.m[("p3", s, int(K), m, tuple((c, N(v)) for c, v in zip(cols, vals)))] = float(g.uar.mean())

    def __call__(self, spec) -> float:
        return self.m.get(csv_key(spec), np.nan)


def iemocap_excluded_selection(out: Path, T: str, fz: dict) -> list:
    """9.4 節の 3：IEMOCAP を除いた 2 つの仮のターゲットの 20 単位で選び直し，固定の値と一致するかを数える（値は書かない）。"""
    if T == "IEMOCAP":
        return []
    p1, p2, p3 = (SEL.load(out, T, w) for w in (1, 2, 3))
    check_pseudo_rows(T, fz, p1, p2, p3)
    keep = [d[d.c != "IEMOCAP"] for d in (p1, p2, p3)]
    n_units = len(SEL.unit_names(keep[2]))
    rows = []
    teth = dai1.select_tether(keep[0])
    for (kind, K), v in sorted(teth.items()):
        rows.append(dict(T=T, category="tether", key=f"{kind}|{K}", reselected=json.dumps(v),
                         match=tuple(float(x) for x in v) == tuple(float(x) for x in fz["tether"][f"{kind}|{K}"])))
    w = dai1.select_w(keep[1])
    for K, v in sorted(w.items()):
        rows.append(dict(T=T, category="w", key=str(K), reselected=json.dumps(v), match=tuple(v) == tuple(fz["w"][str(K)])))
    r = SEL.select_T({"tether": fz["tether"], "w": fz["w"]}, *keep, n_units=n_units)
    for key, ch in r["select"].items():
        a = {k: N(x) for k, x in ch.items() if k != "pseudo_uar"}
        b = {k: N(x) for k, x in fz["select"][key].items() if k != "pseudo_uar"}
        rows.append(dict(T=T, category="select", key=key, reselected=json.dumps(a, sort_keys=True), match=a == b))
    for K, v in r["main"].items():
        rows.append(dict(T=T, category="main", key=K, reselected=v, match=v == fz["main"][K]))
    for key, p in r["partner"].items():
        q = fz["partner"][key]
        rows.append(dict(T=T, category="partner", key=key, reselected=f"{p['method']}|{p['version']}",
                         match=(p["method"], p["version"]) == (q["method"], q["version"])))
    for x in rows:
        x["n_units"] = n_units
    return rows


# ------------------------------------------------------------------ 固定のファイルから数える注（成績の値は使わない）

def note_counts(frozen: dict, targets, ss: pd.DataFrame) -> list:
    """7 節の共正則化の注・格子の注，9.2 節の記述に回したセル，族 4 の「同じ識別器」，族 3 の相手の内訳を，T ごとと 4 つの合計で数える．
    設定の一致は固定のファイルから（(T, 生徒, K) ごとに 1 回），判定の一致は same_setting の行（(T, c, 生徒, K)）から数える。"""
    out = []

    def add(T, item, n, of, source, scope=""):
        out.append(dict(T=T, item=item, scope=scope, count=int(n), of=int(of), source=source))

    acc = defaultdict(lambda: [0, 0])
    for T in targets:
        fz = frozen[T]
        loc = defaultdict(lambda: [0, 0])

        def inc(key, hit):
            loc[key][0] += int(bool(hit)); loc[key][1] += 1

        for s in DD.STUDENTS:
            for K in KS_ALL:
                fam = K in ST.KS_MAIN
                scopes = ("all_K", "family_K") if fam else ("all_K",)
                inc(("coreg_selected_rho0", "all_K"), float(sel_params(fz, s, K, "coreg", "cr")["rho"]) == 0.0)
                for v in ("sel", "one"):
                    inc(("joint_selected_a1", "all_K"), int(float(sel_params(fz, s, K, "joint", v)["a"])) == 1)
                    inc(("joint_same_setting_as_joint_a1", "all_K"), same_setting(fz, s, K, "joint", v, "joint_a1", v))
                v = main_ver(fz, K)
                j = sel_params(fz, s, K, "joint", v)
                part = fz["partner"][f"{s}|{K}"]
                pp = sel_params(fz, s, K, part["method"], part["version"])
                for sc in scopes:
                    inc(("fam4_selected_equals_default", sc), v == "one" and float(j["rho"]) == 1.0 and int(float(j["a"])) == 1)
                    if s != "AVT":
                        inc(("partner_coreg_rho0_single_student", sc), part["method"] == "coreg" and float(pp["rho"]) == 0.0)
                    inc(("partner_kd_temp1", sc), part["method"] == "kd" and float(pp.get("temp", 0)) == 1.0)
                    inc(("partner_versions_differ", sc), part["method"] in ("kd", "agree", "mix") and part["version"] != v)
                    if s == "AVT":
                        inc(("partner_cotrain_avt", sc), part["method"] == "cotrain")
                    for pm in SEL.PARTNERS:
                        inc((f"partner_is_{pm}", sc), part["method"] == pm)
        det = fz.get("detect", {})
        for fam, (_, _, ks) in ST.FAMS.items():
            for s in DD.STUDENTS:
                for K in ks:
                    key = f"{fam}|{s}|{K}"
                    inc((f"fam{fam}_descriptive_cell", "family_K"), key in det.get("descriptive_cells", []))
                    inc((f"fam{fam}_same_level_possible", "family_K"), key in det.get("same_level_cells", []))
        for (item, sc), (n, of) in sorted(loc.items()):
            add(T, item, n, of, "frozen", sc)
            acc[(item, sc)][0] += n; acc[(item, sc)][1] += of
    for (item, sc), (n, of) in sorted(acc.items()):
        add("ALL", item, n, of, "frozen", sc)
    if len(ss):
        for col in ("pred_joint_eq_a1_main", "pred_fam4_eq_default", "pred_sel_eq_one", "pred_partner_eq_none"):
            for sc, d in (("all_K", ss), ("family_K", ss[ss.in_family])):
                for T, g in list(d.groupby("T")) + [("ALL", d)]:
                    add(T, col, g[col].sum(), len(g), "predictions（(T, c, 生徒, K) の行．全部の単位で判定が同じ）", sc)
    return out


def grid_edges(out: Path, frozen: dict, targets) -> list:
    """格子の端を選んだ数をつまみごとに（2 節の段 source の C，5 節の C′，7 節の手法のつまみ．比較の検査の 3 回目の 4，
    定義の 5 回目の記録 3）．3 点以上の格子だけを数え，joint_a1 の a は数えない（`dai1_select.select_T` の edge と同じ規則）。"""
    rows = []
    sm = pd.read_csv(out / "source" / "summary.csv")
    lo, hi = min(dai1.C_GRID), max(dai1.C_GRID)
    rows.append(dict(T="-", stage="source", method="source", knob="C", n_low=int(np.isclose(sm.C, lo).sum()),
                     n_high=int(np.isclose(sm.C, hi).sum()), n_cells=len(sm), grid=json.dumps(list(dai1.C_GRID))))
    cw_grid = sorted({cw for cw, _ in dai1.TETHER})
    for T in targets:
        fz = frozen[T]
        cws = [float(v[0]) for v in fz["tether"].values()]
        rows.append(dict(T=T, stage="tether", method="none", knob="Cw", n_low=int(sum(np.isclose(cws, cw_grid[0]))),
                         n_high=int(sum(np.isclose(cws, cw_grid[-1]))), n_cells=len(cws), grid=json.dumps(cw_grid)))
        acc = defaultdict(lambda: [0, 0, 0])
        for key, ch in fz["select"].items():
            meth = key.split("|")[2]
            gm = "joint" if meth == "joint_a1" else meth
            for knob, grid in SEL.GRIDS[gm]:
                if len(grid) < 3 or knob not in ch or (meth == "joint_a1" and knob == "a"):
                    continue
                g = [float(x) for x in grid]
                a = acc[(meth, knob)]
                a[0] += int(np.isclose(float(ch[knob]), min(g))); a[1] += int(np.isclose(float(ch[knob]), max(g))); a[2] += 1
        for (meth, knob), (nl, nh, n) in sorted(acc.items()):
            g = [float(x) for x in dict(SEL.GRIDS["joint" if meth == "joint_a1" else meth])[knob]]
            rows.append(dict(T=T, stage="select", method=meth, knob=knob, n_low=nl, n_high=nh, n_cells=n, grid=json.dumps(g)))
    return rows


# ------------------------------------------------------------------ 進め方

def degenerate_check(ctx: Ctx, s: str, K: int) -> dict:
    """退化の検算：同じ行どうしの差は全部の単位で判定が同じで，効果・区間・標準誤差が 0 になる。"""
    R = fam_rows(ctx.fz, s, K)
    e = diff_stats(ctx, R["prop"], R["prop"], K, "degenerate")
    return dict(T=ctx.T, c=ctx.cname, student=s, K=K, effect=e["effect"], lo=e["lo"], hi=e["hi"], se=e["se"],
                same_pred_all_units=e["same_pred_all_units"],
                ok=bool(e["effect"] == 0 and e["lo"] == 0 and e["hi"] == 0 and e["se"] == 0 and e["same_pred_all_units"]))


def run_pair(job: dict) -> dict:
    """1 つの (T, c) の全部の表．戻り：{表の名前: 行の並び}，照合の数，φ の値，退化の検算。"""
    t0 = time.time()
    mode, out, T, c, enc = job["mode"], Path(job["out"]), job["T"], job["c"], job["enc"]
    fz = json.load(open(out / "select" / "frozen.json"))[T]
    lab = Labels(c, enc)
    chk = Counter()
    cname = "ALL" if mode == "eval" else c
    R = int(fz["n_seeds_eval"]) if mode == "eval" else 2
    units = [Unit(out / mode / "units" / f"{T}_{cname}_f{f}_r{r}", f, r, lab, chk) for f in range(DD.N_FOLDS) for r in range(R)]
    if mode == "pseudo" and {u.commit for u in units} != {str(fz["pseudo_commit"])}:
        sys.exit(f"{T} c={c}: 仮のターゲットの単位のコミットが固定のファイルの pseudo_commit と違う")
    g_eff, _ = ST.g_eff_of(c)
    tables = defaultdict(list)
    ctx = Ctx(mode, out, T, c, fz, units, lab, g_eff, tables, chk)
    srcs = DD.others(T) if mode == "eval" else [x for x in DD.others(T) if x != c]
    pm = PseudoMeans(out, T, fz)
    phi = {}
    degen = []
    for s in DD.STUDENTS:
        for K in KS_ALL:
            t_levels(ctx, s, K)
            t_outside(ctx, s, K)
            t_nonmain(ctx, s, K)
            t_teacher_types(ctx, s, K)
            t_ablation(ctx, s, K)
            t_fam3_notes(ctx, s, K)
            t_regret(ctx, s, K, pm)
            t_recall(ctx, s, K)
            t_tether(ctx, s, K)
            t_sensitivity(ctx, s, K)
            t_same_setting(ctx, s, K)
            degen.append(degenerate_check(ctx, s, K))
            ctx.clear()
        print(f"  {T} c={c} 生徒 {s} 済み（{time.time() - t0:.0f} 秒）", flush=True)
    t_srcstd(ctx)
    t_oracle(ctx)
    t_marker_only(ctx)
    t_start_C_curve(ctx, srcs)
    t_notes_weights(ctx)
    t_convergence(ctx)
    t_assumptions(ctx, srcs, phi)
    ctx.clear()
    commits = sorted({u.commit for u in units})
    fsha = sorted({u.frozen_sha for u in units})
    return dict(T=T, c=c, tables=dict(tables), chk=dict(chk), phi={f"{k[0]}|{k[1]}": v for k, v in phi.items()}, degen=degen,
                units=len(units), units_commit=commits, units_frozen_sha=fsha, sec=time.time() - t0)


def guard_eval(out: Path, targets) -> tuple[str, str]:
    fzp = out / "select" / "frozen.json"
    cp = ROOT / "frozen" / f"{out.name}_frozen.json"
    if not fzp.exists() or not cp.exists() or dai1.sha(fzp) != dai1.sha(cp):
        sys.exit(f"固定のファイルが無いか，{cp.relative_to(ROOT)} と一致しない")
    if not dai1.committed_clean(cp):
        sys.exit("固定のファイルの写しがコミットされていない")
    commit, dirty = dai1.git_commit()
    if dirty:
        sys.exit("作業木に未コミットの変更がある（exp・splits・measure・features・data/metadata）．評価の段は流さない")
    fz = json.load(open(fzp))
    h = dai1.sha(fzp)
    commits = set()
    for T in targets:
        R = int(fz[T]["n_seeds_eval"])
        for f in range(DD.N_FOLDS):
            for r in range(R):
                base = out / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
                for w in (1, 2, 3):
                    for ext in ("npz", "csv"):
                        if not Path(f"{base}_p{w}.{ext}").exists():
                            sys.exit(f"評価の回の単位がそろっていない：{base.name}_p{w}.{ext}")
                z = np.load(f"{base}_p3.npz")
                if str(z["frozen_sha256"]) != h:
                    sys.exit(f"{base.name} の frozen_sha256 が固定のファイルと違う")
                commits.add(str(z["commit"]))
    if len(commits) != 1:
        sys.exit(f"評価の回の単位のコミットが 1 つでない：{sorted(commits)}")
    ucommit = commits.pop()
    if commit != ucommit:                                              # 集計の値が依るほかのファイル（dai3・dai1_select・dai1・dai_data・measure・分割の表）も固定する（gate-code の 5 回目の 1）
        sys.exit(f"今のコミット {commit[:10]} が評価の回の単位のコミット {ucommit[:10]} と違う．評価の回のコミットの作業木で流す（run/eval_procedure.md の 8）")
    for name in FIXED_SCRIPTS:                                          # 集計の仕方は評価の回の前に固定した（D1，比較の 5 回目 2-I）
        same = script_at(ucommit, name)
        if same is not True:
            sys.exit(f"exp/{name} が評価の回の単位のコミット {ucommit[:10]} の版と違う（{same}）．評価の値を見た後に集計を変えない")
    return h, ucommit


FIXED_SCRIPTS = ("dai1_desc.py", "dai1_stats.py")                      # eval の段で，評価の回の単位のコミットの版と一致を求める台本
RECORDED_SCRIPTS = ("dai1_desc.py", "dai1_stats.py", "dai1_select.py", "dai1.py", "dai3.py", "dai_data.py")   # 一致を記録だけする


def script_at(commit: str, name: str):
    """exp/<name> の，コミット commit の版が今のファイルと同じなら True，違えば理由の文字列。"""
    if not commit:
        return "コミットが無い"
    r = subprocess.run(["git", "show", f"{commit}:./exp/{name}"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        return "そのコミットに無い"
    return True if hashlib.sha256(r.stdout).hexdigest() == dai1.sha(HERE / name) else "sha256 が違う"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("eval", "pseudo"))
    ap.add_argument("place", nargs="?", default=str(ROOT / "results" / "dai1"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--targets", default=",".join(DD.CORPORA))
    ap.add_argument("--encoder", type=int, choices=(1, 2), default=None)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.place).resolve()
    targets = [t for t in a.targets.split(",") if t]
    if a.mode == "eval" and targets != list(DD.CORPORA):
        sys.exit("評価の段は 4 つの T の全部で流す（--targets は試走だけ）")
    enc_n = a.encoder or (2 if out.name.endswith("enc2") else 1)
    enc = DD.ENC1 if enc_n == 1 else DD.ENC2
    t0 = time.time()
    if a.mode == "eval":
        fsha, ucommit = guard_eval(out, targets)
    else:
        fsha, ucommit = dai1.sha(out / "select" / "frozen.json"), ""
    dest = Path(a.out_dir) if a.out_dir else out / "stats" / ("desc" if a.mode == "eval" else "desc_pseudo")
    if a.mode == "eval" and dest.exists() and any(dest.iterdir()):
        sys.exit(f"{dest} にすでに出力がある（上書きしない）")
    dest.mkdir(parents=True, exist_ok=True)
    commit, dirty = dai1.git_commit()
    jobs = [dict(mode=a.mode, out=str(out), T=T, c=c, enc=enc) for T in targets
            for c in ([T] if a.mode == "eval" else DD.others(T))]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(run_pair, jobs))
    else:
        res = [run_pair(j) for j in jobs]
    tables = defaultdict(list)
    chk = Counter()
    phi = defaultdict(list)
    degen = []
    for r in res:
        for k, v in r["tables"].items():
            tables[k] += v
        for k, v in r["chk"].items():
            chk[tuple(k) if isinstance(k, list) else k] += v
        for k, v in r["phi"].items():
            phi[k] += v
        degen += r["degen"]
    marker_same_direction(tables["sensitivity"])
    frozen = json.load(open(out / "select" / "frozen.json"))
    for T in targets:
        if T in ("AFEW", "MOSEI", "MELD"):
            tables["iemocap_excluded_selection"] += iemocap_excluded_selection(out, T, frozen[T])
    tables["counts"] = note_counts(frozen, targets, pd.DataFrame(tables["same_setting"]))
    tables["grid_edges"] = grid_edges(out, frozen, targets)
    for key, v in sorted(phi.items()):
        T, K = key.split("|")
        arr = np.array(v, float)
        ok = arr[np.isfinite(arr)]
        med = float(np.median(ok)) if len(ok) else np.nan
        tables["assumptions"].append(dict(T=T, c="ALL" if a.mode == "eval" else "pseudo(30 単位)", assumption="積のモデルの誤りの独立",
                                          key=f"K={K}", value=med, criterion="< 0.2", holds=bool(med < 0.2),
                                          detail=json.dumps({"n_values": len(arr), "n_undefined": int((~np.isfinite(arr)).sum())})))
    chk_rows = sorted({k[0] for k in chk})
    tables["check"] = [dict(category=cat, checked=chk[(cat, "checked")], mismatch=chk[(cat, "mismatch")],
                            missing_row=chk[(cat, "missing_row")]) for cat in chk_rows]
    tables["degenerate_check"] = degen
    tables["fam3_candidates"] = [dict(method=m, knobs=",".join(k for k, _ in SEL.GRIDS[m]),
                                      n_points=int(np.prod([len(v) for _, v in SEL.GRIDS[m]])),
                                      versions=",".join(SEL.PARTNER_VERSIONS.get(m, ["sel", "one"]))) for m in SEL.PARTNERS]
    stamp = dict(mode=a.mode, commit=commit, dirty=dirty, frozen_sha256=fsha, units_commit=ucommit)
    counts = {}
    for name, rows in sorted(tables.items()):
        df = pd.DataFrame(rows)
        for k, v in stamp.items():
            df[k] = v
        df.to_csv(dest / f"{name}.csv", index=False)
        counts[name] = len(df)
    ssc = {f"{r['T']}|{r['item']}|{r['scope']}": f"{r['count']}/{r['of']}" for r in tables["counts"]}
    scripts = {n: dict(sha256=dai1.sha(HERE / n), same_as_units_commit=script_at(ucommit, n) if ucommit else None)
               for n in RECORDED_SCRIPTS}
    man = dict(stamp, place=str(out), dest=str(dest), targets=targets, encoder=enc_n, rows=counts, scripts=scripts,
               units={f"{r['T']}|{r['c']}": r["units"] for r in res},
               units_commits={f"{r['T']}|{r['c']}": r["units_commit"] for r in res},
               units_frozen_sha={f"{r['T']}|{r['c']}": r["units_frozen_sha"] for r in res},
               check={f"{k[0]}|{k[1]}": v for k, v in sorted(chk.items())},
               check_total=dict(checked=sum(v for k, v in chk.items() if k[1] == "checked"),
                                mismatch=sum(v for k, v in chk.items() if k[1] == "mismatch"),
                                missing_row=sum(v for k, v in chk.items() if k[1] == "missing_row")),
               degenerate_ok=f"{sum(d['ok'] for d in degen)}/{len(degen)}", same_setting_counts=ssc,
               sec={f"{r['T']}|{r['c']}": round(r["sec"]) for r in res}, total_sec=round(time.time() - t0),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(man, open(dest / "manifest.json", "w"), indent=1, ensure_ascii=False)
    print("表の行の数:", json.dumps(counts, ensure_ascii=False))
    print("照合:", json.dumps(man["check_total"]), "退化の検算:", man["degenerate_ok"])
    print("書いた:", dest, f"（{man['total_sec']} 秒）")


if __name__ == "__main__":
    main()
