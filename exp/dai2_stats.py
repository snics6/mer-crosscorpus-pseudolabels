"""DAI 実験 2 の集計（設計 `docs/設計/実験2_設計.md` 2.3 節・4 節・6 節．比べ方の公平さの関門の指摘 2-I（`checks/DAI-2_3/compare_v4.md`））。

評価の値を見る前に集計の仕方を固めるための台本である．評価の回の前にコミットし，sha256 を `frozen/eval_gate.json` の required に
`"exp/dai2_stats.py"` の鍵で入れる（無いと eval の段は止まる）．
全部の量を，保存した int8 のテストの判定と，`measure/data.py` の load_group を通したラベルと塊（`dai3.corpus_df` → `dai_data.load_corpus`）から
作り直す．区間は実験 1 の 9.3 節の規則で，`dai1_stats.effect_stats` をそのまま呼ぶ（書き直さない）．単位の UAR は `dai1_stats.uar_full`
（テストに現れる感情の再現率の平均．rows.csv と同じ定義），区間の中の塊を除いた値と 2 組の値は `dai1_stats.uar_rows`（30 本未満か感情が
欠ければ NaN．30 本の規則）．塊ごとの数は `dai1_stats.counts`，θ は `dai1_stats.theta`．p 値は付けない（実験 2 には族を置かない．設計 4 節）．

使い方：
  python exp/dai2_stats.py eval   [置き場]   評価の回の出力だけを読む（既定の置き場 results/dai2/stats）．次のときだけ動く：
        frozen/dai2_frozen.json と results/dai2/select/frozen.json の sha256 が一致し，frozen/dai1_frozen.json と
        results/dai1/select/frozen.json が一致し，両方の写しがコミット済みで，実験 2 の固定のファイルが使った実験 1 の固定のファイルの
        sha256 が今のものと同じで，作業木（exp，splits，measure，features，data/metadata．未追跡のファイルを含む）が汚れておらず，
        この台本がコミット済みで，`frozen/eval_gate.json` がコミット済みで required にこの台本の今の sha256 があり，
        results/dai1/stats/families.csv（`python exp/dai1_stats.py eval` の出力）があり，その行の dirty がすべて偽．
        単位はすべてそろうことを求める（線形の手法と CORAL は r ∈ {0, …, R−1}，R は実験 1 の固定のファイルの n_seeds_eval．
        勾配法の手法は r ∈ {0, 1, 2}．固定のファイルの grad_seeds_eval）．入力を替える引数（--dai1-dir など）と --targets・--no-null は受け付けない．
        照合の表の CRITICAL（単位の sample_id，引き戻しの鍵，各単位の保存の固定のファイルの sha256，格子，行の重複，未コミットの変更で
        計算した行，ラベルと splits）のどれかが 0 でなければ表を書かずに止まる．
  python exp/dai2_stats.py pseudo [置き場]   試走．仮のターゲットの回の出力（実験 1 は results/dai1/pseudo/units の p1・p3，実験 2 は
        results/dai2/pseudo/units）に同じ集計を当てる（既定の置き場 results/dai2/stats_pseudo．results/dai2/stats には書かない）．
        評価のターゲット T ごとに，仮のターゲット c を 1 つの「コーパス」とし，ソースは残りの 2 つ．提案と手法の設定は T の固定の値．
        族 3 は `dai1_stats.collect`・`effect_stats`・`holm` で (T, c) ごとに作り（families_pseudo.csv），記述に回すセルは実験 1 の固定の
        ファイルの c の detect.descriptive_cells．試走には選んだ点の解き直しが無いので，選んだ点の解き直しを使う量（5・6 の役 selected，
        3 の *_rep）は空になる．--targets，--pseudo-c で絞り，--allow-partial でそろった単位だけを使う（試走だけ）．
  python exp/dai2_stats.py selftest   作った数で，退化した場合（2 つの行の判定が同じ）に θ・区間・標準誤差が 0 になること，
        分割の符号の反転で区間が 0 について反転すること，Welch–Satterthwaite の自由度の範囲を確かめる．
成績の値は画面に出さない（行の数，件数，照合の食い違いの数だけ）．

語：「コーパス」は評価の回では評価のターゲット T，試走では仮のターゲット c．「単位」は (分割 f, K 本の抜き方の種 r)．「点」は格子の点．
「選んだ点」は `results/dai2/select/frozen.json` の `<生徒>|<K>|<手法>` の値．「選択なしの点」は dai2.DEFAULTS・dai2_grad.DEFAULTS．
提案は実験 1 の固定のファイルの主の版（`main[K]`）の同時に解く版の選んだ設定（`joint|<版>` の ρ・a．`dai1_stats.unit_preds` の prop と同じ
取り出し方）．比較の基準 none は実験 1 の段 1 の none[K]（固定の引き戻しの行．K = 0 は出発点）．
手法の群：線形（2.1 節の self，entmin，im，t3a，t3a_b，lp，own，st，saerens と 2.3 節の coral）は r ∈ {0, …, R−1}．勾配法（dann，mme，mapctl，
xmuda）は r ∈ {0, 1, 2}（s = r）で，区間は R = 3 の式（effect_stats の種の項が 3 つの種の不偏分散になる）．xmuda は生徒 A・V・T，lp は K ≥ 1．
saerens は付録の確かめだけ（設計 2.1 節），t3a_b は記述の版（同）．S+T の K = 0 は 1 問（β の 3 行は同じ値）なので β = 1 の判定を使う．
解き直しの役の名前（設計 2.3 節の名前と保存の実際の名前は同じ）：map の解き直しの行（method `<手法>_rep`）の roles は default，selected，
ctl_of_dann_default，ctl_of_mme_default，ctl_of_xmuda_ref，ctl_of_dann_selected，ctl_of_mme_selected，ctl_of_xmuda_selected．xmuda の
解き直しの行（xmuda_rep）の roles は ref と selected_<生徒>．xmuda の行は roles に selected_<その生徒> か ref を含む行だけを使う（3 つの生徒を
同時に学ぶので，ほかの生徒のために選んだ点の行にも値が入る．gate-code の 2 回目の記録 2・3）．役 selected は評価の回だけにある．

出す表（どれも csv．行に mode・commit・dirty を書く．区間の列は effect（θ），lo，hi，se，ci_rule，df，v_cluster（塊の項．MOSEI は塊の
引き直しの分散），v_seed（区間に入れた種の項），n_folds，n_units，identical（標準誤差が 0），same_preds（2 つの行の判定が全部の単位で同じ），
above（下端 > 0．「上回る」），below（上端 < 0．「下回る」）．uar_prop・uar_other は差の 2 つの行の θ の形の平均（同じ単位の上の値））：
 1. prop_minus_method.csv（設計 4 節「差の作り方」と書き方 2-E）：(コーパス, 生徒, K, 手法) ごとに「提案 − 手法（選んだ点）」．共通の単位の上の
    単位ごとの UAR の差 d_fr から effect_stats で θ と区間を作る．列 point は手法の選んだ点，prop_version・prop_point は提案の版と設定．
    線形の手法の K = 0 は手法も提案も種に依らないので，実験 1 の evaluate と同じく全部の種を渡し（種の項は 0），n_units を分割の数 5 にする
    （seed_varies が偽になる）．勾配法の手法の K = 0 は 15 単位（初期化の種が違う．設計 4 節の K = 0 の単位）．
    種の項の割合 seed_share = v_seed ÷ (v_cluster + v_seed) を全部の行に，分割の中の種の間の差の幅（max − min）の分割にわたる平均と最大
    （seed_range_mean・seed_range_max．設計 4 節「種の間の幅を付録に」）を並べる．勾配法の手法との差で seed_share > 0.5 の行には
    Welch–Satterthwaite の自由度の区間（df_ws，lo_ws，hi_ws，above_ws，below_ws）も並べる（設計 4 節，統計の 5 回目の記録 2）：
      df_ws = V² / (V_cl²/df_cl + Σ_f (s_f²/(F² R_f))² / (R_f − 1))．V = V_cl + V_seed．V_cl・df_cl は effect_stats の塊の項と自由度
      （MOSEI の塊の引き直しは df_cl = ∞ で項が消える）．和は種の項に入った分割（塊を除くと分割ごと外れる分割を除く）だけ．
    主の判定は effect_stats の区間で，WS の区間は記述．own の行が「提案 − own」（予測の出どころの差．設計 2.1 節の own の行）．
    列 group は設計の節（2.1，2.2，2.3），gradient は勾配法か，appendix は saerens，descriptive_version は t3a_b．
    CORAL の点は d_pca（部分空間の次元．格子の r ∈ {32, 64, 128}）と書き，種の r と書き分ける（統計の 5 回目の記録）．ほかの表も同じ．
 2. prop_one_minus_own.csv（設計 4 節の 2-H）：「提案（重みなし．版 (1,1,1) の選んだ設定 joint|one）− own（選んだ点）」．全部の (コーパス,
    生徒, K) で作り，主の版が sel（指数を選んだ積）の (T, K) にだけ design_row = True を付ける．偽の行は主の版が one なので 1 の own の行と
    同じ量になり，台本はその一致を照合の表（prop_one_own_dup_mismatch）に数える．
 3. xmuda9.csv・fam3_cells.csv・corpus_sentence.csv（設計 4 節「族 3 の書き方の規則で使う xMUDA 型との差」と実験 1 の設計 9.3 節の族 3 の
    書き方の規則）：xmuda9 はコーパスごとに K ∈ {0, 5, 25} × 生徒 A・V・T の 9 セルの「提案 − xMUDA 型（選んだ点）」（1 の行と同じ値．
    本番の束の値）と，同じ点の解き直しの値（1 問ずつの束．役 selected_<生徒> の行）で作った区間を記述の列 effect_rep・lo_rep・hi_rep・
    above_rep・below_rep に並べる（選んだ点が全部の単位で解き直されているときだけ．評価の回では必ずある）．fam3_cells は実験 1 の族 3 の
    12 セル（評価の回は results/dai1/stats/families.csv の family = 3 の行，試走は families_pseudo）を 1 行ずつ，相手の手法と版，記述に回した
    セルか（実験 1 の固定のファイルの detect.descriptive_cells に鍵 `3|<生徒>|<K>` があるか），正で p_holm < 0.05 か，(a) に数えるかを並べる．
    corpus_sentence はコーパスごとに (a) 記述に回していない族 3 のセルで差が正かつ p_holm < 0.05 の数が 7 以上，(b) 9 セルで「上回る」
    （下端 > 0．本番の束の値）が 5 以上，(c) 9 セルで「下回る」（上端 < 0．本番の束の値）が 0，の 3 つを機械的に当て，族 3 が 12 セル・
    xMUDA 型が 9 セルそろい 3 つがそろうときだけ corpus_sentence = True．記述に回したセルも含めた数（n_fam3_pos_sig_all）と，解き直しの
    値での (b)・(c) の数（n_above9_rep・n_below9_rep．記述）を参考の列に並べる．
 4. ctl_decomp.csv・k0_ctl_none.csv（設計 2.3 節の 2-C・2-G・C2-5）：勾配法の単位の上で「手法 − mapctl」（手法 = dann，mme，xmuda の選んだ点），
    「mapctl − none」．mapctl の点は 2 つ：matched（手法の選んだ点と同じ学習率・反復・μ の mapctl．xMUDA 型では模倣の重みだけを 0 にした問．
    本文の値）と ctl_selected（mapctl として選んだ点．並べる値）．行 comparison は `<手法>-mapctl[matched]`，`<手法>-mapctl[ctl_selected]`，
    `mapctl[matched_<手法>]-none`，`mapctl[selected]-none`．k0_ctl_none は K = 0 の mapctl の全部の点と none（出発点）の判定が全部の発話で
    一致する単位の数と一致の割合の最小（設計 2.3 節の K = 0 の検算）．
 5. da_term.csv・da_quantities.csv（設計 2.3 節「DA の項が働いたか」，6 節の 2 行目）：(コーパス, 手法 ∈ {dann, mme, xmuda}, 生徒, K, 役) ごとに，
    勾配法の各単位を，その点の本番の束の行の dis_ctl を d_ctl として 3 つに分ける．
      (i)   d_ctl ≤ e_ctl + 0.01．e_ctl は mapctl_rep の行（その点と同じ学習率・反復・μ・K の mapctl の解き直し）のうち roles に
            ctl_of_<手法>_<役> を持つ行の dis_rep → ほとんど動かさなかった
      (ii)  (i) でなく d_ctl ≤ e_m．e_m は行 <手法>_rep（同じ点の解き直し）のうち roles に役（map は selected・default，xmuda は
            selected_<生徒>・ref）を持つ行の dis_rep → 動かしたが，動かした先は束で変わる
      (iii) それ以外 → 揺れを超えて動かした
    役は selected（選んだ点．評価の回だけ．判定に使う行）と，default（dann・mme の選択なしの点）・ref（xmuda の XM_REF）（記述）．
    (i)〜(iii) の単位の数の横に，同じ単位の「手法 − mapctl（matched）」の UAR の差（本番の束）の平均・最大・最小を並べる．not_worked
    （「働かなかった」の印）は (i) の単位の数が勾配法の単位の総数の半分を超えるときだけ真．偽を「働いた」と読まない（設計 2.3 節・4 節）．
    行が欠けた単位は n_missing に数え，(i)〜(iii) に入れない．da_quantities は同じ点の本番の束の行の ‖A − I‖_F（anorm），‖W·A − W0‖，
    ‖W − W0‖，判別器の損失（dom）と予測のエントロピーの最初と最後，d_ctl，学び終えた直後の分布の量（mdiff・cdiff・dloss・dacc の前と後），
    xMUDA 型の KL の最初と最後の，単位にわたる平均・最小・最大（設計 2.3 節の R2）．
 6. rep_shift.csv・rep_agreement.csv・prop_minus_method_rep.csv（設計 2.3 節「束による揺れの報告」，6 節の 1 行目）：
    rep_shift は (コーパス, 手法, 生徒, K, 役) ごとに，解き直しと本番の束の判定の入れ替わりの割合（保存した判定から作り直した値．行の
    dis_rep との照合は checks の dis_rep_mismatch）と「解き直しの UAR − 本番の束の UAR」の，単位にわたる平均・最小・最大．役は default，
    selected，ref と ctl（mapctl の解き直しの行で roles が ctl_of_* だけのもの）．
    rep_agreement は (単位, 問) ごとの判定の一致の割合（1 − 入れ替わりの割合）を，(手法, 役) ごとに，T ごとと 4 つの T を合わせた値（T = ALL）で，
    単位にわたる最小と中央値．reproducible は，合わせた中央値と各 T の中央値がすべて 0.99 以上のときだけ真（T の行と合わせた行に同じ値）．
    試走では T の中に 3 つの c の単位が入る．
    prop_minus_method_rep は勾配法の手法の「提案 − 手法」を，同じ点の本番の束の判定（which = prod）と解き直しの判定（which = rep）で作った
    2 つの区間（記述）．役の点の解き直しの行が全部の勾配法の単位で，その役の名前を roles に持つときだけ作る（評価の回は selected と
    default・ref，試走は default・ref）．
 7. marks.csv・marks_points.csv（設計 2.3 節「発散の印」）：本番の束の格子の問ごとに 2 つの印を付ける．判別器の印は dann の問だけで
    dloss_after > 10 ln 2．膨張の印は dann・mme・mapctl で cdiff_after > 100 × cdiff_before，xmuda でどれかのモダリティ（同じ単位・同じ問の
    A・V・T の行）の anorm > 100（xmuda の印は問ごとなので 3 つの生徒の行で同じ数になる）．marks は (コーパス, 手法, 学習率, 生徒, K) ごとに
    問の数（単位 × 点）と 2 つの印の数を分けて数え（n_disc は dann だけ），dann・mme・mapctl の dloss_after の平均と最大を参考に並べる
    （mme・mapctl には印を付けない．学んでいない判別器での値）．selected_point_here・default_point_here はその学習率の行に選んだ点・
    選択なしの点があるか，*_point_marked はその点がどれかの単位でどちらかの印に当たったか．marks_points は選んだ点と選択なしの点
    （dann・mme・mapctl）ごとに印に当たった単位の数（表の注に使う）．
 8. regret.csv（設計 4 節，実験 1 の 9.3 節 S8）：(コーパス, 生徒, K, 手法) ごとに，格子の点の θ（単位の UAR を種で平均し分割で平均）が最大の
    点（同じ値なら格子の並びで先）と選んだ点の差（取りこぼし．作りの上で 0 以上）と，その 2 点の差の effect_stats の区間（ci_*）．
    ci_* は最良の点をテストで選んだ後の区間なので記述で，検定には使わず zero_level と並べて読む（gate-code の 7 回目の記録 4）．
    差が無いときの水準 zero_level は，単位ごとにテストを `dai3.test_halves`（実験 3 の設計 1 節の 2 組．同じ話者が両方の組に入る）で
    2 組に分け，組 A の θ で最良の点を選んで組 B で「その点 − 選んだ点」を測り（zero_level_ab），入れ替え（zero_level_ba），平均した値．
    組の UAR は uar_rows（30 本未満か感情が欠ける組の単位は，その組の θ から外す）．
 9. null_counts.csv（設計 4 節「数え上げの文」．統計の 5 回目の記録 1）：コーパスごとに，分割ごとの差の符号を，全部のセルで同じ分割に
    同じ符号を当てて反転した 2^5 = 32 通り（反転した分割では提案と手法の判定を入れ替えて effect_stats を当て直す．塊の引き直しの乱数は
    セルごとに固定し，32 通りで同じものを使う．反転なしの 1 通りが 1 の値）で，「上回る」と「下回る」のセルの数の分布の平均・上側 5% 点
    （32 通りの逆累積分布の 0.95．上から 2 番目の値）・最大と，観測した数以上になる割合．数える群は all（1 の全部の行），all_main（saerens と
    t3a_b を除く．設計の約 900 の区間のコーパスの分），method:<手法>，xmuda9（3 の 9 セル）．数え上げの文に使う群は all_main で，all は付録，
    method:<手法> と xmuda9 は記述（gate-code の 7 回目の S-ambig．評価の値を見る前に決める）．
 10. source_mix.csv（設計 4 節 b4）：(コーパス) ごとに，DANN の DA の項が一様に引く合わせたソース（評価の回は残り 3 つ，試走は残り 2 つの
    コーパスの全発話．`dai2.st_sources` と同じ発話）のコーパスごとの本数と割合（source_share_of_merged），感情ごとの本数と割合（合わせた
    全体の中 share_of_merged，そのソースの中 share_of_source）．ラベルの件数だけで作る．source = ALL の行は合わせた全体．
 11. checks.csv・meta.json：照合の数．実験 2 の全部の行（格子の点，解き直し，none，saerens，CORAL の曲線）の uar と判定から作った UAR の
    食い違い（rows_uar_mismatch / rows_checked），実験 1 の提案・none・重みなしの提案の行の uar との食い違い（d1_rows_mismatch．試走の族 3 は
    `dai1_stats.collect` の照合の数も families_pseudo.csv に残る），dis_ctl・dis_rep の作り直し，単位の sample_id と K 本の一致，格子の点の
    過不足（設計の格子から削り方を引いたもの），行の重複，行の無い判定・判定の無い行，引き戻しの鍵，各単位の保存の固定のファイルの
    sha256（評価の回），未コミットの変更で計算した行（rows_dirty，d1_rows_dirty），ラベルと splits の一致（split_mismatch），
    WS の種の項の作り直し（ws_vseed_mismatch），実験 1 の段 1 の引き戻しの格子 14 点の none[K] の行の uar と判定（d1_tether_rows_*．
    12 の相手），12 の曲線のセルの数・欠け・選んだ引き戻しの点と本番の行の判定の食い違い（coral_tether_*），退化したセルの数と，そのうち θ・区間・標準誤差が 0 にならなかった数
    （degenerate_cells，degenerate_interval_nonzero）．行のコミットの一覧（row_commits，d1_row_commits）．meta.json は入力の sha256，
    単位の数，表の行の数，照合の合計．
 12. coral_tether_curve.csv・coral_tether_sign.csv（設計 2.3 節の CORAL の行「評価の回で引き戻しの格子 14 点の曲線も解く（C2-6）」と
    範囲の外の項．形は実験 1 の引き戻しの曲線 `dai1_desc.py` の tether_curve・tether_sign にそろえた）：評価の回だけ．(コーパス, 生徒,
    K ≥ 1) ごとに，選んだ d_pca の新しい出発点から引き戻しの格子 14 点（`dai1.TETHER`）で解いた coral_tether の行と，同じ引き戻しの
    none[K]（実験 1 の段 1 の保存 p1 の `<K>|<C′>|<C_b>` の行）について，kind = level（row = coral，none．正解 0 の行との差として
    effect_stats を当てた水準の区間．dai1_desc の zero_of と同じ）と kind = diff（row = coral − none）の区間を並べる．単位は CORAL と同じ
    線形の r ∈ {0, …, R−1}．selected_tether は実験 1 の固定のファイルの引き戻しの点か．coral_tether_sign は 14 点のうち「CORAL − none」の
    θ の符号が選んだ引き戻しの点の符号と同じ点の数（n_same_sign，all_same_sign，n_pos，n_neg，n_zero）と，選んだ引き戻しの曲線の点の判定が
    本番の CORAL の行（selected_point_matches_main）と none の行（selected_none_matches_main）と全単位で同じか．仮のターゲットの回の保存に
    曲線の行は無いので，pseudo の段では 2 つの表は空になる（列も無い）．曲線の行が一部の (生徒, K) でしかそろわなければ
    coral_tether_missing に数え，その (生徒, K) を外す．

設計で決まっておらず台本で決めた細部：
  - 4 の mapctl の点 matched を本文の値，ctl_selected を並べる値とする（dann・mme にも xMUDA 型と同じ 2 つを出す）．
  - 5 は選択なしの点（default・ref）にも同じ分け方を当てて記述として並べる．判定に使うのは役 selected の行．
  - 6 の役 ctl は mapctl の解き直しの行のうち，自分の default・selected の役を持たず ctl_of_* だけを持つ行．
  - 6 の prop_minus_method_rep は，役の名前を roles に持つ解き直しの行だけを使う（5 の行の選び方と同じ）．
  - 7 の xmuda の膨張の印は，同じ (単位, 学習率, 反復, κ, μ, K) の A・V・T の行の anorm の最大で付ける．
  - 8 の最良の点の同点は格子の並び（設計の格子の積の順）で先．取りこぼしの区間は線形の K = 0 で n_units を 5 にする．
  - 9 の上側 5% 点は 32 通りの逆累積分布（numpy の inverted_cdf）の 0.95．分割が 5 つそろわない試走では 2^(分割の数) 通り．
  - WS の区間は勾配法の手法との差（1・3・4・6 の表で，手法か mapctl の相手が勾配法の行）で seed_share > 0.5 のときだけ値を入れる．
    標準誤差が 0 のセルの seed_share は NaN．
  - 区間の乱数はセルの鍵の sha256 から作る（並べる順や並列の数で値が変わらない）．2 の行は 1 の own の行と同じ鍵の乱数を使う
    （主の版が one なら 2 つの行の値が MOSEI の塊の引き直しでも同じになる）．
  - S+T の K = 0 の選んだ点は β = 1 の行とする（3 行が同じ判定）．
  - 12 の曲線の相手は，同じ引き戻しで解いた none[K]（実験 1 の段 1．比較の基準は段 1 で 14 点を解いてある．実験 1 の 9.1 節）とし，
    提案との差は並べない（実験 2 の設計の範囲の外の項は CORAL の曲線を求めるが，相手を決めていない）．符号の数え方は dai1_desc の
    tether_sign と同じ（選んだ引き戻しの点の θ の符号と同じ符号の点の数）．
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                 # 集計は GPU を使わない（判定とラベルの件数だけ）
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai1_stats as ST  # noqa: E402
import dai2  # noqa: E402
import dai2_grad as G  # noqa: E402
import dai2_select as SEL  # noqa: E402
import dai3  # noqa: E402
import dai_data as DD  # noqa: E402

ROOT = DD.ROOT
KS_ALL = dai1.KS_ALL
KS9 = (0, 5, 25)
LIN_PART = ("self", "entmin", "im", "t3a", "t3a_b", "lp", "own", "st", "saerens")
MAP_M = ("dann", "mme", "mapctl")
GRAD_M = MAP_M + ("xmuda",)
METHODS = LIN_PART + ("coral",) + GRAD_M
DA_M = ("dann", "mme", "xmuda")
GRIDS = dict(dai2.GRIDS) | dict(G.GRIDS)
SEL.GRIDS.update(G.GRIDS)                                              # dai2_select.main と同じく勾配法と CORAL の格子を足す（grid_expected が読む）
DEFAULTS = dict(dai2.DEFAULTS) | dict(G.DEFAULTS)
GROUP = {**{m: "2.1" for m in LIN_PART}, "coral": "2.3", "dann": "2.3", "mme": "2.3", "mapctl": "2.3", "xmuda": "2.2"}
DISC_MAX = 10 * math.log(2)                                            # 判別器の印（2.3 節）
CDIFF_RATIO, ANORM_MAX = 100.0, 100.0                                  # 膨張の印（同）
SLACK_I = 0.01                                                         # DA の項の (i) の余裕（テストの発話の 1%．同）
REPRO_MIN = 0.99                                                       # 6 節の再現の基準
SEED0 = 20261104
EMO = ("Neutral", "Happy", "Sad", "Angry")
PROD = {"dai1": ROOT / "results" / "dai1", "dai2": ROOT / "results" / "dai2"}
RUN = {"mode": "", "commit": "", "dirty": True, "gate": ""}
CRITICAL = ("unit_sample_mismatch", "tether_key_mismatch", "frozen_sha_mismatch", "d1_frozen_sha_mismatch", "grid_mismatch", "rows_duplicate",
            "rows_dirty", "d1_rows_dirty", "split_mismatch")


# ------------------------------------------------------------------ 小道具

def norm(v) -> str:
    return ST.norm(v)


def gkeys(m: str) -> list:
    return [k for k, _ in GRIDS[m]]


def pkey(m: str, d) -> tuple:
    """点の鍵：その手法の格子のつまみの (名前, 正規化した値) の並び（手法ごとに順は格子の順）。"""
    return tuple((k, norm(d[k])) for k in gkeys(m))


def pdict(pk: tuple) -> dict:
    return {k: float(v) for k, v in pk}


def pjson(m: str, pk: tuple) -> str:
    """表に書く点（JSON）．CORAL の部分空間の次元は d_pca と書き，種の r と書き分ける（統計の 5 回目の記録）。"""
    return json.dumps({("d_pca" if (m == "coral" and k == "r") else k): v for k, v in pdict(pk).items()})


def students_of(m: str) -> tuple:
    return ("A", "V", "T") if m == "xmuda" else DD.STUDENTS


def ks_of(m: str) -> tuple:
    return DD.KS if m == "lp" else KS_ALL


def ctl_pk(pk: tuple) -> tuple:
    """dann・mme・xmuda の点と同じ学習率・反復・μ の mapctl の点。"""
    d = dict(pk)
    return pkey("mapctl", {k: float(d[k]) for k in ("lr", "iters", "mu")})


def grid_order(m: str, cut: dict) -> list:
    """設計の格子から削り方を引いた点を，格子の積の順に並べた鍵の list。"""
    gr = SEL.grid_expected(m, cut)
    return [tuple((k, norm(v)) for (k, _), v in zip(gr, combo)) for combo in itertools.product(*[vals for _, vals in gr])]


def cell_seed(key) -> int:
    return int(hashlib.sha256(json.dumps(key, default=str).encode()).hexdigest()[:12], 16)


def tokens(roles) -> set:
    if isinstance(roles, (list, tuple, set)):
        return set(roles)
    if roles is None or (isinstance(roles, float) and math.isnan(roles)):
        return set()
    return set(str(roles).split("|"))


def finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def stamp(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(mode=RUN["mode"], commit=RUN["commit"], dirty=RUN["dirty"])


def qhigh(x) -> float:
    return float(np.quantile(np.asarray(x, float), 0.95, method="inverted_cdf"))


# ------------------------------------------------------------------ 区間（effect_stats をそのまま呼ぶ）と WS と符号の反転

def ws_part(e: dict, A: dict, B: dict) -> dict:
    """Welch–Satterthwaite の自由度（統計の 5 回目の記録 2）と種の項の割合．effect_stats の戻り e の塊の項と，同じ d_fr から作った種の項を使う。"""
    keys = sorted(A)
    folds = sorted({f for f, _ in keys})
    F = len(folds)
    d = {f: [ST.uar_full(A[k]) - ST.uar_full(B[k]) for k in keys if k[0] == f] for f in folds}
    dropped = {int(x) for x in str(e.get("folds_dropped", "") or "").split(",") if x.strip() not in ("", "nan")}
    terms = [(float(np.var(d[f], ddof=1)) / len(d[f]) / F ** 2, len(d[f]) - 1) for f in folds if f not in dropped and len(d[f]) > 1]
    v_seed = float(sum(t for t, _ in terms))
    v_cl = float(e["v_jk"])
    V = v_cl + float(e["v_seed"])
    df_cl = float(e["df"]) if finite(e.get("df")) else math.inf
    den = (v_cl ** 2 / df_cl if math.isfinite(df_cl) else 0.0) + sum(t ** 2 / k for t, k in terms)
    df_ws = V ** 2 / den if den > 0 else float("nan")
    out = dict(seed_share=(float(e["v_seed"]) / V if V > 0 else float("nan")), df_ws=df_ws, v_seed_check=abs(v_seed - float(e["v_seed"])))
    rng_ = [float(np.ptp(d[f])) for f in folds if len(d[f]) > 1]          # 分割の中の種の間の差の幅（設計 4 節「種の間の幅を付録に」）
    out.update(seed_range_mean=float(np.mean(rng_)) if rng_ else float("nan"), seed_range_max=float(np.max(rng_)) if rng_ else float("nan"))
    if finite(df_ws):
        q = float(st.t.ppf(0.975, df_ws))
        out.update(lo_ws=e["effect"] - q * math.sqrt(V), hi_ws=e["effect"] + q * math.sqrt(V))
    else:
        out.update(lo_ws=float("nan"), hi_ws=float("nan"))
    return out


def cell_task(args) -> tuple:
    """1 つのセル：effect_stats の区間（反転なし）と，null が真なら分割の符号を反転した 2^F 通りの「上回る」「下回る」の印。"""
    key, A, B, g_eff, seed, null = args
    folds = sorted({f for f, _ in A})
    pats = list(itertools.product((1, -1), repeat=len(folds))) if null else [(1,) * len(folds)]
    e0, above, below = None, [], []
    for pat in pats:
        sg = dict(zip(folds, pat))
        A2 = {k: (A[k] if sg[k[0]] == 1 else B[k]) for k in A}
        B2 = {k: (B[k] if sg[k[0]] == 1 else A[k]) for k in A}
        e = ST.effect_stats(A2, B2, g_eff, np.random.default_rng([SEED0, seed]))
        if e0 is None:
            e0 = e
        above.append(bool(e["lo"] > 0))
        below.append(bool(e["hi"] < 0))
    w = ws_part(e0, A, B)
    ua = ST.theta({f: [ST.uar_full(A[k]) for k in sorted(A) if k[0] == f] for f in folds})
    ub = ST.theta({f: [ST.uar_full(B[k]) for k in sorted(B) if k[0] == f] for f in folds})
    return key, e0, w, ua, ub, above, below


def run_cells(tasks: list, jobs: int) -> dict:
    if not tasks:
        return {}
    out = {}
    if jobs <= 1:
        for t in tasks:
            r = cell_task(t)
            out[r[0]] = r[1:]
        return out
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(cell_task, tasks, chunksize=max(1, len(tasks) // (jobs * 8))):
            out[r[0]] = r[1:]
    return out


def effect_cols(res, linear_k0: bool, grad: bool) -> dict:
    """run_cells の 1 つの結果 → 表の列（p 値は付けない）。"""
    e, w, ua, ub, above, below = res
    n_units = int(e["n_folds"]) if linear_k0 else int(e["n_units"])
    ws_on = bool(grad and finite(w["seed_share"]) and w["seed_share"] > 0.5)
    d = dict(uar_prop=ua, uar_other=ub, effect=e["effect"], lo=e["lo"], hi=e["hi"], se=e["se"], ci_rule=e["method"], df=e["df"],
             v_cluster=e["v_jk"], v_seed=e["v_seed"], seed_share=w["seed_share"], seed_range_mean=w["seed_range_mean"],
             seed_range_max=w["seed_range_max"], identical=bool(e["identical"]),
             seed_varies=bool(e["seed_varies"]), n_folds=int(e["n_folds"]), n_units=n_units,
             folds_dropped=e.get("folds_dropped", ""), boot_lo=e.get("boot_lo", np.nan), boot_hi=e.get("boot_hi", np.nan),
             above=bool(e["lo"] > 0), below=bool(e["hi"] < 0), ws_applies=ws_on,
             df_ws=w["df_ws"] if ws_on else np.nan, lo_ws=w["lo_ws"] if ws_on else np.nan, hi_ws=w["hi_ws"] if ws_on else np.nan)
    d["above_ws"] = bool(ws_on and d["lo_ws"] > 0)
    d["below_ws"] = bool(ws_on and d["hi_ws"] < 0)
    return d


# ------------------------------------------------------------------ 単位の読み込み

class Labels:
    """コーパスのラベルと塊（`dai3.corpus_df` → `dai_data.load_corpus` → `measure/data.py` の load_group．特徴を GPU に載せない）。"""
    cache: dict = {}

    def __init__(self, c: str):
        df = dai3.corpus_df(c)
        self.df = df
        self.y = dict(zip(df["sample_id"].astype(str), df["label_4"].astype(int)))
        self.cl = dict(zip(df["sample_id"].astype(str), df["cluster"].astype(str)))
        sp = pd.read_csv(ROOT / "splits" / f"{c}.csv", keep_default_na=False)
        m = sp.merge(df[["sample_id", "cluster", "label_4"]].astype(str), on="sample_id", how="outer", suffixes=("_s", "_d"), indicator=True)
        self.split_mismatch = int((m["_merge"] != "both").sum() + (m["cluster_s"].astype(str) != m["cluster_d"].astype(str)).sum()
                                  + (m["label_4_s"].astype(str) != m["label_4_d"].astype(str)).sum())

    @classmethod
    def get(cls, c):
        if c not in cls.cache:
            cls.cache[c] = Labels(c)
        return cls.cache[c]


def lin_preds(z, s: str) -> dict:
    """lin の npz → {(手法, 生徒, K, 点の鍵): 判定}（dai2.py の保存の形．none2 は dai2 の解き直した none[K]）。"""
    out = {}
    for K in KS_ALL:
        if f"none_{s}_K{K}_pred" not in z:
            continue
        out[("none2", s, K, ())] = z[f"none_{s}_K{K}_pred"]
        out[("saerens", s, K, ())] = z[f"sae_{s}_K{K}_pred"]
        for pre in ("self", "lin", "nc"):
            for it, pr in zip(z[f"{pre}_{s}_K{K}_items"], z[f"{pre}_{s}_K{K}_pred"]):
                it = json.loads(str(it))
                out[(it["method"], s, K, pkey(it["method"], it))] = pr
        for j, M in enumerate(dai2.T3A_M):
            out[("t3a", s, K, pkey("t3a", {"M": M}))] = z[f"t3a_{s}_K{K}_pred"][j]
            out[("t3a_b", s, K, pkey("t3a_b", {"M": M}))] = z[f"t3a_{s}_K{K}_pred_b"][j]
    if f"st_{s}_items" in z:
        for it, pr in zip(z[f"st_{s}_items"], z[f"st_{s}_pred"]):
            it = json.loads(str(it))
            out[("st", s, int(it["K"]), pkey("st", it))] = pr
    return out


def coral_preds(z, s: str) -> dict:
    out = {}
    if f"coral_{s}_start_pred" not in z:
        return out
    for j, r in enumerate(G.CORAL_R):
        out[("coral", s, 0, pkey("coral", {"r": r}))] = z[f"coral_{s}_start_pred"][j]
    for it, pr in zip(z[f"coral_{s}_items"], z[f"coral_{s}_pred"]):
        it = json.loads(str(it))
        if it["method"] == "coral":
            out[("coral", s, int(it["K"]), pkey("coral", it))] = pr
        else:                                                           # 評価の回の引き戻しの曲線（照合だけ）
            out[("coral_tether", s, int(it["K"]), pkey("coral", it) + (("Cw", norm(it["Cw"])), ("Cb", norm(it["Cb"]))))] = pr
    return out


def map_preds(z, s: str) -> tuple[dict, dict]:
    out, roles = {}, {}
    if f"map_{s}_items" not in z:
        return out, roles
    for it, pr in zip(z[f"map_{s}_items"], z[f"map_{s}_pred"]):
        it = json.loads(str(it))
        out[(it["method"], s, int(it["K"]), pkey(it["method"], it))] = pr
    for it, pr in zip(z[f"maprep_{s}_items"], z[f"maprep_{s}_pred"]):
        it = json.loads(str(it))
        k = ("rep:" + it["method"], s, int(it["K"]), pkey(it["method"], it))
        out[k] = pr
        roles[k] = tokens(it["roles"])
    return out, roles


def xm_preds(z, s: str) -> tuple[dict, dict]:
    out, roles = {}, {}
    if f"xm_{s}_pred" not in z:
        return out, roles
    for it, pr in zip(z["xm_items"], z[f"xm_{s}_pred"]):
        it = json.loads(str(it))
        out[("xmuda", s, int(it["K"]), pkey("xmuda", it))] = pr
    for it, pr in zip(z["xmrep_items"], z[f"xmrep_{s}_pred"]):
        it = json.loads(str(it))
        k = ("rep:xmuda", s, int(it["K"]), pkey("xmuda", it))
        out[k] = pr
        roles[k] = tokens(it["roles"])
    return out, roles


def row_key(r) -> tuple | None:
    """csv の行 → 判定の鍵（lin_preds などと同じ形）．照らさない行は None。"""
    m = str(r["method"])
    s, K = str(r["student"]), int(r["K"])
    if m == "none":
        return ("none2", s, K, ())
    if m == "saerens":
        return ("saerens", s, K, ())
    if m == "st" and str(r.get("beta_dummy")) == "True":
        return None
    if m in GRIDS:
        return (m, s, K, pkey(m, r))
    if m == "coral_tether":
        return ("coral_tether", s, K, pkey("coral", r) + (("Cw", norm(r["Cw"])), ("Cb", norm(r["Cb"]))))
    if m.endswith("_rep") and m[:-4] in GRIDS:
        return ("rep:" + m[:-4], s, K, pkey(m[:-4], r))
    return None


class Unit:
    """1 単位の実験 1 と実験 2 の判定と行（コーパス c のテスト）。"""

    def __init__(self, job: dict, f: int, r: int, parts: tuple):
        self.f, self.r = f, r
        b1 = job["d1"] / f"{job['T']}_{job['cname']}_f{f}_r{r}"
        self.up = ST.UnitPreds(b1)
        self.sid = self.up.sid
        lab = Labels.get(job["corpus"])
        self.y = np.array([lab.y[x] for x in self.sid])
        self.cl = np.array([lab.cl[x] for x in self.sid])
        self.half = dai3.test_halves(self.y, self.cl, self.sid)
        self.P, self.roles, self.rows, self.z = {}, {}, {}, {}
        for part in parts:
            p = Path(f"{job['d2']}/{job['T']}_{job['cname']}_f{f}_r{r}_{part}")
            z = np.load(f"{p}.npz")
            self.z[part] = z
            self.rows[part] = pd.read_csv(f"{p}.csv", low_memory=False)
            for s in DD.STUDENTS:
                if part == "lin":
                    self.P.update(lin_preds(z, s))
                elif part == "coral":
                    self.P.update(coral_preds(z, s))
                elif part == "map":
                    a, b = map_preds(z, s); self.P.update(a); self.roles.update(b)
                elif part == "xmuda" and s != "AVT":
                    a, b = xm_preds(z, s); self.P.update(a); self.roles.update(b)
        fz1 = job["fz1"]
        self.d1 = {}
        for s in DD.STUDENTS:
            for K in KS_ALL:
                v = fz1["main"][str(K)]
                j = fz1["select"][f"{s}|{K}|joint|{v}"]
                j1 = fz1["select"][f"{s}|{K}|joint|one"]
                self.d1[("prop", s, K)] = (self.up.p3item(s, K, method="joint", teacher=v, rho=j["rho"], a=j["a"]), ("joint", v, j["rho"], j["a"]))
                self.d1[("prop_one", s, K)] = (self.up.p3item(s, K, method="joint", teacher="one", rho=j1["rho"], a=j1["a"]),
                                               ("joint", "one", j1["rho"], j1["a"]))
                self.d1[("none", s, K)] = (self.up.none(s, K, fz1["tether"]), None)
        self.grid = {}
        for k in self.P:
            self.grid.setdefault(k[:3], set()).add(k[3])
        self._cnt, self._uar = {}, {}

    def pred(self, key):
        if key[0] in ("prop", "prop_one", "none"):
            return self.d1[key][0]
        if key[0] == "none_t":                                          # 実験 1 の段 1 の none[K]（引き戻しの格子の 1 点．CORAL の曲線の相手）
            return self.none_t(*key[1:])
        if key[0] == "zero":                                            # 水準の区間の相手（正解 0．dai1_desc の zero_of と同じ量）
            return np.full(len(self.y), -1, np.int8)
        return self.P[key]

    def none_t(self, s: str, K: int, cw: float, cb: float):
        """実験 1 の段 1 の保存の none[K]（引き戻し (C′, C_b)）の判定．無ければ None。"""
        items = [str(x) for x in self.up.p1[f"p1_{s}_items"]]
        name = f"{K}|{cw}|{cb}"
        return self.up.p1[f"p1_{s}_pred"][items.index(name)] if name in items else None

    def cnt(self, key):
        """塊ごとの数（区間に渡す．`dai1_stats.counts`）。"""
        if key not in self._cnt:
            self._cnt[key] = ST.counts(self.y, self.pred(key).astype(int), self.cl)
        return self._cnt[key]

    def _tot(self, pr: np.ndarray, msk=None) -> tuple:
        """塊を 1 つにまとめた数 (名前, 正解の数 (1, 4), 件数 (1, 4))．塊で足した数と同じなので uar_full・uar_rows の値は変わらない。"""
        y = self.y if msk is None else self.y[msk]
        p = pr if msk is None else pr[msk]
        cor = np.bincount(y[p == y], minlength=4).astype(float)[None]
        tot = np.bincount(y, minlength=4).astype(float)[None]
        return np.array(["*"]), cor, tot

    def uar(self, key) -> float:
        """単位のテストの UAR（`dai1_stats.uar_full`．rows.csv と同じ定義）。"""
        if key not in self._uar:
            self._uar[key] = ST.uar_full(self._tot(self.pred(key).astype(int)))
        return self._uar[key]

    def half_uar(self, key) -> tuple[float, float]:
        """2 組（`dai3.test_halves`）それぞれの UAR（`dai1_stats.uar_rows`．30 本未満か感情が欠ければ NaN）。"""
        pr = self.pred(key).astype(int)
        out = []
        for h in (0, 1):
            _, cor, tot = self._tot(pr, self.half == h)
            out.append(float(ST.uar_rows(cor[0], tot[0])))
        return tuple(out)


def check_unit(u: Unit, job: dict, chk: dict) -> None:
    """照合：(1) 実験 2 の行の uar と判定から作った UAR，(2) 実験 1 の提案・none・重みなしの提案の行，(3) sample_id の一致，
    (4) dis_ctl・dis_rep の作り直し，(5) 引き戻しの鍵と固定のファイルの sha256（評価の回）．数を chk に足す。"""
    p1 = u.up.p1
    for part, z in u.z.items():
        same = np.array_equal(z["sample_test"], p1["sample_test"]) and all(np.array_equal(z[f"sample_K{K}"], p1[f"sample_K{K}"]) for K in DD.KS)
        chk["unit_sample_mismatch"] += int(not same)
        chk["tether_key_mismatch"] += int(str(z["tether_key"]) != job["tether_key"])
        if job["mode"] == "eval":
            chk["frozen_sha_mismatch"] += int(str(z["frozen_sha256"]) != job["fz2_sha"])
        rows = u.rows[part]
        chk["rows_dirty"] += int((rows["dirty"].astype(str) == "True").sum())
        job["_commits"].update(rows["commit"].astype(str))
        seen = set()
        for _, r in rows.iterrows():
            k = row_key(r) if not str(r["method"]).startswith("_") else None
            if k is None:
                continue
            if k in seen:
                chk["rows_duplicate"] += 1
                continue
            seen.add(k)
            if k not in u.P:
                chk["rows_without_pred"] += 1
                continue
            chk["rows_checked"] += 1
            chk["rows_uar_mismatch"] += int(not abs(u.uar(k) - float(r["uar"])) <= 1e-6)
            if part in ("map", "xmuda") and k[0].startswith("rep:"):
                m = k[0][4:]
                prod = (m, k[1], k[2], k[3])
                chk["dis_rep_checked"] += 1
                chk["dis_rep_mismatch"] += int(not abs(float((u.P[k] != u.P[prod]).mean()) - float(r["dis_rep"])) <= 1e-9)
            if part in ("map", "xmuda") and k[0] in ("dann", "mme", "xmuda") and finite(r.get("dis_ctl")):
                ck = ("mapctl", k[1], k[2], ctl_pk(k[3]))
                if ck in u.P:
                    chk["dis_ctl_checked"] += 1
                    chk["dis_ctl_mismatch"] += int(not abs(float((u.P[k] != u.P[ck]).mean()) - float(r["dis_ctl"])) <= 1e-9)
        own = {k for k in u.P if (k[0] in GRIDS or k[0] in ("none2", "saerens")) and
               ((part == "lin" and k[0] in LIN_PART + ("none2",)) or (part == "coral" and k[0] == "coral") or
                (part == "map" and k[0] in MAP_M) or (part == "xmuda" and k[0] == "xmuda"))}
        chk["preds_without_row"] += len(own - seen)
    r3 = pd.read_csv(f"{job['d1']}/{job['T']}_{job['cname']}_f{u.f}_r{u.r}_p3.csv", low_memory=False)
    r1 = pd.read_csv(f"{job['d1']}/{job['T']}_{job['cname']}_f{u.f}_r{u.r}_p1.csv", low_memory=False)
    for rr_ in (r1, r3):
        chk["d1_rows_dirty"] += int((rr_["dirty"].astype(str) == "True").sum())
        job["_d1_commits"].update(rr_["commit"].astype(str))
    for (name, s, K), (_pr, spec) in u.d1.items():
        if name == "none":
            if K == 0:
                rr = r1[(r1.stage == 1) & (r1.method == "start") & (r1.kind == s)]
            else:
                cw, cb = job["fz1"]["tether"][f"{s}|{K}"]
                rr = r1[(r1.stage == 1) & (r1.method == "none") & (r1.kind == s) & (r1.K == K) & np.isclose(r1.Cw, cw) & np.isclose(r1.Cb, cb)]
        else:
            _, v, rho, a = spec
            rr = r3[(r3.stage == 3) & (r3.method == "joint") & (r3.student == s) & (r3.K == K) & (r3.teacher == v)
                    & np.isclose(r3.rho.astype(float), float(rho)) & np.isclose(r3.a.astype(float), float(a))]
        chk["d1_rows_checked"] += 1
        chk["d1_rows_mismatch"] += int(len(rr) != 1 or not abs(u.uar((name, s, K)) - float(rr.uar.iloc[0])) <= 1e-6)
    for s in DD.STUDENTS:                                              # CORAL の曲線の相手：引き戻しの格子 14 点の none[K] の行
        for K in DD.KS:
            for cw, cb in dai1.TETHER:
                pr = u.none_t(s, K, cw, cb)
                if pr is None:
                    chk["d1_tether_missing"] += 1
                    continue
                rr = r1[(r1.stage == 1) & (r1.method == "none") & (r1.kind == s) & (r1.K == K) & np.isclose(r1.Cw, cw) & np.isclose(r1.Cb, cb)]
                chk["d1_tether_rows_checked"] += 1
                chk["d1_tether_rows_mismatch"] += int(len(rr) != 1 or not abs(u.uar(("none_t", s, K, cw, cb)) - float(rr.uar.iloc[0])) <= 1e-6)
    if job["mode"] == "eval":
        for w in (1, 3):
            chk["d1_frozen_sha_mismatch"] += int(str(getattr(u.up, f"p{w}")["frozen_sha256"]) != job["fz1_sha"])


# ------------------------------------------------------------------ 1 つのコーパスの集計

def selected(job: dict, m: str, s: str, K: int) -> tuple:
    if m == "saerens":
        return ()
    if m == "st" and K == 0:
        return pkey("st", {"beta": dai2.ST_BETA[0]})                    # K = 0 の S+T は 1 問（β の 3 行は同じ値．dai2.py）
    d = {k: v for k, v in job["fz2"]["select"][f"{s}|{K}|{m}"].items() if k != "pseudo_uar"}
    return pkey(m, d)


def prop_point(fz1: dict, s: str, K: int, version: str | None = None) -> str:
    """提案（同時に解く版）の版と選んだ設定（実験 1 の固定のファイル．version が None なら主の版）。"""
    v = version or fz1["main"][str(K)]
    j = fz1["select"][f"{s}|{K}|joint|{v}"]
    return json.dumps({"version": v, "rho": float(j["rho"]), "a": float(j["a"])})


def role_point(job: dict, m: str, s: str, K: int, role: str) -> tuple:
    if role == "selected":
        return selected(job, m, s, K)
    if role == "ref":
        return pkey("xmuda", G.XM_REF)
    return pkey(m, DEFAULTS[m])


def role_kinds(job: dict, m: str) -> list:
    base = ["ref"] if m == "xmuda" else (["default"] if m in MAP_M else [])
    return (["selected"] + base) if job["mode"] == "eval" else base


def own_token(m: str, s: str, role: str) -> str:
    return f"selected_{s}" if (m == "xmuda" and role == "selected") else role


def cells(units: list, key_of) -> dict:
    return {(u.f, u.r): u.cnt(key_of(u)) for u in units}


def aggregate(job: dict, jobs: int, null: bool) -> dict:
    t0 = time.time()
    T, corpus = job["T"], job["corpus"]
    cut = job["cut"]
    chk = {k: 0 for k in ("unit_sample_mismatch", "tether_key_mismatch", "frozen_sha_mismatch", "rows_duplicate", "rows_without_pred",
                          "rows_checked", "rows_uar_mismatch", "dis_rep_checked", "dis_rep_mismatch", "dis_ctl_checked", "dis_ctl_mismatch",
                          "preds_without_row", "d1_rows_checked", "d1_rows_mismatch", "d1_frozen_sha_mismatch", "grid_mismatch",
                          "prop_one_own_dup_mismatch", "ws_vseed_mismatch", "rows_dirty", "d1_rows_dirty",
                          "degenerate_cells", "degenerate_interval_nonzero", "d1_tether_rows_checked", "d1_tether_rows_mismatch",
                          "d1_tether_missing", "coral_tether_cells", "coral_tether_missing", "coral_tether_main_mismatch")}
    job["_commits"], job["_d1_commits"] = set(), set()
    U = {}
    for f, r in job["lin_units"]:
        parts = ("lin", "coral") + (("map", "xmuda") if (f, r) in job["grad_units"] else ())
        U[(f, r)] = Unit(job, f, r, parts)
    for (f, r) in job["grad_units"]:
        if (f, r) not in U:
            U[(f, r)] = Unit(job, f, r, ("map", "xmuda"))
    for u in U.values():
        check_unit(u, job, chk)
    lin_u = [U[k] for k in job["lin_units"]]
    grad_u = [U[k] for k in job["grad_units"]]
    units_of = lambda m: grad_u if m in GRAD_M else lin_u  # noqa: E731
    g_eff, _ = ST.g_eff_of(corpus)
    tag = dict(T=T, c=job["c"], corpus=corpus)
    # 格子の点の過不足（設計の格子から削り方を引いたもの）
    for m in METHODS:
        if m == "saerens":
            continue
        for s in students_of(m):
            for K in ks_of(m):
                want = set(grid_order(m, cut)) if not (m == "st" and K == 0) else {selected(job, m, s, K)}
                for u in units_of(m):
                    chk["grid_mismatch"] += int(u.grid.get((m, s, K), set()) != want)
    # ---- 区間のセル
    tasks, meta = [], {}

    same_of = {}                                                        # 2 つの行の判定が全部の単位で同じセル（退化した場合の検算）

    def add(name, A_of, B_of, us, extra, null_=False, seed_key=None):
        key = (name,) + tuple(extra.values())
        meta[key] = (name, extra)
        same_of[key] = all(np.array_equal(u.pred(A_of(u)), u.pred(B_of(u))) for u in us)
        tasks.append((key, cells(us, A_of), cells(us, B_of), g_eff, cell_seed((corpus, T, job["c"]) + (seed_key or key)), null_))

    for m in METHODS:
        for s in students_of(m):
            for K in ks_of(m):
                pk = selected(job, m, s, K)
                add("pm", lambda u, s=s, K=K: ("prop", s, K), lambda u, m=m, s=s, K=K, pk=pk: (m, s, K, pk), units_of(m),
                    dict(student=s, K=K, method=m), null_=null)
    for s in DD.STUDENTS:
        for K in KS_ALL:
            pk = selected(job, "own", s, K)
            add("po", lambda u, s=s, K=K: ("prop_one", s, K), lambda u, s=s, K=K, pk=pk: ("own", s, K, pk), lin_u, dict(student=s, K=K),
                seed_key=("pm", s, K, "own"))                           # 1 の own の行と同じ塊の引き直し（主の版が one なら同じ値になる）
    for m in DA_M:
        for s in students_of(m):
            for K in KS_ALL:
                pk = selected(job, m, s, K)
                pc = {"matched": ctl_pk(pk), "ctl_selected": selected(job, "mapctl", s, K)}
                for cname_, cpk in pc.items():
                    add("cd", lambda u, m=m, s=s, K=K, pk=pk: (m, s, K, pk), lambda u, s=s, K=K, cpk=cpk: ("mapctl", s, K, cpk), grad_u,
                        dict(student=s, K=K, comparison=f"{m}-mapctl[{cname_}]", method=m, ctl_point=cname_))
                add("cd", lambda u, s=s, K=K, cpk=pc["matched"]: ("mapctl", s, K, cpk), lambda u, s=s, K=K: ("none", s, K), grad_u,
                    dict(student=s, K=K, comparison=f"mapctl[matched_{m}]-none", method="mapctl", ctl_point=f"matched_{m}"))
    for s in DD.STUDENTS:
        for K in KS_ALL:
            add("cd", lambda u, s=s, K=K: ("mapctl", s, K, selected(job, "mapctl", s, K)), lambda u, s=s, K=K: ("none", s, K), grad_u,
                dict(student=s, K=K, comparison="mapctl[selected]-none", method="mapctl", ctl_point="ctl_selected"))
    # 解き直しの値で作った「提案 − 手法」（記述）．役の点が全部の勾配法の単位で，その役の名前（map は selected・default，xmuda は
    # selected_<生徒>・ref）を roles に持つ解き直しの行で解き直されているときだけ（gate-code の 2 回目の記録 2・3 の行の選び方）
    rep_cells = []
    for m in GRAD_M:
        for s in students_of(m):
            for K in KS_ALL:
                roles = ["selected"] + (["ref"] if m == "xmuda" else ["default"])
                for role in roles:
                    pk = role_point(job, m, s, K, role)
                    rk = ("rep:" + m, s, K, pk)
                    if not all(rk in u.P and own_token(m, s, role) in u.roles.get(rk, set()) for u in grad_u):
                        continue
                    rep_cells.append((m, s, K, role, pk))
                    for which, kf in (("prod", lambda u, m=m, s=s, K=K, pk=pk: (m, s, K, pk)), ("rep", lambda u, rk=rk: rk)):
                        add("pr", lambda u, s=s, K=K: ("prop", s, K), kf, grad_u, dict(student=s, K=K, method=m, role=role, which=which))
    # 取りこぼし（θ が最大の点 − 選んだ点）と差が無いときの水準
    regret_rows = []
    for m in METHODS:
        if m == "saerens":
            continue
        for s in students_of(m):
            for K in ks_of(m):
                us = units_of(m)
                pts = grid_order(m, cut) if not (m == "st" and K == 0) else [selected(job, m, s, K)]
                th, tA, tB = {}, {}, {}
                for pk in pts:
                    key = (m, s, K, pk)
                    th[pk] = ST.theta({f: [u.uar(key) for u in us if u.f == f] for f in sorted({u.f for u in us})})
                    hv = {u: u.half_uar(key) for u in us}
                    for h, tt in ((0, tA), (1, tB)):
                        per_f = {}
                        for u in us:
                            if np.isfinite(hv[u][h]):
                                per_f.setdefault(u.f, []).append(hv[u][h])
                        tt[pk] = ST.theta(per_f) if per_f else float("nan")
                sel = selected(job, m, s, K)
                best = max(pts, key=lambda p: (th[p], -pts.index(p)))
                ga = max(pts, key=lambda p: (tA[p] if np.isfinite(tA[p]) else -np.inf, -pts.index(p)))
                gb = max(pts, key=lambda p: (tB[p] if np.isfinite(tB[p]) else -np.inf, -pts.index(p)))
                z_ab, z_ba = tB[ga] - tB[sel], tA[gb] - tA[sel]
                regret_rows.append(dict(**tag, student=s, K=K, method=m, n_grid=len(pts), point_selected=pjson(m, sel),
                                        point_best=pjson(m, best), best_is_selected=best == sel, regret=th[best] - th[sel],
                                        zero_level=0.5 * (z_ab + z_ba), zero_level_ab=z_ab, zero_level_ba=z_ba,
                                        n_units=len(us) if not (m not in GRAD_M and K == 0) else len({u.f for u in us})))
                add("rg", lambda u, m=m, s=s, K=K, b=best: (m, s, K, b), lambda u, m=m, s=s, K=K, p=sel: (m, s, K, p), us,
                    dict(student=s, K=K, method=m))
    # CORAL の引き戻しの曲線（評価の回だけ．選んだ d_pca で K ≥ 1 を引き戻しの格子 14 点で解いた coral_tether の行）
    curve = {}
    for s in DD.STUDENTS:
        for K in DD.KS:
            pk = selected(job, "coral", s, K)
            tk = {(cw, cb): ("coral_tether", s, K, pk + (("Cw", norm(cw)), ("Cb", norm(cb)))) for cw, cb in dai1.TETHER}
            have = [all(k in u.P for u in lin_u) for k in tk.values()]
            if not any(have):
                continue                                                # 仮のターゲットの回には曲線が無い
            if not all(have) or any(u.none_t(s, K, cw, cb) is None for u in lin_u for cw, cb in dai1.TETHER):
                chk["coral_tether_missing"] += 1
                continue
            curve[(s, K)] = (pk, tk)
            for (cw, cb), ck in tk.items():
                nk = ("none_t", s, K, cw, cb)
                ex = dict(student=s, K=K, Cw=cw, Cb=cb)
                add("ctl", lambda u, ck=ck: ck, lambda u, ck=ck: ("zero", ck), lin_u, ex | dict(row="coral"))
                add("ctl", lambda u, nk=nk: nk, lambda u, nk=nk: ("zero", nk), lin_u, ex | dict(row="none"))
                add("ctd", lambda u, ck=ck: ck, lambda u, nk=nk: nk, lin_u, ex | dict(row="coral − none"))
    res = run_cells(tasks, jobs)
    for v in res.values():
        chk["ws_vseed_mismatch"] += int(not v[1]["v_seed_check"] <= 1e-9)
    for key, same in same_of.items():                                  # 退化した場合：θ・区間の両端・標準誤差が 0 で identical の印
        if same:
            e = res[key][0]
            chk["degenerate_cells"] += 1
            chk["degenerate_interval_nonzero"] += int(not (e["effect"] == 0 and e["lo"] == 0 and e["hi"] == 0 and e["se"] == 0 and e["identical"]))

    def ecols(key, linear_k0: bool, grad: bool) -> dict:
        return effect_cols(res[key], linear_k0, grad) | {"same_preds": same_of[key]}
    # ---- 表 1
    T1 = []
    for key, (name, ex) in meta.items():
        if name != "pm":
            continue
        m = ex["method"]
        r = res[key]
        pk = selected(job, m, ex["student"], ex["K"])
        T1.append(dict(**tag, student=ex["student"], K=ex["K"], method=m, group=GROUP[m], gradient=m in GRAD_M,
                       appendix=m == "saerens", descriptive_version=m == "t3a_b",
                       point=pjson(m, pk),
                       prop_version=job["fz1"]["main"][str(ex["K"])], prop_point=prop_point(job["fz1"], ex["student"], ex["K"]),
                       **ecols(key, m not in GRAD_M and ex["K"] == 0, m in GRAD_M)))
    T1 = pd.DataFrame(T1)
    # ---- 表 2
    T2 = []
    for key, (name, ex) in meta.items():
        if name != "po":
            continue
        s, K = ex["student"], ex["K"]
        c = ecols(key, K == 0, False)
        is_sel = job["fz1"]["main"][str(K)] == "sel"
        T2.append(dict(**tag, student=s, K=K, design_row=is_sel, main_version=job["fz1"]["main"][str(K)],
                       prop_point=prop_point(job["fz1"], s, K, "one"), point=pjson("own", selected(job, "own", s, K)), **c))
        if not is_sel:                                                  # 主の版が one なら 1 の own の行と同じ量になる（照合）
            o = T1[(T1.student == s) & (T1.K == K) & (T1.method == "own")].iloc[0]
            chk["prop_one_own_dup_mismatch"] += int(not (np.isclose(o.effect, c["effect"]) and np.isclose(o.lo, c["lo"], equal_nan=True)))
    T2 = pd.DataFrame(T2)
    # ---- 表 4
    T4 = []
    for key, (name, ex) in meta.items():
        if name == "cd":
            T4.append(dict(**tag, **ex, **ecols(key, False, True)))
    T4 = pd.DataFrame(T4)
    k0 = []
    for s in DD.STUDENTS:
        for pk in grid_order("mapctl", cut):
            ag = [float((u.pred(("mapctl", s, 0, pk)) == u.pred(("none", s, 0))).mean()) for u in grad_u]
            k0.append(dict(**tag, student=s, point=json.dumps(pdict(pk)), n_units=len(ag), n_units_identical=int(sum(a == 1.0 for a in ag)),
                           agree_min=min(ag) if ag else np.nan))
    k0 = pd.DataFrame(k0)
    # ---- 表 6（解き直し）
    T6r = []
    for key, (name, ex) in meta.items():
        if name == "pr":
            T6r.append(dict(**tag, **ex, **ecols(key, False, True)))
    T6r = pd.DataFrame(T6r)
    # ---- 表 3
    x9 = T1[(T1.method == "xmuda") & T1.K.isin(KS9)].copy()
    if len(T6r):
        rp = T6r[(T6r.method == "xmuda") & (T6r.role == "selected") & (T6r.which == "rep")].set_index(["student", "K"])
        for col in ("effect", "lo", "hi", "above", "below"):
            x9[f"{col}_rep"] = [rp[col].get((s, K), np.nan) if len(rp) else np.nan for s, K in zip(x9.student, x9.K)]
    else:
        for col in ("effect", "lo", "hi", "above", "below"):
            x9[f"{col}_rep"] = np.nan
    fam = job["families"]
    desc = set(job["descriptive"])
    f3 = fam[(fam.family == 3)]
    tested = [r for _, r in f3.iterrows() if f"3|{r.student}|{int(r.K)}" not in desc]
    na = sum(1 for r in tested if r.effect > 0 and r.p_holm < ST.ALPHA)
    F3 = []                                                             # 族 3 の 12 セルの 1 つずつ（(a) の数え方の確かめ）
    for _, r in f3.iterrows():
        part = job["fz1"]["partner"][f"{r.student}|{int(r.K)}"]
        dsc = f"3|{r.student}|{int(r.K)}" in desc
        F3.append(dict(**tag, student=r.student, K=int(r.K), partner=part["method"], partner_version=part["version"], effect=r.effect,
                       lo=r.lo, hi=r.hi, p_holm=r.p_holm, descriptive=dsc, pos_sig=bool(r.effect > 0 and r.p_holm < ST.ALPHA),
                       counted_for_a=bool(not dsc and r.effect > 0 and r.p_holm < ST.ALPHA)))
    na_all = int(((f3.effect > 0) & (f3.p_holm < ST.ALPHA)).sum())
    n_ab, n_be = int(x9.above.sum()), int(x9.below.sum())
    rep_ok = bool(len(x9) and x9["above_rep"].notna().all())
    cs = dict(**tag, n_fam3_cells=len(f3), n_fam3_descriptive=len(f3) - len(tested), n_fam3_pos_sig_tested=na, n_fam3_pos_sig_all=na_all,
              a_ok=na >= 7, n_xmuda9_cells=len(x9), n_above9=n_ab, n_below9=n_be, b_ok=n_ab >= 5, c_ok=n_be == 0,
              n_above9_rep=int(x9["above_rep"].fillna(False).astype(bool).sum()) if rep_ok else np.nan,
              n_below9_rep=int(x9["below_rep"].fillna(False).astype(bool).sum()) if rep_ok else np.nan,
              fam3_source=job["fam_source"])
    cs["corpus_sentence"] = bool(cs["a_ok"] and cs["b_ok"] and cs["c_ok"] and len(f3) == 12 and len(x9) == 9)
    # ---- 表 10（符号の反転）
    nulls = []
    groups = {"all": T1.index, "all_main": T1.index[~(T1.appendix | T1.descriptive_version)], "xmuda9": x9.index}
    groups.update({f"method:{m}": T1.index[T1.method == m] for m in METHODS})
    pm_keys = [k for k, (n, _) in meta.items() if n == "pm"]
    key_of_row = {i: k for i, k in zip(T1.index, pm_keys)}
    if null:
        npat = len(res[pm_keys[0]][4])
        for gname, idx in groups.items():
            for what, pos in (("above", 4), ("below", 5)):
                cnt = np.zeros(npat, int)
                for i in idx:
                    cnt += np.array(res[key_of_row[i]][pos], int)
                obs = int(cnt[0])
                nulls.append(dict(**tag, group=gname, what=what, n_cells=len(idx), observed=obs, null_mean=float(cnt.mean()),
                                  null_q95=qhigh(cnt), null_max=int(cnt.max()), frac_ge_observed=float((cnt >= obs).mean()), n_patterns=npat))
    # ---- 表 5（DA の項）と da_quantities
    T5, DQ = [], []
    num_cols = {"map": ["anorm", "wa_dist", "w_dist", "ent_first", "ent_last", "dom_first", "dom_last", "dis_ctl", "mdiff_before", "mdiff_after",
                        "cdiff_before", "cdiff_after", "dloss_before", "dloss_after", "dacc_before", "dacc_after"],
                "xmuda": ["anorm", "wa_dist", "w_dist", "ent_first", "ent_last", "kl_first", "kl_last", "dis_ctl"]}
    idx_rows = {}
    for u in grad_u:
        for part in ("map", "xmuda"):
            for _, r in u.rows[part].iterrows():
                k = row_key(r) if not str(r["method"]).startswith("_") else None
                if k is not None:
                    idx_rows[(u.f, u.r) + k] = r
    for m in DA_M:
        part = "xmuda" if m == "xmuda" else "map"
        for s in students_of(m):
            for K in KS_ALL:
                for role in role_kinds(job, m):
                    pk = role_point(job, m, s, K, role)
                    cls, dm, miss, vals = [], [], 0, {c: [] for c in num_cols[part]}
                    for u in grad_u:
                        prod = idx_rows.get((u.f, u.r, m, s, K, pk))
                        rm = idx_rows.get((u.f, u.r, "rep:" + m, s, K, pk))
                        rc = idx_rows.get((u.f, u.r, "rep:mapctl", s, K, ctl_pk(pk)))
                        if prod is not None:
                            for c_ in num_cols[part]:
                                vals[c_].append(float(prod[c_]))
                        ok = (prod is not None and rm is not None and rc is not None and own_token(m, s, role) in tokens(rm["roles"])
                              and f"ctl_of_{m}_{role}" in tokens(rc["roles"]) and finite(prod["dis_ctl"]))
                        if not ok:
                            miss += 1
                            continue
                        d, e_m, e_c = float(prod["dis_ctl"]), float(rm["dis_rep"]), float(rc["dis_rep"])
                        cls.append(1 if d <= e_c + SLACK_I else (2 if d <= e_m else 3))
                        dm.append(u.uar((m, s, K, pk)) - u.uar(("mapctl", s, K, ctl_pk(pk))))
                    cls, dm = np.array(cls), np.array(dm)
                    row = dict(**tag, method=m, student=s, K=K, role=role, point=json.dumps(pdict(pk)), n_units=len(grad_u), n_missing=miss)
                    for i, nm in ((1, "i"), (2, "ii"), (3, "iii")):
                        sel_ = dm[cls == i] if len(cls) else np.array([])
                        row[f"n_{nm}"] = int((cls == i).sum()) if len(cls) else 0
                        row[f"uar_diff_mean_{nm}"] = float(sel_.mean()) if len(sel_) else np.nan
                        row[f"uar_diff_max_{nm}"] = float(sel_.max()) if len(sel_) else np.nan
                        row[f"uar_diff_min_{nm}"] = float(sel_.min()) if len(sel_) else np.nan
                    row["not_worked"] = bool(row["n_i"] > len(grad_u) / 2)
                    T5.append(row)
                    q = dict(**tag, method=m, student=s, K=K, role=role, point=json.dumps(pdict(pk)), n_units=len(vals[num_cols[part][0]]))
                    for c_, v in vals.items():
                        v = np.array(v, float)
                        q.update({f"{c_}_mean": float(np.nanmean(v)) if len(v) and np.isfinite(v).any() else np.nan,
                                  f"{c_}_min": float(np.nanmin(v)) if len(v) and np.isfinite(v).any() else np.nan,
                                  f"{c_}_max": float(np.nanmax(v)) if len(v) and np.isfinite(v).any() else np.nan})
                    DQ.append(q)
    # ---- 表 6（揺れと一致）
    T6, AG = [], []
    for m in GRAD_M:
        for s in students_of(m):
            for K in KS_ALL:
                acc = {}
                for u in grad_u:
                    for k, toks in u.roles.items():
                        if k[0] != "rep:" + m or k[1] != s or k[2] != K:
                            continue
                        kinds = set()
                        if m == "xmuda":
                            kinds |= {"ref"} & toks
                            kinds |= {"selected"} if f"selected_{s}" in toks else set()
                        else:
                            kinds |= {"default", "selected"} & toks
                            if not kinds and any(t.startswith("ctl_of_") for t in toks):
                                kinds.add("ctl")
                        prod = (m, s, K, k[3])
                        flip = float((u.P[k] != u.P[prod]).mean())
                        du = u.uar(k) - u.uar(prod)
                        for kd in kinds:
                            acc.setdefault(kd, []).append((flip, du))
                for kd, v in acc.items():
                    fl, du = np.array([x[0] for x in v]), np.array([x[1] for x in v])
                    T6.append(dict(**tag, method=m, student=s, K=K, role=kd, n=len(v), flip_mean=fl.mean(), flip_min=fl.min(), flip_max=fl.max(),
                                   uar_diff_mean=du.mean(), uar_diff_min=du.min(), uar_diff_max=du.max()))
                    AG += [dict(T=T, c=job["c"], method=m, role=kd, student=s, K=K, agree=1.0 - x[0]) for x in v]
    # ---- 表 7（2 つの印）
    MK, MP = [], []
    prod_rows = []
    for u in grad_u:
        for part, meths in (("map", MAP_M), ("xmuda", ("xmuda",))):
            d = u.rows[part]
            d = d[d.method.isin(meths)].copy()
            d["f_"], d["r_"] = u.f, u.r
            prod_rows.append(d)
    PR = pd.concat(prod_rows, ignore_index=True) if prod_rows else pd.DataFrame()
    if len(PR):
        PR["pk"] = [repr(pkey(m, r)) for m, r in zip(PR.method, PR.to_dict("records"))]
        PR["disc"] = np.where(PR.method == "dann", PR.dloss_after.astype(float) > DISC_MAX, False) if "dloss_after" in PR else False
        PR["expand"] = False
        mp = PR.method.isin(MAP_M)
        PR.loc[mp, "expand"] = PR.loc[mp, "cdiff_after"].astype(float) > CDIFF_RATIO * PR.loc[mp, "cdiff_before"].astype(float)
        xm = PR.method == "xmuda"
        if xm.any():
            keycols = ["f_", "r_", "lr", "iters", "kap", "mu", "K"]
            amax = PR[xm].groupby(keycols)["anorm"].transform("max")
            PR.loc[xm, "expand"] = amax > ANORM_MAX
        hits = {}
        for m in GRAD_M:
            for s in students_of(m):
                for K in KS_ALL:
                    for role in (["selected", "default"] if m in MAP_M else ["selected"]):
                        pk = role_point(job, m, s, K, role)
                        hit = PR[(PR.method == m) & (PR.student == s) & (PR.K == K) & (PR.pk == repr(pk))]
                        nd, ne = (int(hit.disc.sum()) if m == "dann" else np.nan), int(hit.expand.sum())
                        MP.append(dict(**tag, method=m, student=s, K=K, role=role, point=json.dumps(pdict(pk)), n_units=len(hit),
                                       n_units_disc=nd, n_units_expand=ne, marked=bool((m == "dann" and nd > 0) or ne > 0)))
                        hits[(m, s, K, role)] = (float(pdict(pk)["lr"]), MP[-1]["marked"])
        for (m, lr, s, K), g in PR.groupby(["method", "lr", "student", "K"]):
            row = dict(**tag, method=m, lr=float(lr), student=s, K=int(K), n_problems=len(g), n_disc=int(g.disc.sum()) if m == "dann" else np.nan,
                       n_expand=int(g.expand.sum()))
            if m in MAP_M:
                row.update(dloss_after_mean=float(g.dloss_after.astype(float).mean()), dloss_after_max=float(g.dloss_after.astype(float).max()))
            for role in ("selected", "default"):                         # その学習率の行に選んだ点・選択なしの点があり，印に当たったか（表の注）
                h = hits.get((m, s, int(K), role))
                row[f"{role}_point_here"] = bool(h is not None and np.isclose(h[0], float(lr)))
                row[f"{role}_point_marked"] = bool(h[1]) if row[f"{role}_point_here"] else np.nan
            MK.append(row)
    # ---- 表 8
    RG = pd.DataFrame(regret_rows)
    rg_keys = [k for k, (n, _) in meta.items() if n == "rg"]
    if len(RG):
        cols = [ecols(k, meta[k][1]["method"] not in GRAD_M and meta[k][1]["K"] == 0, meta[k][1]["method"] in GRAD_M) for k in rg_keys]
        for c_ in ("effect", "lo", "hi", "se", "ci_rule", "identical", "same_preds"):
            RG[f"ci_{c_}"] = [c[c_] for c in cols]
    # ---- 表 12（CORAL の引き戻しの曲線と符号）
    CT, CTS = [], []
    for (s, K), (pk, tk) in curve.items():
        sel_t = tuple(float(x) for x in job["fz1"]["tether"][f"{s}|{K}"])
        main = (("coral", s, K, pk))
        thetas = {}
        for (cw, cb), ck in tk.items():
            base_ = dict(**tag, student=s, K=K, d_pca=float(pdict(pk)["r"]), Cw=cw, Cb=cb, selected_tether=(cw, cb) == sel_t)
            for name_, row in (("ctl", "coral"), ("ctl", "none"), ("ctd", "coral − none")):
                key = (name_, s, K, cw, cb, row)
                CT.append(dict(base_, kind="level" if name_ == "ctl" else "diff", row=row, **ecols(key, False, False)))
            thetas[(cw, cb)] = res[("ctd", s, K, cw, cb, "coral − none")][0]["effect"]
        chk["coral_tether_cells"] += 1
        same_main = all(np.array_equal(u.pred(tk[sel_t]), u.pred(main)) for u in lin_u) if sel_t in tk else False
        same_none = all(np.array_equal(u.pred(("none_t", s, K) + sel_t), u.pred(("none", s, K))) for u in lin_u) if sel_t in tk else False
        chk["coral_tether_main_mismatch"] += int(not (same_main and same_none))
        ref = float(np.sign(thetas.get(sel_t, np.nan)))
        sg = np.sign(np.array(list(thetas.values()), float))
        CTS.append(dict(**tag, student=s, K=K, d_pca=float(pdict(pk)["r"]), row="coral − none", n_points=len(sg), sign_at_selected=ref,
                        n_same_sign=int((sg == ref).sum()), all_same_sign=bool((sg == ref).all()), n_pos=int((sg > 0).sum()),
                        n_neg=int((sg < 0).sum()), n_zero=int((sg == 0).sum()), selected_point_matches_main=bool(same_main),
                        selected_none_matches_main=bool(same_none)))
    # ---- 表 9
    SM = []
    srcs = job["srcs"]
    tot = {c_: np.bincount(np.array(list(Labels.get(c_).y.values())), minlength=4) for c_ in srcs}
    allv = sum(tot.values())
    for c_ in srcs:
        for k in range(4):
            SM.append(dict(**tag, source=c_, emotion=EMO[k], n=int(tot[c_][k]), share_of_source=tot[c_][k] / tot[c_].sum(),
                           share_of_merged=tot[c_][k] / allv.sum(), source_share_of_merged=tot[c_].sum() / allv.sum()))
    for k in range(4):
        SM.append(dict(**tag, source="ALL", emotion=EMO[k], n=int(allv[k]), share_of_source=allv[k] / allv.sum(), share_of_merged=allv[k] / allv.sum(),
                       source_share_of_merged=1.0))
    chk["split_mismatch"] = Labels.get(corpus).split_mismatch
    print(f"{T} {job['c']}: 単位 線形 {len(lin_u)}・勾配法 {len(grad_u)}，区間 {len(tasks)}，行の照合 {chk['rows_uar_mismatch']}/{chk['rows_checked']}，"
          f"実験 1 の行の照合 {chk['d1_rows_mismatch']}/{chk['d1_rows_checked']}，{time.time() - t0:.0f} 秒", flush=True)
    return dict(T1=T1, T2=T2, X9=x9, CS=pd.DataFrame([cs]), F3=pd.DataFrame(F3), CT=pd.DataFrame(CT), CTS=pd.DataFrame(CTS), T4=T4, K0=k0, T5=pd.DataFrame(T5), DQ=pd.DataFrame(DQ), T6=pd.DataFrame(T6),
                T6r=T6r, AG=pd.DataFrame(AG), MK=pd.DataFrame(MK), MP=pd.DataFrame(MP), RG=RG, SM=pd.DataFrame(SM), NU=pd.DataFrame(nulls),
                CHK=pd.DataFrame([dict(**tag, **chk, row_commits="|".join(sorted(job["_commits"])),
                                       d1_row_commits="|".join(sorted(job["_d1_commits"])))]))


def agreement_table(AG: pd.DataFrame) -> pd.DataFrame:
    """6 節：(手法, 役) ごとに，T ごとと 4 つの T を合わせた (単位, 問) の一致の最小と中央値．合わせた中央値と各 T の中央値がすべて 0.99 以上なら真。"""
    out = []
    if not len(AG):
        return pd.DataFrame()
    for (m, role), g in AG.groupby(["method", "role"]):
        per = {T: gg.agree.to_numpy() for T, gg in g.groupby("T")}
        allv = g.agree.to_numpy()
        ok = bool(np.median(allv) >= REPRO_MIN and all(np.median(v) >= REPRO_MIN for v in per.values()))
        for T, v in per.items():
            out.append(dict(T=T, method=m, role=role, n=len(v), agree_min=float(v.min()), agree_median=float(np.median(v)), reproducible=ok))
        out.append(dict(T="ALL", method=m, role=role, n=len(allv), agree_min=float(allv.min()), agree_median=float(np.median(allv)), reproducible=ok))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 族 3（試走）

def families_pseudo(d1_out: Path, T: str, c: str, fz1: dict, units: list) -> pd.DataFrame:
    """試走の族 3：`dai1_stats.collect`（判定の取り出しと行の照合）・`effect_stats`・`holm` を (T, c) に当てる（evaluate と同じ手順）。"""
    lab = Labels.get(c)
    y_of = lambda sid: np.array([lab.y[x] for x in sid])  # noqa: E731
    cl_of = lambda sid: np.array([lab.cl[x] for x in sid])  # noqa: E731
    cl, mism, total = ST.collect(d1_out, "pseudo", T, c, fz1, units, y_of, cl_of)
    g_eff, _ = ST.g_eff_of(c)
    rng = np.random.default_rng(SEED0)
    res = []
    for s in DD.STUDENTS:
        for K in ST.KS_MAIN:
            e = ST.effect_stats(cl[(s, K, "prop")], cl[(s, K, "partner")], g_eff, rng)
            if K == 0:
                e["n_units"] = len({f for f, _ in units})
            res.append(dict(family=3, T=T, c=c, student=s, K=K, mismatch=mism, checked=total, **e))
    for x, a in zip(res, ST.holm(np.array([x["p"] for x in res]))):
        x["p_holm"] = float(a)
    return pd.DataFrame(res)


# ------------------------------------------------------------------ 進め方

def available(base: Path, parts: tuple) -> bool:
    return all(Path(f"{base}_{p}.npz").exists() and Path(f"{base}_{p}.csv").exists() for p in parts)


def make_jobs(args) -> tuple[list, dict]:
    d1, d2 = Path(args.dai1_dir), Path(args.dai2_dir)
    fz1p = Path(args.dai1_frozen)
    fz2p = Path(args.dai2_frozen) if args.dai2_frozen else d2 / "select" / "frozen.json"
    fz1, fz2 = json.load(open(fz1p)), json.load(open(fz2p))
    info = dict(dai1_frozen=str(fz1p), dai1_frozen_sha256=dai1.sha(fz1p), dai2_frozen=str(fz2p), dai2_frozen_sha256=dai1.sha(fz2p),
                dai2_frozen_dai1_sha256=fz2["_meta"].get("dai1_frozen_sha256"), grad_cut=fz2["_meta"].get("grad_cut"))
    info["dai1_frozen_matches_dai2_meta"] = info["dai1_frozen_sha256"] == info["dai2_frozen_dai1_sha256"]
    cut = fz2["_meta"]["grad_cut"]
    jobs = []
    targets = [T for T in args.targets if T in fz2]
    if args.mode == "eval":
        fam_path = d1 / "stats" / "families.csv"
        if not fam_path.exists():
            sys.exit(f"{fam_path} が無い（`python exp/dai1_stats.py eval` を先に流す）")
        fam = pd.read_csv(fam_path)
        if (fam["dirty"].astype(str) == "True").any():
            sys.exit(f"{fam_path} は未コミットの変更がある作業木で作られた（`python exp/dai1_stats.py eval` を汚れていない作業木で流し直す）")
        info.update(families=str(fam_path), families_sha256=dai1.sha(fam_path), families_commit=sorted(set(fam.commit.astype(str))))
        gseeds = [int(x) for x in fz2["_meta"]["grad_seeds_eval"]]
        seen_commit, seen_gate = set(fam.commit.astype(str)), set()          # 単位・families.csv のコミットと gate_sha256（指摘 2-N）
        if targets != list(DD.CORPORA):                                    # T の完全性（gate-code の 7 回目の記録 1）
            sys.exit(f"固定のファイルの T が 4 つそろわない：{targets}")
        for T in targets:
            if int((fam["T"].astype(str).eq(T) & fam["family"].astype(str).eq("3")).sum()) != 12:
                sys.exit(f"{fam_path} の {T} の族 3 の行が 12 でない")
        info.update(families_mismatch_sum=int(pd.to_numeric(fam.get("mismatch", pd.Series([0])), errors="coerce").fillna(0).sum()))
        for T in targets:
            R = int(fz1[T]["n_seeds_eval"])
            if int(fz2[T].get("n_seeds_eval_linear") or R) != R:
                sys.exit(f"{T}: 実験 2 の固定のファイルの n_seeds_eval_linear が実験 1 の n_seeds_eval と違う")
            lin_units = [(f, r) for f in range(DD.N_FOLDS) for r in range(R)]
            grad_units = [(f, r) for f in range(DD.N_FOLDS) for r in gseeds]
            for f, r in lin_units:
                b2 = d2 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
                b1 = d1 / "eval" / "units" / f"{T}_ALL_f{f}_r{r}"
                if not available(b2, ("lin", "coral") + (("map", "xmuda") if (f, r) in grad_units else ())) or not available(b1, ("p1", "p3")):
                    sys.exit(f"{T}: 評価の回の単位 f{f} r{r} の保存がそろっていない")
                for d_, ps in ((d2, ("lin", "coral") + (("map", "xmuda") if (f, r) in grad_units else ())), (d1, ("p1", "p3"))):
                    for p_ in ps:
                        z_ = np.load(f"{d_ / 'eval' / 'units' / f'{T}_ALL_f{f}_r{r}'}_{p_}.npz")
                        seen_commit.add(str(z_["commit"])); seen_gate.add(str(z_["gate_sha256"]))
                        want_fz = info["dai2_frozen_sha256"] if d_ is d2 else info["dai1_frozen_sha256"]
                        if str(z_["frozen_sha256"]) != want_fz:
                            sys.exit(f"{T}: 単位 f{f} r{r} の {p_} の frozen_sha256 が今の固定のファイルと違う")
            jobs.append(dict(mode="eval", T=T, c="ALL", cname="ALL", corpus=T, srcs=DD.others(T), lin_units=lin_units, grad_units=grad_units,
                             d1=d1 / "eval" / "units", d2=d2 / "eval" / "units", fz1=fz1[T], fz2=fz2[T], cut=cut,
                             descriptive=fz1[T]["detect"]["descriptive_cells"], families=fam[fam["T"] == T], fam_source=str(fam_path),
                             tether_key=dai2.tether_key(dai2.parse_tether(fz1[T]["tether"])), fz1_sha=info["dai1_frozen_sha256"],
                             fz2_sha=info["dai2_frozen_sha256"]))
        if seen_commit != {RUN["commit"]}:
            sys.exit(f"今のコミット {RUN['commit'][:10]} と，評価の回の単位・families.csv のコミット {sorted(x[:10] for x in seen_commit)} が違う．"
                     "評価の回のコミットの作業木で流す（run/eval_procedure.md の 8）")
        if seen_gate != {RUN["gate"]}:
            sys.exit(f"単位の gate_sha256 {sorted(x[:10] for x in seen_gate)} が今の eval_gate.json {RUN['gate'][:10]} と違う")
        return jobs, info
    for T in targets:
        for c in DD.others(T):
            if args.pseudo_c and c not in args.pseudo_c:
                continue
            units = [(f, r) for f in range(DD.N_FOLDS) for r in (0, 1)]
            ok = [(f, r) for f, r in units if available(d2 / "pseudo" / "units" / f"{T}_{c}_f{f}_r{r}", ("lin", "coral", "map", "xmuda"))
                  and available(d1 / "pseudo" / "units" / f"{T}_{c}_f{f}_r{r}", ("p1", "p3"))]
            if ok != units and not args.allow_partial:
                sys.exit(f"{T} {c}: 仮のターゲットの単位が {len(ok)}/{len(units)} しかない（試走は --allow-partial）")
            if not ok:
                continue
            d1u = [(f, r) for f, r in units if available(d1 / "pseudo" / "units" / f"{T}_{c}_f{f}_r{r}", ("p1", "p3"))]
            fam = families_pseudo(d1, T, c, fz1[T], d1u)
            jobs.append(dict(mode="pseudo", T=T, c=c, cname=c, corpus=c, srcs=[x for x in DD.others(T) if x != c], lin_units=ok, grad_units=ok,
                             d1=d1 / "pseudo" / "units", d2=d2 / "pseudo" / "units", fz1=fz1[T], fz2=fz2[T], cut=cut,
                             descriptive=fz1[c]["detect"]["descriptive_cells"], families=fam, fam_source=f"dai1_stats（試走．単位 {len(d1u)}）",
                             tether_key=dai2.tether_key(dai2.parse_tether(fz1[T]["tether"])), fz1_sha=info["dai1_frozen_sha256"],
                             fz2_sha=info["dai2_frozen_sha256"]))
    return jobs, info


def eval_guard(args) -> dict:
    """評価の段を流す条件（指摘 2-I．docstring の eval の項）．戻り：照らした sha256。"""
    if Path(args.dai1_dir).resolve() != PROD["dai1"].resolve() or Path(args.dai2_dir).resolve() != PROD["dai2"].resolve() or args.dai2_frozen \
            or Path(args.dai1_frozen).resolve() != (ROOT / "frozen" / "dai1_frozen.json").resolve() or args.allow_partial or args.pseudo_c:
        sys.exit("eval は本番の置き場（results/dai1，results/dai2）と frozen の固定のファイルだけを読む（入力を替える引数は受け付けない）")
    if RUN["dirty"]:
        sys.exit("eval は作業木に未コミットの変更があると流さない（exp，splits，measure，features，data/metadata）")
    pairs = {"dai2": (ROOT / "frozen" / "dai2_frozen.json", PROD["dai2"] / "select" / "frozen.json"),
             "dai1": (ROOT / "frozen" / "dai1_frozen.json", PROD["dai1"] / "select" / "frozen.json")}
    out = {}
    for n, (cp, fz) in pairs.items():
        if not cp.exists() or not fz.exists() or dai1.sha(cp) != dai1.sha(fz):
            sys.exit(f"{cp.relative_to(ROOT)} と {fz.relative_to(ROOT)} が無いか sha256 が違う")
        if not dai1.committed_clean(cp):
            sys.exit(f"{cp.relative_to(ROOT)} がコミットされていない")
        out[n] = dai1.sha(cp)
    if json.load(open(pairs["dai2"][0]))["_meta"].get("dai1_frozen_sha256") != out["dai1"]:
        sys.exit("実験 2 の固定のファイルが使った実験 1 の固定のファイルが今のものと違う")
    gp = ROOT / "frozen" / "eval_gate.json"
    if not gp.exists() or not dai1.committed_clean(gp):
        sys.exit("frozen/eval_gate.json が無いかコミットされていない")
    me = dai1.sha(Path(__file__))
    if json.load(open(gp))["required"].get("exp/dai2_stats.py") != me:
        sys.exit("eval_gate.json の required にこの台本の今の sha256 が無い（集計の台本は評価の回の前に固定する．指摘 2-I）")
    if not dai1.committed_clean(Path(__file__)):
        sys.exit("この台本がコミットされていない")
    out["eval_gate"] = dai1.sha(gp)
    RUN["gate"] = out["eval_gate"]
    return out


def selftest() -> None:
    """作った数で確かめる：(1) 同じ判定の 2 行は θ = 0・se = 0・p = 1・identical，32 通りのどれでも「上回る」「下回る」にならない．
    (2) 全部の分割を反転すると θ の符号が反転し，区間も 0 について反転する．(3) WS の自由度は塊の項と種の項の自由度の最小と和の間。"""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    cl = np.array([f"g{i % 40}" for i in range(400)])
    fold = np.array([int(x[1:]) % 5 for x in cl])
    A, B = {}, {}
    for f in range(5):
        m = fold == f
        for r in range(3):
            pa = np.where(rng.random(m.sum()) < 0.6, y[m], rng.integers(0, 4, m.sum()))
            pb = np.where(rng.random(m.sum()) < 0.5, y[m], rng.integers(0, 4, m.sum()))
            A[(f, r)] = ST.counts(y[m], pa, cl[m])
            B[(f, r)] = ST.counts(y[m], pb, cl[m])
    for g_eff in (8.0, 30.0):
        _, e, w, ua, ub, ab, be = cell_task(("x", A, A, g_eff, 1, True))
        assert e["effect"] == 0 and e["se"] == 0 and e["p"] == 1.0 and e["identical"] and not any(ab) and not any(be), e
        _, e1, w1, *_ = cell_task(("x", A, B, g_eff, 1, True))
        flipA = {k: B[k] for k in A}; flipB = {k: A[k] for k in A}
        _, e2, *_ = cell_task(("x", flipA, flipB, g_eff, 1, False))
        assert np.isclose(e1["effect"], -e2["effect"]) and np.isclose(e1["lo"], -e2["hi"]) and np.isclose(e1["hi"], -e2["lo"]), (e1, e2)
        df_cl = e1["df"] if finite(e1["df"]) else math.inf
        assert w1["v_seed_check"] < 1e-9
        assert (min(df_cl, 2) - 1e-9 <= w1["df_ws"] <= df_cl + 10 + 1e-9) or not finite(w1["df_ws"]), (w1, df_cl)
        print(f"selftest g_eff={g_eff}：同じ判定 → θ 0・p 1・identical，反転 → 符号が反転，WS の自由度 {w1['df_ws']:.2f}（塊 {df_cl}，種 10）")
    print("selftest：通った")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("eval", "pseudo", "selftest"))
    ap.add_argument("out", nargs="?", default=None, help="出力の置き場（既定：eval は results/dai2/stats，pseudo は results/dai2/stats_pseudo）")
    ap.add_argument("--dai1-dir", default=str(PROD["dai1"]))
    ap.add_argument("--dai2-dir", default=str(PROD["dai2"]))
    ap.add_argument("--dai1-frozen", default=str(ROOT / "frozen" / "dai1_frozen.json"))
    ap.add_argument("--dai2-frozen", default=None, help="既定は <dai2-dir>/select/frozen.json（pseudo だけ替えられる）")
    ap.add_argument("--targets", nargs="*", default=list(DD.CORPORA))
    ap.add_argument("--pseudo-c", nargs="*", default=None, help="試走だけ：仮のターゲットを絞る")
    ap.add_argument("--allow-partial", action="store_true", help="試走だけ：そろった単位だけで集計する")
    ap.add_argument("--no-null", action="store_true", help="試走だけ：符号の反転を省く")
    ap.add_argument("--jobs", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args()
    if args.mode == "selftest":
        selftest(); return
    RUN["mode"] = args.mode
    RUN["commit"], RUN["dirty"] = dai1.git_commit()
    guard = {}
    if args.mode == "eval":
        if args.no_null or args.targets != list(DD.CORPORA):
            sys.exit("eval は 4 つの T の全部の表を作る（--no-null と --targets は受け付けない）")
        guard = eval_guard(args)
    out = Path(args.out) if args.out else Path(args.dai2_dir) / ("stats" if args.mode == "eval" else "stats_pseudo")
    if args.mode == "eval" and out.exists() and any(out.iterdir()):     # 流し直した事実を残す（gate-code の 7 回目の記録 3．手順 8）
        sys.exit(f"{out} に中身がある．消してから流し直し，台帳に書く")
    if args.mode == "pseudo" and out.resolve() == (PROD["dai2"] / "stats").resolve():
        sys.exit("試走の出力を評価の回の置き場に書かない")
    t0 = time.time()
    jobs, info = make_jobs(args)
    if args.mode == "pseudo" and not info["dai1_frozen_matches_dai2_meta"]:
        print("注意：実験 2 の固定のファイルが使った実験 1 の固定のファイルと，今の実験 1 の固定のファイルの sha256 が違う（試走として進める）", flush=True)
    res = []
    for j in jobs:
        r = aggregate(j, args.jobs, not args.no_null)
        bad = {k: int(r["CHK"][k].iloc[0]) for k in CRITICAL if int(r["CHK"][k].iloc[0])}
        if bad and args.mode == "eval":
            sys.exit(f"{j['T']}: 単位の取り合わせか固定のファイルか格子が食い違う {bad}（表を書かずに止める）")
        if bad:
            print(f"注意：{j['T']} {j['c']} の食い違い {bad}（試走として進める）", flush=True)
        res.append(r)
    out.mkdir(parents=True, exist_ok=True)
    names = {"T1": "prop_minus_method", "T2": "prop_one_minus_own", "X9": "xmuda9", "CS": "corpus_sentence", "F3": "fam3_cells", "T4": "ctl_decomp",
             "K0": "k0_ctl_none", "T5": "da_term", "DQ": "da_quantities", "T6": "rep_shift", "T6r": "prop_minus_method_rep",
             "MK": "marks", "MP": "marks_points", "RG": "regret", "SM": "source_mix", "NU": "null_counts", "CT": "coral_tether_curve",
             "CTS": "coral_tether_sign", "CHK": "checks"}
    counts = {}
    for k, nm in names.items():
        df = pd.concat([r[k] for r in res if len(r[k])], ignore_index=True) if any(len(r[k]) for r in res) else pd.DataFrame()
        stamp(df).to_csv(out / f"{nm}.csv", index=False)
        counts[nm] = len(df)
    AG = pd.concat([r["AG"] for r in res if len(r["AG"])], ignore_index=True) if any(len(r["AG"]) for r in res) else pd.DataFrame()
    stamp(agreement_table(AG)).to_csv(out / "rep_agreement.csv", index=False)
    if args.mode == "pseudo":
        fams = pd.concat([j["families"] for j in jobs], ignore_index=True) if jobs else pd.DataFrame()
        stamp(fams).to_csv(out / "families_pseudo.csv", index=False)
    chk = pd.concat([r["CHK"] for r in res], ignore_index=True) if res else pd.DataFrame()
    meta = dict(mode=args.mode, argv=sys.argv, time=time.strftime("%Y-%m-%d %H:%M:%S"), commit=RUN["commit"], dirty=RUN["dirty"],
                script_sha256=dai1.sha(Path(__file__)), dai1_stats_sha256=dai1.sha(HERE / "dai1_stats.py"), guard=guard, **info,
                jobs=[dict(T=j["T"], c=j["c"], n_lin_units=len(j["lin_units"]), n_grad_units=len(j["grad_units"])) for j in jobs],
                rows=counts, checks_total={c: int(chk[c].sum()) for c in chk.columns if c not in ("T", "c", "corpus", "row_commits", "d1_row_commits")}
                if len(chk) else {}, row_commits=sorted({x for v in chk.get("row_commits", []) for x in str(v).split("|") if x}),
                d1_row_commits=sorted({x for v in chk.get("d1_row_commits", []) for x in str(v).split("|") if x}),
                sec=time.time() - t0)
    if args.mode == "eval":                                              # CORAL の曲線の (生徒, K) は T ごとに 16 個のはず（gate-code の 7 回目の記録 2）
        meta["coral_tether_cells_expected"] = 16 * len(DD.CORPORA)
        print("CORAL の曲線の (生徒, K)：", meta["checks_total"].get("coral_tether_cells"), "/", meta["coral_tether_cells_expected"], flush=True)
    json.dump(meta, open(out / "meta.json", "w"), indent=1, ensure_ascii=False, default=str)
    print("書いた:", out, {k: v for k, v in counts.items()}, flush=True)
    print("照合の合計:", meta["checks_total"], flush=True)


if __name__ == "__main__":
    main()
