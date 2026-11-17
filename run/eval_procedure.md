# 評価の回の手順（2026-11-04．gate-code の評価の回の起動の台本の検査 code.md・code_v2.md の I-commit・I-gather・I-frozen2・v2-5 を手順にしたもの）

1. 実験 2 の仮のターゲットの回の出力を 4 台の Pod から研究室の PC に集める．
   - `results/dai2/pseudo/units/` はそのまま合わせる（名前に T が入るので重ならない）．
   - `results/dai2/st_cv/` は，4 台で同じ名前のファイルの sha256 が一致することを確かめてから合わせる．
   - `results/dai2/provenance_log.jsonl` は `results/dai2/provenance_log_<Pod 名>.jsonl` の名前で置く．
   - gate-code に出力の欠けと重複を点検させる．
2. 実験 2 の選ぶ段（`python exp/dai2.py --stage select`）を流し，`_meta.grad_cut` が設計 5 節の決定 {"mme_sw01": true, "iter200": false} であることを照らしてから，`frozen/dai2_frozen.json` をコミットする（比較の 5 回目 2-K）．実験 2 の集計の台本 `exp/dai2_stats.py` と実験 1 の記述の数の集計の台本 `exp/dai1_desc.py` を gate-code に通してコミットし，`eval_gate.json` の required に足す（比較の 5 回目 2-I，定義の 5 回目）．
3. 査読者 2 人（`dai-review-ml`・`dai-review-affect`）に，評価の回の前の最終の確認を頼む（設計の変更：削り方の決定，範囲の外の文，報告の書き方の足し）．査読者に `dai2_frozen.json` の値を渡さない．ここで設計を変えたら台帳に 1 行足す（漏れの 5 回目の 7）．
4. `frozen/eval_gate.json` を作ってコミットする（`dai1.EVAL_REQUIRED` と 3 つの固定のファイル，7 つの役の最新の報告の sha256）．
   その後，評価の回のコミットで `./.venv/bin/python checks/DAI-eval/code_run/m3_frozen_keys.py dai2` を流し，食い違いが 0 であることを確かめる．
5. 8 台の Pod を評価の回のコミットにそろえ，次を送る．
   - 全部：特徴の npz 12 本，`results/dai1/{source,select,provenance_log.jsonl}`．
   - 実験 2 の Pod：`results/dai2/{select,st_cv}`．
   - 実験 1・3 の Pod：`results/dai3/{frozen.json,validate,detect}`．
6. 実験 1・3 の Pod（4 台）：`run/dai_eval.sh 1 <T>`，終わったら `run/dai_eval.sh 3 <T>`．実験 2 の Pod（4 台）：`run/dai_eval.sh 2 <T>`．
   - 起動した後に `logs/dai_eval_<実験>_<T>.inputs` を見て，commit が 8 台で同じで汚れの数が 0 であることを照らす．記録に GPU の行（H200）と commit の行があることも確かめる（v3-1）．8 台すべてに `results/dai1/provenance_log.jsonl` を，実験 2 の Pod には空でない `results/dai2/st_cv/` を送る（v3-2）．
   - `.failed` ができたら，流し直す前に，止めた処理が最後に書き終えた単位より後に更新されたファイルを消す（v2-5）．止まるまでに書き終えた評価の出力は人が開かない．流し直した回数と理由を台帳に書く（漏れの 5 回目の 3）．
7. 研究室の PC に集める．
   - `results/dai1/eval/units/`，`results/dai2/eval/units/`，`results/dai3/eval/units/*.pkl` を合わせる．
   - 各 Pod の `results/dai{1,2,3}/provenance_log.jsonl` は `provenance_log_<Pod 名>.jsonl` の名前で置く（先頭の 46 行は研究室の PC から送った行）．
   - 実験 3 の T ごとの表（`results/dai3/eval/*.csv`，`verdicts.json`）は `logs/dai3_eval_tables_<T>/` に写す．
8. 評価の回のコミットの別の作業木で集計する（I-commit）．集計の台本（dai1_desc.py，dai2_stats.py）は，今のコミットが評価の回の単位のコミットと違うと止まる（DAI-1 code_v5 [I-guard]，DAI-2 code_v3 [I-guard]）．
   - `git worktree add --detach <置き場> <評価の回のコミット>`．`.venv` はリンク．`results/` はリポジトリが中の csv を追跡しているので作業木に実在する．`results/dai1`・`results/dai2`・`results/dai3` を個別に本来のものへのリンクにする．`data/cache/` は本来の `*.npz` を 1 本ずつリンクする（`faces_*.csv.gz` は git にあるので，ディレクトリごとリンクにしない）．
   - 流す前に pkl が 200 個そろい，pkl の `_run.commit` が 1 つであることを数える．
   - `python exp/dai3.py --stage eval`（4 つの T）を流し，出力をファイルに残す．そのファイルに計算した印の行（`eval <T>_f…: … 秒`）が 0 であることを数える．
   - `python exp/dai1_stats.py evaluate`，`OMP_NUM_THREADS=1 python exp/dai1_desc.py eval --jobs 4`，`OMP_NUM_THREADS=1 python exp/dai1_desc.py pseudo --jobs 4`（設計 14 節の φ の正本．pseudo の段は出力先を上書きする），`python exp/dai2_stats.py eval` も同じ作業木で流す（dai2_stats は families.csv を読むので dai1_stats の後）．表の書き出しの途中で止まったら，出力先を消してから流し直し，台帳に書く．
9. gate-code に結果ファイルの欠けと重複を点検させ（`checks/DAI-1/code_run_post/c9_predcheck.py` を評価の回の出力にも流させる．統計の 5 回目の記録），結果の文を `gate-claim` に照合させてから文書に書く．
