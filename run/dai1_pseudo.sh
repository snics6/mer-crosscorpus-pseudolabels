#!/bin/bash
# DAI 実験 1 の仮のターゲットの回を，1 つの評価のターゲット T について分割ごとに並べて回す（設計 10 節）．
# 使い方：run/dai1_pseudo.sh <T> [置き場]．段 source の出力が <置き場>/source にあること．
# 段 1（分割ごとに 5 本並べる）→ 段 2（全単位．引き戻しと指数を選んで tether_w.json）→ 段 3（分割ごとに 5 本）．
# どれかの処理が失敗したら止まり，logs/dai1_<T>.failed を書く．
set -u
cd "$(dirname "$0")/.."
T=$1
OUT=${2:-results/dai1}
PY=${PY:-python}
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MERDA_GPU_FRACTION=${MERDA_GPU_FRACTION:-0.15}
run_folds() {   # $1 = 段
  local pids=() f
  for f in 0 1 2 3 4; do
    $PY exp/dai1.py --stage pseudo --targets "$T" --folds $f --pass "$1" --out-dir "$OUT" > "logs/dai1_${T}_p$1_f$f.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    if ! wait "$p"; then echo "段 $1 の処理 $p が失敗" > "logs/dai1_${T}.failed"; exit 1; fi
  done
}
run_folds 1
$PY exp/dai1.py --stage pseudo --targets "$T" --pass 2 --out-dir "$OUT" > "logs/dai1_${T}_p2.log" 2>&1 || { echo "段 2 が失敗" > "logs/dai1_${T}.failed"; exit 1; }
run_folds 3
date > "logs/dai1_${T}.done"
