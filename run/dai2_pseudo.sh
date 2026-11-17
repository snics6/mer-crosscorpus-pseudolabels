#!/bin/bash
# DAI 実験 2 の仮のターゲットの回を，1 つの評価のターゲット T について分割ごとに並べて回す（設計 4・5 節）．
# 使い方：run/dai2_pseudo.sh <T> [実験 2 の置き場] [実験 1 の置き場]．CUT に削り方の引数（例 "--cut-mme-sw01"）を入れられる．
# 実験 1 の置き場に source/，pseudo/<T>_tether_w.json，pseudo/units/<T>_*_p1.npz があること．
# 線形の手法（dai2.py）と勾配法の手法・CORAL（dai2_grad.py）を分割ごとに並べる．どれかが失敗したら logs/dai2_<T>.failed を書く．
set -u
cd "$(dirname "$0")/.."
T=$1
OUT=${2:-results/dai2}
D1=${3:-results/dai1}
CUT=${CUT:-}
PY=${PY:-python}
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MERDA_GPU_FRACTION=${MERDA_GPU_FRACTION:-0.12}
pids=()
for f in 0 1 2 3 4; do
  $PY exp/dai2.py --stage pseudo --targets "$T" --folds $f --out-dir "$OUT" --dai1-dir "$D1" > "logs/dai2_${T}_lin_f$f.log" 2>&1 &
  pids+=($!)
  $PY exp/dai2_grad.py --stage pseudo --targets "$T" --folds $f --out-dir "$OUT" --dai1-dir "$D1" --parts map xmuda coral $CUT > "logs/dai2_${T}_grad_f$f.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do
  if ! wait "$p"; then echo "処理 $p が失敗" > "logs/dai2_${T}.failed"; exit 1; fi
done
date > "logs/dai2_${T}.done"
