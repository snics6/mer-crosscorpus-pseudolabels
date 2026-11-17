#!/bin/bash
# DAI 実験 1〜3 の評価のターゲットの回を，1 つの実験と 1 つの評価のターゲット T について回す
# （実験 1 の設計 10 節，実験 2 の設計 4 節，実験 3 の設計 9 節）．
# 使い方：run/dai_eval.sh <1|2|3> <T>．置き場は本番の results/dai1・results/dai2・results/dai3 に固定する．
#   1：実験 1 の eval を分割ごとに 5 本並べる（GPU のメモリの上限は 1 本あたり既定 0.15．仮のターゲットの回と同じ）．
#   2：実験 2 の eval（線形の手法と勾配法の手法）を分割ごとに 10 本並べる（既定 0.12．仮のターゲットの回と同じ）．
#      results/dai1/{source,provenance_log.jsonl} と results/dai2/{select,st_cv} が要る（dai2.check_encoder が provenance を読む）．
#   3：実験 3 の eval を T だけで 1 本流す（既定 0.5）．同じ機械で実験 1 の eval（この台本の 1）が終わっていること．
#      単位ごとの保存 results/dai3/eval/units/<T>_f<f>_r<r>.pkl を作る．4 つの T の判定の規則は，全部の T の保存を
#      1 か所に集め，評価の回と同じコミットの作業木で --targets を 4 つにして流し直したときに当たる（保存を読むだけ）．
# GPU のメモリの上限は FRAC で替えられる（例 FRAC=0.2）．
# 前提：frozen/eval_gate.json・dai1_frozen.json・dai2_frozen.json・dai3_frozen.json がコミット済みで，作業木が汚れていないこと
# （台本の側の check_eval_gate などが確かめる）．
# 入力（特徴・発話一覧・分割の表・段 source の識別器・固定のファイル・provenance）の sha256，ライブラリの版，GPU を最初に記録する．
# どれかの処理が失敗したら残りの処理を止め，logs/dai_eval_<実験>_<T>.failed を書いて止まる．終わったら .done を書く．
# 起動のたびに前の .done・.failed を消す．ログは追記する（前の失敗の記録を残す）．
set -u
[ "${BASH_VERSINFO[0]}${BASH_VERSINFO[1]}" -ge 51 ] 2>/dev/null || { echo "bash 5.1 以上が要る（wait -n -p）．今は $BASH_VERSION"; exit 1; }
cd "$(dirname "$0")/.."
E=$1
T=$2
PY=${PY:-python}
mkdir -p logs
TAG="${E}_${T}"
rm -f "logs/dai_eval_${TAG}.done" "logs/dai_eval_${TAG}.failed"
pids=()
fail() {
  [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null
  wait 2>/dev/null
  echo "$(date) $1" > "logs/dai_eval_${TAG}.failed"
  exit 1
}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
{
  echo "== $(date) host $(hostname) commit $(git rev-parse HEAD) status $(git status --porcelain exp splits run measure features data/metadata | wc -l)"
  $PY -c "import numpy, pandas, sklearn, scipy, torch; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  sha256sum data/cache/audio_wavlm_*.npz data/cache/video_*.npz data/cache/text_roberta_*.npz data/metadata/*.csv splits/* frozen/*.json
  sha256sum results/dai1/source/*.npz results/dai1/select/frozen.json results/dai1/provenance_log*.jsonl
  case "$E" in
    2) sha256sum results/dai2/select/frozen.json results/dai2/st_cv/* ;;
    3) sha256sum results/dai3/frozen.json ;;
  esac
} >> "logs/dai_eval_${TAG}.inputs" 2>&1 || fail "入力の記録を書けない"
tail -n +"$(grep -n "^== " "logs/dai_eval_${TAG}.inputs" | tail -1 | cut -d: -f1)" "logs/dai_eval_${TAG}.inputs" | grep -qiE "No such file|cannot|error|Traceback" && fail "入力の記録に誤りの行がある（logs/dai_eval_${TAG}.inputs）"
case "$E" in
  1)
    export MERDA_GPU_FRACTION=${FRAC:-0.15}
    echo "MERDA_GPU_FRACTION $MERDA_GPU_FRACTION" >> "logs/dai_eval_${TAG}.inputs"
    for f in 0 1 2 3 4; do
      $PY exp/dai1.py --stage eval --targets "$T" --folds $f --out-dir results/dai1 >> "logs/dai1_eval_${T}_f$f.log" 2>&1 &
      pids+=($!)
    done ;;
  2)
    export MERDA_GPU_FRACTION=${FRAC:-0.12}
    echo "MERDA_GPU_FRACTION $MERDA_GPU_FRACTION" >> "logs/dai_eval_${TAG}.inputs"
    [ -f results/dai1/provenance_log.jsonl ] || fail "results/dai1/provenance_log.jsonl が無い（dai2.check_encoder が止まる）"
    for f in 0 1 2 3 4; do
      $PY exp/dai2.py --stage eval --targets "$T" --folds $f --out-dir results/dai2 --dai1-dir results/dai1 >> "logs/dai2_eval_${T}_lin_f$f.log" 2>&1 &
      pids+=($!)
      $PY exp/dai2_grad.py --stage eval --targets "$T" --folds $f --out-dir results/dai2 --dai1-dir results/dai1 --parts map xmuda coral >> "logs/dai2_eval_${T}_grad_f$f.log" 2>&1 &
      pids+=($!)
    done ;;
  3)
    export MERDA_GPU_FRACTION=${FRAC:-0.5}
    echo "MERDA_GPU_FRACTION $MERDA_GPU_FRACTION" >> "logs/dai_eval_${TAG}.inputs"
    [ -f "logs/dai_eval_1_${T}.done" ] || fail "この機械で実験 1 の eval（run/dai_eval.sh 1 $T）が終わっていない"
    n=$(ls results/dai1/eval/units/${T}_ALL_f*_r*_p3.npz 2>/dev/null | wc -l)
    m=$(ls results/dai1/eval/units/${T}_ALL_f*_r*_p3.csv 2>/dev/null | wc -l)
    [ "$n" -ge 50 ] && [ "$m" -ge 50 ] || fail "実験 1 の eval の段 3 の単位が npz $n 個，csv $m 個しかない"
    $PY exp/dai3.py --stage eval --targets "$T" >> "logs/dai3_eval_${T}.log" 2>&1 &
    pids+=($!) ;;
  *) fail "実験の番号は 1・2・3 のどれか" ;;
esac
left=${#pids[@]}
while [ "$left" -gt 0 ]; do
  wait -n -p fin; st=$?
  left=$((left - 1))
  [ "$st" -eq 0 ] || fail "処理 ${fin:-?} が失敗（終了コード $st）"
done
date > "logs/dai_eval_${TAG}.done"
