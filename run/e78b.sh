#!/bin/bash
# E78（DAI 論文の比較実験の第 2 波）を，logs/e77.done ができてから 1 本ずつ順に回す（各 GPU メモリの 50%）．
# run/e78.sh の 2 本並列は，E77 で GPU のメモリの合計が上限に近づき遅くなったので使わない（2026-11-01）．終わると logs/e78.done を書く．
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
PY=./.venv/bin/python
until [ -f logs/e77.done ]; do sleep 300; done
date "+開始 %F %T" > logs/e78b.started
for arg in "--world fuse-audio-video-text" "--world fuse-audio-video" "--kind text_roberta" "--kind video" "--kind audio_wavlm"; do
  name=$(echo $arg | awk '{print $2}')
  $PY exp/compare_dai2.py $arg --gpu-fraction 0.5 >> logs/e78b_${name}.log 2>&1 || echo "失敗 $name $?" >> logs/e78_fail.log
done
date "+終了 %F %T" > logs/e78.done
