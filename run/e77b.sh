#!/bin/bash
# E77（DAI 論文の比較実験の第 1 波）を 1 本ずつ順に回す（2026-11-01 08:15 に run/e77.sh の 2 本並列を止めて替えた．
# GPU のメモリの合計が上限の 24 GB に近づき，WSL2 で 5〜10 倍遅くなっていたため．各 GPU メモリの 50%）．終わると logs/e77.done を書く．
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
PY=./.venv/bin/python
date "+開始 %F %T" > logs/e77b.started
for arg in "--world fuse-audio-video-text" "--world fuse-audio-video" "--kind text_roberta" "--kind video" "--kind audio_wavlm"; do
  name=$(echo $arg | awk '{print $2}')
  $PY exp/compare_dai.py $arg --gpu-fraction 0.5 >> logs/e77b_${name}.log 2>&1 || echo "失敗 $name $?" >> logs/e77_fail.log
done
date "+終了 %F %T" > logs/e77.done
