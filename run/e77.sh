#!/bin/bash
# E77（DAI 論文の比較実験の第 1 波）．2 本の流れを GPU に並べて回す（各 GPU メモリの 40%）．
# 設計：docs/exploratory-analyses.md E77．終わると logs/e77.done を書く．
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
PY=./.venv/bin/python
stream1() {
  $PY exp/compare_dai.py --world fuse-audio-video-text --gpu-fraction 0.4 >> logs/e77_avt.log 2>&1 || echo "失敗 AVT $?" >> logs/e77_fail.log
  $PY exp/compare_dai.py --kind text_roberta --gpu-fraction 0.4 >> logs/e77_text.log 2>&1 || echo "失敗 text $?" >> logs/e77_fail.log
  $PY exp/compare_dai.py --kind video --gpu-fraction 0.4 >> logs/e77_video.log 2>&1 || echo "失敗 video $?" >> logs/e77_fail.log
}
stream2() {
  $PY exp/compare_dai.py --world fuse-audio-video --gpu-fraction 0.4 >> logs/e77_av.log 2>&1 || echo "失敗 AV $?" >> logs/e77_fail.log
  $PY exp/compare_dai.py --kind audio_wavlm --gpu-fraction 0.4 >> logs/e77_audio.log 2>&1 || echo "失敗 audio $?" >> logs/e77_fail.log
}
date "+開始 %F %T" > logs/e77.started
stream1 &
stream2 &
wait
date "+終了 %F %T" > logs/e77.done
