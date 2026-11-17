#!/bin/bash
# E78（DAI 論文の比較実験の第 2 波）．logs/e77.done ができてから，2 本の流れを GPU に並べて回す（各 GPU メモリの 40%）．
# 設計：docs/exploratory-analyses.md E78．終わると logs/e78.done を書く．
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
PY=./.venv/bin/python
until [ -f logs/e77.done ]; do sleep 120; done
stream1() {
  $PY exp/compare_dai2.py --world fuse-audio-video-text --gpu-fraction 0.4 >> logs/e78_avt.log 2>&1 || echo "失敗 AVT $?" >> logs/e78_fail.log
  $PY exp/compare_dai2.py --kind text_roberta --gpu-fraction 0.4 >> logs/e78_text.log 2>&1 || echo "失敗 text $?" >> logs/e78_fail.log
  $PY exp/compare_dai2.py --kind video --gpu-fraction 0.4 >> logs/e78_video.log 2>&1 || echo "失敗 video $?" >> logs/e78_fail.log
}
stream2() {
  $PY exp/compare_dai2.py --world fuse-audio-video --gpu-fraction 0.4 >> logs/e78_av.log 2>&1 || echo "失敗 AV $?" >> logs/e78_fail.log
  $PY exp/compare_dai2.py --kind audio_wavlm --gpu-fraction 0.4 >> logs/e78_audio.log 2>&1 || echo "失敗 audio $?" >> logs/e78_fail.log
}
date "+開始 %F %T" > logs/e78.started
stream1 &
stream2 &
wait
date "+終了 %F %T" > logs/e78.done
