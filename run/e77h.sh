#!/bin/bash
# E77 の残りを RunPod の H100 で回す（DAI 論文の短縮 a・b・c と entmin・IM の種 1・2 つき．2026-11-01 夜）．
# 引数は流れの番号 1〜3．比べの中心の AVT・AV の世界を先に，筋 4 の判定に使う種類を後に回す．終わると logs/e77h_<番号>.done を書く．
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
PY=./.venv/bin/python
F="--ks 0 5 25 --copy-pseudo --coreg-eval-only --nc-seeds 1 2 --gpu-fraction 0.22"
run() { name=$1; shift; $PY exp/compare_dai.py "$@" $F >> logs/e77h_${name}.log 2>&1 || echo "失敗 $name $* $?" >> logs/e77h_fail.log; }
case $1 in
  1) run avt --world fuse-audio-video-text; run text --kind text_roberta ;;
  2) run av1 --world fuse-audio-video --target IEMOCAP-impro IEMOCAP-script MOSEI; run video --kind video ;;
  3) run av2 --world fuse-audio-video --target MELD AFEW CREMA-D; run audio --kind audio_wavlm ;;
esac
date "+終了 %F %T" > logs/e77h_$1.done
