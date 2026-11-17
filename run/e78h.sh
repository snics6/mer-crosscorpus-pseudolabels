#!/bin/bash
# E78 を RunPod の H100 で回す（Co-teaching なし，DAI 論文の短縮 a・c．2026-11-01 夜）．
# 引数は流れの番号 1〜3．AVT・AV の世界を先に，種類を後に回す．終わると logs/e78h_<番号>.done を書く．
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
PY=./.venv/bin/python
F="--ks 0 5 25 --copy-pseudo --methods dann mme mapctl xmuda coral --gpu-fraction 0.22"
run() { name=$1; shift; $PY exp/compare_dai2.py "$@" $F >> logs/e78h_${name}.log 2>&1 || echo "失敗 $name $* $?" >> logs/e78h_fail.log; }
case $1 in
  1) run avt --world fuse-audio-video-text; run text --kind text_roberta ;;
  2) run av1 --world fuse-audio-video --target IEMOCAP-impro IEMOCAP-script MOSEI; run video --kind video ;;
  3) run av2 --world fuse-audio-video --target MELD AFEW CREMA-D; run audio --kind audio_wavlm ;;
esac
date "+終了 %F %T" > logs/e78h_$1.done
