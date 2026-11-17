#!/usr/bin/env bash
# DAI-23 の本計算（設計 docs/設計/実験23_設計.md 第 4 版 8 節）．固定のコミットの別の作業木の DAI/ で流す．
#   (1) 流れの照合（--check-identity．書き出さない）
#   (2) 操作の確かめ → results/dai23/fc.json（fc を決める．一度だけ）
#   (3) 低域通過の特徴（fc は fc.json から）と逆再生の特徴 → data/cache/audio_wavlm-{lp<fc>,rev}_<コーパス>.npz
#   (4) 単位の段 → (5) 集計の段
# 段ごとに logs/dai23_<段>.done を作り，済んだ段は飛ばす．失敗は logs/dai23.failed．終わると logs/dai23.done．
# 検査を通した台本の sha256 を DAI23_SHA（exp/dai23.py），DAI23_FEAT_SHA（features/build_audio_lowpass.py），
# DAI23_ASR_SHA（features/check_lowpass_asr.py）で必ず渡す．
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
trap 'touch logs/dai23.failed' ERR
for pair in "exp/dai23.py:${DAI23_SHA:-}" "features/build_audio_lowpass.py:${DAI23_FEAT_SHA:-}" "features/check_lowpass_asr.py:${DAI23_ASR_SHA:-}"; do
  f=${pair%%:*}; want=${pair#*:}
  if [ -z "$want" ]; then echo "$f の検査を通した sha256 が無い"; touch logs/dai23.failed; exit 1; fi
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then echo "$f の sha256 が検査を通した版と違う：$got"; touch logs/dai23.failed; exit 1; fi
done
if [ -n "$(git status --porcelain exp splits measure features data/metadata)" ]; then
  echo "exp・splits・measure・features・data/metadata に未コミットの変更がある"; touch logs/dai23.failed; exit 1
fi
export CUDA_VISIBLE_DEVICES=0
export MERDA_GPU_FRACTION=0.6
PY=.venv/bin/python
echo "開始 $(date '+%F %T')　コミット $(git rev-parse HEAD)"

if [ ! -e logs/dai23_identity.done ]; then
  $PY features/build_audio_lowpass.py --op lp --fc 500 --limit 64 --check-identity > logs/dai23_identity.log 2>&1
  touch logs/dai23_identity.done; echo "流れの照合を終えた $(date '+%F %T')"
fi
if [ ! -e logs/dai23_asr.done ]; then
  $PY features/check_lowpass_asr.py > logs/dai23_asr.log 2>&1
  touch logs/dai23_asr.done; echo "操作の確かめを終えた $(date '+%F %T')　$(cat results/dai23/fc.json | tr -d '\n' | cut -c1-80)"
fi
for op in lp rev; do
  if [ ! -e "logs/dai23_feat_${op}.done" ]; then
    $PY features/build_audio_lowpass.py --op "$op" > "logs/dai23_feat_${op}.log" 2>&1
    $PY features/build_audio_lowpass.py --op "$op" --merge >> "logs/dai23_feat_${op}.log" 2>&1
    touch "logs/dai23_feat_${op}.done"; echo "特徴（${op}）を終えた $(date '+%F %T')"
  fi
done
if [ ! -e logs/dai23_units.done ]; then
  $PY exp/dai23.py --stage units > logs/dai23_units.log 2>&1
  touch logs/dai23_units.done; echo "単位の段を終えた $(date '+%F %T')"
fi
if [ ! -e logs/dai23_summary.done ]; then
  $PY exp/dai23.py --stage summary > logs/dai23_summary.log 2>&1
  touch logs/dai23_summary.done; echo "集計の段を終えた $(date '+%F %T')"
fi
touch logs/dai23.done
echo "終わり $(date '+%F %T')"
