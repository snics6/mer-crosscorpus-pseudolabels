#!/usr/bin/env bash
# 実験 16 の下調べ（docs/設計/実験16_下調べ_設計.md）の K = 5・25 を流す．units → summary → 計算の後の量（posthoc）．
# 2026-11-13 の本計算は台本の冒頭の使い方のコマンドを手で流した（ログ logs/dai16_probe_units.log・dai16_probe_summary.log）．
# この起動の台本は 2026-11-15 の整理のときに，その手順を 1 本にまとめて書いた．K = 10 の追補は run/dai_k10.sh．
# 終わると logs/dai16_probe.done（失敗は logs/dai16_probe.failed）．検査を通した台本の sha256 を SHA16 で渡す．
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
trap 'touch logs/dai16_probe.failed' ERR
rm -f logs/dai16_probe.done logs/dai16_probe.failed
[ "$(sha256sum exp/dai16_probe.py | cut -d' ' -f1)" = "${SHA16:?}" ] || { echo "dai16_probe.py の sha256 が違う"; touch logs/dai16_probe.failed; exit 1; }
PY=.venv/bin/python
for STAGE in units summary; do
  echo "=== dai16 $STAGE $(date '+%F %T')"
  $PY exp/dai16_probe.py --stage "$STAGE" 2>&1 | tee -a "logs/dai16_probe_${STAGE}.log"
done
CUDA_VISIBLE_DEVICES="" $PY exp/dai16_posthoc.py 2>&1 | tee -a logs/dai16_probe_posthoc.log
date '+%F %T' > logs/dai16_probe.done
echo "EXIT=0"
