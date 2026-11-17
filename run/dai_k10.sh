#!/usr/bin/env bash
# DAI-16・17 の下調べの K = 10 の追補（docs/設計/実験16・17_下調べ_K10_追補.md）．DAI-16 → DAI-17 → 計算の後の量 を続けて流す．
# 終わると logs/dai_k10.done（失敗は logs/dai_k10.failed）．検査を通した台本の sha256 を SHA16・SHA17 で渡す．
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
trap 'touch logs/dai_k10.failed' ERR
rm -f logs/dai_k10.done logs/dai_k10.failed
[ "$(sha256sum exp/dai16_probe.py | cut -d' ' -f1)" = "${SHA16:?}" ] || { echo "dai16_probe.py の sha256 が違う"; touch logs/dai_k10.failed; exit 1; }
[ "$(sha256sum exp/dai17_probe.py | cut -d' ' -f1)" = "${SHA17:?}" ] || { echo "dai17_probe.py の sha256 が違う"; touch logs/dai_k10.failed; exit 1; }
PY=.venv/bin/python
# 既存の K = 5・25 の置き場が書き換わらないことを，前後の sha256 で確かめる
snap() { find results/dai16_probe results/dai17_probe -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1; }
BEFORE=$(snap)
for STAGE in units summary; do
  echo "=== dai16 $STAGE $(date '+%F %T')"
  $PY exp/dai16_probe.py --stage "$STAGE" --ks 10 --out-dir results/dai16_probe_k10 2>&1 | tee -a "logs/dai16_k10_${STAGE}.log"
done
for STAGE in units reweight students summary; do
  echo "=== dai17 $STAGE $(date '+%F %T')"
  $PY exp/dai17_probe.py --stage "$STAGE" --ks 10 --out-dir results/dai17_probe_k10 2>&1 | tee -a "logs/dai17_k10_${STAGE}.log"
done
CUDA_VISIBLE_DEVICES="" $PY exp/dai16_posthoc.py --out-dir results/dai16_probe_k10 2>&1 | tee -a logs/dai16_k10_posthoc.log
CUDA_VISIBLE_DEVICES="" $PY exp/dai17_posthoc.py --out-dir results/dai17_probe_k10 2>&1 | tee -a logs/dai17_k10_posthoc.log
AFTER=$(snap)
[ "$BEFORE" = "$AFTER" ] || { echo "既存の置き場（results/dai16_probe・dai17_probe）が書き換わった"; touch logs/dai_k10.failed; exit 1; }
date '+%F %T' > logs/dai_k10.done
echo "EXIT=0"
