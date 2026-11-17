#!/usr/bin/env bash
# 実験 18 の下調べ（docs/設計/実験18_下調べ_設計.md）．6 コーパスで source → base → tether → units → students → summary を続けて流す．
# 終わると logs/dai18_probe.done（失敗は logs/dai18_probe.failed）．検査を通した台本の sha256 を SHA18 で渡す．
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
trap 'touch logs/dai18_probe.failed' ERR
rm -f logs/dai18_probe.done logs/dai18_probe.failed
[ -n "${SHA18:-}" ] || { echo "SHA18 が渡されていない"; touch logs/dai18_probe.failed; exit 1; }
[ "$(sha256sum exp/dai18_probe.py | cut -d' ' -f1)" = "${SHA18:?}" ] || { echo "dai18_probe.py の sha256 が違う"; touch logs/dai18_probe.failed; exit 1; }
PY=.venv/bin/python
# 実験 1 と 4 コーパスの実験 16・17 の置き場が書き換わらないことを，前後の sha256 で確かめる
snap() { find results/dai1/source results/dai1/pseudo results/dai16_probe results/dai17_probe results/dai16_probe_k10 results/dai17_probe_k10 frozen -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1; }
BEFORE=$(snap)
$PY exp/dai18_probe.py --stage selftest 2>&1 | tee -a logs/dai18_selftest.log
for STAGE in source base tether units students summary; do
  echo "=== dai18 $STAGE $(date '+%F %T')"
  $PY exp/dai18_probe.py --stage "$STAGE" 2>&1 | tee -a "logs/dai18_${STAGE}.log"
done
AFTER=$(snap)
[ "$BEFORE" = "$AFTER" ] || { echo "実験 1 か 4 コーパスの置き場が書き換わった"; touch logs/dai18_probe.failed; exit 1; }
date '+%F %T' > logs/dai18_probe.done
echo "EXIT=0"
