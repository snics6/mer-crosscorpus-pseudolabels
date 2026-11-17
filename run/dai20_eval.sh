#!/usr/bin/env bash
# 実験 20（docs/設計/実験20_設計.md）の評価の回：units → summary を続けて流す．
# 終わると logs/dai20_eval.done（失敗は logs/dai20_eval.failed）．検査を通した台本の sha256 を SHA20 で渡す．
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
trap 'touch logs/dai20_eval.failed' ERR
rm -f logs/dai20_eval.done logs/dai20_eval.failed
[ -n "${SHA20:-}" ] || { echo "SHA20 が渡されていない"; touch logs/dai20_eval.failed; exit 1; }
[ "$(sha256sum exp/dai20_eval.py | cut -d' ' -f1)" = "${SHA20:?}" ] || { echo "dai20_eval.py の sha256 が違う"; touch logs/dai20_eval.failed; exit 1; }
PY=.venv/bin/python
# 実験 1・2 の集計，実験 1 の評価の回の保存（名前と大きさ），下調べの集計，固定のファイルが書き換わらないことを前後で確かめる
snap() { { find results/dai1/stats results/dai1/source results/dai2/stats results/dai16_probe/summary results/dai16_probe_k10/summary frozen -type f -print0 | sort -z | xargs -0 sha256sum;
           find results/dai1/eval/units -type f -printf '%p %s %T@\n' | sort; } | sha256sum | cut -d' ' -f1; }
BEFORE=$(snap)
$PY exp/dai20_eval.py --stage selftest 2>&1 | tee -a logs/dai20_selftest.log
for STAGE in units summary; do
  echo "=== dai20 $STAGE $(date '+%F %T')"
  $PY exp/dai20_eval.py --stage "$STAGE" 2>&1 | tee -a "logs/dai20_${STAGE}.log"
done
# 実験 1 の評価の回の単位の保存の sha256 を残す（由来の記録）
find results/dai1/eval/units -name '*.npz' -print0 | sort -z | xargs -0 sha256sum > results/dai20_eval/dai1_eval_units_sha256.txt
AFTER=$(snap)
[ "$BEFORE" = "$AFTER" ] || { echo "実験 1・2 か下調べの置き場か固定のファイルが書き換わった"; touch logs/dai20_eval.failed; exit 1; }
date '+%F %T' > logs/dai20_eval.done
echo "EXIT=0"
