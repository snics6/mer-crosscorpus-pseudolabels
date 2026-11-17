#!/usr/bin/env bash
# data/ を生データ /mnt/d/Databases/（読むだけ）から作り直す台本．
# 作り方と出来上がりの説明は data/README.md（4 節が特徴の作り方）．
#
#   ./run/build_data.sh manifest        発話一覧だけ
#   ./run/build_data.sh feat1 feat2     特徴だけ
#   ./run/build_data.sh                 全部（manifest → check → feat1 → feat2）
#
# | 名前 | 中身 | 見込み |
# |---|---|---|
# | `manifest` | 発話一覧 6 本を作り直して監査する | 分 |
# | `check`    | 話者の診断と話者分割の検査 | 分 |
# | `feat1`    | 第 1 エンコーダ（ResNet-50・WavLM・RoBERTa） | 半日．GPU |
# | `feat2`    | 第 2 エンコーダ（CLIP・HuBERT・DeBERTa） | 半日．GPU |
#
# **既存の data/ を上書きする．**特に manifest はラベルの正本を書き換えるので，
# 走らせる前に git で data/metadata/ の状態を確かめ，終わったら差分を見る．
# **feat2 の映像（CLIP）は feat1 が保存した顔の座標を読む．**順序を入れ替えない．
# AFEW の書き起こし（Whisper）は git に入れた data/metadata/afew_transcript_whisper.csv を
# manifest が読む．この台本では作り直さない（作り直すなら features/build_transcript_afew.py）．
# 抽出は途中保存（data/cache/parts/）を見て飛ばすので，止まっても流し直せばよい．
set -u
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/bin/python
L=logs/build_data.log
mkdir -p logs

say() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
run() { local d="$1"; shift; say "▶ $d"; "$@" >> "$L" 2>&1
        local c=$?; say "■ $d 終了コード $c"; return 0; }

do_manifest() {
  say "==== 発話一覧を作る ===="
  for C in iemocap mosei meld afew; do
    run "マニフェスト $C" $PY -u features/build_manifest_$C.py
  done
  run "マニフェスト RAVDESS・CREMA-D" $PY -u features/build_manifest_extra.py
  run "マニフェストの監査" $PY -u measure/audit_manifest.py
}

do_check() {
  say "==== 話者の検査 ===="
  run "話者の診断"     $PY -u measure/check_speakers.py
  run "話者分割の検査" $PY -u measure/split.py
}

do_feat1() {
  say "==== 第 1 エンコーダの特徴 ===="
  run "映像 ResNet-50" $PY -u features/build_video.py --corpus all
  run "映像の結合"     $PY -u features/build_video.py --corpus all --merge
  run "音声 WavLM"     $PY -u features/build_audio.py --corpus all --budget 240 --max-batch 64
  run "音声の結合"     $PY -u features/build_audio.py --corpus all --merge
  for C in IEMOCAP MOSEI MELD AFEW; do
    run "言葉 RoBERTa $C" $PY -u features/build_text.py --corpus "$C"
  done
  run "言葉の結合" $PY -u features/build_text.py --corpus all --merge
}

do_feat2() {
  say "==== 第 2 エンコーダの特徴 ===="
  run "音声 HuBERT" $PY -u features/build_audio.py --corpus all --budget 240 --max-batch 64 \
      --model facebook/hubert-large-ll60k
  run "音声 HuBERT の結合" $PY -u features/build_audio.py --corpus all --merge \
      --model facebook/hubert-large-ll60k
  for C in IEMOCAP MOSEI MELD AFEW; do
    run "言葉 DeBERTa $C" $PY -u features/build_text.py --corpus "$C" \
        --model microsoft/deberta-v3-base
  done
  run "言葉 DeBERTa の結合" $PY -u features/build_text.py --corpus all --merge \
      --model microsoft/deberta-v3-base
  run "映像 CLIP"        $PY -u features/build_video_clip.py --corpus all
  run "映像 CLIP の結合" $PY -u features/build_video_clip.py --corpus all --merge
}

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(manifest check feat1 feat2)
say "########## 実行: ${STAGES[*]} ##########"
for s in "${STAGES[@]}"; do
  case "$s" in
    manifest) do_manifest ;; check) do_check ;;
    feat1) do_feat1 ;; feat2) do_feat2 ;;
    *) say "**知らない段階: $s**" ;;
  esac
done
say "########## 終わった ##########"
