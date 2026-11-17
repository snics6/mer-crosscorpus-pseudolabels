# Code for "Other modalities make up for missing labels: emotion-wise weighted pseudo-labels for multimodal emotion recognition on a new corpus with few labels"

This repository contains the code, the configuration fixed before each evaluation, the speaker-independent splits (identifiers only) and the aggregated result tables used for the tables and figures of the paper. Comments and docstrings in the scripts are mostly in Japanese.

**Not included**: the corpora, the features, the encoder weights, the emotion labels and transcripts of the corpora, per-utterance outputs and other large intermediate results. The corpora must be obtained from their distributors, and the labels are rebuilt from the corpora's own annotation files by the scripts in `features/`.

## Environment

- Create the Python environment at `./.venv` (the run scripts call `.venv/bin/python`) and run all commands from the repository root; create `logs/` before running the run scripts.
- Python 3.12.3. Library versions are fixed in `requirements.txt` (main ones: NumPy 2.5.2, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.11.0 with CUDA 12.6, Transformers 4.44.0); see the header of `requirements.txt` for installing PyTorch.
- Hardware used for the reported results:
  - LOCO setting, evaluation run of distillation and the 11 compared methods (Sects. 4.1 and 4.3 of the paper): NVIDIA H200, PyTorch 2.8.0 with CUDA 12.8.
  - All-source setting (Sect. 4.2; experiments E77, E78, E80): NVIDIA H100 80GB, PyTorch 2.11.0 with CUDA 12.6.
  - Evaluation of the proposal (Sects. 4.4–4.6) and Supplementary Information S3: NVIDIA RTX A5000 (24 GB), PyTorch 2.11.0 with CUDA 12.6.
- Rerunning on other hardware can change some values: in our checks, most units were identical, but a few cross-validation cells and pseudo-label units differed by up to 0.45 percentage points, and the xMUDA-style method with learning rate 0.01 differed by up to 28.2 points across machines.
- Each result folder in `results/` has `provenance.json` with the commit, script and input hashes, constants and software versions of the run.

## Encoders (weights not included)

| Modality | Encoder | Source |
|---|---|---|
| Audio | WavLM-large, last layer averaged over time | `microsoft/wavlm-large` (Hugging Face) |
| Audio (single-source analysis only) | HuBERT-large | `facebook/hubert-large-ll60k` |
| Visual, face detection | YOLOv8n-face | https://github.com/derronqi/yolov8-face (GPL-3.0); sha256 in `data/weights.sha256` |
| Visual, expression features | ResNet-50 trained on AffectNet (`FER_static_ResNet50_AffectNet.pt`) | from the implementation of Ryumina et al.; sha256 in `data/weights.sha256`; architecture in `features/resnet50_affectnet.py` |
| Text | RoBERTa-base, last layer averaged over tokens | `roberta-base` |
| Transcripts of AFEW and check of reversed speech | Whisper large-v3 | `openai/whisper-large-v3` |
| Earlier-stage features read by the first-half analyses | CLIP ViT-L/14, DeBERTa-v3-base | `openai/clip-vit-large-patch14`, `microsoft/deberta-v3-base` |

Place the two face weights in `weights/` and check them against `data/weights.sha256`.

## Corpora (not included)

IEMOCAP, CMU-MOSEI, MELD, AFEW, CREMA-D and RAVDESS, obtained from their distributors (IEMOCAP and AFEW under licence agreements). Set the corpus locations in `features/build_manifest_*.py` (default `/mnt/d/Databases/<corpus>`).

## Pipeline

1. **Utterance lists**: `run/build_data.sh manifest` rebuilds `data/metadata/*_utterance_manifest.csv` (labels, speakers, transcripts) from the corpora; AFEW transcripts with `features/build_transcript_afew.py`. `data/metadata_ids/` lists the utterance identifiers and grouping fields of our lists (no labels or transcripts) for checking the rebuilt lists.
2. **Features**: `run/build_data.sh feat1` (ResNet-50, WavLM, RoBERTa) and `feat2` (CLIP, HuBERT, DeBERTa). Features are always read through `measure/data.py` (`load_group`).
3. **Splits**: `python exp/dai_data.py` rebuilds `splits/` (five cluster-preserving folds per corpus, made once with seed 0). The released `splits/` contain identifiers, clusters and folds only, for checking the rebuilt splits.
4. **LOCO setting (Sects. 4.1, 4.3–4.6)**, in this order:

   | Step | Experiment | Command |
   |---|---|---|
   | 1 | Distillation, source classifiers | `python exp/dai1.py --stage source` |
   | 2 | Distillation, provisional targets | `run/dai1_pseudo.sh <T>` |
   | 3 | Distillation, selection | `python exp/dai1.py --stage select` (writes `frozen/dai1_frozen.json`) |
   | 4 | 11 methods, provisional targets and selection | `run/dai2_pseudo.sh <T>`, `python exp/dai2.py --stage select` |
   | 5 | Evaluation run (experiments 1–3) | `run/dai_eval.sh 1 <T>`, `run/dai_eval.sh 3 <T>`, `run/dai_eval.sh 2 <T>` (procedure in `run/eval_procedure.md`) |
   | 6 | Aggregation | `python exp/dai1_stats.py eval`, `python exp/dai1_desc.py eval`, `python exp/dai2_stats.py eval` |
   | 7 | Predictability of text decisions from audio (SI S3) | `run/dai14_run.sh`; reversed speech `run/dai23.sh` |
   | 8 | Design of the proposal (Sect. 3.6) | `run/dai16_probe.sh`, `run/dai_k10.sh` |
   | 9 | Six corpora (CREMA-D hyperparameters) | `run/dai18_probe.sh` |
   | 10 | Evaluation of the proposal | `python exp/dai20_eval.py --stage freeze`, then `run/dai20_eval.sh` |
   | 11 | Retrained AVT and CREMA-D | `python exp/dai21_eval5.py --stage freeze`, then `run/dai21_eval5.sh` |
   | 12 | Post hoc analyses | `tools/dai1_teacher_margin.py` (Sect. 4.3), `tools/dai20_select_strongest.py` (Sect. 4.4), `tools/dai20_strongest_from_k.py` (Sect. 4.5) |

   The run scripts take the sha256 of the experiment script and stop if it differs. The evaluation stages of experiments 1–3 check, through `frozen/eval_gate.json`, design documents and review reports that are not included in this repository, so they do not run here as released; the configuration they fixed is in `frozen/`, and the scripts are kept unchanged so that their hashes match `provenance.json`.
5. **All-source and single-source settings (Sect. 4.2, SI S2)**: `exp/reanalysis_e75.py` (E75), `exp/relearn_e76.py`, `exp/relearn_e76_finish.py`, `exp/analyze_e76.py` (E76), `run/e77*.sh` and `run/e78*.sh` (E77, E78), `exp/e80_teacher_j.py`, `exp/e80_stage_c.py` (E80), `exp/analyze_e79.py --stage delta|select|measure|judge` (E79). These analyses read outputs of earlier exploratory runs (maps of source and target scores, candidate pseudo-label providers). The small tables among those inputs are included in `results/`; per-utterance outputs and the scripts of the earlier exploratory runs are not included.
6. **Figures**: `paper/scripts/` (values in `paper/figures/data/`).

## Paper items and scripts

| Paper | Script | Results |
|---|---|---|
| Table 1, Table S1 | `features/build_manifest_*.py`, `exp/dai18_probe.py` | `results/dai18_probe/splits_new.csv` |
| Sect. 4.1, Table 2, Fig. 3, Tables S3–S4 | `exp/dai1*.py`, `exp/dai2*.py`, `paper/scripts/make_figs_v4.py` | `results/dai1/stats/`, `results/dai2/stats/prop_minus_method.csv` |
| Sect. 4.2, Table 3 | `exp/compare_dai.py`, `exp/compare_dai2.py`, `exp/e80_*.py`, `exp/analyze_e79.py` | `results/e79/judge/`, `results/e79_p4/judge/` |
| Sect. 4.3, Fig. 4 | `tools/dai1_teacher_margin.py`, `paper/scripts/make_figs_slides.py` | `results/dai1/stats/teacher_margin*.csv` |
| Sect. 3.6, Fig. 2, SI S5 | `exp/dai16_probe.py`, `paper/scripts/make_figs_prop_v2.py` | `results/dai16_probe*/` |
| Sects. 4.4–4.6, Table 4, Fig. 5, Tables S6–S8, S10 | `exp/dai20_eval.py`, `exp/dai21_eval5.py`, `tools/dai20_*.py`, `paper/scripts/make_figs_v4.py` | `results/dai20_eval/`, `results/dai21_eval5/` |
| SI S2 (single-source analysis, fit of the approximation) | `exp/reanalysis_e75.py`, `exp/*e76*.py` | `results/e75/`, `results/e76/`, `results/e76_analysis/` |
| SI S3 | `exp/dai14.py`, `exp/dai23.py`, `features/build_audio_lowpass.py` | `results/dai14/summary/`, `results/dai23/summary/` |
| SI S4 | `exp/dai1.py`, `exp/dai3.py` | `results/dai1/stats/`, `results/dai3/` |

## Licence

TBD.
