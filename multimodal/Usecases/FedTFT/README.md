# Immune-Edge

Latency-aware knowledge distillation framework for self-healing intrusion detection with adversarial hardening.

## What is implemented

- Cloud-side robust teacher training with PGD/FGSM augmentation.
- Edge-side quantized student (8-bit/4-bit ready).
- Teacher-student knowledge distillation with temperature scaling.
- Hybrid GAN hardening loop where the generator targets student failure regions.
- Latency-aware objective:

  `L_total = alpha * L_detection + beta * L_adversarial + gamma * L_compute`

- Dataset adapters for ToN-IoT and CSE-CIC-IDS2018 (CSV pipeline).
- Checkpointing, export to TorchScript/ONNX, and edge budget estimation.

## Directory structure

- `immune_edge/`: source package
- `configs/`: YAML templates for each phase
- `scripts/`: phase entrypoints
- `docs/`: research and implementation notes

## Environment

Virtual environment created at:

`/home/ccl/Desktop/akeel_folder/MMFL_Flower/akeel_research_env`

Activate when needed:

```bash
source /home/ccl/Desktop/akeel_folder/MMFL_Flower/akeel_research_env/bin/activate
```

Preflight check (memory, deps, dataset presence):

```bash
python check_resources.py
```

## Path configuration

Configs use explicit environment variables for dataset/checkpoint paths.

1. Create your local env file from template and fill absolute paths:

```bash
cp configs/env.example.sh configs/env.sh
```

2. Load it before running scripts:

```bash
source configs/env.sh
```

If any variable is missing, config loading fails fast with an explicit error.

## Dataset download scripts

The repository now includes full dataset bootstrap scripts:

- `scripts/download_nids_datasets.py`: downloads NF-ToN-IoT and NF-CSE-CIC-IDS2018.
- `scripts/prepare_ids_splits.py`: creates `train.csv`, `val.csv`, `test.csv`.
- `scripts/generate_env_from_splits.py`: writes `configs/env.sh`.
- `scripts/bootstrap_datasets.py`: runs all steps end-to-end.

One-command bootstrap:

```bash
python scripts/bootstrap_datasets.py
```

This will:

- download raw datasets into `datasets/raw` and `datasets/extracted`
- create split CSVs in `datasets/processed/ton_iot` and `datasets/processed/cse_cic_ids2018`
- generate `configs/env.sh` with absolute paths for all required env vars

Then load paths:

```bash
source configs/env.sh
```

## Suggested phase order

1. Train robust teacher:

```bash
python scripts/train_teacher.py --config configs/teacher_cse_cic.yaml
```

2. Distill student on edge target data:

```bash
python scripts/distill_student.py --config configs/distill_ton_iot.yaml
```

3. Harden student using hybrid GAN:

```bash
python scripts/harden_student.py --config configs/harden_ton_iot.yaml
```

4. Export for deployment:

```bash
python scripts/export_student.py --config configs/export_student.yaml
```

## Paper protocol

For publication-quality reporting, run both:

1. Same-domain (main claim):
   - Teacher on ToN-IoT
   - Distill on ToN-IoT
   - Harden on ToN-IoT

2. Cross-domain transfer (generalization claim):
   - Teacher on CSE-CIC
   - Distill/harden on ToN-IoT

Same-domain end-to-end launcher:

```bash
bash scripts/run_ton_iot_same_domain_pipeline.sh
```

Cross-domain (teacher fixed on CSE-CIC) launcher:

```bash
bash scripts/run_uploaded_ton_iot_post_teacher.sh
```

## Resume-safe training

- `train_teacher.py`, `distill_student.py`, and `harden_student.py` now auto-resume from:
  - `checkpoints/teacher_last.pt`
  - `checkpoints/student_last.pt`
  - `checkpoints/hardening_last.pt`
- Resume can be overridden with `training.resume_from_checkpoint` in YAML.
- Disable auto-resume by setting `training.auto_resume: false`.

## Evaluation (clean + adversarial + latency)

Run publication-style evaluation on a student checkpoint:

```bash
python scripts/evaluate_student.py \
  --config configs/harden_ton_iot_uploaded.yaml \
  --checkpoint artifacts/harden_ton_iot_uploaded/checkpoints/hardening_best.pt \
  --split test \
  --adv
```

This writes `eval_report.json` under the config output directory.

## NeurIPS Suite (multi-seed + baselines + ablations)

Run full paper-style suite on uploaded ToN-IoT using the saved teacher:

```bash
python scripts/run_neurips_suite.py \
  --python-bin /home/ccl/Desktop/akeel_folder/MMFL_Flower/akeel_research_env/bin/python \
  --env-file configs/env_ton_iot_uploaded.sh \
  --teacher-checkpoint artifacts/teacher_cse_cic/checkpoints/teacher_best.pt \
  --seeds 123,231,777 \
  --scenarios all \
  --cuda-visible-devices 0,1
```

Suite outputs:

- `artifacts/neurips_suite/per_run_results.csv`
- `artifacts/neurips_suite/summary_by_scenario.csv`
- `artifacts/neurips_suite/summary_by_scenario.md`

## Low-resource defaults

- CPU-first settings (`runtime.device: cpu`).
- Small batch sizes (`16` by default).
- `num_workers: 0` to avoid background worker overhead.
- Minimal attack steps in hardening config to control compute.

## Important

- No experiments were executed during implementation.
