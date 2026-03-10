# CLAUDE.md
## Project: ISIC Lesion Urgency Model (Research)

You are an expert senior machine learning engineer and research engineer. Build a research-grade system that predicts lesion urgency from ISIC images.

Do not use emojis anywhere in this repository (including README and code comments).

This project is research only, not a diagnostic device. All user-facing surfaces must contain a clear disclaimer.

---

## Objective

Train an AI model that determines lesion urgency using ISIC image data.

Preferred output is a safe triage framing:
- urgent (biopsy or urgent referral recommended)
- monitor (safe to monitor routinely)
- uncertain (abstain and recommend professional assessment)

Also providing the reasoning for the prediction.

The dataset provides diagnosis labels; urgency labels must be derived via a clearly documented mapping from diagnosis to urgency (malignancy-risk proxy). The mapping must be configurable, versioned, and explicitly recorded in reports.

---

## Engineering Requirements

### Code quality
- Clean architecture with clear module boundaries
- Simple, tidy, readable code with minimal magic
- Typed Python (type hints everywhere practical)
- Consistent formatting (ruff/black or ruff format)
- Clear naming, small functions, low coupling
- No notebook-only workflows; everything runnable via CLI
- Dont over engineer.
### Modern stack (use this unless there is a strong reason not to)
- Python 3.11+
- PyTorch
- timm for backbones
- Lightning (or a minimal training loop if you can keep it equally clean)
- Hydra or pydantic-settings for config
- MLflow or W&B optional, but at minimum TensorBoard logs
- FastAPI for inference API
- pytest for tests
- pre-commit hooks

### Reproducibility
- Deterministic seeds and saved split files
- Patient-level splitting when patient_id exists
- Save full config and git commit hash for every run
- Store metrics and artifacts under reports/

### Safety
- Always include an uncertain/abstain option
- Calibrate probabilities (temperature scaling at minimum)
- Threshold selection must target high sensitivity for urgent class
- Never claim clinical diagnosis; present as research triage support only

---

## Dataset: ISIC

The dataset comes from ISIC. Support these input modes:
1) A prepared local dataset folder:
   - data/images/...
   - data/labels.csv with at least: image_path, diagnosis, patient_id (if available)
2) A converter script that transforms ISIC metadata exports into labels.csv.

Implement a robust dataset loader that can validate:
- missing files
- invalid labels
- class distribution
- duplicates (image-level or hash-based if possible)

---

## Repository Structure

Create the following structure:

project/
  README.md
  requirements.txt or pyproject.toml
  configs/
    default.yaml
    model/
    data/
    train/
  src/
    urgency/
      __init__.py
      cli.py
      config.py
      data/
        dataset.py
        splits.py
        transforms.py
      models/
        backbones.py
        multimodal.py
      training/
        trainer.py
        losses.py
        callbacks.py
      evaluation/
        metrics.py
        calibration.py
        error_analysis.py
        plots.py
      inference/
        infer.py
        api.py
      utils/
        logging.py
        io.py
        seed.py
        checks.py
  scripts/
    prepare_isic.py
    make_submission_bundle.py
  tests/
    test_data_loading.py
    test_splits.py
    test_infer_smoke.py
  reports/
    (generated)
  paper/
    (optional, generated)

Keep the project import path simple: src/urgency as the main package.

---

## CLI Commands (must exist)

All commands should be documented in README and work end-to-end.

- Prepare data (optional converter):
  python -m urgency.cli prepare-isic --input <path> --output data/

- Create splits:
  python -m urgency.cli make-splits --labels data/labels.csv --out data/splits/

- Train:
  python -m urgency.cli train --config configs/default.yaml

- Evaluate:
  python -m urgency.cli evaluate --run_dir runs/<run_id> --split test

- Calibrate:
  python -m urgency.cli calibrate --run_dir runs/<run_id> --split val

- Infer (CLI):
  python -m urgency.cli infer --run_dir runs/<run_id> --image path/to.jpg

- Serve API:
  python -m urgency.cli serve --run_dir runs/<run_id> --host 0.0.0.0 --port 8000

---

## Modeling Approach

### Baseline
- Transfer learning with a timm backbone (EfficientNet or ConvNeXt recommended as a strong baseline)
- Input: image only
- Output: urgency risk (urgent probability) and triage decision with thresholds and abstain region

### Optional metadata
If available (age/sex/anatomic site), add a simple tabular encoder and fuse with image features. Keep it simple and well-documented.

### Urgency label mapping
Implement a configurable mapping:
- urgent_diagnoses: list[str]
- monitor_diagnoses: list[str]
Any diagnosis not mapped must either be excluded or assigned to an explicit bucket. This behavior must be set in config and logged.

---

## Evaluation Requirements

Compute and save to reports/ for each run:
- confusion matrix
- accuracy, macro F1
- urgent sensitivity and specificity
- ROC-AUC and PR-AUC for urgent vs rest (as applicable)
- calibration: ECE, reliability diagram
- bootstrap confidence intervals for key metrics
- error analysis: top false negatives and false positives with paths (do not copy images)

Create:
- reports/metrics.json
- reports/leaderboard.csv (one row per run)
- reports/calibration_plot.png
- reports/confusion_matrix.png
- reports/error_analysis.md
- reports/run_summary.md

---

## Triage Decision Logic

Implement triage thresholds:
- if p_urgent >= T_high: urgent
- elif p_urgent <= T_low: monitor
- else: uncertain

Thresholds must be tuned on validation set and logged.
Default strategy: optimize for high urgent sensitivity with acceptable specificity.

---

## Iterative Improvement Loop (Required)

After building the baseline system, you must run an improvement loop that:
- Tests the full pipeline (unit tests + smoke train/eval on a tiny subset)
- Runs at least one full training/evaluation baseline
- Proposes improvements only when supported by evidence

Define a stop rule:
- Stop after 3 consecutive iterations with no meaningful improvement (default: macro F1 +0.5% or urgent sensitivity +1% without major regression), or after a maximum budget (default: 12 total runs), whichever comes first.

Each iteration must:
1) Write an experiment plan:
   - Hypothesis
   - Change(s) (max 3)
   - Expected effect
   - Risk
   - Success criteria
2) Execute training and evaluation
3) Compare to the best run so far
4) Keep the best checkpoint and mark it as best_model
5) Append results to reports/leaderboard.csv
6) Write a brief decision record to reports/research_log.md

Improvements can include:
- backbone choice (EfficientNet vs ConvNeXt vs ViT)
- image size
- augmentation tuning
- optimizer and LR schedule (AdamW + cosine or OneCycle)
- class imbalance handling (weighted CE vs focal)
- label smoothing
- calibration improvements (temperature scaling)
- threshold tuning and abstain optimization

Do not introduce excessive complexity. Prefer small, controlled changes.

---

## Self-Critique and Mistake Feedback (Required)

Every time you encounter a mistake (bug, failing test, wrong assumption, pipeline break, poor metric handling, leakage risk):
- Treat it as feedback.
- Add a short entry to reports/mistakes.md with:
  - what happened
  - root cause
  - fix
  - prevention (test, assertion, validation, or guardrail)
- Implement the prevention immediately where possible so the same mistake cannot recur.

Examples of prevention:
- add a unit test
- add a data validation check
- add an assertion with a clear error message
- add a linter rule or type check

---

## Testing Requirements

Minimum tests:
- Dataset loader validates files and labels
- Patient-level split does not leak patient_id across splits
- Inference smoke test loads a checkpoint and runs one image through pipeline

Also include a small smoke configuration:
- configs/smoke.yaml that trains for 1 epoch on a tiny subset

CI is optional, but tests must run locally via:
- pytest

---

## README Requirements

Keep README simple and short:
- one paragraph describing the project
- setup instructions
- commands to prepare data, train, evaluate, run inference, run API
- a disclaimer
No emojis.

---

## First Actions

1) Implement the repository scaffold exactly as specified.
2) Implement the data loader and split logic first (patient-level).
3) Add smoke tests and a smoke config.
4) Train a baseline and generate all required reports.
5) Start the iterative improvement loop until stop conditions are met.
