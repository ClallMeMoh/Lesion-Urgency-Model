# ISIC Lesion Urgency Model

A research-grade system for predicting skin lesion urgency from dermoscopy images using the
ISIC dataset. The model classifies each image into one of three triage categories: urgent,
monitor, or uncertain. It is intended to support research into automated triage assistance
and must not be used for clinical decision-making.

---

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+, PyTorch 2.2+, and optionally a CUDA-capable GPU.

---

## Prepare Data

Download ISIC metadata and images from https://www.isic-archive.com and place images under
`data/images/`. Then convert the metadata export:

```bash
python -m urgency.cli prepare-isic --input <isic_metadata.csv> --output data/
```

Create train/val/test splits:

```bash
python -m urgency.cli make-splits --labels data/labels.csv --out data/splits/
```

---

## Train

```bash
python -m urgency.cli train --config configs/default.yaml
```

To run a quick smoke test (1 epoch, tiny subset):

```bash
python -m urgency.cli train --config configs/smoke.yaml
```

Override any config value:

```bash
python -m urgency.cli train --config configs/default.yaml run_name=run_002 train.lr=0.0005
```

---

## Calibrate

After training, fit temperature scaling and tune triage thresholds on the validation set:

```bash
python -m urgency.cli calibrate --run_dir runs/run_001
```

---

## Evaluate

```bash
python -m urgency.cli evaluate --run_dir runs/run_001 --split test
```

Reports are written to `reports/`:

- `metrics.json` - Full metric suite with bootstrap confidence intervals
- `confusion_matrix.png`
- `calibration_plot.png`
- `roc_curve.png`
- `error_analysis.md` - Top false positives and false negatives
- `run_summary.md`
- `leaderboard.csv` - One row per run for comparison

---

## Infer

Run inference on a single image:

```bash
python -m urgency.cli infer --run_dir runs/run_001 --image path/to/image.jpg
```

---

## Serve API

Start the FastAPI inference server:

```bash
python -m urgency.cli serve --run_dir runs/run_001 --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` - Check model status
- `POST /predict` - Upload an image file and receive a triage prediction

---

## Run Tests

```bash
pytest
```

---

## Disclaimer

This system is a research tool only. It is not a medical device, has not been clinically
validated, and must not be used to make or inform diagnostic or treatment decisions. All
outputs are for research purposes only. Always consult a qualified dermatologist for any
skin-related concerns.
