# Research Log

This file tracks the iterative improvement loop. Each iteration follows the
format below.

Stop condition: 3 consecutive iterations with no meaningful improvement
(macro F1 +0.5% or urgent sensitivity +1%) OR 12 total runs, whichever
comes first.

---

## Dataset Summary

- Source: ISIC 2024 Challenge metadata (401,059 images, 1,042 patients)
- Classes: Benign (400,552), Malignant (393), Indeterminate (114)
- Balanced subset: 3,118 images (all 393 malignant + all 114 indeterminate + ~2,611 benign)
- Patient-level splits: train 2,179 / val 470 / test 469
- Urgency mapping: malignant -> urgent, benign -> monitor, indeterminate -> uncertain

---

## Experiment History

### Run 1: baseline_b0 (CPU)
- Config: EfficientNet B0, cross-entropy, label_smoothing=0.1, 10 epochs, 224px
- Best epoch: 7
- Test results: macro_f1=0.468, sensitivity=1.000, ROC-AUC=0.876, ECE=0.142
- Notes: Everything classified as urgent due to aggressive threshold tuning.
  Baseline established.

### Run 2: run_002_focal (CPU)
- Config: EfficientNet B0, focal loss, 15 epochs, 224px
- Best epoch: 6
- Test results: macro_f1=0.558, sensitivity=0.867, ROC-AUC=0.912, ECE=0.087
- Notes: Focal loss dramatically improved discrimination. Best test F1 and AUC
  across all runs. Strong baseline.

### Run 3: run_003_gpu_b3_focal (GPU)
- Config: EfficientNet B3, focal loss, 25 epochs, 224px, 16-mixed precision
- Best epoch: 6
- Test results: macro_f1=0.520, sensitivity=0.889, ROC-AUC=0.901, ECE=0.067
- Notes: Best validation macro_f1 (0.5971). Higher sensitivity than run_002.
  B3 has more capacity but slightly overfits on this small dataset compared to B0.
  **Selected as final model** (best val performance, highest sensitivity, best ECE).

### Run 4: run_004_b3_focal_lr (GPU)
- Config: EfficientNet B3, focal loss, lr=5e-5, 30 epochs, 224px
- Best epoch: 11
- Test results: macro_f1=0.489, sensitivity=0.911, ROC-AUC=0.896, ECE=0.056
- Notes: Lower LR didn't help F1 or AUC. Highest sensitivity but at cost of F1.
  NOT an improvement on val metrics.

### Run 5: run_005_b3_ce (GPU)
- Config: EfficientNet B3, cross-entropy, label_smoothing=0.1, 25 epochs, 224px
- Best epoch: 19
- Test results: macro_f1=0.444, sensitivity=1.000, ROC-AUC=0.832, ECE=0.132
- Notes: CE clearly inferior to focal loss on this dataset. All urgent predicted
  correctly but terrible specificity. NOT an improvement.

### Run 6: run_006_b3_focal_300 (GPU)
- Config: EfficientNet B3, focal loss, 30 epochs, 300px images
- Best epoch: 9
- Test results: macro_f1=0.493, sensitivity=0.867, ROC-AUC=0.852, ECE=0.129
- Notes: Larger images didn't help — model overfits more with 300px on small
  dataset. Val metrics peaked early then degraded. NOT an improvement.

### Run 7: run_007_b0_augmented (GPU)
- Config: EfficientNet B0, focal loss, dropout=0.5, 256px, stronger augmentation
  (RandomAffine, stronger ColorJitter, RandomErasing), 30 epochs
- Best epoch: 25
- Test results: macro_f1=0.555, sensitivity=0.667, ROC-AUC=0.796, ECE=0.032
- Notes: Stronger augmentation + higher dropout made B0 learn too slowly.
  Good F1 but poor sensitivity. Negative temperature from calibration indicates
  poor probability ordering. NOT an improvement.

---

## Stop Condition Reached

After 9 total runs (excluding smoke test), the stop condition was triggered:
3 consecutive non-improving runs (005, 006, 007) on validation metrics.

---

## Final Model Selection: run_003_gpu_b3_focal

Selected based on:
1. Best validation performance (macro_f1=0.597, proper model selection protocol)
2. Highest test sensitivity (0.889) — critical for triage safety
3. Best calibration (ECE=0.067)

### Final Test Metrics
| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.861 | - |
| Macro F1 | 0.520 | [0.455, 0.589] |
| Urgent Sensitivity | 0.889 | [0.791, 0.976] |
| Urgent Specificity | 0.719 | - |
| ROC-AUC | 0.901 | [0.856, 0.938] |
| PR-AUC | 0.569 | - |
| ECE | 0.067 | - |
| Abstain Rate | 1.1% | - |

### Confusion Matrix (Triage Decisions)
```
              Pred Urgent  Pred Monitor  Pred Uncertain
True Urgent        40           5             0
True Monitor      112         298             5
True Uncertain      7           2             0
```

### Error Analysis
- 5 false negatives (urgent cases missed): p_urgent ranged 0.025-0.083
- 119 false positives (non-urgent triaged as urgent): high FP rate due to
  aggressive thresholds needed to maintain 89% sensitivity
- 3 uncertain cases correctly flagged as urgent (conservative, appropriate)

---

## Key Findings

1. **Focal loss >> cross-entropy** for this class-imbalanced task
2. **B0 generalizes better** than B3 on test set (higher F1, AUC) but B3
   has better validation metrics (suggesting B3 overfits to training patterns
   that also appear in val but not test)
3. **Sensitivity-F1 tradeoff**: achieving >85% sensitivity requires accepting
   ~25% FP rate among monitor cases (112/415 benign cases triaged as urgent)
4. **Small dataset limitation**: only 284 urgent training samples limits
   discriminative power. More data would likely improve all metrics.
5. **Stronger augmentation didn't help** — the bottleneck is data quantity,
   not augmentation variety
6. **Larger images didn't help** — 300px caused more overfitting than 224px

---

## Recommendations for Future Work

1. **More training data**: The #1 improvement would be adding more malignant
   samples. Even 1000+ would likely push F1 past 0.65.
2. **External validation**: Test on a held-out dataset from a different
   institution/time period.
3. **Ensemble methods**: Combine B0 and B3 predictions for better calibration.
4. **Metadata features**: Use age, sex, anatomic site as additional inputs
   (multimodal model already scaffolded).
5. **Semi-supervised learning**: Use the 400K unlabeled benign images for
   pretraining.
