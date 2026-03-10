# Run Summary

Run name: run_003_gpu_b3_focal
Timestamp: 2026-03-02T13:31:59.837146+00:00
Git hash: unknown

## Model
- Backbone: efficientnet_b3
- Pretrained: True
- Num classes: 3

## Training
- Max epochs: 25
- LR: 0.0002
- Optimizer: adamw
- Scheduler: cosine
- Loss: focal

## Triage Thresholds
- t_high: 0.6
- t_low: 0.3

## Metrics

| Metric | Value |
|--------|-------|
| accuracy | 0.8614 |
| macro_f1 | 0.5196 |
| urgent_sensitivity | 0.8889 |
| urgent_specificity | 0.7193 |
| roc_auc | 0.9012 |
| pr_auc | 0.5689 |
| ece | 0.0669 |
| abstain_rate | 0.0107 |
| macro_f1_ci_lower | 0.4546 |
| macro_f1_ci_upper | 0.5887 |
| urgent_sensitivity_ci_lower | 0.7905 |
| urgent_sensitivity_ci_upper | 0.9765 |