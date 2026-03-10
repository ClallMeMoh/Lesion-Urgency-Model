# Error Analysis

Generated: 2026-03-02T13:31:59.819315+00:00

Total false negatives (urgent missed): 5
Total false positives (non-urgent predicted urgent): 119

## Top-20 False Negatives (Urgent Missed)

_Sorted by ascending p_urgent — most missed urgent cases at the top._

| image_path | true_label | pred_label | p_urgent |
|------------|------------|------------|----------|
| ISIC_6561814.jpg | urgent | monitor | 0.0251 |
| ISIC_9544804.jpg | urgent | monitor | 0.0502 |
| ISIC_9821013.jpg | urgent | monitor | 0.0645 |
| ISIC_5661573.jpg | urgent | monitor | 0.0683 |
| ISIC_6095295.jpg | urgent | monitor | 0.0828 |

## Top-20 False Positives (Non-Urgent Predicted Urgent)

_Sorted by descending p_urgent — most confidently wrong predictions at the top._

| image_path | true_label | pred_label | p_urgent |
|------------|------------|------------|----------|
| ISIC_1459374.jpg | uncertain | urgent | 0.8651 |
| ISIC_5942958.jpg | monitor | urgent | 0.8004 |
| ISIC_1209339.jpg | monitor | urgent | 0.7866 |
| ISIC_6208101.jpg | monitor | urgent | 0.7762 |
| ISIC_5583077.jpg | monitor | urgent | 0.7678 |
| ISIC_7112661.jpg | monitor | urgent | 0.6789 |
| ISIC_8556155.jpg | monitor | urgent | 0.6760 |
| ISIC_6862297.jpg | monitor | urgent | 0.6740 |
| ISIC_7650870.jpg | monitor | urgent | 0.6555 |
| ISIC_6745510.jpg | monitor | urgent | 0.6415 |
| ISIC_9279160.jpg | monitor | urgent | 0.6326 |
| ISIC_1847080.jpg | monitor | urgent | 0.6273 |
| ISIC_3324431.jpg | uncertain | urgent | 0.6170 |
| ISIC_1107832.jpg | monitor | urgent | 0.6113 |
| ISIC_3654241.jpg | monitor | urgent | 0.5955 |
| ISIC_0790835.jpg | uncertain | urgent | 0.5954 |
| ISIC_9887713.jpg | monitor | urgent | 0.5940 |
| ISIC_8628073.jpg | monitor | urgent | 0.5923 |
| ISIC_3150405.jpg | monitor | urgent | 0.5849 |
| ISIC_4353467.jpg | monitor | urgent | 0.5704 |