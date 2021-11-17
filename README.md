# Introduction
This repo contains evaluation metric codes used in Microsoft Cognitive Services Computer Vision for tasks such as classification and object detection.

# Functionalities
This repo currently offers evaluation metrics for two vision tasks:

- **Image classification**:
    - `TopKAccuracyEvaluator`: computes the top-k accuracy for multi-class classification problem. A prediction is considered correct, if the ground truth label is within the labels with top k confidences.
    - `ThresholdAccuracyEvaluator`: computes the threshold based accuracy (mainly for multi-label classification problem), i.e., accuracy of the predictions with confidence over a certain threshold.
    - `AveragePrecisionEvaluator`: computes the average precision, i.e., precision averaged across different confidence thresholds.
    - `PrecisionEvaluator`: computes precision
    - `RecallEvaluator`: computes recall
    - `F1ScoreEvaluator`: computes f1-score (recall and precision will be reported as well)
    - `EceLossEvaluator`: computes the [ECE loss](https://arxiv.org/pdf/1706.04599.pdf), i.e., the expected calibration error, given the model confidence and true labels for a set of data points.
- **Object detection**:
    - `CocoMeanAveragePrecisionEvaluator`: Coco mean average precision (mAP) computation across different classes, under multiple [IoU(s)](https://en.wikipedia.org/wiki/Jaccard_index).

While different machine learning problems/applications prefer different metrics, below are some general recommendations:
- **Multiclass classification**: Top-1 accuracy and Top-5 accuracy
- **Multilabel classification**: Average precision, Precision/recall/precision@k/threshold, where k and threshold can be very problem-specific
- **Object detection**: mAP@IoU=30 and mAP@IoU=50
