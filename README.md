# Introduction 
This repo contains evaluation metric codes used in Microsoft Cognitive Services Computer Vision for tasks such as classification and object detection.

# Functionalities
This repo currently offers evaluation metrics for two vision tasks:

- Image classification:
    - `evaluators.TopKAccuracyEvaluator`: computes the top-k accuracy, i.e., accuracy of the top k predictions with highest confidence.
    - `evaluators.AveragePrecisionEvaluator`: computes the average precision, precision averaged across different confidence thresholds.
    - `evaluators.ThresholdAccuracyEvaluator`: computes the threshold based accuracy, i.e., accuracy of the predictions with confidence over a certain threshold.
    - `evaluators.EceLossEvaluator`: computes the [ECE loss](https://arxiv.org/pdf/1706.04599.pdf), i.e., the expected calibration error, given the model confidence and true labels for a set of data points. 
- Object detection:
    - `evaluators.MeanAveragePrecisionEvaluatorForSingleIOU`, `evaluators.MeanAveragePrecisionEvaluatorForMultipleIOUs`: computes the mean average precision (mAP), i.e. mean average precision across different classes, under single or multiple [IoU(s)](https://en.wikipedia.org/wiki/Jaccard_index).
    