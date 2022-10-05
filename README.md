# Vision Evaluation

## Introduction

This repo contains evaluation metric codes used in Microsoft Cognitive Services Computer Vision for tasks such as classification, object detection, image caption, and image matting.

If you only need the image classification or object detection evaluation pipeline, JRE is not required.
This repo

- contains evaluation metric codes used in Microsoft Cognitive Services Computer Vision for tasks such as classification and object detection.
- defines the contract for metric calculation code in `Evaluator` class, for bringing custom evaluators under the same interface

This repo isn't trying to re-invent the wheel, but to provide centralized defaults for most metrics across different vision tasks so dev/research teams can compare model performance on the same page. As expected, you can find many implementations backed up by well-known sklearn or pycocotools.

## Functionalities

This repo currently offers evaluation metrics for three vision tasks:

- **Image classification**:
  - `TopKAccuracyEvaluator`: computes the top-k accuracy for multi-class classification problem. A prediction is considered correct, if the ground truth label is within the labels with top k confidences.
  - `ThresholdAccuracyEvaluator`: computes the threshold based accuracy (mainly for multi-label classification problem), i.e., accuracy of the predictions with confidence over a certain threshold.
  - `AveragePrecisionEvaluator`: computes the average precision, i.e., precision averaged across different confidence thresholds.
  - `PrecisionEvaluator`: computes precision.
  - `RecallEvaluator`: computes recall.
  - `BalancedAccuracyScoreEvaluator`: computes balanced accuracy, i.e., average recall across classes, for multiclass classification.
  - `RocAucEvaluator`: computes Area under the Receiver Operating Characteristic Curve.
  - `F1ScoreEvaluator`: computes f1-score (recall and precision will be reported as well).
  - `EceLossEvaluator`: computes the [ECE loss](https://arxiv.org/pdf/1706.04599.pdf), i.e., the expected calibration error, given the model confidence and true labels for a set of data points.
- **Object detection**:
  - `CocoMeanAveragePrecisionEvaluator`: Coco mean average precision (mAP) computation across different classes, under multiple [IoU(s)](https://en.wikipedia.org/wiki/Jaccard_index).
- **Image caption**:
  - `BleuScoreEvaluator`: computes the Bleu score. For more details, refer to [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf).
  - `METEORScoreEvaluator`: computes the Meteor score. For more details, refer to [Project page](http://www.cs.cmu.edu/~alavie/METEOR/). We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor).
  - `ROUGELScoreEvaluator`: computes the Rouge-L score. Refer to [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf) for more details.
  - `CIDErScoreEvaluator`:  computes the CIDEr score. Refer to [CIDEr: Consensus-based Image Description Evaluation](http://arxiv.org/pdf/1411.5726.pdf) for more details.
  - `SPICEScoreEvaluator`:  computes the SPICE score. Refer to [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822) for more details.
- **Image matting**:
  - `MeanIOUEvaluator`: computes the mean intersection-over-union score. 
  - `ForegroundIOUEvaluator`: computes the foreground intersection-over-union evaluator score.
  - `BoundaryMeanIOUEvaluator`: computes the boundary mean intersection-over-union score. 
  - `BoundaryForegroundIOUEvaluator`:  computes the boundary foreground intersection-over-union score.
  - `L1ErrorEvaluator`:  computes the L1 error.
- **Image regression**:
  - `MeanLpErrorEvaluator`: computes the mean Lp error (e.g. L1 error for p=1, L2 error for p=2, etc.).
  
While different machine learning problems/applications prefer different metrics, below are some general recommendations:

- **Multiclass classification**: Top-1 Accuracy and Top-5 Accuracy
- **Multilabel classification**: Average Precision, Precision/Recall/Precision@k/threshold, where k and threshold can be very problem-specific
- **Object detection**: mAP@IoU=30 and mAP@IoU=50
- **Image caption**: Bleu, METEOR, ROUGE-L, CIDEr, SPICE
- **Image matting**: Mean IOU, Foreground IOU, Boundary mean IOU, Boundary Foreground IOU, L1 Error
- **Image regression**: Mean L1 Error, Mean L2 Error

## Additional Requirements

The image caption evaluators requires Jave Runtime Environment (JRE) (Java 1.8.0). This is not required for other evaluators.
