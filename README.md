# Introduction 
This repo contains evaluation metric codes used in Microsoft Cognitive Services Computer Vision for tasks such as classification, object detection, and image caption.

# Functionalities
This repo currently offers evaluation metrics for three vision tasks:

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
    - `MeanAveragePrecisionEvaluatorForSingleIOU`, `evaluators.MeanAveragePrecisionEvaluatorForMultipleIOUs`: computes the mean average precision (mAP), i.e. mean average precision across different classes, under single or multiple [IoU(s)](https://en.wikipedia.org/wiki/Jaccard_index).
- **Image caption**:
    - `BleuScoreEvaluator`: computes the Blue score. For more details, refer to [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf).
    - `METEORScoreEvaluator`: computes the Meteor score. For more details, refer to [Project page](http://www.cs.cmu.edu/~alavie/METEOR/). We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor).
    - `ROUGELScoreEvaluator`: computes the Rouge-L score. Refer to [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf) for more details.
    - `CIDErScoreEvaluator`:  computes the CIDEr score. Refer to [CIDEr: Consensus-based Image Description Evaluation](http://arxiv.org/pdf/1411.5726.pdf) for more details.
    - `SPICEScoreEvaluator`:  computes the SPICE score. Refer to [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822) for more details.

While different machine learning problems/applications prefer different metrics, below are some general recommendations:
- **Multiclass classification**: Top-1 accuracy and Top-5 accuracy
- **Multilabel classification**: Average precision, Precision/recall/precision@k/threshold, where k and threshold can be very problem-specific
- **Object detection**: mAP@IoU=30 and mAP@IoU=50
- **Image caption**: Bleu, METEOR, ROUGE-L, CIDEr, SPICE
