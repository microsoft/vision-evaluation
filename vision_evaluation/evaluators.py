import collections
import statistics

import sklearn.metrics as sm
import numpy as np
from abc import ABC, abstractmethod

import cv2
from pycocotools.cocoeval import Params, COCOeval
from sklearn.metrics import balanced_accuracy_score

from .prediction_filters import TopKPredictionFilter, ThresholdPredictionFilter
from functools import reduce


def _targets_to_mat(targets, n_class):
    if len(targets.shape) == 1:
        target_mat = np.zeros((len(targets), n_class), dtype=int)
        for i, t in enumerate(targets):
            target_mat[i, t] = 1
    else:
        target_mat = targets

    return target_mat


class Evaluator(ABC):
    """Class to evaluate model outputs and report the result.
    """

    def __init__(self):
        self.custom_fields = {}
        self.reset()

    @abstractmethod
    def add_predictions(self, predictions, targets):
        raise NotImplementedError

    @abstractmethod
    def get_report(self, **kwargs):
        raise NotImplementedError

    def add_custom_field(self, name, value):
        self.custom_fields[name] = str(value)

    def reset(self):
        self.custom_fields = {}


class EvaluatorAggregator(Evaluator):
    def __init__(self, evaluators):
        self.evaluators = evaluators
        super(EvaluatorAggregator, self).__init__()

    def add_predictions(self, predictions, targets):
        for evaluator in self.evaluators:
            evaluator.add_predictions(predictions, targets)

    def get_report(self, **kwargs):
        return reduce(lambda x, y: x.update(y) or x, [evalator.get_report(**kwargs) for evalator in self.evaluators])

    def reset(self):
        for evaluator in self.evaluators:
            evaluator.reset()


class MemorizingEverythingEvaluator(Evaluator, ABC):
    """
    Base evaluator that memorize all ground truth and predictions
    """

    def __init__(self, prediction_filter=None):
        self.all_targets = np.array([])
        self.all_predictions = np.array([])

        super(MemorizingEverythingEvaluator, self).__init__()
        self.prediction_filter = prediction_filter

    def reset(self):
        super(MemorizingEverythingEvaluator, self).reset()
        self.all_targets = np.array([])
        self.all_predictions = np.array([])

    def add_predictions(self, predictions, targets):
        """ Add a batch of predictions for evaluation.
        Args:
            predictions: the model output array. Shape (N, num_class)
            targets: the ground truths. Shape (N, num_class) for multi-label or (N,) for multi-class
        """

        assert len(predictions) == len(targets)

        target_mat = _targets_to_mat(targets, predictions.shape[1])
        predictions = self.prediction_filter.filter(predictions, 'vec') if self.prediction_filter else predictions

        if self.all_predictions.size != 0:
            self.all_predictions = np.append(self.all_predictions, predictions, axis=0)
        else:
            self.all_predictions = np.copy(predictions)

        if self.all_targets.size != 0:
            self.all_targets = np.append(self.all_targets, target_mat, axis=0)
        else:
            self.all_targets = np.copy(target_mat)

    def calculate_score(self, average='macro'):
        """
        average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.
        """
        assert self.all_targets.size == self.all_predictions.size
        result = 0.0
        if self.all_targets.size > 0:
            non_empty_idx = np.where(np.invert(np.all(self.all_targets == 0, axis=0)))[0]
            if non_empty_idx.size != 0:
                result = self._calculate(self.all_targets[:, non_empty_idx], self.all_predictions[:, non_empty_idx], average=average)

        return result

    @abstractmethod
    def _calculate(self, targets, predictions, average):
        pass

    @abstractmethod
    def _get_id(self):
        pass

    def get_report(self, **kwargs):
        average = kwargs.get('average', 'macro')
        return {self._get_id(): self.calculate_score(average)}


class TopKAccuracyEvaluator(Evaluator):
    """
    Top k accuracy evaluator for multiclass classification
    """

    def __init__(self, k):
        self.total_num = 0
        self.topk_correct_num = 0

        super(TopKAccuracyEvaluator, self).__init__()
        self.prediction_filter = TopKPredictionFilter(k)

    def reset(self):
        super(TopKAccuracyEvaluator, self).reset()
        self.total_num = 0
        self.topk_correct_num = 0

    def add_predictions(self, predictions, targets):
        """ Evaluate a batch of predictions.
        Args:
            predictions: the model output numpy array. Shape (N, num_class)
            targets: the golden truths. Shape (N,)
        """
        assert len(predictions) == len(targets)
        assert len(targets.shape) == 1

        n_sample = len(predictions)

        top_k_predictions = self.prediction_filter.filter(predictions, 'indices')
        self.topk_correct_num += len([1 for sample_idx in range(n_sample) if targets[sample_idx] in top_k_predictions[sample_idx]])
        self.total_num += n_sample

    def get_report(self, **kwargs):
        return {f'accuracy_{self.prediction_filter.identifier}': float(self.topk_correct_num) / self.total_num if self.total_num else 0.0}


class ThresholdAccuracyEvaluator(Evaluator):
    """
    Threshold-based accuracy evaluator for multilabel classification, calculated in a sample-based flavor
    Note that
        1. this could be used for multi-class classification, but does not make much sense
        2. sklearn.metrics.accuracy_score actually is computing exact match ratio for multi-label classification, which is too harsh
    """

    def __init__(self, threshold):
        self.num_sample = 0
        self.sample_accuracy_sum = 0

        super().__init__()
        self.prediction_filter = ThresholdPredictionFilter(threshold)

    def add_predictions(self, predictions, targets):
        """ Add a batch of predictions for evaluation.
        Args:
            predictions: the model output array. Shape (N, num_class)
            targets: the ground truths. Shape (N, num_class) for multi-label (or (N,) for multi-class)
        """

        assert len(predictions) == len(targets)

        num_samples = len(predictions)
        target_mat = _targets_to_mat(targets, predictions.shape[1])

        prediction_over_threshold = self.prediction_filter.filter(predictions, 'vec')
        n_correct_predictions = np.multiply(prediction_over_threshold, target_mat).sum(1)  # shape (N,)
        n_total = (np.add(prediction_over_threshold, target_mat) >= 1).sum(1)  # shape (N,)
        n_total[n_total == 0] = 1  # To avoid zero-division. If n_total==0, num should be zero as well.
        self.sample_accuracy_sum += (n_correct_predictions / n_total).sum()
        self.num_sample += num_samples

    def get_report(self, **kwargs):
        return {f'accuracy_{self.prediction_filter.identifier}': float(self.sample_accuracy_sum) / self.num_sample if self.num_sample else 0.0}

    def reset(self):
        super(ThresholdAccuracyEvaluator, self).reset()
        self.num_sample = 0
        self.sample_accuracy_sum = 0


class F1ScoreEvaluator(EvaluatorAggregator):
    """
    F1 score evaluator for both multi-class and multi-label classification, which also reports precision and recall
    """

    def __init__(self, prediction_filter):
        super().__init__([RecallEvaluator(prediction_filter), PrecisionEvaluator(prediction_filter)])
        self._filter_id = prediction_filter.identifier

    def get_report(self, **kwargs):
        average = kwargs.get('average', 'macro')
        report = super(F1ScoreEvaluator, self).get_report(average=average)
        prec = report[f'precision_{self._filter_id}']
        recall = report[f'recall_{self._filter_id}']
        report[f'f1_score_{self._filter_id}'] = 2 * (prec * recall) / (prec + recall) if prec + recall > 0 else 0.0

        return report


class PrecisionEvaluator(MemorizingEverythingEvaluator):
    """
    Precision evaluator for both multi-class and multi-label classification
    """

    def __init__(self, prediction_filter):
        super().__init__(prediction_filter)

    def _get_id(self):
        return f'precision_{self.prediction_filter.identifier}'

    def _calculate(self, targets, predictions, average):
        return sm.precision_score(targets, predictions, average=average)


class RecallEvaluator(MemorizingEverythingEvaluator):
    """
    Recall evaluator for both multi-class and multi-label classification
    """

    def __init__(self, prediction_filter):
        super().__init__(prediction_filter)

    def _get_id(self):
        return f'recall_{self.prediction_filter.identifier}'

    def _calculate(self, targets, predictions, average):
        return sm.recall_score(targets, predictions, average=average)


class AveragePrecisionEvaluator(MemorizingEverythingEvaluator):
    """
    Average Precision evaluator for both multi-class and multi-label classification
    """

    def __init__(self):
        super().__init__()

    def _get_id(self):
        return 'average_precision'

    def _calculate(self, targets, predictions, average):
        return sm.average_precision_score(targets, predictions, average=average)


class TagWiseAccuracyEvaluator(Evaluator):
    """
    Tag wise accuracy for multiclass classification
    """

    def _get_id(self):
        return 'tag_wise_accuracy'

    def reset(self):
        super(TagWiseAccuracyEvaluator, self).reset()
        self.confusion_matrix = 0

    def add_predictions(self, predictions, targets):
        """ Evaluate a batch of predictions.
        Args:
            predictions: the model output numpy array. Shape (N, num_class)
            targets: the golden truths. Shape (N,)
        """
        assert len(predictions) == len(targets)
        assert len(targets.shape) == 1

        prediction_cls = np.argmax(predictions, axis=1)
        self.confusion_matrix = np.add(self.confusion_matrix, sm.confusion_matrix(targets, prediction_cls, labels=np.arange(predictions.shape[1])))

    def get_report(self, **kwargs):
        normalized_cm = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
        per_class_accuracy = np.nan_to_num(normalized_cm.diagonal())  # avoid nan output

        return {self._get_id(): list(per_class_accuracy)}


class TagWiseAveragePrecisionEvaluator(MemorizingEverythingEvaluator):
    """
    Tag wise average precision for multiclass and multilabel classification
    """

    def _get_id(self):
        return 'tag_wise_average_precision'

    def _calculate(self, targets, predictions, average):
        """
        Average is ignored and set to be None, calcluate average precision for each class
        """
        return sm.average_precision_score(targets, predictions, average=None)

    def get_report(self, **kwargs):
        """ Get per class accuracy report.
        return:
            performance: list of float
        """
        per_class_ap = self.calculate_score()
        return {self._get_id(): list(per_class_ap) if not isinstance(per_class_ap, float) else [per_class_ap]}


class EceLossEvaluator(Evaluator):
    """
    Computes the expected calibration error (ECE) given the model confidence and true labels for a set of data points.

    Works for multi-class classification only.

    https://arxiv.org/pdf/1706.04599.pdf
    """

    def __init__(self, n_bins=15):
        # Calibration ECE, Divide the probability into nbins
        self.n_bins = n_bins
        bins = np.linspace(0, 1, self.n_bins + 1)
        self.bin_lower_bounds = bins[:-1]
        self.bin_upper_bounds = bins[1:]
        self.prediction_filter = TopKPredictionFilter(1)
        super(EceLossEvaluator, self).__init__()

    def add_predictions(self, predictions, targets):
        """ Evaluate a batch of predictions.
        Args:
            predictions: the model output numpy array. Shape (N, num_class)
            targets: the golden truths. Shape (N,)
        """

        self.total_num += len(predictions)

        indices = np.array(self.prediction_filter.filter(predictions, 'indices')).flatten()
        confidence = predictions[np.arange(len(predictions)), indices]
        correct = (indices == targets)
        for bin_i in range(self.n_bins):
            bin_lower_bound, bin_upper_bound = self.bin_lower_bounds[bin_i], self.bin_upper_bounds[bin_i]
            in_bin = np.logical_and(confidence > bin_lower_bound, confidence <= bin_upper_bound)
            self.total_correct_in_bin[bin_i] += correct[in_bin].astype(int).sum()
            self.sum_confidence_in_bin[bin_i] += confidence[in_bin].astype(float).sum()

    def get_report(self, **kwargs):
        return {'calibration_ece': float(np.sum(np.abs(self.total_correct_in_bin - self.sum_confidence_in_bin)) / self.total_num) if self.total_num else 0.0}

    def reset(self):
        super(EceLossEvaluator, self).reset()
        self.total_num = 0
        self.total_correct_in_bin = np.zeros(self.n_bins)
        self.sum_confidence_in_bin = np.zeros(self.n_bins)


class RocAucEvaluator(Evaluator):
    """
    Utilize sklearn.metrics.roc_auc_score to Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    Check https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score for details.
    """

    def reset(self):
        super(RocAucEvaluator, self).reset()
        self.all_targets = None
        self.all_predictions = None

    def add_predictions(self, predictions, targets):
        """ add predictions and targets.
        Args:
            predictions: predictions of array-like of shape (n_samples,) or (n_samples, n_classes)
            targets: targets of array-like of shape (n_samples,) or (n_samples, n_classes)

        """
        self.all_targets = np.concatenate([self.all_targets, np.array(targets)]) if self.all_targets else np.array(targets)
        self.all_predictions = np.concatenate([self.all_predictions, np.array(predictions)]) if self.all_predictions else np.array(predictions)

    def get_report(self, **kwargs):
        average = kwargs.get('average', 'macro')
        sample_weight = kwargs.get('sample_weight')
        max_fpr = kwargs.get('max_fpr')
        multi_class = kwargs.get('multi_class', 'raise')
        labels = kwargs.get('labels')

        if len(self.all_targets.shape) == 1 and len(self.all_predictions.shape) == 2 and self.all_predictions.shape[1] == 2:
            all_predictions = self.all_predictions[:, 1]
        else:
            all_predictions = self.all_predictions
        return {
            'roc_auc': sm.roc_auc_score(y_true=self.all_targets, y_score=all_predictions, average=average, sample_weight=sample_weight, max_fpr=max_fpr, multi_class=multi_class, labels=labels)
        }


class MeanAveragePrecisionEvaluatorForSingleIOU(Evaluator):
    def __init__(self, iou=0.5, report_tag_wise=False):
        """
        Args:
            iou: float, single IoU for matching
            report_tag_wise: if assigned True, also return the per class average precision
        """
        super(MeanAveragePrecisionEvaluatorForSingleIOU, self).__init__()
        self.iou = iou
        self.report_tag_wise = report_tag_wise

    def add_predictions(self, predictions, targets):
        """ Evaluate list of image with object detection results using single IOU evaluation.
        Args:
            predictions: list of predictions [[[label_idx, probability, L, T, R, B], ...], [...], ...]
            targets: list of image targets [[[label_idx, L, T, R, B], ...], ...]
        """

        assert len(predictions) == len(targets)

        eval_predictions = collections.defaultdict(list)
        eval_ground_truths = collections.defaultdict(dict)
        for img_idx, prediction in enumerate(predictions):
            for bbox in prediction:
                label = int(bbox[0])
                eval_predictions[label].append([img_idx, float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])])

        for img_idx, target in enumerate(targets):
            for bbox in target:
                label = int(bbox[0])
                if img_idx not in eval_ground_truths[label]:
                    eval_ground_truths[label][img_idx] = []
                eval_ground_truths[label][img_idx].append([float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])])

        class_indices = set(list(eval_predictions.keys()) + list(eval_ground_truths.keys()))
        for class_index in class_indices:
            is_correct, probabilities = self._evaluate_predictions(eval_ground_truths[class_index], eval_predictions[class_index], self.iou)
            true_num = sum([len(t) for t in eval_ground_truths[class_index].values()])

            self.is_correct[class_index].extend(is_correct)
            self.probabilities[class_index].extend(probabilities)
            self.true_num[class_index] += true_num

    @staticmethod
    def _calculate_area(rect):
        w = rect[2] - rect[0] + 1e-5
        h = rect[3] - rect[1] + 1e-5
        return float(w * h) if w > 0 and h > 0 else 0.0

    @staticmethod
    def _calculate_iou(rect0, rect1):
        rect_intersect = [max(rect0[0], rect1[0]),
                          max(rect0[1], rect1[1]),
                          min(rect0[2], rect1[2]),
                          min(rect0[3], rect1[3])]
        calc_area = MeanAveragePrecisionEvaluatorForSingleIOU._calculate_area
        area_intersect = calc_area(rect_intersect)
        return area_intersect / (calc_area(rect0) + calc_area(rect1) - area_intersect)

    def _is_true_positive(self, prediction, ground_truth, already_detected, iou_threshold):
        image_id = prediction[0]
        prediction_rect = prediction[2:6]
        if image_id not in ground_truth:
            return False, already_detected

        ious = np.array([self._calculate_iou(prediction_rect, g) for g in ground_truth[image_id]])
        best_bb = np.argmax(ious)
        best_iou = ious[best_bb]

        if best_iou < iou_threshold or (image_id, best_bb) in already_detected:
            return False, already_detected

        already_detected.add((image_id, best_bb))
        return True, already_detected

    def _evaluate_predictions(self, ground_truths, predictions, iou_threshold):
        """ Evaluate the correctness of the given predictions.
        Args:
            ground_truths: List of ground truths for the class. {image_id: [[left, top, right, bottom], [...]], ...}
            predictions: List of predictions for the class. [[image_id, probability, left, top, right, bottom], [...], ...]
            iou_threshold: Minimum IOU threshold to be considered as a same bounding box.
        """

        # Sort the predictions by the probability
        sorted_predictions = sorted(predictions, key=lambda x: -x[1])
        already_detected = set()
        is_correct = []
        for prediction in sorted_predictions:
            correct, already_detected = self._is_true_positive(prediction, ground_truths, already_detected,
                                                               iou_threshold)
            is_correct.append(correct)

        is_correct = np.array(is_correct)
        probabilities = np.array([p[1] for p in sorted_predictions])

        return is_correct, probabilities

    @staticmethod
    def _calculate_average_precision(is_correct, probabilities, true_num, average='macro'):
        if true_num == 0:
            return 0
        if not is_correct or not any(is_correct):
            return 0
        recall = float(np.sum(is_correct)) / true_num
        return sm.average_precision_score(is_correct, probabilities, average=average) * recall

    def get_report(self, **kwargs):
        average = kwargs.get('average', 'macro')
        for class_index in self.is_correct:
            ap = MeanAveragePrecisionEvaluatorForSingleIOU._calculate_average_precision(self.is_correct[class_index], self.probabilities[class_index], self.true_num[class_index], average)
            self.aps[class_index] = ap

        mean_ap = float(statistics.mean([self.aps[x] for x in self.aps])) if self.aps else 0.0
        key_name = f'mAP_{int(self.iou * 100)}'
        report = {key_name: mean_ap}
        if self.report_tag_wise:
            report[f'tag_wise_AP_{int(self.iou * 100)}'] = [self.aps[class_index] for class_index in self.aps]
        return report

    def reset(self):
        self.is_correct = collections.defaultdict(list)
        self.probabilities = collections.defaultdict(list)
        self.true_num = collections.defaultdict(int)
        self.aps = collections.defaultdict(float)
        super(MeanAveragePrecisionEvaluatorForSingleIOU, self).reset()


class MeanAveragePrecisionEvaluatorForMultipleIOUs(EvaluatorAggregator):
    DEFAULT_IOU_VALUES = [0.3, 0.5, 0.75, 0.9]

    def __init__(self, ious=DEFAULT_IOU_VALUES, report_tag_wise=None):
        if not report_tag_wise:
            report_tag_wise = len(ious) * [False]

        assert len(ious) == len(report_tag_wise)
        evaluators = [MeanAveragePrecisionEvaluatorForSingleIOU(ious[i], report_tag_wise[i]) for i in range(len(ious))]
        super(MeanAveragePrecisionEvaluatorForMultipleIOUs, self).__init__(evaluators)


class CocoMeanAveragePrecisionEvaluator(Evaluator):
    """ Coco mAP evaluator. Adapted to have the same interface as other evaluators.
    Source: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    This evaluator is an alternative of MeanAveragePrecisionEvaluatorForMultipleIOUs. Difference is
    the way to compute average precision: Coco computes area under curve with trapezoidal rule and
    linear interpolation, while the latter uses sklearn implementation. Coco can be too optimistic.
    """

    DEFAULT_IOU_VALUES = [0.3, 0.5, 0.75, 0.9]

    def __init__(self, ious=DEFAULT_IOU_VALUES, report_tag_wise=None, coordinates='absolute', max_dets=300):
        """ Initialize evaluator by specified ious and indicators of whether to report tag-wise mAP. For richer settings, please overwrite self.coco_eval_params
        Args:
            ious: list of ious.
            report_tag_wise: None or list of booleans with the same size as `ious`. True value means the
                for the corresponding iou, mAPs of each tag will be reported.
            coordinates: 'absolute' or 'relative'
            max_dets: max number of boxes
        """
        super(CocoMeanAveragePrecisionEvaluator, self).__init__()
        if not report_tag_wise:
            report_tag_wise = len(ious) * [False]
        assert len(ious) == len(report_tag_wise)
        self.report_tag_wise = report_tag_wise

        self.coco_eval_params = Params(iouType='bbox')
        self.coco_eval_params.areaRngLbl = ['all']
        if coordinates == 'relative':
            self.coco_eval_params.areaRng = [[0, 1.0]]

        self.coco_eval_params.maxDets = [max_dets]
        self.coco_eval_params.iouThrs = ious

    def add_predictions(self, predictions, targets):
        """ Evaluate list of image with object detection results using mscoco evaluation. Specify whether coordinates are 'absolute' or 'relative' in ctor
        Args:
            predictions: list of predictions [[[label_idx, probability, L, T, R, B], ...], [...], ...]
            targets: list of image targets [[[label_idx, L, T, R, B], ...], ...], or [[[label_idx, is_crowd, L, T, R, B], ...], ...]

        """
        self.targets += targets
        self.predictions += predictions

    def _coco_eval(self):
        from .coco_wrapper import COCOWrapper

        coco_ground_truths = COCOWrapper.convert(self.targets, 'gt')
        coco_predictions = COCOWrapper.convert(self.predictions, 'prediction')

        coco_eval = COCOeval(coco_ground_truths, coco_predictions, 'bbox')
        self.coco_eval_params.catIds = coco_eval.params.catIds
        self.coco_eval_params.imgIds = coco_eval.params.imgIds
        coco_eval.params = self.coco_eval_params

        coco_eval.evaluate()
        coco_eval.accumulate()

        return coco_eval

    @staticmethod
    def _summarize(eval_result, ap=1, iouThr=None, areaRng='all', maxDets=300, catId=None):
        # Adapted from https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L427
        p = eval_result.params
        iouThrs = np.array(p.iouThrs)
        # indices of categories, either all categories or the specified catId
        cind = p.catIds if catId is None else [catId]

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = eval_result.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == iouThrs)[0]
                s = s[t]
            s = s[:, :, cind, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = eval_result.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == iouThrs)[0]
                s = s[t]
            s = s[:, cind, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = 0.
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def get_report(self, **kwargs):
        coco_eval_result = self._coco_eval()
        report = {'avg_mAP': self._summarize(coco_eval_result, 1, maxDets=coco_eval_result.params.maxDets[-1])}
        # mAP for each iou
        report.update({f'mAP_{int(iou * 100)}': self._summarize(coco_eval_result, 1, iou, maxDets=coco_eval_result.params.maxDets[-1]) for iou in coco_eval_result.params.iouThrs})

        # tag-wise mAP
        for iou, iou_report_tag_wise in zip(coco_eval_result.params.iouThrs, self.report_tag_wise):
            if iou_report_tag_wise:
                report[f'tag_wise_AP_{int(iou * 100)}'] = [self._summarize(coco_eval_result, 1, iou, maxDets=coco_eval_result.params.maxDets[-1], catId=cat_id) for cat_id in
                                                           coco_eval_result.params.catIds]

        return report

    def reset(self):
        super(CocoMeanAveragePrecisionEvaluator, self).reset()
        self.targets = []
        self.predictions = []


class BalancedAccuracyScoreEvaluator(MemorizingEverythingEvaluator):
    """
    Average of recall obtained on each class, for multiclass classification problem
    """

    def _calculate(self, targets, predictions, average):
        single_targets = np.argmax(targets, axis=1)
        y_single_preds = np.argmax(predictions, axis=1)
        return balanced_accuracy_score(single_targets, y_single_preds)

    def _get_id(self):
        return 'balanced_accuracy'


class MeanAveragePrecisionNPointsEvaluator(MemorizingEverythingEvaluator):
    """
    N-point interpolated average precision, averaged over classes
    """

    def __init__(self, n_points=11):
        super().__init__()
        self.ap_n_points_eval = []
        self.n_points = n_points

    def _calculate(self, targets, predictions, average):
        n_class = predictions.shape[1]
        return np.mean([self._per_class_calc(predictions[:, i], targets[:, i]) for i in range(n_class)])

    def _per_class_calc(self, predictions, targets):
        """ Evaluate a batch of predictions.
        Args:
            predictions: the probability of the data to be 'positive'. Shape (N,)
            targets: the binary ground truths in {0, 1} or {-1, 1}. Shape (N,)
        """
        assert len(predictions) == len(targets)
        assert len(targets.shape) == 1

        precision, recall, _ = sm.precision_recall_curve(targets, predictions)
        recall_thresholds = np.linspace(1, 0, self.n_points, endpoint=True).tolist()
        precision_sum = 0
        recall_idx = 0
        precision_tmp = 0
        for threshold in recall_thresholds:
            while recall_idx < len(recall) and threshold <= recall[recall_idx]:
                precision_tmp = max(precision_tmp, precision[recall_idx])
                recall_idx += 1
            precision_sum += precision_tmp
        return precision_sum / self.n_points

    def _get_id(self):
        return f'mAP_{self.n_points}_points'


class ImageCaptionEvaluatorBase(Evaluator):
    """
    Base class for image caption metric evaluator
    """

    def __init__(self, metric):
        self.targets = []
        self.predictions = []
        super(ImageCaptionEvaluatorBase, self).__init__()
        self.metric = metric

    def add_predictions(self, predictions, targets):
        """ Evaluate list of image with image caption results using pycocoimcap tools.
        Args:
            predictions: list of string predictions [caption1, caption2, ...], shape: (N, ), type: string
            targets: list of string ground truth for image caption task: [[gt1, gt2, ...], [gt1, gt2, ...], ...], type: string
        """
        self.targets += targets
        self.predictions += predictions

    def reset(self):
        super(ImageCaptionEvaluatorBase, self).reset()
        self.targets = []
        self.predictions = []

    def get_report(self, **kwargs):
        from .coco_evalcap_utils import ImageCaptionCOCOEval, ImageCaptionCOCO, ImageCaptionWrapper
        imcap_predictions, imcap_targets = ImageCaptionWrapper.convert(self.predictions, self.targets)
        coco = ImageCaptionCOCO(imcap_targets)
        cocoRes = coco.loadRes(imcap_predictions)
        cocoEval = ImageCaptionCOCOEval(coco, cocoRes, self.metric)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
        result = cocoEval.eval
        return result


class BleuScoreEvaluator(ImageCaptionEvaluatorBase):
    """
    BLEU score evaluator for image caption task. For more details, refer to http://www.aclweb.org/anthology/P02-1040.pdf.
    """

    def __init__(self):
        super().__init__(metric='Bleu')
        self.predictions = []
        self.targets = []


class METEORScoreEvaluator(ImageCaptionEvaluatorBase):
    """
    METEOR score evaluator for image caption task. For more details, refer to http://www.cs.cmu.edu/~alavie/METEOR/.
    """

    def __init__(self):
        super().__init__(metric='METEOR')
        self.predictions = []
        self.targets = []


class ROUGELScoreEvaluator(ImageCaptionEvaluatorBase):
    """
    ROUGE_L score evaluator for image caption task. For more details, refer to http://anthology.aclweb.org/W/W04/W04-1013.pdf
    """

    def __init__(self):
        super().__init__(metric='ROUGE_L')
        self.predictions = []
        self.targets = []


class CIDErScoreEvaluator(ImageCaptionEvaluatorBase):
    """
    CIDEr score evaluator for image caption task. For more details, refer to http://arxiv.org/pdf/1411.5726.pdf.
    """

    def __init__(self):
        super().__init__(metric='CIDEr')
        self.predictions = []
        self.targets = []


class SPICEScoreEvaluator(ImageCaptionEvaluatorBase):
    """
    SPICE score evaluator for image caption task. For more details, refer to https://arxiv.org/abs/1607.08822.
    """

    def __init__(self):
        super().__init__(metric='SPICE')
        self.predictions = []
        self.targets = []


class MattingEvaluatorBase(Evaluator):
    """
    Base class for image matting evaluator
    """
    def __init__(self):
        super(MattingEvaluatorBase, self).__init__()
        self.predictions = []
        self.targets = []
        self.metric = None

    def add_predictions(self, predictions, targets):
        """ Adding predictions and ground truth of images for image matting task
        Args:
            predictions: list of image matting predictions, [matting1, matting2, ...], shape: (N, ), type: PIL image object or Numpy array
            targets: list of image matting ground truth, [gt1, gt2, ...], shape: (N, ), type: PIL image object or Numpy array
        """
        self.targets += targets
        self.predictions += predictions

    def reset(self):
        super(MattingEvaluatorBase, self).reset()
        self.targets = []
        self.predictions = []

    def _convert2binary(self, mask, threshold=128):
        bin_mask = mask.copy()
        bin_mask[mask < threshold] = 0
        bin_mask[mask >= threshold] = 1
        return bin_mask

    def _find_contours(self, matting, thickness=10):
        matting = np.copy(matting)
        opencv_major_version = int(cv2.__version__.split('.')[0])
        if opencv_major_version >= 4:
            contours, _ = cv2.findContours(matting, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(matting, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(matting.shape, np.uint8)

        cv2.drawContours(mask, contours, -1, 255, thickness)
        return mask

    def _create_contour_mask(self, gt_mask, pred_mask, line_width=10):
        contour_mask = self._find_contours((gt_mask * 255).astype('uint8'), thickness=line_width) / 255.0
        gt_contour_mask = gt_mask * contour_mask
        pred_contour_mask = pred_mask * contour_mask
        return gt_contour_mask, pred_contour_mask


class MeanIOUEvaluator(MattingEvaluatorBase):
    """
    Mean intersection-over-union evaluator
    """
    def __init__(self):
        super(MeanIOUEvaluator, self).__init__()
        self.metric = 'mIOU'

    def get_report(self, convert_to_binary=True):
        num_class = 2
        mean_iou = []
        for pred_mask, gt_mask in zip(self.predictions, self.targets):
            pred_mask = np.asarray(pred_mask)
            gt_mask = np.asarray(gt_mask)

            if convert_to_binary:
                pred_binmask = self._convert2binary(pred_mask)
                gt_binmask = self._convert2binary(gt_mask)
            else:
                pred_binmask = pred_mask
                gt_binmask = gt_mask

            label = num_class * gt_binmask.astype('int') + pred_binmask
            count = np.bincount(label.flatten(), minlength=num_class**2)
            confusion_matrix = count.reshape(num_class, num_class)
            iou = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + 1e-10)
            valid = confusion_matrix.sum(axis=1) > 0
            mean_iou_per_image = np.nanmean(iou[valid])
            mean_iou.append(mean_iou_per_image)

        return {self.metric: sum(mean_iou) / len(mean_iou)}


class ForegroundIOUEvaluator(MattingEvaluatorBase):
    """
    Foreground intersection-over-union evaluator
    """
    def __init__(self):
        super(ForegroundIOUEvaluator, self).__init__()
        self.metric = 'fgIOU'

    def get_report(self, convert_to_binary=True):
        num_class = 2
        fg_iou = []
        for pred_mask, gt_mask in zip(self.predictions, self.targets):
            pred_mask = np.asarray(pred_mask)
            gt_mask = np.asarray(gt_mask)

            if convert_to_binary:
                pred_binmask = self._convert2binary(pred_mask)
                gt_binmask = self._convert2binary(gt_mask)
            else:
                pred_binmask = pred_mask
                gt_binmask = gt_mask

            if np.all(gt_binmask == 0):
                res = 1 if np.all(pred_binmask == 0) else 0
                return {self.metric: res}

            label = num_class * gt_binmask.astype('int') + pred_binmask
            count = np.bincount(label.flatten(), minlength=num_class**2)
            confusion_matrix = count.reshape(num_class, num_class)
            iou = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + 1e-10)
            fg_iou.append(iou[1])

        return {self.metric: sum(fg_iou) / len(fg_iou)}


class BoundaryMeanIOUEvaluator(MattingEvaluatorBase):
    """
    Boundary mean intersection-over-union evaluator
    """
    def __init__(self):
        super(BoundaryMeanIOUEvaluator, self).__init__()
        self.metric = 'b_mIOU'
        self.base_evaluator = MeanIOUEvaluator()

    def get_report(self):
        for pred_mask, gt_mask in zip(self.predictions, self.targets):
            pred_mask = np.asarray(pred_mask)
            gt_mask = np.asarray(gt_mask)

            pred_binmask = self._convert2binary(pred_mask)
            gt_binmask = self._convert2binary(gt_mask)
            gt_boundary_mask, pred_boundary_mask = self._create_contour_mask(gt_binmask, pred_binmask)
            self.base_evaluator.add_predictions([pred_boundary_mask.astype(np.int64)], [gt_boundary_mask.astype(np.int64)])
        result = self.base_evaluator.get_report(convert_to_binary=False)
        return {self.metric: result[self.base_evaluator.metric]}


class BoundaryForegroundIOUEvaluator(MattingEvaluatorBase):
    """
    Boundary foreground intersection-over-union evaluator
    """
    def __init__(self):
        super(BoundaryForegroundIOUEvaluator, self).__init__()
        self.metric = 'b_fgIOU'
        self.base_evaluator = ForegroundIOUEvaluator()

    def get_report(self):
        for pred_mask, gt_mask in zip(self.predictions, self.targets):
            pred_mask = np.asarray(pred_mask)
            gt_mask = np.asarray(gt_mask)

            pred_binmask = self._convert2binary(pred_mask)
            gt_binmask = self._convert2binary(gt_mask)
            gt_boundary_mask, pred_boundary_mask = self._create_contour_mask(gt_binmask, pred_binmask)
            self.base_evaluator.add_predictions([pred_boundary_mask.astype(np.int64)], [gt_boundary_mask.astype(np.int64)])
        result = self.base_evaluator.get_report(convert_to_binary=False)
        return {self.metric: result[self.base_evaluator.metric]}


class L1ErrorEvaluator(MattingEvaluatorBase):
    """
    L1 error evaluator
    """
    def __init__(self):
        super(L1ErrorEvaluator, self).__init__()
        self.metric = 'L1Err'

    def get_report(self):
        l1_loss = []
        for pred_mask, gt_mask in zip(self.predictions, self.targets):
            pred_mask = np.asarray(pred_mask)
            gt_mask = np.asarray(gt_mask)
            mean_l1 = np.abs(pred_mask.astype(np.float)-gt_mask.astype(np.float)).mean()
            l1_loss.append(mean_l1)
        return {self.metric: sum(l1_loss) / len(l1_loss)}
