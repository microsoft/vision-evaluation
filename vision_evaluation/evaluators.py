import collections
import statistics
import sklearn.metrics as sm
import numpy as np
from abc import ABC, abstractmethod
from .prediction_filters import PredictionFilter, TopKPredictionFilter, ThresholdPredictionFilter
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
        average = kwargs['average'] if 'average' in kwargs else 'macro'
        return {self._get_id(): self.calculate_score(average)}


class TopKAccuracyEvaluator(Evaluator):
    """
    Top k accuracy evaluator for multiclass classification
    """

    def __init__(self, k):
        self.prediction_filter = TopKPredictionFilter(k)
        super(TopKAccuracyEvaluator, self).__init__()

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


class MeanAveragePrecisionEvaluatorForSingleIOU(Evaluator):
    def __init__(self, iou=0.5):
        super(MeanAveragePrecisionEvaluatorForSingleIOU, self).__init__()
        self.iou = iou

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
        average = kwargs['average'] if 'average' in kwargs else 'macro'
        for class_index in self.is_correct:
            ap = MeanAveragePrecisionEvaluatorForSingleIOU._calculate_average_precision(self.is_correct[class_index], self.probabilities[class_index], self.true_num[class_index], average)
            self.aps[class_index] = ap

        mean_ap = float(statistics.mean([self.aps[x] for x in self.aps])) if self.aps else 0.0
        return {f'mAP_{int(self.iou * 100)}': mean_ap}

    def reset(self):
        self.is_correct = collections.defaultdict(list)
        self.probabilities = collections.defaultdict(list)
        self.true_num = collections.defaultdict(int)
        self.aps = collections.defaultdict(float)
        super(MeanAveragePrecisionEvaluatorForSingleIOU, self).reset()


class MeanAveragePrecisionEvaluatorForMultipleIOUs(EvaluatorAggregator):
    DEFAULT_IOU_VALUES = [0.3, 0.5, 0.75, 0.9]

    def __init__(self, ious=DEFAULT_IOU_VALUES):
        evaluators = [MeanAveragePrecisionEvaluatorForSingleIOU(iou) for iou in ious]
        super(MeanAveragePrecisionEvaluatorForMultipleIOUs, self).__init__(evaluators)
