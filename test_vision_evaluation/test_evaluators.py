import unittest
import numpy as np
import json
import pathlib
from PIL import Image

from vision_evaluation.evaluators import AveragePrecisionEvaluator, F1ScoreEvaluator, TopKAccuracyEvaluator, ThresholdAccuracyEvaluator, MeanAveragePrecisionEvaluatorForSingleIOU, EceLossEvaluator, \
    PrecisionEvaluator, RecallEvaluator, TagWiseAccuracyEvaluator, TagWiseAveragePrecisionEvaluator, \
    MeanAveragePrecisionNPointsEvaluator, BalancedAccuracyScoreEvaluator, CocoMeanAveragePrecisionEvaluator, BleuScoreEvaluator, METEORScoreEvaluator, \
    ROUGELScoreEvaluator, CIDErScoreEvaluator, SPICEScoreEvaluator, RocAucEvaluator, MeanIOUEvaluator, ForegroundIOUEvaluator, BoundaryMeanIOUEvaluator, BoundaryForegroundIOUEvaluator, \
    L1ErrorEvaluator, GroupWiseEvaluator, MeanLpErrorEvaluator
from vision_evaluation.prediction_filters import TopKPredictionFilter, ThresholdPredictionFilter


class TestGroupWiseEvaluator(unittest.TestCase):
    def test_group_wise_classification_evaluator(self):

        gts = [{"targets": np.array([0, 0, 0, 0, 2, 2]), "groups": np.array([0, 0, 1, 1, 0, 0])},
               {"targets": np.array([0, 0, 0, 1, 1, 1]), "groups": np.array([0, 0, 0, 1, 1, 1])}]

        preds = [np.array([[0.8, 0.1, 0.1],
                           [0.8, 0.1, 0.1],
                           [0.1, 0.8, 0.1],
                           [0.1, 0.8, 0.1],
                           [0.1, 0.1, 0.8],
                           [0.1, 0.1, 0.8]]),
                 np.array([[0.8, 0.1],
                           [0.1, 0.8],
                           [0.8, 0.1],
                           [0.1, 0.8],
                           [0.8, 0.1],
                           [0.1, 0.8]])]

        n_group_classes = [2, 2]

        accuracies = [[{0: 1, 1: 0}, {0: 1, 1: 1}], [{0: 0.666, 1: 0.666}, {0: 1, 1: 1}]]

        for k_idx, top_k in enumerate([1, 5]):

            for i, (predictions, ground_truths) in enumerate(zip(preds, gts)):
                eval = GroupWiseEvaluator(lambda: TopKAccuracyEvaluator(top_k))
                eval.add_predictions(predictions, ground_truths)
                group_wise_top_k_acc = eval.get_report()['group_wise_metrics']

                for g in range(n_group_classes[i]):
                    if g in group_wise_top_k_acc:
                        acc = group_wise_top_k_acc[g][f"accuracy_top{top_k}"]
                        self.assertAlmostEqual(acc, accuracies[i][k_idx][g], places=2)

    def test_group_wise_detection_evaluator(self):
        preds = [[[[0, 1.0, 0, 0, 10, 10]],
                  [[1, 1.0, 5, 5, 10, 10]],
                  [[2, 1.0, 1, 1, 5, 5]]]]

        gts = [{"targets": [[[0, 0, 0, 10, 10]], [[1, 5, 5, 10, 10]], [[2, 1, 1, 5, 5]]],
                "groups":  [0, 0, 1]}]

        n_group_classes = [2]

        true_mAP = [{0: 1.0, 1: 1.0}]

        eval = GroupWiseEvaluator(lambda: CocoMeanAveragePrecisionEvaluator(ious=[0.5]))

        for i, (ground_truths, preds) in enumerate(zip(gts, preds)):

            eval.add_predictions(preds, ground_truths)
            report = eval.get_report()['group_wise_metrics']

            for g in range(n_group_classes[i]):
                self.assertAlmostEqual(report[g]["mAP_50"], true_mAP[i][g], places=8)


class TestClassificationEvaluator(unittest.TestCase):
    TARGETS = [
        np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1]),
        np.array([1, 0, 2, 0, 1, 2, 0, 0, 0, 1, 2, 1, 2, 2, 0])]
    PREDICTIONS = [
        np.array([[1, 0],
                  [0, 1],
                  [0.5, 0.5],
                  [0.1, 0.9],
                  [0.44, 0.56],
                  [0.09, 0.91],
                  [0.91, 0.09],
                  [0.37, 0.63],
                  [0.34, 0.66],
                  [0.89, 0.11]]),
        np.array([[0.99, 0.01, 0],
                  [0, 0.99, 0.01],
                  [0.51, 0.49, 0.0],
                  [0.09, 0.8, 0.11],
                  [0.34, 0.36, 0.3],
                  [0.09, 0.90, 0.01],
                  [0.91, 0.06, 0.03],
                  [0.37, 0.60, 0.03],
                  [0.34, 0.46, 0.2],
                  [0.79, 0.11, 0.1],
                  [0.34, 0.16, 0.5],
                  [0.04, 0.56, 0.4],
                  [0.04, 0.36, 0.6],
                  [0.04, 0.36, 0.6],
                  [0.99, 0.01, 0.0]])]

    def test_top_k_accuracy_evaluator(self):
        gts = [[0.4, 1.0, 1.0], [0.4666666, 0.7333333, 1.0]]
        for k_idx, top_k in enumerate([1, 2, 5]):
            for i, (targets, predictions) in enumerate(zip(self.TARGETS, self.PREDICTIONS)):
                eval = TopKAccuracyEvaluator(top_k)
                eval.add_predictions(predictions, targets)
                top_k_acc = eval.get_report()[f"accuracy_top{top_k}"]
                import sklearn.metrics as sm
                if predictions.shape[1] == 2:
                    predictions = predictions[:, 1]
                self.assertAlmostEqual(sm.top_k_accuracy_score(targets, predictions, k=top_k), top_k_acc)
                self.assertAlmostEqual(top_k_acc, gts[i][k_idx], places=5)

    def test_top_1_accuracy_evaluator_equivalent_to_top1_precision_eval(self):
        for targets, predictions in zip(self.TARGETS, self.PREDICTIONS):
            top1_acc_evaluator = TopKAccuracyEvaluator(1)
            top1_acc_evaluator.add_predictions(predictions, targets)

            top1_prec_evaluator = PrecisionEvaluator(TopKPredictionFilter(1))
            top1_prec_evaluator.add_predictions(predictions, targets)

            self.assertEqual(top1_acc_evaluator.get_report()["accuracy_top1"], top1_prec_evaluator.get_report(average='samples')['precision_top1'])

    def test_average_precision_evaluator(self):
        gts = [[0.447682, 0.475744, 0.490476190, 0.65], [0.384352058, 0.485592, 0.50326599, 0.6888888]]
        for i, (targets, predictions) in enumerate(zip(self.TARGETS, self.PREDICTIONS)):
            evaluator = AveragePrecisionEvaluator()
            evaluator.add_predictions(predictions, targets)
            for fl_i, flavor in enumerate(['micro', 'macro', 'weighted', 'samples']):
                self.assertAlmostEqual(evaluator.get_report(average=flavor)['average_precision'], gts[i][fl_i], places=5)

    def test_ece_loss_evaluator(self):
        gts = [0.584, 0.40800000]
        for i, (targets, predictions) in enumerate(zip(self.TARGETS, self.PREDICTIONS)):
            evaluator = EceLossEvaluator()
            evaluator.add_predictions(predictions, targets)
            self.assertAlmostEqual(evaluator.get_report()["calibration_ece"], gts[i], places=5)

    def test_threshold_accuracy_evaluator(self):
        gts = [[0.4, 0.35, 0.2], [0.355555, 0.4, 0.133333]]
        for i, (targets, predictions) in enumerate(zip(self.TARGETS, self.PREDICTIONS)):
            for j, threshold in enumerate(['0.3', '0.5', '0.7']):
                thresh03_evaluator = ThresholdAccuracyEvaluator(float(threshold))
                thresh03_evaluator.add_predictions(predictions, targets)
                self.assertAlmostEqual(thresh03_evaluator.get_report()[f"accuracy_thres={threshold}"], gts[i][j], places=5)

    def test_perclass_accuracy_evaluator(self):
        evaluator = TagWiseAccuracyEvaluator()
        evaluator.add_predictions(self.PREDICTIONS[0], self.TARGETS[0])
        result = evaluator.get_report()
        self.assertAlmostEqual(result['tag_wise_accuracy'][0], 0.33333, 5)
        self.assertEqual(result['tag_wise_accuracy'][1], 0.5)

    def test_perclass_accuracy_evaluator_with_missing_class(self):
        target_missing_class = np.array([0, 1, 0, 0])
        predicitons_missing_class = np.array([[1, 0, 0],
                                              [0, 1, 0],
                                              [0.5, 0.5, 0],
                                              [0.1, 0.9, 0]])
        evaluator = TagWiseAccuracyEvaluator()
        evaluator.add_predictions(predicitons_missing_class, target_missing_class)
        result = evaluator.get_report()
        self.assertEqual(len(result['tag_wise_accuracy']), 3)
        self.assertAlmostEqual(result['tag_wise_accuracy'][0], 0.666666, 5)
        self.assertEqual(result['tag_wise_accuracy'][1], 1.0)
        self.assertEqual(result['tag_wise_accuracy'][2], 0.0)

    def test_perclass_average_precision_evaluator(self):
        evaluator = TagWiseAveragePrecisionEvaluator()
        evaluator.add_predictions(self.PREDICTIONS[0], self.TARGETS[0])
        result = evaluator.get_report()
        self.assertAlmostEqual(result['tag_wise_average_precision'][0], 0.54940, 5)
        self.assertAlmostEqual(result['tag_wise_average_precision'][1], 0.40208, 5)

        # multilabel with only one class, but without negative tags, precision is meaningless
        targets_single_cls = np.array([[1], [1], [1]])
        predictions_single_cls = np.array([[0], [0], [1]])
        evaluator_single_cls = TagWiseAveragePrecisionEvaluator()
        evaluator_single_cls.add_predictions(predictions_single_cls, targets_single_cls)
        result = evaluator_single_cls.get_report()
        self.assertAlmostEqual(result['tag_wise_average_precision'][0], 1, 5)


class TestMultilabelClassificationEvaluator(unittest.TestCase):
    TARGETS = np.array([[1, 0, 0],
                        [0, 1, 1],
                        [1, 1, 1]])
    PROB_PREDICTIONS = np.array([[1, 0.3, 0],
                                [0, 1, 0.5],
                                [0.5, 0.6, 0.5]])
    INDEX_PREDICTIONS = np.array([[0, 1, 2],
                                 [1, 2, 0],
                                 [1, 0, 2]])

    def test_precision_evaluator(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [0.66666, 0.83333, 1.0, 0.66666]
        for i in range(len(thresholds)):
            prec_eval = PrecisionEvaluator(ThresholdPredictionFilter(thresholds[i]))
            prec_eval.add_predictions(self.PROB_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(prec_eval.get_report(average='samples')[f"precision_thres={thresholds[i]}"], expectations[i], places=4)

        ks = [1, 2, 3]
        expectations = [1.0, 0.833333, 0.66666]
        for i in range(len(ks)):
            prec_eval = PrecisionEvaluator(TopKPredictionFilter(ks[i]))
            prec_eval.add_predictions(self.PROB_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(prec_eval.get_report(average='samples')[f"precision_top{ks[i]}"], expectations[i], places=4)

    def test_recall_evaluator(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [1.0, 1.0, 0.61111, 0.5]
        for i in range(len(thresholds)):
            recall_eval = RecallEvaluator(ThresholdPredictionFilter(thresholds[i]))
            recall_eval.add_predictions(self.PROB_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(recall_eval.get_report(average='samples')[f"recall_thres={thresholds[i]}"], expectations[i], places=4)

        ks = [0, 1, 2, 3]
        expectations = [0, 0.61111, 0.88888, 1.0]
        for i in range(len(ks)):
            recall_eval = RecallEvaluator(TopKPredictionFilter(ks[i]))
            recall_eval.add_predictions(self.PROB_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(recall_eval.get_report(average='samples')[f"recall_top{ks[i]}"], expectations[i], places=4)

        for i in range(len(ks)):
            recall_eval = RecallEvaluator(TopKPredictionFilter(ks[i], prediction_mode='indices'))
            recall_eval.add_predictions(self.INDEX_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(recall_eval.get_report(average='samples')[f"recall_top{ks[i]}"], expectations[i], places=4)

    def test_average_precision_evaluator(self):
        targets = np.array([[1, 0, 0, 0],
                            [0, 1, 1, 1],
                            [0, 0, 1, 1],
                            [1, 1, 1, 0]])
        predictions = np.array([[0, 0.3, 0.7, 0],
                                [0, 1, 0.5, 0],
                                [0, 0, 0.5, 0],
                                [0.5, 0.6, 0, 0.5]])
        gts = [0.67328, 0.73611, 0.731481, 0.680555]
        evaluator = AveragePrecisionEvaluator()
        evaluator.add_predictions(predictions, targets)
        for fl_i, flavor in enumerate(['micro', 'macro', 'weighted', 'samples']):
            evaluator.get_report(average=flavor)['average_precision']
            self.assertAlmostEqual(evaluator.get_report(average=flavor)['average_precision'], gts[fl_i], places=5)

    def test_f1_score_evaluator(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = {'f1': [0.8, 0.94118, 0.57142, 0.44444], 'recall': [1.0, 1.0, 0.5, 0.33333], 'precision': [0.66666, 0.88888, 0.66666, 0.66666]}
        for i in range(len(thresholds)):
            recall_eval = F1ScoreEvaluator(ThresholdPredictionFilter(thresholds[i]))
            recall_eval.add_predictions(self.PROB_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"f1_score_thres={thresholds[i]}"], expectations['f1'][i], places=4)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"recall_thres={thresholds[i]}"], expectations['recall'][i], places=4)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"precision_thres={thresholds[i]}"], expectations['precision'][i], places=4)

        ks = [0, 1, 2, 3]
        expectations = {'f1': [0.0, 0.57142, 0.86021, 0.8], 'recall': [0.0, 0.5, 0.83333, 1.0], 'precision': [0, 0.66666, 0.88888, 0.66666]}
        for i in range(len(ks)):
            recall_eval = F1ScoreEvaluator(TopKPredictionFilter(ks[i]))
            recall_eval.add_predictions(self.PROB_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"f1_score_top{ks[i]}"], expectations['f1'][i], places=4)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"recall_top{ks[i]}"], expectations['recall'][i], places=4)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"precision_top{ks[i]}"], expectations['precision'][i], places=4)


class TestMeanAveragePrecisionEvaluatorForSingleIOU(unittest.TestCase):
    def test_perfect_one_image_absolute_coordinates(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)

        predictions = [[[0, 1.0, 0, 0, 10, 10],
                        [1, 1.0, 5, 5, 10, 10],
                        [2, 1.0, 1, 1, 5, 5]]]

        targets = [[[0, 0, 0, 10, 10],
                    [1, 5, 5, 10, 10],
                    [2, 1, 1, 5, 5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 1.0)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_perfect_one_image(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1],
                        [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1],
                    [2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 1.0)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_wrong_one_image(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [0, 1.0, 0.5, 0.5, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 0.75)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_perfect_two_images(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 1.0)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_two_batches(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5, report_tag_wise=True)

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)

        predictions = [[[0, 1.0, 0.9, 0.9, 1, 1],  # Wrong
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 0.75)
        self.assertTrue(isinstance(report["mAP_50"], float))
        self.assertEqual(len(report["tag_wise_AP_50"]), 3)

    def test_iou_threshold(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)

        predictions = [[[0, 1.0, 0.5, 0.5, 1, 1],  # IOU 0.25
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 0.5)
        self.assertTrue(isinstance(report["mAP_50"], float))

        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.2)

        predictions = [[[0, 1.0, 0.5, 0.5, 1, 1],  # IOU 0.25
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_20"], 1.0)
        self.assertTrue(isinstance(report["mAP_20"], float))

    def test_no_predictions(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)

        predictions = [[]]
        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1],
                    [2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 0.0)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_no_targets(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1],
                        [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 0.0)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_empty_result(self):
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)
        report = evaluator.get_report()
        self.assertIn('mAP_50', report)
        self.assertEqual(report["mAP_50"], 0.0)
        self.assertTrue(isinstance(report["mAP_50"], float))


class TestMeanAveragePrecisionNPoints(unittest.TestCase):
    TARGETS = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0]])
    PREDICTIONS = np.array([[1, 0],
                            [0, 1],
                            [0.5, 0.5],
                            [0.1, 0.9],
                            [0.44, 0.56],
                            [0.09, 0.91],
                            [0.91, 0.09],
                            [0.37, 0.63],
                            [0.34, 0.66],
                            [0.89, 0.11]])

    def test_mean_average_precision_n_points(self):
        evaluator = MeanAveragePrecisionNPointsEvaluator(11)
        evaluator.add_predictions(predictions=self.PREDICTIONS, targets=self.TARGETS)
        report = evaluator.get_report()
        self.assertAlmostEqual(report[evaluator._get_id()], 0.7406926406926406, places=4)


class TestBalancedScoreEvaluator(unittest.TestCase):
    TARGETS = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1])
    PREDICTIONS = np.array([[1, 0],
                            [0, 1],
                            [0.6, 0.4],
                            [0.1, 0.9],
                            [0.44, 0.56],
                            [0.09, 0.91],
                            [0.91, 0.09],
                            [0.37, 0.63],
                            [0.34, 0.66],
                            [0.89, 0.11]])

    def test_balanced_evaluator(self):
        evaluator = BalancedAccuracyScoreEvaluator()
        evaluator.add_predictions(predictions=self.PREDICTIONS, targets=self.TARGETS)
        report = evaluator.get_report()
        self.assertAlmostEqual(report[evaluator._get_id()], 0.625, places=4)


class TestCocoMeanAveragePrecisionEvaluator(unittest.TestCase):
    def test_perfect_one_image_absolute_coordinates(self):
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5])

        predictions = [[[0, 1.0, 0, 0, 10, 10],
                        [1, 1.0, 5, 5, 10, 10],
                        [2, 1.0, 1, 1, 5, 5]]]

        targets = [[[0, 0, 0, 10, 10],
                    [1, 5, 5, 10, 10],
                    [2, 1, 1, 5, 5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["mAP_50"], 1.0, places=8)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_perfect_one_image(self):
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], coordinates='relative')

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1],
                        [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1],
                    [2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["mAP_50"], 1.0, places=8)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_wrong_one_image(self):
        # result for tag 0 different from TestMeanAveragePrecisionNPoints
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], coordinates='relative')

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [0, 1.0, 0.5, 0.5, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()

        self.assertAlmostEqual(report["mAP_50"], 1.0, places=8)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_perfect_two_images(self):
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], coordinates='relative')

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["mAP_50"], 1.0, places=8)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_two_batches(self):
        # result for tag 0 different from TestMeanAveragePrecisionNPoints
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], report_tag_wise=[True], coordinates='relative')

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)

        predictions = [[[0, 1.0, 0.9, 0.9, 1, 1],  # Wrong
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)

        report = evaluator.get_report()

        self.assertAlmostEqual(report["mAP_50"], 0.834983498349835, places=8)
        self.assertTrue(isinstance(report["mAP_50"], float))
        self.assertEqual(len(report["tag_wise_AP_50"]), 3)

    def test_is_crowd(self):
        # result for tag 0 different from TestMeanAveragePrecisionNPoints
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], report_tag_wise=[True], coordinates='relative')

        predictions = [[[0, 1.0, 0, 0, 1, 1], [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]],
                       [[0, 1.0, 0.9, 0.9, 1, 1], [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 0, 1, 1], [1, 0, 0.5, 0.5, 1, 1]],
                   [[2, 0, 0.1, 0.1, 0.5, 0.5]],
                   [[0, 1, 0, 0, 1, 1], [1, 0, 0.5, 0.5, 1, 1]],
                   [[2, 0, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)

        report = evaluator.get_report()

        self.assertAlmostEqual(report["mAP_50"], 1.0, places=8)
        self.assertTrue(isinstance(report["mAP_50"], float))
        self.assertEqual(len(report["tag_wise_AP_50"]), 3)

    def test_iou_threshold(self):
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], coordinates='relative')

        predictions = [[[0, 1.0, 0.5, 0.5, 1, 1],  # IOU 0.25
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["mAP_50"], 0.5, places=8)
        self.assertTrue(isinstance(report["mAP_50"], float))

        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.2])

        predictions = [[[0, 1.0, 0.5, 0.5, 1, 1],  # IOU 0.25
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["mAP_20"], 1.0, places=8)
        self.assertTrue(isinstance(report["mAP_20"], float))

    def test_no_predictions(self):
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], coordinates='relative')

        predictions = [[]]
        targets = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1],
                    [2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 0.0)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_no_targets(self):
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], coordinates='relative')

        predictions = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1],
                        [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()
        self.assertEqual(report["mAP_50"], 0.0)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_empty_result(self):
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.5], coordinates='relative')
        report = evaluator.get_report()
        self.assertIn('mAP_50', report)
        self.assertEqual(report["mAP_50"], 0.0)
        self.assertTrue(isinstance(report["mAP_50"], float))

    def test_cat_id_remap(self):
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.2, 0.5], coordinates='relative')

        predictions = [[(0, 1.0, 0, 0, 1, 1),
                        [1, 1.0, 0.5, 0.5, 1, 1],
                        [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets = [[[0, 0, 0, 1, 1],
                    [0, 0.5, 0.5, 1, 1],
                    [2, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions, targets)
        report = evaluator.get_report()

        # Reorder the cat ids so that the target cat ids are continuous: 0->0, 2->1, 1(category not shown in targets)->2
        evaluator = CocoMeanAveragePrecisionEvaluator(ious=[0.2, 0.5], coordinates='relative')

        predictions_remap_cat_id = [[[0, 1.0, 0, 0, 1, 1],
                                     [2, 1.0, 0.5, 0.5, 1, 1],
                                     [1, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        targets_remap_cat_id = [[[0, 0, 0, 1, 1],
                                 [0, 0.5, 0.5, 1, 1],
                                 [1, 0.1, 0.1, 0.5, 0.5]]]

        evaluator.add_predictions(predictions_remap_cat_id, targets_remap_cat_id)
        report_remap_cat_id = evaluator.get_report()

        for k in report.keys():
            self.assertEqual(report[k], report_remap_cat_id[k])


class TestImageCaptionEvaluator(unittest.TestCase):
    predictions_file = pathlib.Path(__file__).resolve().parent / 'data' / 'image_caption_prediction.json'
    ground_truth_file = pathlib.Path(__file__).resolve().parent / 'data' / 'image_caption_gt.json'
    imcap_predictions, imcap_targets = [], []
    predictions_dict = json.loads(predictions_file.read_text())
    ground_truth_dict = json.loads(ground_truth_file.read_text())

    gts_by_id = {}
    predictions_by_id = {pred['image_id']: pred['caption'] for pred in predictions_dict}

    for gt in ground_truth_dict['annotations']:
        if not gt['image_id'] in gts_by_id:
            gts_by_id[gt['image_id']] = []
        gts_by_id[gt['image_id']].append(gt['caption'])
    for key, value in predictions_by_id.items():
        imcap_predictions.append(value)
        imcap_targets.append(gts_by_id[key])

    def test_image_caption_blue_score_evaluator(self):
        evaluator = BleuScoreEvaluator()
        evaluator.add_predictions(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["Bleu_1"], 0.783228681385441)
        self.assertAlmostEqual(report["Bleu_2"], 0.6226378540059051)
        self.assertAlmostEqual(report["Bleu_3"], 0.47542636331846966)
        self.assertAlmostEqual(report["Bleu_4"], 0.3573567238999926)

    def test_image_caption_meteor_score_evaluator(self):
        evaluator = METEORScoreEvaluator()
        evaluator.add_predictions(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["METEOR"], 0.2878681068021112)

    def test_image_caption_rouge_l_score_evaluator(self):
        evaluator = ROUGELScoreEvaluator()
        evaluator.add_predictions(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["ROUGE_L"], 0.5774238052522583)

    def test_image_caption_cider_score_evaluator(self):
        evaluator = CIDErScoreEvaluator()
        evaluator.add_predictions(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["CIDEr"], 1.2346054374217474)

    def test_image_caption_spice_score_evaluator(self):
        evaluator = SPICEScoreEvaluator()
        evaluator.add_predictions(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["SPICE"], 0.2226814382948703)


class TestRocAucEvaluator(unittest.TestCase):
    @staticmethod
    def _get_metric(predictions, targets, **kwargs):
        eval = RocAucEvaluator()
        eval.add_predictions(predictions, targets)
        roc_auc = eval.get_report(**kwargs)['roc_auc']
        return roc_auc

    def test_perfect_predictions(self):
        predictions = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        targets = [0, 0, 0, 1, 1, 1]
        roc_auc = self._get_metric(predictions, targets)
        assert roc_auc == 1.0
        roc_auc = self._get_metric(np.array(predictions), np.array(targets))
        assert roc_auc == 1.0

    def test_perfect_predictions_prob_vec(self):
        predictions = [[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]
        targets = [0, 0, 0, 1, 1, 1]
        roc_auc = self._get_metric(predictions, targets)
        assert roc_auc == 1.0
        roc_auc = self._get_metric(np.array(predictions), np.array(targets))
        assert roc_auc == 1.0

    def test_abysmal_predictions(self):
        predictions = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        targets = [1, 1, 1, 0, 0, 0]
        roc_auc = self._get_metric(predictions, targets)
        assert roc_auc == 0.0
        roc_auc = self._get_metric(np.array(predictions), np.array(targets))
        assert roc_auc == 0.0

    def test_imperfect_predictions(self):
        predictions = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        targets = [0, 0, 0, 1, 0, 1]
        roc_auc = self._get_metric(predictions, targets)
        assert roc_auc == 0.875
        roc_auc = self._get_metric(np.array(predictions), np.array(targets))
        assert roc_auc == 0.875

    def test_perfect_multiclass_predictions(self):
        predictions = [[0.8, 0.2, 0.0], [0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]
        targets = [0, 0, 1, 1, 2, 2]
        roc_auc = self._get_metric(predictions, targets, multi_class='ovr')
        assert roc_auc == 1.0
        roc_auc = self._get_metric(np.array(predictions), np.array(targets), multi_class='ovr')
        assert roc_auc == 1.0

    def test_perfect_multilabel_predictions(self):
        predictions = [[0.8, 0.2, 0.0], [0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]]
        targets = [[1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]]
        roc_auc = self._get_metric(predictions, targets)
        assert roc_auc == 1.0
        roc_auc = self._get_metric(np.array(predictions), np.array(targets))
        assert roc_auc == 1.0


class TestImageMattingEvaluator(unittest.TestCase):

    image_matting_predictions = []
    image_matting_targets = []

    image_matting_predictions.append(Image.open(pathlib.Path(__file__).resolve().parent / 'data' / 'test_0.png'))
    image_matting_predictions.append(Image.open(pathlib.Path(__file__).resolve().parent / 'data' / 'test_1.png'))
    image_matting_targets.append(Image.open(pathlib.Path(__file__).resolve().parent / 'data' / 'gt_0.png'))
    image_matting_targets.append(Image.open(pathlib.Path(__file__).resolve().parent / 'data' / 'gt_1.png'))

    def test_image_matting_mean_iou_evaluator(self):
        evaluator = MeanIOUEvaluator()
        evaluator.add_predictions(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["mIOU"], 0.4530012134867148)

    def test_image_matting_foreground_iou_evaluator(self):
        evaluator = ForegroundIOUEvaluator()
        evaluator.add_predictions(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["fgIOU"], 0.23992256209190865)

    def test_image_matting_l1_error_evaluator(self):
        evaluator = L1ErrorEvaluator()
        evaluator.add_predictions(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["L1Err"], 77.07375)

    def test_image_matting_boundary_mean_iou_evaluator(self):
        evaluator = BoundaryMeanIOUEvaluator()
        evaluator.add_predictions(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["b_mIOU"], 0.6022811774422807)

    def test_image_matting_boundary_foreground_iou_evaluator(self):
        evaluator = BoundaryForegroundIOUEvaluator()
        evaluator.add_predictions(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.get_report()
        self.assertAlmostEqual(report["b_fgIOU"], 0.2460145344436508)


class TestMeanLpErrorEvaluator(unittest.TestCase):
    TARGETS = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0]).astype(float)
    PREDICTIONS = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]).astype(float)

    def test_l1_evaluator(self):
        evaluator_l1 = MeanLpErrorEvaluator(p=1)
        # test that adding in increments works
        evaluator_l1.add_predictions(predictions=self.PREDICTIONS[:5], targets=self.TARGETS[:5])
        evaluator_l1.add_predictions(predictions=self.PREDICTIONS[5:], targets=self.TARGETS[5:])
        report = evaluator_l1.get_report()
        self.assertAlmostEqual(report[evaluator_l1._get_id()], 1, places=4)

    def test_l2_evaluator(self):
        evaluator_l2 = MeanLpErrorEvaluator(p=2)
        # test that adding in increments works
        evaluator_l2.add_predictions(predictions=self.PREDICTIONS[:5], targets=self.TARGETS[:5])
        evaluator_l2.add_predictions(predictions=self.PREDICTIONS[5:], targets=self.TARGETS[5:])
        report = evaluator_l2.get_report()
        self.assertAlmostEqual(report[evaluator_l2._get_id()], np.sqrt(10) / 10, places=4)
