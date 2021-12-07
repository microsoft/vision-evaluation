import unittest
import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
from vision_evaluation.evaluators import AveragePrecisionEvaluator, F1ScoreEvaluator, TopKAccuracyEvaluator, ThresholdAccuracyEvaluator, MeanAveragePrecisionEvaluatorForSingleIOU, EceLossEvaluator, \
    PrecisionEvaluator, RecallEvaluator, TagWiseAccuracyEvaluator, TagWiseAveragePrecisionEvaluator, MeanAveragePrecisionNPointsEvaluator, BalancedAccuracyScoreEvaluator, \
    CocoMeanAveragePrecisionEvaluator
=======
import json
from vision_evaluation.evaluators import AveragePrecisionEvaluator, Evaluator, F1ScoreEvaluator, TopKAccuracyEvaluator, ThresholdAccuracyEvaluator, MeanAveragePrecisionEvaluatorForSingleIOU, EceLossEvaluator, \
    PrecisionEvaluator, RecallEvaluator, TagWiseAccuracyEvaluator, TagWiseAveragePrecisionEvaluator, ImageCaptionEvaluator
>>>>>>> adding pytest, etc
=======
import json
from vision_evaluation.evaluators import AveragePrecisionEvaluator, Evaluator, F1ScoreEvaluator, TopKAccuracyEvaluator, ThresholdAccuracyEvaluator, MeanAveragePrecisionEvaluatorForSingleIOU, EceLossEvaluator, \
    PrecisionEvaluator, RecallEvaluator, TagWiseAccuracyEvaluator, TagWiseAveragePrecisionEvaluator, ImageCaptionEvaluator
>>>>>>> d2f683eac8897648d4b45da1a240a43619964e7a
from vision_evaluation.prediction_filters import TopKPredictionFilter, ThresholdPredictionFilter


class TestClassificationEvaluator(unittest.TestCase):
    TARGETS = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1])
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

    def test_top_k_accuracy_evaluator(self):
        top1_acc_evaluator = TopKAccuracyEvaluator(1)
        top1_acc_evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)

        top5_acc_evaluator = TopKAccuracyEvaluator(5)
        top5_acc_evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)

        self.assertEqual(top1_acc_evaluator.get_report()["accuracy_top1"], 0.4)
        self.assertEqual(top5_acc_evaluator.get_report()["accuracy_top5"], 1.0)

    def test_top_1_accuracy_evaluator_equivalent_to_top1_precision_eval(self):
        top1_acc_evaluator = TopKAccuracyEvaluator(1)
        top1_acc_evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)

        top1_prec_evaluator = PrecisionEvaluator(TopKPredictionFilter(1))
        top1_prec_evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)

        self.assertEqual(top1_acc_evaluator.get_report()["accuracy_top1"], top1_prec_evaluator.get_report(average='samples')['precision_top1'])

    def test_average_precision_evaluator(self):
        evaluator = AveragePrecisionEvaluator()
        evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
        self.assertAlmostEqual(evaluator.get_report(average='micro')["average_precision"], 0.447682, places=5)
        self.assertAlmostEqual(evaluator.get_report(average='macro')["average_precision"], 0.475744, places=5)

    def test_ece_loss_evaluator(self):
        evaluator = EceLossEvaluator()
        evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
        self.assertEqual(evaluator.get_report()["calibration_ece"], 0.584)

    def test_threshold_accuracy_evaluator(self):
        thresh03_evaluator = ThresholdAccuracyEvaluator(0.3)
        thresh03_evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
        self.assertEqual(thresh03_evaluator.get_report()["accuracy_thres=0.3"], 0.4)

        thresh05_evaluator = ThresholdAccuracyEvaluator(0.5)
        thresh05_evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
        self.assertEqual(thresh05_evaluator.get_report()["accuracy_thres=0.5"], 0.35)

    def test_perclass_accuracy_evaluator(self):
        evaluator = TagWiseAccuracyEvaluator()
        evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
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
        evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
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
    PREDICTIONS = np.array([[1, 0.3, 0],
                            [0, 1, 0.5],
                            [0.5, 0.6, 0.5]])

    def test_precision_evaluator(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [0.66666, 0.83333, 1.0, 0.66666]
        for i in range(len(thresholds)):
            prec_eval = PrecisionEvaluator(ThresholdPredictionFilter(thresholds[i]))
            prec_eval.add_predictions(self.PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(prec_eval.get_report(average='samples')[f"precision_thres={thresholds[i]}"], expectations[i], places=4)

        ks = [1, 2, 3]
        expectations = [1.0, 0.833333, 0.66666]
        for i in range(len(ks)):
            prec_eval = PrecisionEvaluator(TopKPredictionFilter(ks[i]))
            prec_eval.add_predictions(self.PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(prec_eval.get_report(average='samples')[f"precision_top{ks[i]}"], expectations[i], places=4)

    def test_recall_evaluator(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [1.0, 1.0, 0.61111, 0.5]
        for i in range(len(thresholds)):
            recall_eval = RecallEvaluator(ThresholdPredictionFilter(thresholds[i]))
            recall_eval.add_predictions(self.PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(recall_eval.get_report(average='samples')[f"recall_thres={thresholds[i]}"], expectations[i], places=4)

        ks = [0, 1, 2, 3]
        expectations = [0, 0.61111, 0.88888, 1.0]
        for i in range(len(ks)):
            recall_eval = RecallEvaluator(TopKPredictionFilter(ks[i]))
            recall_eval.add_predictions(self.PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(recall_eval.get_report(average='samples')[f"recall_top{ks[i]}"], expectations[i], places=4)

    def test_f1_score_evaluator(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = {'f1': [0.8, 0.94118, 0.57142, 0.44444], 'recall': [1.0, 1.0, 0.5, 0.33333], 'precision': [0.66666, 0.88888, 0.66666, 0.66666]}
        for i in range(len(thresholds)):
            recall_eval = F1ScoreEvaluator(ThresholdPredictionFilter(thresholds[i]))
            recall_eval.add_predictions(self.PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"f1_score_thres={thresholds[i]}"], expectations['f1'][i], places=4)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"recall_thres={thresholds[i]}"], expectations['recall'][i], places=4)
            self.assertAlmostEqual(recall_eval.get_report(average='macro')[f"precision_thres={thresholds[i]}"], expectations['precision'][i], places=4)

        ks = [0, 1, 2, 3]
        expectations = {'f1': [0.0, 0.57142, 0.86021, 0.8], 'recall': [0.0, 0.5, 0.83333, 1.0], 'precision': [0, 0.66666, 0.88888, 0.66666]}
        for i in range(len(ks)):
            recall_eval = F1ScoreEvaluator(TopKPredictionFilter(ks[i]))
            recall_eval.add_predictions(self.PREDICTIONS, self.TARGETS)
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


<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> d2f683eac8897648d4b45da1a240a43619964e7a
class TestImageCaptionEvaluator(unittest.TestCase):
    def assertImageCaptionMetricsEqual(self, report):
        self.assertEqual(report["Bleu_1"], 0.783228681385441)
        self.assertEqual(report["Bleu_2"], 0.6226378540059051)
        self.assertEqual(report["Bleu_3"], 0.47542636331846966)
        self.assertEqual(report["Bleu_4"], 0.3573567238999926)
        self.assertEqual(report["METEOR"], 0.2878681068021112)
        self.assertEqual(report["ROUGE_L"], 0.5774238052522583)
        self.assertEqual(report["CIDEr"], 1.2346054374217474)
        self.assertEqual(report["SPICE"], 0.2226814382948703)

    def test_image_caption_evaluator_from_files(self):
        predictions_file = './test/image_caption_prediction.json'
        ground_truth_file = './test/image_caption_gt.json'
        evaluator = ImageCaptionEvaluator()
        evaluator.add_predictions(predictions=predictions_file, targets=ground_truth_file)
        report = evaluator.get_report()
        self.assertImageCaptionMetricsEqual(report)

    def test_image_caption_evaluator_from_variables(self):
        predictions_file = './test/image_caption_prediction.json'
        ground_truth_file = './test/image_caption_gt.json'
        predictions = json.load(open(predictions_file, 'r'))
        targets = json.load(open(ground_truth_file, 'r'))
        evaluator = ImageCaptionEvaluator()
        evaluator.add_predictions(predictions=predictions, targets=targets)
        report = evaluator.get_report()
<<<<<<< HEAD
        self.assertImageCaptionMetricsEqual(report)
>>>>>>> adding pytest, etc
=======
        self.assertImageCaptionMetricsEqual(report)
>>>>>>> d2f683eac8897648d4b45da1a240a43619964e7a
