import unittest
import numpy as np
from vision_evaluation.evaluators import AveragePrecisionEvaluator, TopKAccuracyEvaluator, ThresholdAccuracyEvaluator, MeanAveragePrecisionEvaluatorForSingleIOU, EceLossEvaluator


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

        self.assertEqual(top1_acc_evaluator.get_report(average='micro')["top1_accuracy"], 0.4)
        self.assertEqual(top5_acc_evaluator.get_report(average='micro')["top5_accuracy"], 1.0)

        self.assertEqual(top1_acc_evaluator.get_report(average='macro')["top1_accuracy"], 0.4)
        self.assertEqual(top5_acc_evaluator.get_report(average='macro')["top5_accuracy"], 1.0)

    def test_average_precision_evaluator(self):
        evaluator = AveragePrecisionEvaluator()
        evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
        self.assertEqual(evaluator.get_report(average='micro')["average_precision"], 0.4476823176823177)
        self.assertEqual(evaluator.get_report(average='macro')["average_precision"], 0.47574404761904765)

    def test_ece_loss_evaluator(self):
        evaluator = EceLossEvaluator()
        evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
        self.assertEqual(0.584, evaluator.get_report()["calibration_ece"])

    def test_threshold_accuracy_evaluator(self):
        thresh03_evaluator = ThresholdAccuracyEvaluator(0.3)
        thresh03_evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
        self.assertEqual(0.4, thresh03_evaluator.get_report()["accuracy_0.3"])

        thresh05_evaluator = ThresholdAccuracyEvaluator(0.5)
        thresh05_evaluator.add_predictions(self.PREDICTIONS, self.TARGETS)
        self.assertEqual(0.3, thresh05_evaluator.get_report()["accuracy_0.5"])


class TestMeanAveragePrecisionEvaluatorForSingleIOU(unittest.TestCase):
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
        evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(iou=0.5)

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
