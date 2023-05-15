import unittest
import numpy as np

from vision_evaluation.retrieval_evaluators import MeanAveragePrecisionAtK, PrecisionAtKEvaluator, PrecisionRecallCurveNPointsEvaluator, RecallAtKEvaluator


class TestInformationRetrievalMetrics(unittest.TestCase):
    PREDICTIONS = [np.array([[5, 4, 3, 2, 1]]),
                   np.array([[5, 4, 3, 2, 1]]),
                   np.array([[1, 2, 3, 4, 5]]),
                   np.array([[5, 4, 3, 2, 1]]),
                   np.array([[5, 4, 3, 2, 1],
                             [5, 4, 3, 2, 1]]),
                   np.array([[5, 4, 3, 2, 1],
                             [5, 4, 3, 2, 1]]),
                   np.array([[1]]),
                   np.array([[2],
                             [3]])]
    TARGETS = [np.array([[1, 1, 0, 0, 1]]),
               np.array([[1, 1, 0, 0, 1]]),
               np.array([[1, 0, 0, 1, 1]]),
               np.array([[0, 0, 0, 0, 1]]),
               np.array([[0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1]]),
               np.array([[1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1]]),
               np.array([[1]]),
               np.array([[1],
                         [0]])]

    def test_recall_at_k(self):
        ks = [6, 8, 6, 6, 6, 6, 4, 4]
        expectations = [[0, 0.33333, 0.66666, 0.66666, 0.66666, 1.0],
                        [0, 0.33333, 0.66666, 0.66666, 0.66666, 1.0, 1.0, 1.0],
                        [0, 0.33333, 0.66666, 0.66666, 0.66666, 1.0],
                        [0, 0, 0, 0, 0, 1.0],
                        [0, 0, 0, 0, 0, 1.0],
                        [0, 0.25, 0.25, 0.25, 0.25, 1.0],
                        [0, 1.0, 1.0, 1.0],
                        [0, 0.5, 0.5, 0.5]]
        assert len(self.PREDICTIONS) == len(self.TARGETS) == len(ks) == len(expectations)
        for preds, tgts, exps, k in zip(self.PREDICTIONS, self.TARGETS, expectations, ks):
            for i in range(k):
                recall_eval = RecallAtKEvaluator(i)
                recall_eval.add_predictions(preds, tgts)
                self.assertAlmostEqual(recall_eval.get_report()[f"recall_at_{i}"], exps[i], places=4)

    def test_precision_at_k(self):
        ks = [6, 8, 6, 6, 6, 6, 4, 4]
        expectations = [[0, 1.0, 1.0, 0.66666, 0.5, 0.6],
                        [0, 1.0, 1.0, 0.66666, 0.5, 0.6, 0.6, 0.6],
                        [0, 1.0, 1.0, 0.66666, 0.5, 0.6],
                        [0, 0, 0, 0, 0, 0.2],
                        [0, 0, 0, 0, 0, 0.2],
                        [0, 0.5, 0.25, 0.16666, 0.125, 0.3],
                        [0, 1.0, 1.0, 1.0],
                        [0, 0.5, 0.5, 0.5]]
        assert len(self.PREDICTIONS) == len(self.TARGETS) == len(ks) == len(expectations)
        for preds, tgts, exps, k in zip(self.PREDICTIONS, self.TARGETS, expectations, ks):
            for i in range(k):
                precision_eval = PrecisionAtKEvaluator(i)
                precision_eval.add_predictions(preds, tgts)
                self.assertAlmostEqual(precision_eval.get_report()[f"precision_at_{i}"], exps[i], places=4)

    def test_precision_recall_curve(self):
        predictions = [np.array([[5, 4, 3, 2, 1]]),
                       np.array([[1, 3, 2, 5, 4]]),
                       np.array([[5, 4, 3, 2, 1]]),
                       np.array([[5, 4, 3, 2, 1],
                                 [5, 4, 3, 2, 1]])]
        targets = [np.array([[0, 0, 0, 0, 1]]),
                   np.array([[1, 0, 0, 0, 0]]),
                   np.array([[1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1]])]

        expectations_recall = [np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.39999, 0.29999, 0.19999, 0.09999, 0.0]),
                               np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.39999, 0.29999, 0.19999, 0.09999, 0.0]),
                               np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.39999, 0.29999, 0.19999, 0.09999, 0.0]),
                               np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.39999, 0.29999, 0.19999, 0.09999, 0.0])]

        expectations_precision = [np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1., ]),
                                  np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1., ]),
                                  np.array([0.4, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]),
                                  np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 1., ])]

        assert len(predictions) == len(targets) == len(expectations_recall) == len(expectations_precision)
        for preds, tgts, exp_r, exp_p in zip(predictions, targets, expectations_recall, expectations_precision):
            n_points = 11
            evaluator = PrecisionRecallCurveNPointsEvaluator(n_points)
            evaluator.add_predictions(predictions=preds, targets=tgts)
            result = evaluator.get_report()
            result_recall = result[f"PR_Curve_{n_points}_point_interp"]['recall']
            precision_recall = result[f"PR_Curve_{n_points}_point_interp"]['precision']
            self.assertAlmostEqual(np.sum(np.abs(result_recall - exp_r)), 0.0, places=4)
            self.assertAlmostEqual(np.sum(np.abs(precision_recall - exp_p)), 0.0, places=4)

    def test_mean_average_precision_at_k_evaluator(self):
        targets = [np.array([[1, 0, 1, 1],
                             [1, 0, 0, 1]]),
                   np.array([[1, 0, 1, 1]]),
                   np.array([[1, 0, 0, 1]]),
                   np.array([[]]),
                   np.array([[1, 0, 1, 1]]),
                   np.array([[1, 0, 1, 1]]),
                   np.array([[1, 0, 1, 1]]),
                   np.array([[1, 0, 1, 1]]),
                   np.array([[1, 0, 1, 1]]),
                   np.array([[1, 0, 1, 1]]),
                   np.array([[0, 0, 0, 0]]),
                   np.array([[1, 0, 1]]),
                   np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]])]
        predictions = [np.array([[5, 4, 3, 2],
                                 [5, 4, 3, 2]]),
                       np.array([[5, 4, 3, 2]]),
                       np.array([[5, 4, 3, 2]]),
                       np.array([[]]),
                       np.array([[2, 3, 5, 4]]),
                       np.array([[4, 2, 3, 5]]),
                       np.array([[4, 2, 3, 5]]),
                       np.array([[2, 3, 5, 4]]),
                       np.array([[2, 3, 5, 4]]),
                       np.array([[2, 3, 5, 4]]),
                       np.array([[2, 3, 5, 4]]),
                       np.array([[2, 3, 5]]),
                       np.array([[2, 3, 5, 4, 2, 3, 5, 4, 2, 3, 5, 4, 2, 3, 5, 7, 3, 4]])]
        rank = [4, 4, 4, 4, 4, 4, 3, 3, 5, 2, 4, 4, 8]
        expectations = [0.77777, 0.80555, 0.75, 0.0, 0.91666, 1.0, 1.0, 0.66666, 0.91666, 1.0, 0.0, 0.83333, 0.61339]

        assert len(targets) == len(predictions) == len(rank) == len(expectations)

        for preds, tgts, exps, k in zip(predictions, targets, expectations, rank):
            evaluator = MeanAveragePrecisionAtK(k)
            evaluator.add_predictions(preds, tgts)
            self.assertAlmostEqual(evaluator.get_report()[f"map_at_{k}"], exps, places=4)
