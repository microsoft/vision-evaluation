import unittest
import numpy as np
from visionmetrics import TopKAccuracy


class TestClassification(unittest.TestCase):
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
                eval = TopKAccuracy(top_k)
                eval.update(predictions, targets)
                top_k_acc = eval.compute()[f'accuracy_top{top_k}']
                import sklearn.metrics as sm
                if predictions.shape[1] == 2:
                    predictions = predictions[:, 1]
                self.assertAlmostEqual(sm.top_k_accuracy_score(targets, predictions, k=top_k), top_k_acc.numpy())
                self.assertAlmostEqual(top_k_acc.numpy(), gts[i][k_idx], places=5)
