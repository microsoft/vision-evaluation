import numpy as np
from abc import ABC, abstractmethod


class PredictionFilter(ABC):
    """Filter predictions with a certain criteria
    """

    @abstractmethod
    def filter(self, predictions, return_mode):
        pass

    @property
    @abstractmethod
    def identifier(self):
        pass


class TopKPredictionFilter(PredictionFilter):

    def __init__(self, k: int, prediction_mode='prob'):
        """
        Args:
            k: k predictions with highest confidence
            prediction_mode: can be 'indices' or 'prob', indicating whether the predictions are a set of class indices or predicted probabilities.
        """
        assert k >= 0
        assert prediction_mode == 'prob' or prediction_mode == 'indices', f"Prediction mode {prediction_mode} is not supported!"

        self.prediction_mode = prediction_mode
        self.k = k

    def filter(self, predictions, return_mode):
        """ Return k class predictions with highest confidence.
        Args:
            predictions:
                when 'prediction_mode' is 'prob', refers to predicted probabilities of N samples belonging to C classes. Shape (N, C)
                when 'prediction_mode' is 'indices', refers to indices of M highest confidence of C classes in descending order, for each of the N samples. Shape (N, M)
            return_mode: can be 'indices' or 'vec', indicating whether return value is a set of class indices or 0-1 vector

        Returns:
            k labels with highest probabilities, for each sample
        """

        k = min(predictions.shape[1], self.k)

        if self.prediction_mode == 'prob':
            if k == 0:
                top_k_pred_indices = np.array([[] for i in range(predictions.shape[1])], dtype=int)
            elif k == 1:
                top_k_pred_indices = np.argmax(predictions, axis=1)
                top_k_pred_indices = top_k_pred_indices.reshape((-1, 1))
            else:
                top_k_pred_indices = np.argpartition(predictions, -k, axis=1)[:, -k:]
        else:
            top_k_pred_indices = predictions[:, :k]

        if return_mode == 'indices':
            return list(top_k_pred_indices)
        else:
            preds = np.zeros_like(predictions, dtype=bool)
            row_index = np.repeat(range(len(predictions)), k)
            col_index = top_k_pred_indices.reshape((1, -1))
            preds[row_index, col_index] = True

            return preds

    @property
    def identifier(self):
        return f'top{self.k}'


class ThresholdPredictionFilter(PredictionFilter):
    def __init__(self, threshold: float):
        """
        Args:
            threshold: confidence threshold
        """

        self.threshold = threshold

    def filter(self, predictions, return_mode):
        """ Return predictions over confidence over threshold
        Args:
            predictions: the model output numpy array. Shape (N, num_class)
            return_mode: can be 'indices' or 'vec', indicating whether return value is a set of class indices or 0-1 vector

        Returns:
            labels with probabilities over threshold, for each sample
        """
        if return_mode == 'indices':
            preds_over_thres = [[] for _ in range(len(predictions))]
            for indices in np.argwhere(predictions >= self.threshold):
                preds_over_thres[indices[0]].append(indices[1])

            return preds_over_thres
        else:
            return predictions >= self.threshold

    @property
    def identifier(self):
        return f'thres={self.threshold}'
