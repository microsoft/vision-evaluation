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

    def __init__(self, k):
        """
        Args:
            k: k predictions with highest confidence
        """
        self.k = k

    def filter(self, predictions, return_mode):
        """ Return k class predictions with highest confidence.
        Args:
            predictions: predicted probabilities of N samples belonging to C classes. Shape (N, C)
            return_mode: can be 'indices' or 'vec', indicating whether return value is a set of class indices or 0-1 vector

        Returns:
            k labels with highest probabilities, for each sample
        """

        k = min(predictions.shape[1], self.k)
        top_k_pred_indices = np.argsort(-predictions, axis=1)[:, :k]
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
    def __init__(self, threshold):
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
