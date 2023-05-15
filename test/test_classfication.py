import unittest
import torch
from visionmetrics import DoNothingMetric


class TestDoNothingMetric(unittest.TestCase):
    def test_simple(self):
        metric = DoNothingMetric()
        metric.update(torch.tensor(1.0), torch.tensor(1.0))
        self.assertEqual(metric.compute(), "DoNothingMetric")
