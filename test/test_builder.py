import unittest
from visionmetrics import EvaluatorBuilder


class TestEvaluatorBuilder(unittest.TestCase):
    def test_classification_multiclass(self):
        builder = EvaluatorBuilder('classification_multiclass')
        top1_accuracy = builder.add('topk_accuracy', k=1)
        top5_accuracy = builder.add('topk_accuracy', k=5)
        self.assertNotEqual(top1_accuracy, top5_accuracy)
        evaluator = builder.create()

        report = evaluator.get_report()
        self.assertIn(top1_accuracy, report)
        self.assertIn(top5_accuracy, report)

    def test_classification_multilabel(self):
        builder = EvaluatorBuilder('classification_multilabel')
        th05_accuracy = builder.add('threshold_accuracy', threshold=0.5)
        th08_accuracy = builder.add('threshold_accuracy', threshold=0.8)
        self.assertNotEqual(th05_accuracy, th08_accuracy)
        evaluator = builder.create()

        report = evaluator.get_report()
        self.assertIn(th05_accuracy, report)
        self.assertIn(th08_accuracy, report)

    def test_classification_objectdetection(self):
        builder = EvaluatorBuilder('object_detection')
        coco_05 = builder.add('coco_mean_average_precision', iou=0.5)
        coco_03 = builder.add('coco_mean_average_precision', iou=0.3)
        self.assertNotEqual(coco_05, coco_03)
        evaluator = builder.create()

        report = evaluator.get_report()

        self.assertIn(coco_05, report)
        self.assertIn(coco_03, report)

    def test_get_names(self):
        names = EvaluatorBuilder('classification_multiclass').get_available_metrics_names()
        self.assertTrue(names)
