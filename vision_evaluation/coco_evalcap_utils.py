import json
from collections import defaultdict

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class ImageCaptionCOCO(COCO):
    def __init__(self, data_or_file=None):
        """
        Args:
            data_or_file: a dictionary data or a path to the json file containing the dictionary data
        """
        super().__init__()
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if data_or_file:
            if isinstance(data_or_file, dict):
                dataset = data_or_file
            else:
                with open(data_or_file, 'r') as f:
                    dataset = json.load(f)
            self.dataset = dataset
            self.createIndex()


class ImageCaptionCOCOEval(COCOEvalCap):
    """
    Evaluate on the image caption predictions using pycocoevalcap.
    """
    def __init__(self, coco, cocoRes, metric):
        super().__init__(coco, cocoRes)
        self.scores = []
        if metric == "Bleu":
            self.scores = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]
        elif metric == "METEOR":
            self.scores = [(Meteor(), "METEOR")]
        elif metric == "ROUGE_L":
            self.scores = [(Rouge(), "ROUGE_L")]
        elif metric == "CIDEr":
            self.scores = [(Cider(), "CIDEr")]
        elif metric == "SPICE":
            self.scores = [(Spice(), "SPICE")]
        else:
            raise ValueError(f'Not supported image caption metric: {metric}. Supported metric list: [Bleu, METEOR, ROUGE_L, CIDEr, SPICE]')

    def reset(self):
        self.scores = []

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        scorers = self.scores

        # Compute scores
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        self.setEvalImgs()


class ImageCaptionWrapper:
    """
    Convert the data to pycocoevalcap format in order to use pycocoevalcap
    """
    @staticmethod
    def convert(imcap_predictions, imcap_targets):
        predictions = []
        gts = {'annotations': [], 'images': []}
        pred_id = 1
        for index, pred in enumerate(imcap_predictions):
            pred_dict = {}
            pred_dict['image_id'] = index
            pred_dict['caption'] = pred
            predictions.append(pred_dict)

            image_dict = {}
            image_dict['id'] = index
            image_dict['file_name'] = f'tmp_file_name_{index}'
            gts['images'].append(image_dict)

            for gt in imcap_targets[index]:
                caption_dict = {}
                caption_dict['image_id'] = index
                caption_dict['caption'] = gt
                caption_dict['id'] = pred_id
                gts['annotations'].append(caption_dict)
                pred_id += 1

        return predictions, gts
