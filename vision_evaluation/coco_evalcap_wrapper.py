from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class ImageCaptionCOCOEvalCaption(COCOEvalCap):
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
            raise Exception('Not supported image caption metric. Supported metric list: [Bleu, METEOR, ROUGE_L, CIDEr, SPICE]')

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
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        self.setEvalImgs()
