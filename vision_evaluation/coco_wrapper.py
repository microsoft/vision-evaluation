from pycocotools.coco import COCO


class COCOWrapper:
    """convert the data to coco format http://mscoco.org/dataset/#format in order to use coco_eval
    """
    @staticmethod
    def convert(gt_or_pred, phase='prediction'):
        """
        convert ground-truth or predictions to coco format
        """
        anno = []
        count = 1
        category_set = set()
        img_list = []
        for img_idx, bboxes in enumerate(gt_or_pred):
            img_list.append({'id': img_idx})
            for bbox in bboxes:
                coco_bbox = [bbox[-4], bbox[-3], bbox[-2]-bbox[-4], bbox[-1]-bbox[-3]]
                if phase == 'prediction':
                    item = {
                        'id': count,
                        'image_id': img_idx,
                        'category_id': bbox[0],
                        'bbox': coco_bbox,
                        'area': coco_bbox[-2] * coco_bbox[-1],
                        'iscrowd': 0,
                        'score': bbox[1]
                    }
                elif phase == "gt":
                    item = {
                        'id': count,
                        'image_id': img_idx,
                        'category_id': bbox[0],
                        'bbox': coco_bbox,
                        'area': coco_bbox[-2] * coco_bbox[-1],
                        'iscrowd': 0 if len(bbox) == 5 else bbox[1]
                    }
                else:
                    raise Exception('not supported annotation')
                category_set.add(bbox[0])
                count += 1
                anno.append(item)

        results = COCO()
        results.dataset['annotations'] = anno
        results.dataset['images'] = img_list
        results.dataset['categories'] = [{'id': idx} for idx in category_set]
        results.createIndex()

        return results
