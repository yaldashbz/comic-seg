import numpy as np

from tqdm import tqdm
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from .utils import *


class ComicSemanticEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.metadata = MetadataCatalog.get(dataset_name)
        self.num_classes = len(self.metadata.thing_classes)
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def process(self, inputs, outputs, **kwargs):
        for input_, output in zip(inputs, outputs):
            pred_mask = output["sem_seg"].argmax(dim=0).cpu().numpy()
            annotations = input_["annotations"]
            height = input_["height"]
            width = input_["width"]
            image_id = input_["image_id"]

            self._update_confusion_matrix(
                pred_mask, annotations, height, width, image_id, **kwargs
            )

    def _update_confusion_matrix(
        self, pred_mask, annotations, height, width, image_id, **kwargs
    ):
        cropped_box = kwargs.get("cropped_box", None)

        for annotation in tqdm(annotations):
            category_id = annotation["category_id"]
            gt_mask = annotation["segmentation"]
            try:
                target_mask = coco_segmentation_to_mask(gt_mask, height, width)
            except ValueError:
                print(f"Problem in annotations image_id = {image_id} in class {category_id}!")
                continue

            if cropped_box is not None:
                target_mask = np.array(Image.fromarray(target_mask).crop(cropped_box))

            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            self.confusion_matrix[category_id, category_id] += intersection
            self.confusion_matrix[category_id, :] += np.sum(pred_mask == category_id, axis=(0, 1))
            self.confusion_matrix[:, category_id] += np.sum(target_mask == category_id, axis=(0, 1))
            self.confusion_matrix[category_id, category_id] += union - intersection
    
    def _compute_iou(self):
        diagonal = np.diag(self.confusion_matrix)
        row_sum = np.sum(self.confusion_matrix, axis=1)
        col_sum = np.sum(self.confusion_matrix, axis=0)
        union = row_sum + col_sum - diagonal
        iou = np.divide(diagonal.astype(float), 
                        union.astype(float), 
                        out=np.zeros_like(diagonal, dtype=np.float64),
                        where=union != 0)
        return iou

    def evaluate(self):
        ious = self._compute_iou()
        mean_iou = np.mean(ious)

        return {
            "mean_iou": mean_iou,
            "iou_per_class": dict(enumerate(ious))
        }
