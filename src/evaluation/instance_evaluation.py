import numpy as np
import pycocotools.mask as mask_util

from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from src.dataset.register_comic_instance import COMIC_CATEGORY2CLASS_ID
from src.dataset.coco2comic import COCO2COMIC
from .utils import *


class ComicInstanceEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.metadata = MetadataCatalog.get(dataset_name)
        self.num_classes = len(self.metadata.thing_classes)
        coco_train_metadata = MetadataCatalog.get("coco_2017_train")
        self.coco_category_names = coco_train_metadata.thing_classes
        self.comic_category2class_id = {k: v - 1 for k, v in COMIC_CATEGORY2CLASS_ID.items()}
        self.iou_per_class = {class_id: [] for class_id in self.metadata.get("thing_classes")}
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def process(self, inputs, outputs, **kwargs):
        for input_, output in zip(inputs, outputs):
            pred_classes = output["instances"].pred_classes.cpu().numpy()
            pred_masks = output["instances"].pred_masks.cpu().numpy()
            annotations = input_["annotations"]
            height = input_["height"]
            width = input_["width"]
            image_id = input_["image_id"]

            for pred_class, pred_mask in zip(pred_classes, pred_masks):
                self._update_confusion_matrix(
                    pred_mask, pred_class, annotations, height, width, image_id, **kwargs
                )

    def _update_confusion_matrix(
        self, pred_mask, pred_class, annotations, height, width, image_id, **kwargs
    ):
        comic_pred_category = COCO2COMIC[self.coco_category_names[pred_class]]
        pred_class = self.comic_category2class_id[comic_pred_category]
        cropped_box = kwargs.get("cropped_box", None)

        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id != pred_class:
                continue
            
            gt_mask = annotation["segmentation"]
            try:
                target_mask = coco_segmentation_to_mask(gt_mask, height, width)
            except ValueError:
                print(f"Problem in annotations image_id = {image_id} in class {comic_pred_category}!")
                continue
            
            if cropped_box is not None:
                target_mask = np.array(Image.fromarray(target_mask).crop(cropped_box))

            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            self.confusion_matrix[pred_class, category_id] += intersection
            self.confusion_matrix[pred_class, :] += np.sum(pred_mask, dtype=np.int64)
            self.confusion_matrix[:, category_id] += np.sum(target_mask, dtype=np.int64)
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

        return dict(segm={
            "mean_iou": mean_iou,
            **{self.metadata.thing_classes[class_id]: iou 
               for class_id, iou in dict(enumerate(ious)).items()}
        })
