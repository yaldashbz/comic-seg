import os
from typing import Optional
import cv2
import copy
import torch
import numpy as np
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures.boxes import BoxMode

from src.dataset.helpers import get_panels

transform_list = [
    T.RandomRotation(angle=(-10, 10)),  # Randomly rotate the image by -10 to +10 degrees
    T.RandomBrightness(0.8, 1.8),
    T.RandomContrast(0.6, 1.3),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
]


def comic_mapper_panel_wise(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    panels, cropped_boxes = get_panels(dataset_dict)
    assert len(panels) == len(cropped_boxes)
    image_id = dataset_dict['image_id']
    new_dataset_dicts = []
    for i, (_, cropped_box) in enumerate(zip(panels, cropped_boxes)):
        image = utils.read_image(dataset_dict["file_name"], format="RGB")
        box = BoxMode.convert(cropped_box, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        box = list(map(int, cropped_box))
        image, transforms = T.apply_transform_gens(
            [T.CropTransform(*box), *transform_list], 
            image
        )
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        instances = utils.filter_empty_instances(instances)
        new_sample = {
            'file_name': dataset_dict["file_name"],
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
            'height': image.shape[1],
            'width': image.shape[0],
            'image_id': int(f'{image_id}{i}'),
            'annotations': annos,
            'instances': instances
        }
        new_dataset_dicts.append(new_sample)
    
    return new_dataset_dicts


def comic_mapper_page_wise(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="RGB")
    image, transforms = T.apply_transform_gens(transform_list, image)
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    dataset_dict["annotations"] = annos
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return dataset_dict
