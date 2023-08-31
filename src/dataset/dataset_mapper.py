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


def panel_mapper(sample):
    """
    Map a dataset_dict to multiple samples of panels using CropTransform
    """
    panels, cropped_boxes = get_panels(sample)
    assert len(panels) == len(cropped_boxes)
    image_id = sample['image_id']
    new_dataset_dicts = []
    for i, cropped_box in enumerate(cropped_boxes):
        dataset_dict = copy.deepcopy(sample) # super important
        image = utils.read_image(dataset_dict["file_name"], format="RGB")
        box = BoxMode.convert(cropped_box, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        box = list(map(int, box))
        image, transforms = T.apply_transform_gens(
            [T.CropTransform(*box)], 
            image
        )
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]
        new_sample = {
            'image': image,
            'height': image.shape[1],
            'width': image.shape[0],
            'image_id': int(f'{image_id}{i}'),
            'annotations': [ann for ann in annos if len(ann['segmentation'])],
        }
        new_dataset_dicts.append(new_sample)
    
    return new_dataset_dicts


def image2tensor_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    if 'image' not in dataset_dict:
        image = utils.read_image(dataset_dict["file_name"], format="RGB")
    else:
        image = dataset_dict['image']
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return dataset_dict


def comic_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    if 'image' not in dataset_dict:
        image = utils.read_image(dataset_dict["file_name"], format="RGB")
    else:
        image = dataset_dict['image']
    image, transforms = T.apply_transform_gens(transform_list, image)
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    dataset_dict["annotations"] = [ann for ann in annos if len(ann['segmentation'])]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return dataset_dict
