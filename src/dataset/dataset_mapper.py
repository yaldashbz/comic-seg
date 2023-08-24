import copy
import torch
import numpy as np
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils


transform_list = [
    T.RandomRotation(angle=(-10, 10)),  # Randomly rotate the image by -10 to +10 degrees
    T.RandomBrightness(0.8, 1.8),
    T.RandomContrast(0.6, 1.3),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
]


def comic_mapper(dataset_dict):
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
