from typing import Any, Dict
import copy
import torch
import numpy as np
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures.boxes import BoxMode
from detectron2.structures import BitMasks


from src.dataset.helpers import get_panels

TRAIN_TRANSFORM_LIST = [
    T.RandomRotation(angle=(-10, 10)),  # Randomly rotate the image by -10 to +10 degrees
    T.RandomBrightness(0.8, 1.8),
    T.RandomContrast(0.6, 1.3),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
]

TEST_TRANSFORM_LIST = [
    T.NoOpTransform()
]

TRANSFORM_LISTS = {
    'train': TRAIN_TRANSFORM_LIST,
    'test': TEST_TRANSFORM_LIST
}


def build_sem_seg_train_aug(cfg, augs):
    if cfg.INPUT.CROP.ENABLED:
        augs.extend([
            T.ResizeShortestEdge(
                short_edge_length=min(cfg.INPUT.CROP.SIZE)
            ),
            T.PadTransform(size=cfg.INPUT.CROP.SIZE, pad_value=0)
            # T.RandomCrop_CategoryAreaConstraint(
            #     cfg.INPUT.CROP.TYPE,
            #     cfg.INPUT.CROP.SIZE,
            #     cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
            #     cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            # )
        ])
    return augs


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
            'height': image.shape[0],
            'width': image.shape[1],
            'image_id': int(f'{image_id}{i}'),
            'annotations': [ann for ann in annos if len(ann['segmentation'])],
        }
        new_dataset_dicts.append(new_sample)
    
    return new_dataset_dicts


def convert_instances_to_sem_seg(instances, image_size):
    segmentation_masks = instances.gt_masks
    image_height, image_width = image_size
    bitmasks = BitMasks.from_polygon_masks(segmentation_masks, image_width, image_height)
    sem_seg = bitmasks.tensor.numpy()
    print(sem_seg.shape)
    
    # ignore panel
    sem_seg[sem_seg == 24] = float('-inf')
    
    if np.any(sem_seg):
        max_class_id = np.argmax(sem_seg, axis=0)

        # Create a mask of pixels where the maximum value is non-zero
        max_pixel_mask = np.max(sem_seg, axis=0) > 0

        # Build the semantic segmentation map using np.where
        seg_map = np.where(max_pixel_mask, max_class_id, 0)
    else:
        seg_map = np.zeros(image_size, dtype=np.int32)
    return torch.from_numpy(seg_map)


def comic_mapper(sample, transform_list, classes_to_keep=None, has_sem_seg=False):
    dataset_dict = copy.deepcopy(sample)
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
    if classes_to_keep:
        annos = [ann for ann in annos if ann['category_id'] in classes_to_keep]
    dataset_dict["annotations"] = [ann for ann in annos if len(ann['segmentation'])]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    instances = utils.filter_empty_instances(instances)
    if has_sem_seg:
        dataset_dict["sem_seg"] = convert_instances_to_sem_seg(instances, image.shape[:2])
    dataset_dict["instances"] = instances
    dataset_dict["image"] = image
    dataset_dict["height"] = image.shape[0]
    dataset_dict["width"] = image.shape[1]
    return dataset_dict


class ComicDatasetMapper:
    def __init__(self, cfg, is_train=True, max_size=None, classes_to_keep=None) -> None:
        self.cfg = cfg
        mode = 'train' if is_train else 'test'
        transform_list = TRANSFORM_LISTS[mode]
        self.max_size = max_size 
        self.classes_to_keep = classes_to_keep
        
        # took from deeplab build_train_loader code
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            self.max_size = None # to handle the resizing and padding in deeplab
            transform_list = build_sem_seg_train_aug(cfg, transform_list)
        
        self.transform_list = transform_list
    
    def __call__(self, dataset_dict: Dict) -> Any:
        has_sem_seg = "SemanticSegmentor" in self.cfg.MODEL.META_ARCHITECTURE
        dataset_dict = comic_mapper(
            dataset_dict, self.transform_list, self.classes_to_keep, has_sem_seg)
        if self.max_size:
            h, w = dataset_dict['height'], dataset_dict['width']
            transform_list = [T.ResizeShortestEdge(min(h, w, self.max_size), self.max_size)]
            dataset_dict = comic_mapper(dataset_dict, transform_list)
        image = dataset_dict['image']
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return dataset_dict
