import os

from tqdm import tqdm
from copy import deepcopy
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data import MetadataCatalog

from .utils import *


DATASET_NAME = 'sinergia_comic_instances'
PLACID_NAME = 'sinergia_placid_instances'
YVES_NAME = 'sinergia_yves_instances'
SINERGIA_ROOT = "/sinergia/shabanza/datasets/sinergia"
SINERGIA_IMG_DIR = os.path.join(SINERGIA_ROOT, "images", "default")
SINERGIA_INSTANCES_FILE = os.path.join(SINERGIA_ROOT, "coco", "annotations", "instances_default.json")
SINERGIA_INSTANCES_MODIFIED_FILE = os.path.join(SINERGIA_ROOT, "coco", "annotations", "instances_default_modified.json")
PLACID_INSTANCES_FILE = os.path.join(SINERGIA_ROOT, "coco", "annotations", "instances_placid.json")
YVES_INSTANCES_FILE = os.path.join(SINERGIA_ROOT, "coco", "annotations", "instances_yves.json")
LABEL_COLOR_FILE = os.path.join(SINERGIA_ROOT, 'label_colors.txt')
COMIC_STUFF_CATEGORIES = ['Horizon', 'Background', 'Comic Bubble'] # TODO for later (ask)


def extract_categories(label_colors_file, categories):
    category2color = {}
    with open(label_colors_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            color = list(map(int, line[:3]))
            category2color[' '.join(line[3:])] = color

    class_id2color = {}
    class_id2category = {}
    for category in categories:
        class_id2category[category['id']] = category['name']
        class_id2color[category['id']] = category2color[category['name']]
    
    return class_id2color, class_id2category


def _update_categories(label_colors_file, categories, fill_isthing=False):
    """
    Add color and isthing to the data
    """
    class_id2color, _ = extract_categories(LABEL_COLOR_FILE, categories)
    class_id2color, _ = extract_categories(label_colors_file, categories)
    for category in categories:
        category['color'] = class_id2color[category['id']]
        if fill_isthing:
            category['isthing'] = 0 if category in COMIC_STUFF_CATEGORIES else 1
    return categories


def _fix_segmentations(annotations):
    """
    Fill segmentations which don't have segmentation key (using bbox)
    Fix segmentations which are in dict format {'size': ..., 'counts': ...}
    """
    for ann in tqdm(annotations):
        if 'segmentation' not in ann or len(ann['segmentation']) == 0:
            x, y, w, h = ann['bbox']
            ann['segmentation'] = [[x, y, x+w, y, x+w, y+h, x, y+h]]

        # if isinstance(ann['segmentation'], dict):
        #     ann['segmentation'] = convert_list_counts_to_seg_list(ann['segmentation'])


def _get_comic_meta(categories):
    thing_ids = [k["id"] for k in categories]
    assert len(thing_ids) == 28, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 27]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in categories]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def get_group_instances(instances, group):
    removed_ids = []
    images = []
    for image in instances['images']:
        if group not in image['file_name']:
            removed_ids.append(image['id'])
        else:
            images.append(image)
    
    annotations = []
    for ann in tqdm(instances['annotations']):
        if ann['image_id'] not in removed_ids:
            annotations.append(ann)
    
    new_instances = deepcopy(instances)
    new_instances['images'] = images
    new_instances['annotations'] = annotations
    return new_instances


def _register_group_instances(json_file, dataset_name, group_instances):
    if not os.path.exists(json_file):
        save_json(json_file, group_instances)

    register_coco_instances(
        dataset_name,
        _get_comic_meta(group_instances['categories']),
        json_file,
        SINERGIA_IMG_DIR
    )


def register_comic_instances():    
    if not os.path.exists(SINERGIA_INSTANCES_MODIFIED_FILE):
        # open un-modified dataset
        sinergia_instances = open_json(SINERGIA_INSTANCES_FILE)
        
        _update_categories(LABEL_COLOR_FILE, sinergia_instances['categories'])
        _fix_segmentations(sinergia_instances['annotations'])
        
        # save modified dataset
        save_json(SINERGIA_INSTANCES_MODIFIED_FILE, sinergia_instances)
    else:
        print("loading sinergia json ...")
        sinergia_instances = open_json(SINERGIA_INSTANCES_MODIFIED_FILE)
        print("Sinergia Json loaded.")

    register_coco_instances(
        DATASET_NAME,
        _get_comic_meta(sinergia_instances['categories']),
        SINERGIA_INSTANCES_MODIFIED_FILE,
        SINERGIA_IMG_DIR
    )
    

def register_placid_instances():
    assert DATASET_NAME in MetadataCatalog.list()
    
    sinergia_instances = open_json(SINERGIA_INSTANCES_MODIFIED_FILE)
    _register_group_instances(
        PLACID_INSTANCES_FILE, 
        PLACID_NAME,
        get_group_instances(sinergia_instances, 'placid')
    )
    

def register_yves_instances():
    assert DATASET_NAME in MetadataCatalog.list()

    sinergia_instances = open_json(SINERGIA_INSTANCES_MODIFIED_FILE)
    _register_group_instances(
        YVES_INSTANCES_FILE, 
        YVES_NAME,
        get_group_instances(sinergia_instances, 'yves')
    )


if DATASET_NAME not in MetadataCatalog.list():
    print("registering comic ...")
    register_comic_instances()


if PLACID_NAME not in MetadataCatalog.list():
    print("registering placid ...")
    register_placid_instances()


if YVES_NAME not in MetadataCatalog.list():
    print("registering yves ...")
    register_yves_instances()


COMIC_CATEGORIES = open_json(SINERGIA_INSTANCES_MODIFIED_FILE)['categories']
COMIC_CATEGORY2CLASS_ID = {k['name']: k['id'] for k in COMIC_CATEGORIES}
COMIC_CLASS_ID2CATEGORY = {v: k for k, v in COMIC_CATEGORY2CLASS_ID.items()}
