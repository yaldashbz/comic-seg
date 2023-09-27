from detectron2.data import MetadataCatalog

from src.dataset import DATASET_NAME, COMIC_CLASS_ID2CATEGORY

sinergia_specific_classes = [
    'Building', 'Background', 'Face', 'Hand', 'Horizon', 'Panel',
    'Comic Bubble', 'Text', 'Plant'
]

comic2coco_names = {
    'Character': ['person'],
    'Car': ['car', 'truck'],
    'Motorbike': ['motorcycle'],
    'Bus': ['bus'],
    'Airplane': ['airplane'],
    'Bicycle': ['bicycle'],
    'Boat': ['boat'],
    'Train': ['train'],
    'Cat': ['cat'],
    'Dog': ['dog'], 
    'Cow': ['cow'], 
    'Horse': ['horse'],
    'Sheep': ['sheep'],
    'Bird': ['bird'],
    'Chair': ['chair', 'bench'],
    'Sofa': ['couch'],
    'Table': ['dining table'],
    'Generic Animal': ['elephant', 'bear', 'zebra', 'giraffe'],
    'Generic Object': [
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant',
        'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
}

for comic in sinergia_specific_classes:
    comic2coco_names.update({comic: []})


COCO2COMIC = {}
for k, v in comic2coco_names.items():
    if isinstance(v, list):
        for item in v:
            COCO2COMIC[item] = k


COMIC_METADATA = MetadataCatalog.get(DATASET_NAME)
COMIC_CID2DID = {v: k for k, v in COMIC_METADATA.thing_dataset_id_to_contiguous_id.items()}
COCO_METADATA = MetadataCatalog.get("coco_2017_train")


def comic2coco(comic_cid):
    """
    cid is the id which is in detectron dataset
    (contiguous id)
    """
    dataset_id = COMIC_CID2DID[comic_cid]
    comic_label = COMIC_CLASS_ID2CATEGORY[dataset_id]
    coco_label = comic2coco_names[comic_label][0]
    return COCO_METADATA.thing_classes.index(coco_label)
