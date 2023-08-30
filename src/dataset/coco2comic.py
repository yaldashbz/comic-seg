sinergia_specific_classes = [
    'Building', 'Background', 'Face', 'Hand', 'Horizon', 'Panel',
    'Comic Bubble', 'Text', 'Plant'
]

comic2coco = {
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
    comic2coco.update({comic: []})

COCO2COMIC = {}
for k, v in comic2coco.items():
    if isinstance(v, list):
        for item in v:
            COCO2COMIC[item] = k
