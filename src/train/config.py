from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

from src.dataset import (
    NAME_MAPPER, 
    register_train_test, 
    register_panels
)


def _base_setup():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl'

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    cfg.SOLVER.OPTIMIZER = 'SGD'
    # cfg.OUTPUT_DIR = '/sinergia/shabanza/outputs/'
    cfg.OUTPUT_DIR = '/home/yalda/IVRL_backup/shabanza_sinergia/outputs/'

    cfg.MODEL.PIXEL_MEAN = [184.70014834, 158.68679797, 118.3750071]
    cfg.MODEL.PIXEL_STD = [45.54069698, 40.70228227, 40.9410987]
    
    return cfg

def setup(data_mode, cropped=True, test_size=0.2, random_state=42):
    dataset_name = NAME_MAPPER[data_mode]
    print(dataset_name)
    dataset_train_name, dataset_test_name = register_train_test(
        dataset_name, 
        test_size, 
        random_state
    )
    cfg = _base_setup()
    
    if cropped:
        dataset_train_name = register_panels(dataset_train_name, 'train')
        dataset_test_name = register_panels(dataset_test_name, 'test')
        
    cfg.DATASETS.TRAIN = (dataset_train_name, )
    cfg.DATASETS.TEST = (dataset_test_name, )
    return cfg
