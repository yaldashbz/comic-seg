from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

from src.dataset.register_train_test import *

def setup(args):
    # register train test
    print('registering train test dataset ...')
    new_train_name, new_test_name = register_train_test(
        args.dataset_name, 
        args.test_size, 
        args.random_state
    )
    
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl'

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

    cfg.DATASETS.TRAIN = (new_train_name, )
    cfg.DATASETS.TEST = (new_test_name, )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    cfg.SOLVER.OPTIMIZER = 'SGD'
    cfg.OUTPUT_DIR = '/sinergia/shabanza/outputs/'

    cfg.MODEL.PIXEL_MEAN = [184.70014834, 158.68679797, 118.3750071]
    cfg.MODEL.PIXEL_STD = [45.54069698, 40.70228227, 40.9410987]
    
    return cfg
