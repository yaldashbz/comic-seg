import warnings

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from Mask2Former.mask2former import add_maskformer2_config

from src.dataset import (
    NAME_MAPPER, 
    register_train_test, 
    register_panels,
    get_min_max_sizes
)


def base_setup(**kwargs):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    
    # model
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 28
    cfg.MODEL.PIXEL_MEAN = [184.70014834, 158.68679797, 118.3750071]
    cfg.MODEL.PIXEL_STD = [45.54069698, 40.70228227, 40.9410987]

    # data
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # optimizer
    cfg.SOLVER.IMS_PER_BATCH = kwargs.get('batch_size', 1)
    cfg.SOLVER.BASE_LR = kwargs.get('lr', 0.0001)
    cfg.SOLVER.OPTIMIZER = 'SGD'
    # cfg.OUTPUT_DIR = '/sinergia/shabanza/outputs/'
    cfg.OUTPUT_DIR = '/home/yalda/IVRL_backup/shabanza_sinergia/outputs/'
    
    return cfg


def setup(args):
    dataset_name = NAME_MAPPER[args.data_mode]
    print(dataset_name)
    dataset_train_name, dataset_test_name = register_train_test(
        dataset_name, 
        args.test_size, 
        args.random_state,
        eval_type=args.eval_type
    )
    cfg = base_setup(batch_size=args.batch_size, lr=args.lr)
    
    warnings.filterwarnings('ignore')
    if args.cropped:
        dataset_train_name = register_panels(dataset_train_name, 'train', eval_type=args.eval_type)
        dataset_test_name = register_panels(dataset_test_name, 'test', eval_type=args.eval_type)
    warnings.resetwarnings()
        
    cfg.DATASETS.TRAIN = (dataset_train_name, )
    cfg.DATASETS.TEST = (dataset_test_name, )
    min_size, max_size = get_min_max_sizes(dataset_train_name)
    cfg.INPUT.MIN_SIZE_TRAIN = min_size
    cfg.INPUT.MAX_SIZE_TRAIN = max_size
    return cfg
