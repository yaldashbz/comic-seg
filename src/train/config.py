import warnings

from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from Mask2Former.mask2former import add_maskformer2_config

from src.dataset import (
    NAME_MAPPER, 
    register_train_test, 
    register_panels,
    get_min_max_sizes
)

def mask2former_setup(args):
    dataset_name = NAME_MAPPER[args.data_mode]
    print(dataset_name)
    dataset_train_name, dataset_test_name = register_train_test(
        dataset_name, 
        args.test_size, 
        args.random_state,
        eval_type=args.eval_type
    )
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # model
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
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.OPTIMIZER = 'SGD'
    
    # train
    cfg.FN_MODE = args.fn_mode
    
    warnings.filterwarnings('ignore')
    if args.cropped:
        dataset_train_name = register_panels(dataset_train_name, 'train', eval_type=args.eval_type)
        dataset_test_name = register_panels(dataset_test_name, 'test', eval_type=args.eval_type)
    warnings.resetwarnings()
        
    cfg.DATASETS.TRAIN = (dataset_train_name, )
    cfg.DATASETS.TEST = (dataset_test_name, )
    cfg.KEEP_CLASS_IDS = args.keep_class_ids
    
    epochs = args.epochs
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    train_size = len(DatasetCatalog.get(dataset_train_name))
    one_epoch = train_size / (batch_size * num_gpus)
    
    # for logging
    cfg.ONE_EPOCH = int(one_epoch)
    cfg.TEST.EVAL_PERIOD = cfg.ONE_EPOCH
    # solver
    cfg.SOLVER.MAX_ITER = int(one_epoch * epochs)
    print('LR is: ', cfg.SOLVER.BASE_LR)
    print('MAX ITER is: ', cfg.SOLVER.MAX_ITER)
    print('STEPS is: ', cfg.SOLVER.STEPS)
    return cfg


def deeplab_setup(args):
    dataset_name = NAME_MAPPER[args.data_mode]
    print(dataset_name)
    dataset_train_name, dataset_test_name = register_train_test(
        dataset_name, 
        args.test_size, 
        args.random_state,
        eval_type=args.eval_type
    )
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 28
    cfg.MODEL.PIXEL_MEAN = [184.70014834, 158.68679797, 118.3750071]
    cfg.MODEL.PIXEL_STD = [45.54069698, 40.70228227, 40.9410987]
    # optimizer
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    
    # train
    cfg.FN_MODE = args.fn_mode
    
    warnings.filterwarnings('ignore')
    if args.cropped:
        dataset_train_name = register_panels(dataset_train_name, 'train', eval_type=args.eval_type)
        dataset_test_name = register_panels(dataset_test_name, 'test', eval_type=args.eval_type)
    warnings.resetwarnings()
        
    cfg.DATASETS.TRAIN = (dataset_train_name, )
    cfg.DATASETS.TEST = (dataset_test_name, )
    print('dataset train name: ', dataset_train_name)
    cfg.INPUT.CROP.SIZE = (128, 128)
    
    cfg.KEEP_CLASS_IDS = args.keep_class_ids
    
    epochs = args.epochs
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    train_size = len(DatasetCatalog.get(dataset_train_name))
    one_epoch = train_size / (batch_size * num_gpus)
    
    # for logging
    cfg.ONE_EPOCH = int(one_epoch)
    cfg.TEST.EVAL_PERIOD = cfg.ONE_EPOCH
    # solver
    cfg.SOLVER.MAX_ITER = int(one_epoch * epochs)
    print('LR is: ', cfg.SOLVER.BASE_LR)
    print('MAX ITER is: ', cfg.SOLVER.MAX_ITER)
    print('STEPS is: ', cfg.SOLVER.STEPS)
    return cfg
