import os
import sys
import cv2
import matplotlib.pyplot as plt


# import some common detectron2 utilities
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config


def get_predictor():
    from Mask2Former.mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("./Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 28
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 28
    cfg.MODEL.WEIGHTS = '/sinergia/shabanza/outputs/coco/model_final.pth'

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    predictor = ComicPredictor(cfg)
    return predictor


def cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input')
    parser.add_argument('--output-dir', default='./outputs')
    parser.add_argument('--eval', action='store_true', default=True)
    args = parser.parse_args()
    return args


def predict(predictor, img_path):
    im = cv2.imread(img_path)
    outputs = predictor(im)
    metadata = MetadataCatalog.get(DATASET_NAME)
    instance_pred = get_instance_pred(im, outputs, metadata)
    semantic_pred = get_semantic_pred(im, outputs, metadata)
    return instance_pred, semantic_pred


def main():
    args = cli()
    predictor = get_predictor()
    instance_pred, semantic_pred = predict(predictor, args.input)
    
    output_path = args.output_dir
    if not os.path.exists(output_path):
        os.mkdir(args.output_dir)
    
    plt.imsave(os.path.join(output_path, f'instance_{args.input.split("/")[-1]}'), instance_pred)
    plt.imsave(os.path.join(output_path, f'semantic_{args.input.split("/")[-1]}'), semantic_pred)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'Mask2Former'))
    
    from src.train.mask2former_trainer import ComicPredictor
    from src.dataset.helpers import get_instance_pred, get_semantic_pred
    from src.dataset.register_comic_instance import DATASET_NAME


    main()
