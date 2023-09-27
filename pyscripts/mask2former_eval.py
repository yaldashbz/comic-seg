# Some basic setup:
# Setup detectron2 logger
import os
import sys

sys.path.append(os.getcwd())
import json
from src.dataset.register_comic_instance import *
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")


# import some common detectron2 utilities
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# import Mask2Former project
from Mask2Former.mask2former import add_maskformer2_config



def get_model():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("./Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl'

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    predictor = DefaultPredictor(cfg)
    return predictor


def cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset-name', default='sinergia_comic_instances')
    parser.add_argument('--panel-wise', action='store_true', default=1)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    return args


def main():
    from src.evaluation.helpers import evaluate_page_wise, evaluate_panel_wise
    from src.evaluation.instance_evaluation import ComicInstanceEvaluator

    args = cli()
    dataset_name = args.dataset_name
    dataset_dicts = DatasetCatalog.get(dataset_name)
    predictor = get_model()
    evaluator = ComicInstanceEvaluator(dataset_name)
    
    print("Evaluation started ...")
    if args.panel_wise:
        print("Panel wise")
        evaluate_panel_wise(evaluator, predictor, dataset_dicts)
    else:
        print("Page wise")
        evaluate_page_wise(evaluator, predictor, dataset_dicts)

    result = evaluator.evaluate()
    f = open(args.output_dir, 'w')
    json.dumps(result, f)
    f.close()


if __name__ == '__main__':
    main()