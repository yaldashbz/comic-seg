#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import DatasetEvaluators
from detectron2.projects.deeplab.build_solver import build_lr_scheduler
from detectron2.data import (
    build_detection_test_loader, 
    build_detection_train_loader,
    MetadataCatalog
)

from src.evaluation import ComicInstanceEvaluator, ComicSemanticEvaluator
from src.dataset.dataset_mapper import ComicDatasetMapper
from src.dataset.helpers import EvalType
from src.train.utils import freeze_deeplab



def build_sem_seg_train_aug(cfg):
    augs = []
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    return augs


class DeepLabComicTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """
    
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        freeze_deeplab(model, mode=cfg.FN_MODE)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == EvalType.COMIC_INSTANCE.value:
            evaluator_list.append(ComicInstanceEvaluator(dataset_name))

        if evaluator_type == EvalType.COMIC_SEM_SEG.value:
            evaluator_list.append(ComicSemanticEvaluator(dataset_name))

        if len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        classes_to_keep = list(map(int, cfg.KEEP_CLASS_IDS)) if cfg.KEEP_CLASS_IDS else None
        return build_detection_train_loader(
            cfg, mapper=ComicDatasetMapper(cfg=cfg, is_train=True, max_size=500, classes_to_keep=classes_to_keep), 
            prefetch_factor=2
        )
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # dataset_name is same as cfg.DATASET.TEST[0]
        return build_detection_test_loader(
            cfg, dataset_name,
            mapper=ComicDatasetMapper(cfg=cfg, is_train=False)
        )

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)
