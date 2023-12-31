import os
import copy
import itertools
import logging
import torch
from collections import OrderedDict
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import detectron2.data.transforms as T
from detectron2.projects.deeplab.build_solver import build_lr_scheduler
from detectron2.evaluation import DatasetEvaluators
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import (
    build_detection_test_loader, 
    build_detection_train_loader,
    MetadataCatalog
)
from detectron2.data import MetadataCatalog
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from src.evaluation import ComicInstanceEvaluator, ComicSemanticEvaluator
from src.dataset.dataset_mapper import ComicDatasetMapper
from src.dataset.helpers import EvalType
from src.train.utils import freeze_mask2former

from Mask2Former.mask2former.test_time_augmentation import SemanticSegmentorWithTTA


class ComicPredictor(DefaultPredictor):
    def __init__(self, cfg, weights_path=None):
        if weights_path: cfg.MODEL.WEIGHTS = weights_path
        super().__init__(cfg)
        self.aug = T.NoOpTransform()

    def __call__(self, original_image):
         with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class Mask2FormerComicTrainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        freeze_mask2former(model, mode=cfg.FN_MODE)
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

        if evaluator_type == EvalType.COCO.value:
            evaluator_list.append(COCOEvaluator(dataset_name))

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

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
