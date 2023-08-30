import numpy as np
import pycocotools.mask as mask_util

from src.dataset.helpers import convert_rle_mask_to_list


def coco_segmentation_to_mask(segmentation, height, width):
    """
    Function to convert coco format segmentation to binary masks
    """
    rle = mask_util.frPyObjects(segmentation, height, width)
    binary_mask = mask_util.decode(rle)
    binary_mask = np.any(binary_mask, axis=2).astype(np.uint8)
    return binary_mask


def compute_iou(predicted_mask, target_mask):
    """
    Function to compute IoU between two binary masks
    """
    intersection = np.logical_and(predicted_mask, target_mask).sum()
    union = np.logical_or(predicted_mask, target_mask).sum()
    iou = intersection / (union + 1e-6)  # Add a small epsilon to avoid division by zero
    return iou
