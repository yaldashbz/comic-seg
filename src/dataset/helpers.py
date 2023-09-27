import os
import cv2
import numpy as np
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures.boxes import BoxMode
from detectron2.data import DatasetCatalog
import pycocotools.mask as mask_utils

from .utils import *


class EvalType(Enum):
    COMIC_INSTANCE = 'comic_instance'
    COMIC_SEM_SEG = 'comic_sem_seg'
    COCO = 'coco'


def visualize_sample_anns(sample, category_id=None):
    image = Image.open(sample["file_name"])
    annotations = sample["annotations"]

    # Draw the annotations on the image
    if category_id is not None:
        annotations = filter(lambda x: x['category_id'] == category_id, annotations)
    for annotation in annotations:
        segment = annotation["segmentation"]
        if isinstance(segment, dict):
            segment = convert_rle_mask_to_coords(segment)
        for polygon in segment:
            polygon = [(x, y) for x, y in zip(polygon[0::2], polygon[1::2])]
            polygon = [tuple(map(int, point)) for point in polygon]
            draw = ImageDraw.Draw(image)
            draw.polygon(polygon, outline='green', width=2)

    # Display the image
    image.show()


def visualize_dataset_dict(sample, metadata):
    im = cv2.imread(sample["file_name"])
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    r = v.draw_dataset_dict(sample).get_image()
    plt.figure(figsize=(30,30))
    plt.axis('off')
    plt.imshow(r)


def get_instance_pred(im, outputs, metadata):
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    return instance_result


def get_semantic_pred(im, outputs, metadata):
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
    return semantic_result
    

def show_seg_predictions(im, outputs, metadata, axis=1):
    """
    Show panoptic/instance/semantic predictions
    """
    
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
    print("Panoptic segmentation (top), instance segmentation (middle), semantic segmentation (bottom)")
    res = np.concatenate((panoptic_result, instance_result, semantic_result), axis=axis)[:, :, ::-1]
    plt.figure(figsize=(40, 30))
    plt.imshow(res)
    plt.axis('off')
    plt.show()


def extract_panel(image, segmentation):
    mask = np.zeros_like(image[:, :, 0])
    segments = [np.array(segment).reshape((-1, 2)).astype(np.int32) for segment in segmentation]
    cv2.fillPoly(mask, segments, 255)
    panel = cv2.bitwise_and(image, image, mask=mask)
    return panel


def get_panels(sample):
    """
    Return panels and cropped_boxes, in XYXY format
    """
    panels = []
    cropped_boxes = []
    pil_im = Image.open(sample['file_name'])

    for ann in sample['annotations']:
        if ann['category_id'] == 24: # panel
            bbox_mode = ann['bbox_mode']
            if bbox_mode == BoxMode.XYXY_ABS:
                cropped_box = ann['bbox']
            if bbox_mode == BoxMode.XYWH_ABS:
                x, y, w, h = ann['bbox']
                cropped_box = (x, y, x + w, y + h)
            cropped_image = np.array(pil_im.crop(cropped_box))
            panels.append(cropped_image)
            cropped_boxes.append(cropped_box)
    return panels, cropped_boxes


def get_all_panels_dataset_dicts(dataset_dicts, save=False):
    new_dataset_dicts = []
    for sample in tqdm(dataset_dicts):
        panels, cropped_boxes = get_panels(sample)
        
        assert len(panels) == len(cropped_boxes)
        
        page_file_name = sample['file_name']
        image_id = sample['image_id']
        for i, (panel, cropped_box) in enumerate(zip(panels, cropped_boxes)):

            panel_file_name = f'{page_file_name.split(".png")[0]}_panel_{i}.png'
            if save and not os.path.exists(panel_file_name):
                cv2.imwrite(panel_file_name, panel)
            
            new_sample = {
                'file_name': panel_file_name if save else page_file_name,
                'height': panel.shape[1],
                'width': panel.shape[0],
                'image_id': int(f'{image_id}{i}'),
                'annotations': sample['annotations'],
                'panel': panel,
                'cropped_box': cropped_box,
                'cropped_box_mode': BoxMode.XYXY_ABS
            }

            new_dataset_dicts.append(new_sample)

    return new_dataset_dicts


def mask_to_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_list = [contour.flatten().tolist() for contour in contours]
    return contours_list


def convert_rle_mask_to_coords(rle_mask):
    binary_mask = mask_utils.decode(rle_mask)
    # Convert the binary mask to a list of [x, y] coordinates
    return mask_to_contours(binary_mask)


def compute_pixel_mean_std(dataset_dicts):
    pixel_values = []
    for dataset_dict in tqdm(dataset_dicts):
        image = cv2.imread(dataset_dict['file_name'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixel_values.extend(image.reshape(-1, 3))
    return np.mean(pixel_values, axis=0), np.std(pixel_values, axis=0)


def get_min_max_sizes(dataset_name):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    min_size = min([min(data["height"], data["width"]) for data in dataset_dicts])
    max_size = max([max(data["height"], data["width"]) for data in dataset_dicts])
    return min_size, max_size
