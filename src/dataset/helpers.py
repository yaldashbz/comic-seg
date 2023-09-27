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


def decode_rle_segmentation(rle_segmentation, cropped_box):
    cropped_mask = mask_utils.decode(rle_segmentation)
    cropped_box = list(map(int, cropped_box))
    cropped_mask = cropped_mask[
        cropped_box[1]:cropped_box[1] + cropped_box[3], 
        cropped_box[0]:cropped_box[0] + cropped_box[2]
    ]
    cropped_rle_segmentation = mask_utils.encode(np.asfortranarray(cropped_mask.astype(np.uint8)))

    return cropped_rle_segmentation


def crop_annotations(annotations, cropped_box):
    # Don't use for now.
    cropped_annotations = []
    cropped_x_min, cropped_y_min, cropped_x_max, cropped_y_max = cropped_box

    for annotation in annotations:
        bbox = annotation['bbox']
        if annotation['bbox_mode'] == BoxMode.XYWH_ABS:
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        segmentation = annotation['segmentation']

        # Calculate the intersection between the bbox and cropped_box
        x_min = max(bbox[0], cropped_x_min)
        y_min = max(bbox[1], cropped_y_min)
        x_max = min(bbox[2], cropped_x_max)
        y_max = min(bbox[3], cropped_y_max)

        # Check if there is an intersection
        if x_min < x_max and y_min < y_max:
            # Calculate the new cropped bbox coordinates
            cropped_bbox = [
                x_min - cropped_x_min,
                y_min - cropped_y_min,
                x_max - x_min,
                y_max - y_min
            ]

            # Calculate the new cropped segmentation coordinates
            cropped_segmentation = []
            if isinstance(segmentation, dict) and 'size' in segmentation and 'counts' in segmentation:
                cropped_rle_segmentation = decode_rle_segmentation(segmentation, cropped_bbox)
                if cropped_rle_segmentation is not None:
                    cropped_segmentation.append(cropped_rle_segmentation)
            else:
                for segment in segmentation:
                    cropped_segment = [
                        [point[0] - cropped_x_min, point[1] - cropped_y_min]
                        for point in zip(segment[::2], segment[1::2])
                        if (
                            cropped_x_min <= point[0] <= cropped_x_max + 100
                            and cropped_y_min <= point[1] <= cropped_y_max + 100
                        )
                    ]
                    if cropped_segment:
                        cropped_segmentation.append(cropped_segment)

            # Create a new annotation with the cropped bbox and segmentation
            cropped_annotation = {
                'iscrowd': annotation['iscrowd'],
                'bbox': cropped_bbox,
                'category_id': annotation['category_id'],
                'segmentation': cropped_segmentation,
                'bbox_mode': BoxMode.XYWH_ABS
            }

            cropped_annotations.append(cropped_annotation)

    return cropped_annotations


def mask_to_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_list = [contour.flatten().tolist() for contour in contours]
    return contours_list


def convert_rle_mask_to_coords(rle_mask):
    binary_mask = mask_utils.decode(rle_mask)
def mask_to_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_list = [contour.flatten().tolist() for contour in contours]
    return contours_list


def convert_rle_mask_to_coords(rle_mask):
    binary_mask = mask_utils.decode(rle_mask)
    # Convert the binary mask to a list of [x, y] coordinates
    return mask_to_contours(binary_mask)
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
