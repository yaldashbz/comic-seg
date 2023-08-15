import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from detectron2.utils.visualizer import Visualizer, ColorMode


def visualize_sample_anns(sample, category_id=None):
    image = Image.open(sample["file_name"])
    annotations = sample["annotations"]

    # Draw the annotations on the image
    if category_id is not None:
        annotations = filter(lambda x: x['category_id'] == category_id, annotations)
    for annotation in annotations:
        segment = annotation["segmentation"]
        polygons = [segment[0]]
        for polygon in polygons:
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


def show_seg_predictions(im, outputs, metadata):
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
    res = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]
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
    panels = []
    cropped_boxes = []
    for ann in sample['annotations']:
        if ann['category_id'] == 24: # panel
            pil_im = Image.open(sample['file_name'])
            x, y, w, h = ann['bbox']
            cropped_image = np.array(pil_im.crop((x, y, x + w, y + h)))
            panels.append(cropped_image)
            cropped_boxes.append((x, y, x + w, y + h))
    return panels, cropped_boxes
