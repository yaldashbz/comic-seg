import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from src.dataset.register_comic_instance import COMIC_CLASS_ID2CATEGORY
from src.dataset.helpers import get_panels


def evaluate_page_wise(evaluator, predictor, dataset_dicts):
    def process_sample(sample):
        im = cv2.imread(sample['file_name'])
        outputs = predictor(im)
        evaluator.process([sample], [outputs])

    with ThreadPoolExecutor(max_workers=4) as executor:  # Set the number of threads as desired
        futures = [executor.submit(process_sample, sample) 
                   for sample in dataset_dicts]

    # Use tqdm to create a progress bar
    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            future.result()
            pbar.update(1)  # Update the progress bar


def evaluate_panel_wise(evaluator, predictor, dataset_dicts):
    def process_sample(sample):
        panels, cropped_boxes = get_panels(sample)
        for panel, cropped_box in zip(panels, cropped_boxes):
            outputs = predictor(panel)
            evaluator.process([sample], [outputs], cropped_box=cropped_box)
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # Set the number of threads as desired
        futures = [executor.submit(process_sample, sample) 
                   for sample in dataset_dicts]

    # Use tqdm to create a progress bar
    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            future.result()
            pbar.update(1)  # Update the progress bar  

    
def print_evaluator_result(evaluator):
    ious = evaluator._compute_iou()
    mean_iou = np.mean(ious)

    print("mean_iou", mean_iou)
    print("iou_per_class:")
    for class_id, iou in dict(enumerate(ious)).items():
        print(f"{COMIC_CLASS_ID2CATEGORY[class_id + 1]}: {iou}")
