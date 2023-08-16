import json
import numpy as np
import pycocotools.mask as mask_util


def open_json(file_name):
    f = open(file_name, 'r')
    json_obj = json.load(f)
    f.close()
    return json_obj


def save_json(file_name, json_obj):
    f = open(file_name, 'w')
    json.dump(json_obj, f)
    f.close()


def convert_list_counts_to_seg_list(mask):
    mask_size = mask['size']
    counts = mask['counts']
    counts[0] == counts[0] % 10000

    # Initialize variables
    coords = []
    current_x = 0
    current_y = 0

    # Process the counts to create the list of coordinates
    for i in range(0, len(counts), 2):
        run_length = counts[i]
        
        for _ in range(run_length):
            coords.append([current_x, current_y])
            
            current_x += 1
            if current_x == mask_size[0]:
                current_x = 0
                current_y += 1
    
    return coords
