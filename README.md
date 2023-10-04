# comic-seg
Summer@EPFL IVRL 2023

State-of-the-art segmentation investigation (Train, Evaluate) on an old comic dataset.
- src: contains main classes and functions
- pyscripts: contatins python scripts for training, evaluating on datasets
- scripts: contains bash scripts for running pyscripts files

## How to run

Just fix the addresses, then put your bash script in 'scripts' folder.

```
sh scripts/mask2former_predict.sh
```

## Dataset Conversion
- If comic dataset is in cityscapes format, use convert_comic_to_coco_format.ipynb notebook, and then update the path of json_file in [src.dataset.register_comic_instances.py](https://github.com/yaldashbz/comic-seg/blob/main/convert_comic_to_coco_format.ipynb)
- If comic dataset is in coco format, all the src functions will be ok.

## Train/Fine-tune
- All available codes can be found in [src.train](https://github.com/yaldashbz/comic-seg/tree/main/src/train)

## Evaluation
- Both semantic & instance evaluation codes are in [src.evaluation](https://github.com/yaldashbz/comic-seg/tree/main/src/evaluation)

## Data Visualization
- Functions can be found in [src.dataset.helpers](https://github.com/yaldashbz/comic-seg/tree/main/src/dataset/helpers.py)

For more info, check available notebooks in the [notebooks](https://github.com/yaldashbz/comic-seg/tree/main/notebooks) directory.

## Results

Mean IoU on Placid comic:

### Mask2Former
Important Class | Character | Text | Comic Bubble
--- | --- | --- | ---
Pre-trained |  0.8980 | 0.9096 | 0.6771
Pre-trained (w/o sem_seg_head.class_embed) | 0.5235 | 0.2613 | 0.3351
Fine-tuned class-embed | 0.5803 | 0.5124 | 0.3844
Fine-tuned query-embeds | 0.5775 | 0.5083 | 0.3882
Fine-tuned decoder | 0.5711 | 0.3739 | 0.2643


### DeepLabV3
Important Class | Character | Text | Comic Bubble
--- | --- | --- | ---
Pre-trained | 0.5576 | 0.5080 | 0.4104 
Pre-trained (w/o sem_seg_head.predictor) | 0.3423 | 0.2677 | 0.1915
Fine-tuned predictor | 0.1097 | 0.0929 | 0.0735
Fine-tuned decoder | 0.1654 | 0.1423 | 0.1120
Fine-tuned whole model | 0.0908 | 0.0721 | 0.0584
