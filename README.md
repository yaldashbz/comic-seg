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
- If comic dataset is in cityscapes format, use convert_comic_to_coco_format.ipynb notebook, and then update the path of json_file in [src.dataset.register_comic_instances.py](https://github.com/yaldashbz/comic-seg/blob/coco/convert_comic_to_coco_format.ipynb)
- If comic dataset is in coco format, all the src functions will be ok.

## Train/Fine-tune
- All available codes can be found in [src.train](https://github.com/yaldashbz/comic-seg/tree/coco/src/train)

## Evaluation
- Both semantic & instance evaluation codes are in [src.evaluation](https://github.com/yaldashbz/comic-seg/tree/coco/src/evaluation)

## Data Visualization
- Functions can be found in [src.dataset.helpers](https://github.com/yaldashbz/comic-seg/tree/coco/src/dataset/helpers.py)

For more info, check available notebooks in the root directory.

