from tqdm import tqdm
from typing import Tuple
from sklearn.model_selection import train_test_split
from detectron2.data import DatasetCatalog, MetadataCatalog

from src.dataset.helpers import EvalType
from src.dataset.dataset_mapper import panel_mapper

def _register_subset_dataset(dataset, new_name, eval_type, metadata):
    DatasetCatalog.register(new_name, lambda: dataset)
    new_metadata = MetadataCatalog.get(new_name)
    new_metadata.json_file = metadata.json_file
    new_metadata.evaluator_type = eval_type
    new_metadata.thing_dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
    new_metadata.thing_classes = metadata.thing_classes
    new_metadata.stuff_classes = metadata.stuff_classes


def register_train_test(
    dataset_name,
    test_size=0.2, random_state=42,
    eval_type: Tuple[str] = None
) -> Tuple[str, str]:
    print('registering train test dataset ...')
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    new_train_name = f'{dataset_name}_train'
    new_test_name = f'{dataset_name}_test'
    eval_type = EvalType.COCO.value if eval_type is not None else eval_type

    if (new_train_name not in MetadataCatalog.list()) \
        and (new_test_name not in MetadataCatalog.list()):
        train_dataset, test_dataset = train_test_split(
            dataset_dicts, 
            test_size=test_size, 
            random_state=random_state
        )
        _register_subset_dataset(
            train_dataset, new_train_name, eval_type, metadata
        )
        _register_subset_dataset(
            test_dataset, new_test_name, eval_type, metadata
        )
    return new_train_name, new_test_name


def register_panels(dataset_name: str, mode: str, eval_type: Tuple[str] = None):
    assert mode in ['train', 'test']

    eval_type = EvalType.COMIC_SEM_SEG.value if eval_type is None else eval_type
    dataset_dicts = DatasetCatalog.get(dataset_name)
    print(dataset_name, len(dataset_dicts))
    new_cropped_name = f'{dataset_name}_cropped'

    if new_cropped_name in DatasetCatalog.list():
        print(f"{new_cropped_name} already registered...!")
        return new_cropped_name
    
    cropped_dataset_dicts = []
    print(f"Collect all panels for mode {mode} ...")

    for dataset_dict in tqdm(dataset_dicts[:2]):
        cropped_dataset_dicts.extend(panel_mapper(dataset_dict))
    
    _register_subset_dataset(
        cropped_dataset_dicts, 
        new_cropped_name, 
        eval_type,
        MetadataCatalog.get(dataset_name)
    )
    return new_cropped_name
