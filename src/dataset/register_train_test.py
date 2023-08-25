from typing import List, Tuple
from sklearn.model_selection import train_test_split
from detectron2.data import DatasetCatalog, MetadataCatalog

from src.dataset.helpers import EvalType, get_all_panels_dataset_dicts


def _register_subset_dataset(dataset, new_name, eval_type, metadata):
    DatasetCatalog.register(new_name, lambda: dataset)
    new_metadata = MetadataCatalog.get(new_name)
    new_metadata.json_file = metadata.json_file
    new_metadata.evaluator_type = eval_type
    new_metadata.thing_dataset_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
    new_metadata.thing_classes = metadata.thing_classes


def register_train_test(dataset_name, test_size=0.2, random_state=42) -> Tuple[str, str]:
    print('registering train test dataset ...')
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    train_dataset, test_dataset = train_test_split(
        dataset_dicts, 
        test_size=test_size, 
        random_state=random_state
    )
    
    new_train_name = f'{dataset_name}_train'
    _register_subset_dataset(
        train_dataset, new_train_name, EvalType.COMIC_SEM_SEG, metadata
    )

    new_test_name = f'{dataset_name}_test'
    _register_subset_dataset(
        test_dataset, new_test_name, EvalType.COMIC_SEM_SEG, metadata
    )
    return new_train_name, new_test_name


def register_cropped(dataset_name: str, mode: str, new_cropped_dicts) -> Tuple[str, List]:
    metadata = MetadataCatalog.get(dataset_name)
    print(f"Collect all panels for mode {mode} ...")
    new_cropped_name = f'{dataset_name}_cropped'
    _register_subset_dataset(
        new_cropped_dicts, new_cropped_name, EvalType.COMIC_SEM_SEG_PANEL, metadata
    )
    return new_cropped_name
