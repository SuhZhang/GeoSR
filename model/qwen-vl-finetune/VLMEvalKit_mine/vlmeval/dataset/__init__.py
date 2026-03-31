import warnings

from .spatial_reasoning import SPATIAL_REASONING
from .utils import *
from .video_dataset_config import *
from ..smp import *


img_root_map = {}

DATASET_CLASSES = [SPATIAL_REASONING]
SUPPORTED_DATASETS = list(SPATIAL_REASONING.supported_datasets())
SUPPORTED_DATASETS.extend([x for x in supported_video_datasets if x not in SUPPORTED_DATASETS])


def DATASET_TYPE(dataset, *, default: str = "MCQ") -> str:
    if dataset in SPATIAL_REASONING.supported_datasets() or dataset in supported_video_datasets:
        return SPATIAL_REASONING.TYPE
    warnings.warn(f"Dataset {dataset} is not officially supported, will treat as {default}. ")
    return default


def DATASET_MODALITY(dataset, *, default: str = "IMAGE") -> str:
    if dataset is None:
        warnings.warn(f"Dataset is not specified, will treat modality as {default}. ")
        return default
    if dataset in SPATIAL_REASONING.supported_datasets() or dataset in supported_video_datasets:
        return SPATIAL_REASONING.MODALITY
    warnings.warn(f"Dataset {dataset} is not officially supported, will treat modality as {default}. ")
    return default


def build_dataset(dataset_name, **kwargs):
    if dataset_name in supported_video_datasets:
        return supported_video_datasets[dataset_name](**kwargs)
    if dataset_name in SPATIAL_REASONING.supported_datasets():
        return SPATIAL_REASONING(dataset=dataset_name, **kwargs)

    warnings.warn(f"Dataset {dataset_name} is not officially supported. ")
    return None


def infer_dataset_basename(dataset_name):
    basename = "_".join(dataset_name.split("_")[:-1])
    return basename


__all__ = [
    "build_dataset",
    "img_root_map",
    "build_judge",
    "extract_answer_from_item",
    "prefetch_answer",
    "DEBUG_MESSAGE",
    "SUPPORTED_DATASETS",
    "DATASET_TYPE",
    "DATASET_MODALITY",
    "SPATIAL_REASONING",
] + [cls.__name__ for cls in DATASET_CLASSES]
