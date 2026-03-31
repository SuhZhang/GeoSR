import os
import re
from pathlib import Path


GEOSR4D_ROOT = Path(__file__).resolve().parents[4]
GEOSR4D_DATA_ROOT = Path(
    os.environ.get("GEOSR4D_DATA_ROOT", GEOSR4D_ROOT / "data")
)


def _path_from_env(name, default):
    return str(Path(os.environ.get(name, default)))

SPATIAL_REASONING = {
    "annotation_path": _path_from_env(
        "GEOSR4D_TRAIN_QA_PATH",
        GEOSR4D_DATA_ROOT / "spatial_reasoning" / "train_qas.json",
    ),
    "data_path": _path_from_env(
        "GEOSR4D_TRAIN_VIDEO_ROOT",
        GEOSR4D_DATA_ROOT / "spatial_reasoning" / "videos_train",
    ),
}

data_dict = {
    "spatial_reasoning": SPATIAL_REASONING,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["spatial_reasoning"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
