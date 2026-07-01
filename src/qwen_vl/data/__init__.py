import re

SPAR_234K = {
    "annotation_path": "data/train/spar_234k.json",
    "data_path": "data/media",
    "tag": "3d"
}

LLAVA_HOUND_64K = {
    "annotation_path": "data/train/llava_hound_64k.json",
    "data_path": "data/media",
    "tag": "2d"
}

data_dict = {
    "spar_234k": SPAR_234K,
    "llava_hound_64k": LLAVA_HOUND_64K,
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
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["spar_234k", "llava_hound_64k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
