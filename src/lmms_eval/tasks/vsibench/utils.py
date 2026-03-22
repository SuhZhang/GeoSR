
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd

import datasets

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]
if os.path.isdir(dataset_path):
    cache_dir = dataset_path
else:
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(base_cache_dir, cache_name)

def vsibench_doc_to_visual(doc):
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

DIRECTION_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
]

BREAKDOWN_METRIC_SPECS = {
    "vsi_obj_count": ("object_counting", "MRA:.5:.95:.05"),
    "vsi_abs_dist": ("object_abs_distance", "MRA:.5:.95:.05"),
    "vsi_obj_size": ("object_size_estimation", "MRA:.5:.95:.05"),
    "vsi_room_size": ("room_size_estimation", "MRA:.5:.95:.05"),
    "vsi_rel_dist": ("object_rel_distance", "accuracy"),
    "vsi_route_plan": ("route_planning", "accuracy"),
    "vsi_appr_order": ("obj_appearance_order", "accuracy"),
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):
    
    doc['prediction'] = results[0]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {
        "vsibench_score": doc,
        "vsi_obj_count": doc,
        "vsi_abs_dist": doc,
        "vsi_obj_size": doc,
        "vsi_room_size": doc,
        "vsi_rel_dist": doc,
        "vsi_rel_dir": doc,
        "vsi_route_plan": doc,
        "vsi_appr_order": doc,
    }


def _mean_for_question_type(df, question_type, metric):
    rows = df[df["question_type"] == question_type]
    if rows.empty:
        eval_logger.warning(f"No samples for question_type={question_type}; fallback score=0.0")
        return 0.0
    if metric not in rows.columns:
        eval_logger.warning(f"Missing metric={metric} for question_type={question_type}; fallback score=0.0")
        return 0.0
    return float(rows[metric].mean())


def _compute_vsibench_breakdown(results):
    df = pd.DataFrame(results)

    scores = {}
    for key, (question_type, metric) in BREAKDOWN_METRIC_SPECS.items():
        scores[key] = _mean_for_question_type(df, question_type, metric)

    direction_scores = [_mean_for_question_type(df, question_type, "accuracy") for question_type in DIRECTION_QUESTION_TYPES]
    scores["vsi_rel_dir"] = float(np.mean(direction_scores))

    ordered_scores = [
        scores["vsi_obj_count"],
        scores["vsi_abs_dist"],
        scores["vsi_obj_size"],
        scores["vsi_room_size"],
        scores["vsi_rel_dist"],
        scores["vsi_rel_dir"],
        scores["vsi_route_plan"],
        scores["vsi_appr_order"],
    ]
    scores["vsibench_score"] = float(np.mean(ordered_scores))
    return scores


def _log_vsibench_breakdown(scores):
    output = {
        "object_counting_MRA:.5:.95:.05": scores["vsi_obj_count"],
        "object_abs_distance_MRA:.5:.95:.05": scores["vsi_abs_dist"],
        "object_size_estimation_MRA:.5:.95:.05": scores["vsi_obj_size"],
        "room_size_estimation_MRA:.5:.95:.05": scores["vsi_room_size"],
        "object_rel_distance_accuracy": scores["vsi_rel_dist"],
        "object_rel_direction_accuracy": scores["vsi_rel_dir"],
        "route_planning_accuracy": scores["vsi_route_plan"],
        "obj_appearance_order_accuracy": scores["vsi_appr_order"],
        "overall": scores["vsibench_score"],
    }
    eval_logger.info(f"Evaluation results: {output}")

def vsibench_aggregate_results(results):
    scores = _compute_vsibench_breakdown(results)
    _log_vsibench_breakdown(scores)
    return scores["vsibench_score"] * 100.0


def vsibench_aggregate_obj_count(results):
    scores = _compute_vsibench_breakdown(results)
    return scores["vsi_obj_count"] * 100.0


def vsibench_aggregate_abs_dist(results):
    scores = _compute_vsibench_breakdown(results)
    return scores["vsi_abs_dist"] * 100.0


def vsibench_aggregate_obj_size(results):
    scores = _compute_vsibench_breakdown(results)
    return scores["vsi_obj_size"] * 100.0


def vsibench_aggregate_room_size(results):
    scores = _compute_vsibench_breakdown(results)
    return scores["vsi_room_size"] * 100.0


def vsibench_aggregate_rel_dist(results):
    scores = _compute_vsibench_breakdown(results)
    return scores["vsi_rel_dist"] * 100.0


def vsibench_aggregate_rel_dir(results):
    scores = _compute_vsibench_breakdown(results)
    return scores["vsi_rel_dir"] * 100.0


def vsibench_aggregate_route_plan(results):
    scores = _compute_vsibench_breakdown(results)
    return scores["vsi_route_plan"] * 100.0


def vsibench_aggregate_appr_order(results):
    scores = _compute_vsibench_breakdown(results)
    return scores["vsi_appr_order"] * 100.0
