from functools import partial

from .spatial_reasoning import SPATIAL_REASONING


supported_video_datasets = {
    "Spatial-Reasoning": partial(SPATIAL_REASONING, dataset="Spatial-Reasoning"),
}
