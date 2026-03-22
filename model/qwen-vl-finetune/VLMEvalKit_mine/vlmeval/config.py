import os
from pathlib import Path
from functools import partial

from vlmeval.vlm import Qwen2VLChat


GEOSR4D_ROOT = Path(__file__).resolve().parents[4]
GEOSR4D_DATA_ROOT = Path(
    os.environ.get("GEOSR4D_DATA_ROOT", GEOSR4D_ROOT / "data")
)
DEFAULT_MODEL_PATH = GEOSR4D_DATA_ROOT / "models" / "GeoSR4D-Model"


supported_VLM = {
    "Qwen2.5-VL-7B-Instruct-ForVideo-Spatial": partial(
        Qwen2VLChat,
        model_path=str(os.environ.get("GEOSR4D_EVAL_MODEL_PATH", DEFAULT_MODEL_PATH)),
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
}

api_models = {}
