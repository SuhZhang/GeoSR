# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import torch.nn as nn
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from qwen_vl_spatial import (
    Qwen2_5_VLForConditionalGeneration_Spatial,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from safetensors.torch import load_file
local_rank = None


def force_load_checkpoint_keep_device_dtype(model: nn.Module, checkpoint: dict):

    for key, tensor in checkpoint.items():
        parts = key.split('.')
        module_path = parts[:-1]
        param_name = parts[-1]

        # 找到 module
        module = model
        for p in module_path:
            if hasattr(module, p):
                module = getattr(module, p)
            else:
                print(f"Warning: module path {'.'.join(module_path)} not found, skipping {key}")
                module = None
                break
        if module is None:
            continue

        # 确定目标 device/dtype
        if hasattr(module, param_name):
            existing = getattr(module, param_name)
            target_device = existing.device if isinstance(existing, torch.Tensor) else next(module.parameters(), torch.tensor(0)).device
            target_dtype = existing.dtype if isinstance(existing, torch.Tensor) else next(module.parameters(), torch.tensor(0)).dtype
        else:
            # fallback
            try:
                target_device = next(module.parameters()).device
                target_dtype = next(module.parameters()).dtype
            except StopIteration:
                target_device = torch.device("cpu")
                target_dtype = torch.float32

        # 强制替换
        param_tensor = tensor.to(device=target_device, dtype=target_dtype)
        setattr(module, param_name, nn.Parameter(param_tensor))

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    for n, p in model.visual.spatial_encoder.named_parameters():
        p.requires_grad = False
    for n, p in model.visual.spatial_merger.named_parameters():
        p.requires_grad = True
    for n, p in model.visual.q_former_1.named_parameters():
        p.requires_grad = True
    for n, p in model.visual.q_former_2.named_parameters():
        p.requires_grad = True
    model.visual.q_former_queries.requires_grad = True
    for n, p in model.visual.q_former_norm.named_parameters():
        p.requires_grad = True
    for n, p in model.visual.geometry_q_norm.named_parameters():
        p.requires_grad = True
    for n, p in model.visual.geometry_kv_norm.named_parameters():
        p.requires_grad = True
    for n, p in model.visual.geometry_kv_attn.named_parameters():
        p.requires_grad = True
    for n, p in model.visual.gated_fusion_norm_visual.named_parameters():
        p.requires_grad = True
    for n, p in model.visual.gated_fusion_norm_geometry.named_parameters():
        p.requires_grad = True
    for n, p in model.visual.gated_fusion_proj.named_parameters():
        p.requires_grad = True
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration_Spatial.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model.config.vision_mask_apply_prob = model_args.vision_mask_apply_prob
        model.config.vision_mask_prob = model_args.vision_mask_prob
        # Keep backward compatibility for code paths still reading keep-prob.
        model.config.vision_mask_keep_prob = 1.0 - model_args.vision_mask_prob
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
        pi3_params = load_file(model_args.pi3_path)
        pi3_params = {name.replace("encoder.",""):param for name,param in pi3_params.items() if "encoder" in name}
        force_load_checkpoint_keep_device_dtype(model.visual.spatial_encoder, pi3_params)
    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
