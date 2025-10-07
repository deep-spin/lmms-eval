import warnings
from typing import List, Optional, Tuple, Union
import os
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from huggingface_hub import snapshot_download

from vllm.sampling_params import SamplingParams

import io
import base64
from vllm import LLM
from vllm.sampling_params import SamplingParams
from lmms_eval.models.pixtral import Pixtral

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"

@register_model("pixtral_v6")
class Pixtral_v6(Pixtral):
    """
    Custom PIXTRAL model implementation using Hugging Face Transformers and remote code.

    Example usage:

    accelerate launch --num_processes=8 -m lmms_eval \
        --model pixtral \
        --model_args pretrained=mistralai/mistralai/Pixtral-12B-2409 \
        --tasks mme \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "mistralai/Pixtral-12B-2409",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        add_system_prompt: Optional[str] = None,
        tag: Optional[str] = None,
        device_map: str = "",
        use_cache: bool = True,
        max_img_per_msg: int = 10,
        tokenizer_mode: str = "mistral",
        gpu_memory_utilization: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained,
                         device=device,
                         dtype=dtype,
                         batch_size=batch_size,
                         add_system_prompt=add_system_prompt,
                         tag=tag,
                         device_map=device_map,
                         use_cache=use_cache,
                         max_img_per_msg=max_img_per_msg,
                         tokenizer_mode=tokenizer_mode,
                         gpu_memory_utilization=gpu_memory_utilization)