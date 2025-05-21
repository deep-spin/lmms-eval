from lmms_eval.models.llava_hf import LlavaHf
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import (
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

model_map = {
    "llava": LlavaForConditionalGeneration,
    "llava_next": LlavaNextForConditionalGeneration,
}

try:
    from transformers import LlavaOnevisionForConditionalGeneration

    model_map["llava_onevision"] = LlavaOnevisionForConditionalGeneration
except Exception as e:
    eval_logger.debug("Transformers version does not support llava-onevision. Skipping.")



@register_model("llava_v6")
class Llava_v6(LlavaHf):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_frames_num: Optional[int] = 32,
        add_system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained = pretrained,
            revision = revision,
            device = device,
            dtype = dtype,
            batch_size = batch_size,
            trust_remote_code = trust_remote_code,
            attn_implementation = attn_implementation,
            device_map = device_map,
            chat_template = chat_template,
            use_cache = use_cache,
            max_frames_num = max_frames_num,
            add_system_prompt = add_system_prompt,
            **kwargs
            )