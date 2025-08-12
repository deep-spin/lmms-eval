from lmms_eval.models.aya import Aya
import warnings
from typing import Optional, Union
import torch

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<image>"

@register_model("aya_v6")
class Aya_v6(Aya):
    def __init__(
        self,
        pretrained: str = "CohereForAI/aya-vision-8b",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        add_system_prompt: Optional[str] = None,
        device_map: str = "",
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
        pretrained = pretrained,
        device = device,
        dtype = dtype,
        batch_size = batch_size,
        add_system_prompt = add_system_prompt,
        device_map = device_map,
        use_cache = use_cache,)