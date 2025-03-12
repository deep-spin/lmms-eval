import warnings
from typing import List, Optional, Tuple, Union
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

import io
import base64

from transformers import AutoProcessor, AutoModelForImageTextToText

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"

@register_model("aya")
class Aya(lmms):
    """
    Custom AYA model implementation using Hugging Face Transformers and remote code.

    Example usage:

    accelerate launch --num_processes=8 -m lmms_eval \
        --model aya \
        --model_args pretrained=CohereForAI/aya-vision-8b \
        --tasks mme \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "CohereForAI/aya-vision-8b",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        device_map: str = "",
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self._processor = AutoProcessor.from_pretrained(pretrained)
        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, device_map="auto", torch_dtype=torch.float16)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        # Handle distributed setup
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED]
            
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Using DeepSpeed - ensure zero stage is set to 0 in accelerate config")

            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)

            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            if device_map == "auto":
                eval_logger.info("Using pipeline parallelism")
            else:
                eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """Tokenize a string."""
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        """Decode a sequence of tokens to a string."""
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not implemented for NVLM model."""
        raise NotImplementedError("Loglikelihood is not implemented for NVLM model")

    def flatten(self, input):
        """Flatten a nested list."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list


    def get_image_url(self, image_bytes: bytes) -> str:
        buffer = io.BytesIO()
        image_bytes.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{base64_image}"
        return data_url

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text based on the given requests."""
        res = []

        def batch_requests(requests, batch_size):
            for i in range(0, len(requests), batch_size):
                yield [x.arguments for x in requests[i:i + batch_size]]

        chunks = batch_requests(requests, self.batch_size)  
        
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_id, tasks, splits = zip(*chunk)
            visuals = [doc_to_visual(self.task_dict[task][split][ids]) for ids, task, split, doc_to_visual in zip(doc_id, tasks, splits, doc_to_visuals)]
            
            # Use the generation kwargs from the first request in the batch
            gen_kwargs = all_gen_kwargs[0]
            
            # Set default generation parameters if not provided
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 8192
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False

            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]
            visual = visuals[0]
            
            # TODO: handle multiple images / understand the `visuals` object
            if not isinstance(visual, list):
                visual = [visual]

            # vLLM does not work with bytes, so we need to convert it to a data url
            image_urls = [self.get_image_url(v) for v in visual]
            
            # image_url = self.get_image_url(visual)

            # Pixtral expects inputs in a different format, and doesn't work with <image> tokens added in the middle of the prompt.
            context = context.replace(DEFAULT_IMAGE_TOKEN, "")

            # create chat object
            message = [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": context
                      }]
                 +
                 [{"type": "image", "url": image_url} for image_url in image_urls]
                     }]
            
            try:
                inputs = self._processor.apply_chat_template(message, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self._model.device)
                result = self._model.generate(**inputs,
                                             max_new_tokens=gen_kwargs["max_new_tokens"],
                                             do_sample=gen_kwargs["do_sample"],
                                             temperature=gen_kwargs["temperature"],
                                             )
                result = self._processor.tokenizer.decode(result[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                result = ""
            res.append(result)
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Not implemented for AYA model."""
        raise NotImplementedError("Multi-round generation is not implemented for AYA model")