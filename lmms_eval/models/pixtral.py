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



warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"

@register_model("pixtral")
class Pixtral(lmms):
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

        self._model = LLM(model=pretrained, tokenizer_mode="mistral", limit_mm_per_prompt={"image": max_img_per_msg}, gpu_memory_utilization=0.7, max_model_len=16384)
        
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.add_system_prompt = add_system_prompt
        self.tag = tag
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
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)

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
                self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    def download_model(self, model_name: str, model_path: str):
        
        """
        Download the model (given the model_name on hugging face, like 'mistralai/Pixtral-12B-Base-2409') to the specified model_path directory.
        
        Args:
            model_name (str): The name of the model on Hugging Face (e.g., 'mistralai/Pixtral-12B-Base-2409')
            model_path (str, optional): Directory to store the downloaded files. 
                                    If None, uses the default HF cache directory.
        
        Returns:
            str: Path to the directory containing the downloaded model files
        """
        try:
            # If model_path is not provided, create one in the current directory
            if model_path is None:
                tmp_dir  = os.getenv("HF_HOME")
                if tmp_dir is None:
                    tmp_dir = "/mnt/data/cache/huggingface"
                model_path = os.path.join(tmp_dir, model_name.replace("/", "__"))
            
            # Create model_path directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)
            
            # Create a model-specific directory
            model_dir = model_path
            
            # Check if model is already downloaded
            if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
                print(f"Model already exists at {model_dir}")
                return model_dir
                
            print(f"Downloading model files from {model_name}...")
            
            # Download all files from the repo
            local_dir = snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False  # Download actual files, not symlinks
            )
            
            print(f"Download completed. Files saved to: {local_dir}")
            return local_dir
            
        except Exception as e:
            print(f"Error downloading model files: {str(e)}")
            return None
        
    
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

            self._sampling_params = SamplingParams(max_tokens=gen_kwargs["max_new_tokens"], temperature=gen_kwargs["temperature"])
            # TODO: for now, images and text are passed seperatly to the processor
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

            if self.tag:
                context = f"{self.tag} " + context

            # create chat object
            message = [
                {"role": "user",
                 "content": [{"type": "text", "text": context}] + [{"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls]
                     }]
            if self.add_system_prompt is not None:
                message.insert(0, {"role": "system", "content": self.add_system_prompt})
            
            try:
                output = self._model.chat(message, sampling_params=self._sampling_params)
                result = output[0].outputs[0].text
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                result = ""
            res.append(result)
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Not implemented for NVLM model."""
        raise NotImplementedError("Multi-round generation is not implemented for NVLM model")