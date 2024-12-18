import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"

@register_model("nvlm_d")
class NVLM_D(lmms):
    """
    Custom NVLM_D model implementation using Hugging Face Transformers and remote code.

    Example usage:

    accelerate launch --num_processes=8 -m lmms_eval \
        --model nvlm_d \
        --model_args pretrained=Unbabel/qwen2p5-7b-clip-hdr-sft-v3 \
        --tasks mme \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "Unbabel/qwen2p5-7b-clip-hdr-sft-v3",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        #attn_implementation: Optional[str] = best_fit_attn_implementation,
        add_system_prompt: Optional[str] = None,
        device_map: str = "",
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self._model = AutoModel.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
            #attn_implementation=attn_implementation
        )
        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code
        )

        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.add_system_prompt = add_system_prompt
        # Handle distributed setup
        if accelerator.num_processes > 1 and device_map == "":
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

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text based on the given requests."""
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # Group requests by their generation_kwargs
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_id, tasks, splits = zip(*chunk)
            visuals = [doc_to_visual(self.task_dict[task][split][ids]) for ids, task, split, doc_to_visual in zip(doc_id, tasks, splits, doc_to_visuals)]
            
            # Use the generation kwargs from the first request in the batch
            gen_kwargs = all_gen_kwargs[0]
            
            # Set default generation parameters if not provided
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # Process each prompt with its corresponding image
            # prompts = []
            # for context, visual in zip(contexts, visuals):
            #    if DEFAULT_IMAGE_TOKEN not in context:
            #        # If no image token in context, assume the image should be processed first
            #        content = [{"type": "image", "image": visual[0]}]
            #        content.append({"type": "text", "text": context})
            #    else:
            #        # If image token exists, replace it with the actual image
            #        parts = context.split(DEFAULT_IMAGE_TOKEN)
            #        content = []
            #        for i, part in enumerate(parts):
            #            if part:
            #                content.append({"type": "text", "text": part})
            #            if i < len(parts) - 1:  # Don't add image after last text part
            #                content.append({"type": "image", "image": visual[i]})
            #        
            #    prompts.append(content)

            # TODO: for now, images and text are passed seperatly to the processor
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            context = contexts[0]
            visual = visuals[0]
            # TODO: handle multiple images / understand the `visuals` object
            if isinstance(visual, list):
                if len(visual) > 1:
                    eval_logger.warning("More than one image is not supported for now... Using the first one")
                visual = visual[0]
            
            if DEFAULT_IMAGE_TOKEN not in context:
                context = f"{DEFAULT_IMAGE_TOKEN}\n{context}"
                
            # create chat object and apply template
            chat = [{"role": "user", "content": context}]
            if self.add_system_prompt is not None:
                chat.insert(0, {"role": "system", "content": self.add_system_prompt})
            prompt = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            # Process inputs through the processor
            inputs = self._processor(images=[visual], text=[prompt], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate outputs
            # TODO: pass proper arguments to the model
            output_ids = self.model.generate(
                **inputs,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                num_beams=gen_kwargs["num_beams"],
                top_p=gen_kwargs["top_p"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Process and collect results
            for output_id, input_id in zip(output_ids, inputs["input_ids"]):
                generated_id = output_id[len(input_id):]
                generated_text = self.tokenizer.decode(generated_id, skip_special_tokens=True)
                res.append(generated_text)
            
            pbar.update(1)

        # Reorder results back to original order
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Not implemented for NVLM model."""
        raise NotImplementedError("Multi-round generation is not implemented for NVLM model")