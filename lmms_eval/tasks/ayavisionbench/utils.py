from PIL import Image
from io import BytesIO
import numpy as np
import json
from loguru import logger
from pathlib import Path
import io
from copy import deepcopy

from lmms_eval.tasks.ayavisionbench.judge_utils import (
    get_judge_config,
    run_judge,
    compute_results
)

def pil_to_image_dict(pil_img):
    """
    Convert a PIL Image to an image dict with 'bytes' and 'path' keys.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")  # or "JPEG" if you prefer
    img_bytes = buf.getvalue()
    return {'bytes': img_bytes, 'path': None}



def load_baseline_outputs(baseline_model_outputs_path):
    filtered_responses = []
    with open(baseline_model_outputs_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'filtered_resps' in data:
                    # Handle both single string and list of strings cases
                    resps = data['filtered_resps']
                    if isinstance(resps, list):
                        filtered_responses.extend(resps)
                    else:
                        filtered_responses.append(resps)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
                continue
    return filtered_responses


def process_docs(docs):
    """
    Process documents...
    """
    # logger.info(f"processing docs")
    docs = docs.select(range(5)) # filter out some samples!
    def copy_image_fn(example):
        example['copy_image'] = example['image']
        return example
    
    docs = docs.map(copy_image_fn)
    return docs


def gen_doc_to_visual(doc):
    image = doc['image'][0]
    image = image.convert('RGB')
    return [image]


def gen_doc_to_text(doc,lmms_eval_specific_kwargs=None ):
    question = doc["question"]
    if 'pre_prompt' in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    else:
        pre_prompt = ""
    if 'post_prompt' in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    else:
        post_prompt = ""
    return f"{pre_prompt}Question: {question}\n{post_prompt}"


def gen_process_results(doc, results):
    generated_texts = results[0]
    myimg = doc['copy_image'][0]['bytes']
    pil_img = Image.open(io.BytesIO(myimg))
    return {"results": {
        "id": doc["index"],
        "image": pil_img,
        "question": doc["question"],
        "image_category": doc["image_source_category"],
        "prediction": generated_texts
        }
    }

def aggregate_results(results):
    preds = [result["prediction"] for result in results]
    questions = [result["question"] for result in results]
    images = [result["image"] for result in results]
    # Assuming images is a list of PIL Images:
    images_bytes = [pil_to_image_dict(img) for img in images.copy()]

    judge_config = get_judge_config()

    if judge_config["run_judge"]:
        logger.info("Judge is enabled, checking judge config...")
        if judge_config["judge_prompt_type"] == "comparative":
            logger.info("Judgement is set to comparative, checking baseline model outputs...")
            try:
                baseline_model_outputs = load_baseline_outputs(judge_config["baseline_model_outputs_path"])
                assert len(baseline_model_outputs) == len(preds)
                logger.info("Baseline model outputs loaded successfully.")
                judge_results = run_judge(questions,preds,judge_config,baseline_model_outputs,images_bytes)
            except Exception as e:
                logger.error(f"Failed while loading model outputs or running judge.")
                raise e
            
        elif judge_config["judge_prompt_type"] == "direct assessment":
            preds = [result["prediction"] for result in results]
            judge_results = run_judge(questions,preds,judge_config,baseline_model_outputs=None,images=images_bytes)
        else:
            logger.error(f"Invalid judge prompt type: {judge_config['judge_prompt_type']}.Skipping judge...")
            judge_results = None
        
    else:
        judge_results = None
        logger.info("Judge is disabled, skipping judge...")

    if judge_results is None:
        return {"judge_results": None}
    else:
        results = compute_results(judge_results,judge_config)
        return results

