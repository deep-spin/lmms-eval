import base64
from PIL import Image
from io import BytesIO
import numpy as np
import re
import json
import os
from loguru import logger
from pathlib import Path
import yaml

from lmms_eval.tasks.ayavisionbench.judge_templates import (
    COMPARATIVE_GEN_USER_PROMPT,
    COMPARATIVE_GEN_SYSTEM_PROMPT,
    DIRECT_ASSESSMENT_SYSTEM_PROMPT,
    DIRECT_ASSESSMENT_USER_PROMPT 
)
import requests
import time

def get_judge_config():
    with open(Path(__file__).parent / "eval_with_judge_template.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
        config = yaml.safe_load("".join(safe_data))
    return config


def run_judge(questions,preds,judge_config,baseline_model_outputs=None,images=None):
    logger.info(f"Selected judge type: {judge_config['judge_prompt_type']}")
    api_key = judge_config["api_key"]
    api_url = judge_config["api_url"]
    judge_model_name = judge_config["judge_model_name"]
    headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    payload = {
        "model": judge_model_name,
        "max_tokens": judge_config["max_tokens"],
        "temperature": judge_config["temperature"],
    }

    if judge_config["judge_prompt_type"] == "comparative_gen":
        system_prompt = COMPARATIVE_GEN_SYSTEM_PROMPT
        user_prompt_template = COMPARATIVE_GEN_USER_PROMPT
        prompts = [user_prompt_template.format(question=question,answer_1=base_output,answer_2=pred) for question,pred,base_output in zip(questions,preds,baseline_model_outputs)]
    elif judge_config["judge_prompt_type"] == "direct_assessment":
        system_prompt = DIRECT_ASSESSMENT_SYSTEM_PROMPT
        user_prompt_template = DIRECT_ASSESSMENT_USER_PROMPT
        prompts = [user_prompt_template.format(question=question,answer=pred) for question,pred in zip(questions,preds)]
    else:
        raise ValueError(f"Invalid judge prompt type: {judge_config['judge_prompt_type']}")
    
    messages = []
    for image,prompt in zip(images,prompts):
        if image is not None:
            messages.append([
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user",
                    "content": [{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64, {image}"}},],
                    },])
        else:
            messages.append([
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ])
    payload["messages"] = messages

    responses = []
    for attempt in range(judge_config["max_retries"]):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            # print(response_data)
            responses.append(response_data)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed on attempt {attempt+1}: {e}")
            time.sleep(judge_config["wait_time"])
            if attempt == judge_config["max_retries"] - 1:
                logger.info(f"Failed to get response after {judge_config['max_retries']} attempts")
                responses.append(None)
        except Exception as e:
            logger.info(f"Error on attempt {attempt+1}: {e}")
            time.sleep(judge_config["wait_time"])
            responses.append(None)
    return response_data


def process_judge_results(responses):
    import pdb; pdb.set_trace()
    for response in responses:
        if response is None:
            continue
        response_data = response["choices"][0]["message"]["content"]
        responses.append(response_data)
    return responses

def load_baselined_outputs(baseline_model_outputs_path):
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


def base64_to_bytes(base64_string):
    # Remove the header if it exists (e.g., "data:image/jpeg;base64,")
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(base64_string)
    return img_bytes

def process_docs(docs):
    """
    Process documents...
    """
    # logger.info(f"processing docs")
    # Process images in place
    docs = docs.select(range(5)) # filter out some samples!
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


# def doc_to_choice(doc,lmms_eval_specific_kwargs=None ):
#     choices = doc["choices"]
#     choice_list = [f"{label}: {text}" for label, text in zip(choices["label"], choices["text"])]
#     choice_list = choices["text"]
#     return choice_list

# def doc_to_target(doc,lmms_eval_specific_kwargs=None ):
#     choices = doc["choices"]
#     answerKey = doc["answerKey"]
#     import pdb; pdb.set_trace()
#     return choices["label"].index(answerKey)

def gen_process_results(doc, results):
    generated_texts = [res[0] for res in results]
    return {"results": {
        "id": doc["index"],
        "image": doc["image"],
        "question": doc["question"],
        "image_category": doc["image_source_category"],
        "prediction": generated_texts
        } 
    }

def aggregate_results(results):
    preds = [result["prediction"] for result in results]
    questions = [result["question"] for result in results]
    images = [result["image"] for result in results]

    judge_config = get_judge_config()
    logger.info("Checking judge config...")
    if judge_config["run_judge"]:
        logger.info("Judge is enabled, checking judge config...")
        if judge_config["judge_prompt_type"] == "comparative":
            logger.info("Judgement is set to comparative, checking baseline model outputs...")
            try:
                baseline_model_outputs = load_baselined_outputs(judge_config["baseline_model_outputs_path"])
                assert len(baseline_model_outputs) == len(preds)
            except Exception as e:
                logger.error(f"Failed loading baseline model outputs.")
                raise e
            
            logger.info("Baseline model outputs loaded successfully.")
            #TODO: run judge with comparative prompt
            judge_results = run_judge(questions,preds,judge_config,baseline_model_outputs,images)

        elif judge_config["judge_prompt_type"] == "direct assessment":
            preds = [result["prediction"] for result in results]
            #Note: run judge on predicted outputs
            judge_results = run_judge(questions,preds,judge_config,baseline_model_outputs=None,images=images)
        else:
            logger.error(f"Invalid judge prompt type: {judge_config['judge_prompt_type']}.Skipping judge...")
            judge_results = None
        
    else:
        judge_results = None
        logger.info("Judge is disabled, skipping judge...")

    if judge_results is None:
        return {"judge_results": None}
    else:
        results = process_judge_results(judge_results)
        return results


# def cache_judge_outputs(doc, round_res, previous_round_info, save_dir):
#     save_dict = dict(
#         sample_id=doc["index"],
#         question=doc["question"],
#         round_res=round_res,
#     )
#     save_dict.update(previous_round_info)
#     json.dump(save_dict, open(os.path.join(save_dir, f"{save_dict['sample_id']}.json"), "w"), indent=4)
