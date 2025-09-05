from PIL import Image
from io import BytesIO
import numpy as np
import json
from loguru import logger
from pathlib import Path
import io
from copy import deepcopy
import string
import re


def pil_to_image_dict(pil_img):
    """
    Convert a PIL Image to an image dict with 'bytes' and 'path' keys.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")  # or "JPEG" if you prefer
    img_bytes = buf.getvalue()
    return {'bytes': img_bytes, 'path': None}



# def load_baseline_outputs(baseline_model_outputs_path):
#     filtered_responses = []
#     with open(baseline_model_outputs_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line.strip())
#                 if 'filtered_resps' in data:
#                     # Handle both single string and list of strings cases
#                     resps = data['filtered_resps']
#                     if isinstance(resps, list):
#                         filtered_responses.extend(resps)
#                     else:
#                         filtered_responses.append(resps)
#             except json.JSONDecodeError:
#                 print(f"Warning: Skipping invalid JSON line")
#                 continue
#     return filtered_responses


def process_docs(docs):
    """
    Process documents...
    """
    # logger.info(f"processing docs")
    # docs = docs.select(range(10)) # filter out some samples!
    # docs = docs.select(range(0,5))
    return docs


def doc_to_visual(doc):
    if doc['image'] is None:
        return []
    else:
        images = doc['image_file']
        if isinstance(images, list):
            images = [im.convert('RGB') for im in images]
        else:
            images = [images.convert('RGB')]
    return images


def format_options(options):
    """
    Given a list of strings, return a formatted multiple-choice string
    like:
        A) option1
        B) option2
        ...
    """
    letters = string.ascii_uppercase
    choices = "\n".join(f"{letters[i]}){opt}" for i, opt in enumerate(options))
    choices = choices.strip()+"\n"
    return choices

def index_to_option(n: int) -> str:
    """
    Map a 0-indexed number to its corresponding capital letter option.
    Example:
        0 -> 'A'
        1 -> 'B'
        25 -> 'Z'
        26 -> 'AA'
    """
    letters = string.ascii_uppercase
    result = ""
    while True:
        n, r = divmod(n, 26)
        result = letters[r] + result
        if n == 0:
            break
        n -= 1
    return result

def doc_to_text(doc,lmms_eval_specific_kwargs=None ):
    question = doc["question"]
    if 'pre_prompt' in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    else:
        pre_prompt = ""
    if 'post_prompt' in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    else:
        post_prompt = ""
    choices = format_options(doc["options"])
    full_prompt = f"{pre_prompt}Question: {question}\n{choices}{post_prompt}"
    return full_prompt


# def em_doc_to_target(doc, model_specific_target_kwargs=None):
#     answer = doc["answer"]
#     if answer not in range(4):
#         logger.warning(f"Invalid answer in dataset. Dataset seems to contain more than 4 options. Doc sample: {doc}")
#     answer = doc["options"][doc["answer"]]
#     return answer

# def em_exact_match(pred, target):
#     if pred == target:
#         return 1
#     else:
#         return 0

# def em_extract_final_answer(text):
#     match = re.search(r'Final Answer:\s*(.*)', text)
#     if match:
#         return match.group(1)
#     else:
#         return text

# def em_clean_option(answer: str) -> str:
#     """
#     Remove leading multiple-choice markers like 'A) ', 'B) ', 'C) ' 
#     from the beginning of the answer string.
#     If no marker is present, return the string unchanged.
#     """
#     return re.sub(r'^[A-Z]\)\s*', '', answer.strip())

# def em_process_results(doc, results):
#     generated_text = results[0]
#     pred = extract_final_answer(generated_text)
#     clean_pred = clean_option(pred)
#     target_answer = doc["options"][doc["answer"]]
#     if target_answer == None:
#         logger.warning(f"Invalid answer in dataset. Doc sample: {doc}")
#     match = exact_match(clean_pred.strip("."), target_answer.strip("."))
#     breakpoint()
#     return {"match": match}


def doc_to_target(doc, model_specific_target_kwargs=None):
    if isinstance(doc["answer"], int):
        answer = index_to_option(doc["answer"])
    else:
        answer = doc["answer"]
    if answer not in ["A", "B", "C", "D"]:
        logger.warning(f"Invalid answer in dataset. Doc sample: {doc}")
    return answer

def extract_final_answer(text: str) -> str:
    """
    Extract the choice letter from the model's final answer.
    Examples:
        "Final Answer: D) III > I > II" -> "D"
        "Final Answer: B) Paris" -> "B"
        "Final Answer: C" -> "C"
    If no letter is found, return the full text.
    """
    match = re.search(r'Final Answer:\s*([A-Z])\)?', text.strip())
    if match:
        return match.group(1)
    return text

def process_results(doc, results):
    generated_text = results[0]
    pred = extract_final_answer(generated_text).lower().strip()
    target_answer = index_to_option(doc["answer"]).lower().strip()
    if target_answer == None:
        logger.warning(f"Invalid answer in dataset. Doc sample: {doc}")
    if pred == target_answer:
        match = 1
    else:
        match = 0
    return {"accuracy": match}
    
