import base64
from PIL import Image
from io import BytesIO
import numpy as np
import re
import json
import os
from loguru import logger
import io

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
    # docs = docs.select(range(2)) # filter out some samples!
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
    question = doc["instruction"]
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
    generated_text = results[0]
    myimg = doc['copy_image'][0]['bytes']
    pil_img = Image.open(io.BytesIO(myimg))
    return {"results": {
        "id": doc["question_id"],
        "image": pil_img,
        "question": doc["instruction"],
        "language": doc["language"],
        "prediction": generated_text 
        } 
    }


# def save_result_to_cache(doc, round_res, previous_round_info, save_dir):
#     save_dict = dict(
#         sample_id=doc["index"],
#         question=doc["question"],
#         round_res=round_res,
#     )
#     save_dict.update(previous_round_info)
#     json.dump(save_dict, open(os.path.join(save_dir, f"{save_dict['sample_id']}.json"), "w"), indent=4)
