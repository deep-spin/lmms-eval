
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import re
from loguru import logger

def process_docs(docs):
    """
    Process documents...
    """
    # logger.info(f"processing docs")
    # Process images in place
    # docs = docs.select(range(5)) # filter out some samples!
    return docs

# doc_to_text: "Question: {{question}}\nAnswer:"
# doc_to_target: "{{choices.label.index(answerKey)}}"
# doc_to_choice: "{{choices.text}}"

def doc_to_visual(doc):
    return []


def doc_to_text(doc,lmms_eval_specific_kwargs=None ):
    question = doc["question"]
    choices = doc["choices"]
    choices_text = choices["text"]
    choices_label = choices["label"]
    choices = "\n".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])
    my_text = f"Question: {question}\n{choices}\n"
    if 'pre_prompt' in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        my_text = f"{pre_prompt}\n{my_text}"
    else:
        pre_prompt = ""
    if 'post_prompt' in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        my_text = f"{my_text}\n{post_prompt}"
    else:
        post_prompt = ""
    return f"{my_text}"


def doc_to_choice(doc,lmms_eval_specific_kwargs=None ):
    choices = doc["choices"]
    choice_list = [f"{label}: {text}" for label, text in zip(choices["label"], choices["text"])]
    # choice_list = choices["text"]
    return choice_list


def process_results(doc, results):
    losses = [res[0] for res in results]
    pred_index = losses.index(min(losses))
    pred_letter = doc["choices"]["label"][pred_index]
    answer = doc["answerKey"]
    return {"results": {"id": doc["id"], "question": doc["question"], "prediction": pred_letter, "ground_truth": answer} }


def aggregate_results(results):
    preds = [result["prediction"] for result in results]
    gts = [result["ground_truth"] for result in results]
    acc = sum([pred == gt for pred, gt in zip(preds, gts)]) / len(preds)
    return {"acc": acc}
