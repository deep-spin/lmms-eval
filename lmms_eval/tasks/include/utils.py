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




def process_docs(docs):
    """
    Process documents...
    """
    # logger.info(f"processing docs")
    # docs = docs.select(range(10)) # filter out some samples!
    # docs = docs.select(range(0,5))
    return docs


def doc_to_visual(doc):
   return []


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
    choices = format_options([doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]])
    full_prompt = f"{pre_prompt}Question: {question}\n{choices}{post_prompt}"
    return full_prompt


def doc_to_target(doc, model_specific_target_kwargs=None):
    answer = index_to_option(doc["answer"])
    if answer not in ["A", "B", "C", "D"]:
        logger.warning(f"Invalid answer in dataset. Doc sample: {doc}")
    return answer


def extract_final_answer(text: str) -> str:
    # Case 1 - expects the whole text to be a single letter, e.g. "A)", "B"
    pattern_case1 = re.compile(r'^\(?([a-zA-Z])\)?$', re.IGNORECASE)
    # Case 2 - expects the text to be "Final Answer: " or "Answer: " followed by a single letter, e.g. "Final Answer: A)", "Answer: B"
    pattern_case2 = re.compile(r'(?i)(?:Final Answer:|Answer:)\s*\(?([a-zA-Z])\)?', re.DOTALL)
    # Case 3: starts with letter optionally surrounded by parentheses, followed by text
    # e.g. "A) The answer is A", "B) The answer is B"
    pattern_case3 = re.compile(r'(?i)^\(?([a-zA-Z])\)?\)\s+.*', re.DOTALL)
    # Case 4: match a letter at the start of a line that looks like "A),
    # e.g: uquê de rosas brancas ao redor. Portanto, a resposta correta é:\n\nA) Nossa Senhora de Fátima
    pattern_case4 = re.compile(r'(?m)^\s*\(?([A-Za-z])\)?\s*\)', re.MULTILINE)
    text = text.strip()
    
    # Case 1
    match1 = pattern_case1.match(text)
    if match1:
        return match1.group(1).lower().strip()
    
    # Case 2
    match2 = pattern_case2.search(text)
    if match2:
        return match2.group(1).lower().strip()
    
    # Case 3
    match3 = pattern_case3.match(text)
    if match3:
        return match3.group(1).lower().strip()
    
    # Case 4
    match4 = pattern_case4.search(text)
    if match4:
        return match4.group(1).lower().strip()
    
    logger.warning(f"No valid answer letter found in: {text!r}")
    return None

def process_results(doc, results):
    generated_text = results[0]
    pred = extract_final_answer(generated_text)
    target_answer = index_to_option(doc["answer"])
    if target_answer == None:
        logger.warning(f"None target answer parsed from dataset. Doc sample: {doc}")
        return {"accuracy": 0, "pred_answer": pred, "target_answer": target_answer}
    if pred == None:
        logger.warning(f"No valid answer letter found in generated text: {generated_text}. Doc sample: {doc}")
        return {"accuracy": 0, "pred_answer": pred, "target_answer": target_answer}
    elif pred.lower().strip() == target_answer.lower().strip():
        match = 1
    else:
        match = 0
    return {"accuracy": match, "pred_answer": pred.lower().strip(), "target_answer": target_answer.lower().strip()}
    
