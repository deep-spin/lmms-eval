from PIL import Image
import re
import sys
import numpy as np 

def exact_match(pred, target):
    if pred == target:
        return 1
    else:
        return 0
    
def alm_bench_doc_to_visual(doc):
    image = (doc['file_name']).convert('RGB')
    return [image]

def split_answer_options(text):
    option_words = {
        "english": "Options",
        "dutch": "Opties",
        "korean": "옵션",
        "Chinese (Simplified)": "选项",
        "Spanish": "Opciones",
        "Italian": "opzioni",
        "Russian": "Варианты",
        "French": "choix",
        "Portuguese": "Opções",
        "German": "Optionen",
    }
    text = text.strip()
    match = re.match(r"^(.*?)\s*\((?:Options|Opties|옵션|选项|Opciones|opzioni|Варианты|choix |Opções|Optionen):\s*(.*?)\)$", text, re.IGNORECASE)
    if match:
        true_answer = match.group(1).strip()
        choices = re.sub("\s*,\s*", "\n", match.group(2))
        return true_answer, choices
    return None, None

def alm_bench_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["Translated_Question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    _, choices = split_answer_options(doc["Translated_Answer"])
    full_prompt = f"{pre_prompt} {question}\n{choices}{post_prompt}"
    return full_prompt


def alm_bench_process_results(doc, results):
    pred = results[0]
    target, _ = split_answer_options(doc["Translated_Answer"])
    match = exact_match(pred, target)
    return {"results": match}


def alm_bench_doc_to_target(doc, model_specific_target_kwargs):
    true_answer, _ = split_answer_options(doc["Translated_Answer"])
    return true_answer
