from PIL import Image
import re
import sys
import numpy as np 
from lmms_eval.utils import extract_final_answer

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
        "Italian": "Opzioni",
        "Russian": "Варианты",
        "French": "choix ",
        "Portuguese": "Opções",
        "German": "Optionen",
    }
    text = text.strip()
    match = re.match(r"^(.*?)\s*\((?:Options|Opties|옵션|선택|선택사항|선택 |选项|Opciones|opzioni|Варианты|choix |Opções|Optionen|Zutaten|Auswahl):\s*(.*?)\)$", text, re.IGNORECASE)
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
    pred = extract_final_answer(results[0])
    pred = re.sub(r" \n [\s\S]*$", "", pred)
    target, _ = split_answer_options(doc["Translated_Answer"])
    if target == None:
        print(doc["Translated_Answer"])
    match = exact_match(pred, target.strip("."))
    return {"exact_match": match}


def alm_bench_doc_to_target(doc, model_specific_target_kwargs):
    true_answer, _ = split_answer_options(doc["Translated_Answer"])
    return true_answer
