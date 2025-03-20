from PIL import Image
import re
import sys
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
    match = re.match(r"^(.*?)\s*\((?:Options|Opties|옵션|选项|Opciones|opzioni|Варианты|choix|Opções|Optionen):\s*(.*?)\)$", text, re.IGNORECASE)
    if match:
        true_answer = match.group(1).strip()
        choices = re.sub("\s*,\s*", "\n", match.group(2))
        return true_answer, choices
    return None, None

def alm_bench_doc_to_text(doc, lmms_eval_specific_kwargs):
    # Process MCQ question to extract answer, at the moment since we are not filtered the data set, I only process MCQ question types
    question = doc["Translated_Question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs["prompt_format"] == "mcq" and doc["Question_Type"] == "Multiple Choice Questions":
        _, choices = split_answer_options(doc["Translated_Answer"])
        full_prompt = f"{pre_prompt} {question}\n{choices}{post_prompt}"
        return full_prompt
    else:
        return "WRONG PROMPT. Write no answer. "


def alm_bench_process_results(doc, results):
    pred = results[0]
    type = doc["Question_Type"]
    return_dict = {"mcq": 0, "others": 0}
    if type == "Multiple Choice Questions":
        target = doc["Translated_Answer"].split(" (Options: ")[0]
        match = exact_match(pred, target)
        return_dict["mcq"] = match
    else:
        return_dict["others"] = 0
    return return_dict


def alm_bench_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == "mcq":
        true_answer, _ = split_answer_options(doc["Translated_Answer"])
        return true_answer
    # elif model_specific_target_kwargs == "tf":
    #     return doc["Translated_Answer"]
    else:
        return "NO ANSWER"
