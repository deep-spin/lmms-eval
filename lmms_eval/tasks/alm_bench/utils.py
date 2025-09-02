from PIL import Image
import re
import sys
import numpy as np 
from lmms_eval.utils import extract_final_answer


def country_map(lang):
    language_to_country = {
        "German": "Germany",
        "Spanish": "Spain",
        "French": "France",
        "Italian": "Italy",
        "Korean": "South Korea",
        "Dutch": "Netherlands",
        "Russian": "Russia",
        "English": "United States",
        "Portuguese": "Portugal",
        "Chinese (Simplified)": "China",
        "Chinese (Traditional)": "Taiwan",
        "Icelandic": "Iceland",
        "Czech": "Czech Republic",
        "Ukrainian": "Ukraine",
        "Hindi": "India",
        "Japanese": "Japan",
        "Polish": "Poland",
        "Swedish": "Sweden",
        "Hungarian": "Hungary",
        "Romanian": "Romania",
        "Danish": "Denmark",
        "Norwegian": "Norway",
        "Finnish": "Finland"
    }
    return language_to_country[lang]
def exact_match(pred, target):
    if pred == target:
        return 1
    else:
        return 0
    
def alm_bench_doc_to_visual(doc):
    image = (doc['file_name']).convert('RGB')
    return [image]


def split_answer_options(text):
    text = text.strip()

    pattern = (
        r"^(.*?)\s*"                # capture the correct answer
        r"(?:（|\()"                # opening parenthesis (full-width or normal)
        r"(?:Options|Opties|옵션|선택|선택사항|選択肢|選択 |选项|選項|選購|選配|可选|可選|"
        r"Варианты|Опції|Варіанти|choix|Opções|Optionen|Zutaten|Auswahl|"
        r"Vaihtoehdot|Možnosti|विकल्प|オプション|opcje|Alternativ|Opciók|"
        r"Opțiuni|Valgmuligheder|Opsjoner|Valmöguleikar|Valkostir|möguleikar|параметри|Kjör)"
        r"\s*[:：]\s*"              # colon (ASCII or full-width), allow spaces
        r"(.+?)"                    # the list of options
        r"(?:）|\))$"               # closing parenthesis (full-width or normal)
    )

    match = re.match(pattern, text, re.IGNORECASE | re.UNICODE)
    if not match:
        return None, None

    true_answer = match.group(1).strip(" .。")

    raw_choices = match.group(2)
    choices = [opt.strip() for opt in re.split(r"[、，,]", raw_choices)]

    return true_answer, choices


def alm_bench_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["Translated_Question"]
    lang = doc["Language"]
    category = doc["Category"]
    country = country_map(lang)
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    country_specific = f"Provide brief, clear responses in {lang} language. The image represents the {category} in {country}"
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    _, choices = split_answer_options(doc["Translated_Answer"])
    full_prompt = f"{pre_prompt} {country_specific} {question}\n{choices}{post_prompt}"
    return full_prompt


def alm_bench_process_results(doc, results):
    pred = extract_final_answer(results[0])
    target, _ = split_answer_options(doc["Translated_Answer"])
    if target == None:
        print(doc["Translated_Answer"])
    match = exact_match(pred, target.strip("."))
    return {"match": match}


def alm_bench_doc_to_target(doc, model_specific_target_kwargs):
    true_answer, _ = split_answer_options(doc["Translated_Answer"])
    return true_answer