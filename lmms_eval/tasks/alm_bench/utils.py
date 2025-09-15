from PIL import Image
import re
import string
import sys
import numpy as np 
from lmms_eval.utils import extract_final_answer
from loguru import logger

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
    

    image = (doc['file_name']).convert('RGB')
    return [image]

def process_docs(docs):
    # docs = docs.select(range(5))
    for doc in docs:
        true_answer, choices = split_answer_options(doc["Translated_Answer"])
        if true_answer == None or choices == None:
            logger.warning(f"Error encountered while splitting answer options in dataset. Doc sample: {doc}")
    return docs

# def split_answer_options(text):
#     text = text.strip()

#     pattern = (
#         r"^(.*?)\s*"                # capture the correct answer
#         r"(?:（|\()"                # opening parenthesis (full-width or normal)
#         r"(?:Options|Opties|옵션|선택|선택사항|選択肢|選択 |选项|選項|選購|選配|可选|可選|"
#         r"Варианты|Опції|Варіанти|choix|Opções|Optionen|Zutaten|Auswahl|"
#         r"Vaihtoehdot|Možnosti|विकल्प|オプション|opcje|Alternativ|Opciók|"
#         r"Opțiuni|Valgmuligheder|Opsjoner|Valmöguleikar|Valkostir|möguleikar|параметри|Kjör)"
#         r"\s*[:：]\s*"              # colon (ASCII or full-width), allow spaces
#         r"(.+?)"                    # the list of options
#         r"(?:）|\))$"               # closing parenthesis (full-width or normal)
#     )

#     match = re.match(pattern, text, re.IGNORECASE | re.UNICODE)
#     if not match:
#         return None, None

#     true_answer = match.group(1).strip(" .。")

#     raw_choices = match.group(2)
#     choices = [opt.strip() for opt in re.split(r"[、，,]", raw_choices)]

#     return true_answer, choices


def split_answer_options(text):
    text = text.strip()

    pattern = (
        r"^(.*?)\s*"                # capture the correct answer
        r"(?:（|\()"                # opening parenthesis (full-width or normal)
        r"(?:Options|Opties|옵션|선택|선택사항|選択肢|選択 |选项|選項|選購|選配|可选|可選|"
        r"Варианты|Опції|Варіанти|choix|Opções|Optionen|Zutaten|Auswahl|"
        r"Vaihtoehdot|Možnosti|विकल्प|オプション|opcje|Alternativ|Alternativer|Opciók|Muligheder|Indstillinger|muligheder|Mulighed|Lehetőségek|"
        r"Opțiuni|Valgmuligheder|Opsjoner|Valmöguleikar|Valkostir|möguleikar|параметри|Kjör|opzioni|Opzioni|Opciones|opciones|Volby|volby|Варіанти|варианти|Optioner|optioner)"
        r"\s*[:：]\s*"              # colon (ASCII or full-width), allow spaces
        r"(.+?)"                    # the list of options
        r"(?:）|\))\s*[.。]?$"      # closing parenthesis and allow for a period or a dot
        # r"(?:）|\))$"               # closing parenthesis (full-width or normal)
    )

    match = re.match(pattern, text, re.IGNORECASE | re.UNICODE)
    if not match:
        return None, None

    true_answer = match.group(1).strip(" .。")
    raw_choices = match.group(2).strip()

    # # If raw choices contain commas followed by capitals, split there
    # if re.search(r"[、，,]\s*[A-Z]", raw_choices):
    #     choices = [opt.strip() for opt in re.split(r"[、，,](?=\s*[A-Z])", raw_choices)]
    # else:
    #     choices = [raw_choices]

    # Split on commas before a capital letter (Latin or Cyrillic)
    if re.search(r"[、，,]\s*[A-ZА-ЯЇЄІ]", raw_choices):
        choices = [opt.strip() for opt in re.split(r"[、，,](?=\s*[A-ZА-ЯЇЄІ])", raw_choices)]
    else:
        choices = [opt.strip() for opt in re.split(r"[、，,]", raw_choices)]
    return true_answer, choices

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

def alm_bench_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["Translated_Question"]
    lang = doc["Language"]
    category = doc["Category"]
    country = country_map(lang)
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    country_specific = f"Provide brief, clear responses in {lang} language. The image represents the {category} in {country}"
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    true_answer, choices = split_answer_options(doc["Translated_Answer"])
    if true_answer == None or choices == None:
        logger.warning(f"Error encountered while splitting answer options in dataset. Doc sample: {doc}")
    choices = "\n".join([f"{index_to_option(i)}) {choice}" for i, choice in enumerate(choices)])
    full_prompt = f"{pre_prompt} {country_specific} {question}\n{choices}{post_prompt}"
    return full_prompt


# def alm_bench_process_results(doc, results):
#     pred = extract_final_answer(results[0])
#     target, _ = split_answer_options(doc["Translated_Answer"])
#     if target == None:
#         print(doc["Translated_Answer"])
#     match = exact_match(pred, target.strip("."))

#     return {"match": match}


def extract_final_answer(text: str) -> str:
    # match = re.search(r'Final Answer:\s*([A-Z])\)?', text.strip())
    # if match:
    #     return match.group(1)

    pattern_case1 = re.compile(r'^\(?([a-zA-Z])\)?$', re.IGNORECASE)
    pattern_case2 = re.compile(r'(?i)(?:Final Answer:|Answer:)\s*\(?([a-zA-Z])\)?', re.DOTALL)
    # Case 3: starts with letter optionally surrounded by parentheses, followed by text
    pattern_case3 = re.compile(r'(?i)^\(?([a-zA-Z])\)?\)\s+.*', re.DOTALL)


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
    
    logger.warning(f"No valid answer letter found in: {text!r}")
    return None

def transform_target_text_to_letter(target, choices):
    for i in range(len(choices)):
        if choices[i].lower().strip() == target.lower().strip():
            return index_to_option(i).lower().strip()
    return None

def alm_bench_doc_to_target(doc, model_specific_target_kwargs):
    true_answer, choices  = split_answer_options(doc["Translated_Answer"])
    if true_answer == None or choices == None:
        logger.warning(f"Error encountered while splitting answer options in dataset. Doc sample: {doc}")
        breakpoint()
    target_letter = transform_target_text_to_letter(true_answer, choices)
    return target_letter

def process_results(doc, results):
    generated_text = results[0]
    pred = extract_final_answer(generated_text)
    true_answer, choices = split_answer_options(doc["Translated_Answer"])
    target_letter = transform_target_text_to_letter(true_answer, choices)
    if pred is None:
        match = 0
    else:
        if pred.lower().strip() == target_letter.lower().strip():
            match = 1
        else:
            match = 0
    return {"accuracy": match,"parsed_answer": pred,"target_answer": target_letter}