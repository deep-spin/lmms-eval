
import base64
from PIL import Image
from io import BytesIO
import numpy as np

from loguru import logger

from lmms_eval.tasks.cc_ocr.cc_ocr_eval.utils import text_normalize_and_tokenize, calculate_metrics

def base64_to_bytes(base64_string):
    # Remove the header if it exists (e.g., "data:image/jpeg;base64,")
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(base64_string)
    return img_bytes


def ccocr_process_docs(docs):
    """
    Process documents by converting base64 images to PIL Images
    
    Args:
        docs: Dataset object containing documents with base64 encoded images
    Returns:
        Dataset with converted PIL images
    """
    logger.info(f"Converting base64 images to PIL Images...")
    # Process images in place
    docs = docs.map(
        lambda doc: {
            'image': base64_to_bytes(doc['image'])
        }
    )
    return docs

def ccocr_doc_to_visual(doc):
    image = Image.open(BytesIO(doc["image"]))
    image = image.convert('RGB')
    return [image]


def ccocr_doc_to_text(doc, ):
    question = doc["question"]
    return f"{question}"


def ccocr_process_results(doc, results):
    pred = results[0]
    answer = doc["answer"]
    return {"ocr_results": {"index": doc["index"], "image_name": doc["image_name"],"question_type": doc["question"], "prediction": pred, "ground_truth": answer} }



def ccocr_multi_lan_ocr_aggregate_results(results):
    """
    The multi-language-ocr evaluation according to the paper (https://arxiv.org/pdf/2412.02210) is done in different ways depending on the language.
    For "Arabic", "Japanese", "Korean", and "zh" the evaluation procedure is using characters as basic units.
    Otherwise, the evaluation procedure is using words as basic units.
    This function implements the multi-language-ocr evaluation procedure for languages that are not in ["Arabic", "Japanese", "Korean", "zh"].
    The code is taken from https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/Benchmarks/CC-OCR/evaluation/evaluator/ocr_evaluator.py.
    """
    # The code below is implemented according to https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/Benchmarks/CC-OCR/evaluation/evaluator/ocr_evaluator.py

    logger.info("Aggregating results for languages that are not in ['Arabic', 'Japanese', 'Korean', 'zh'].")

    non_basic_languages = ["Arabic", "Japanese", "Korean", "Chinese"]
    pdt_tokenized, gt_tokenized = [], []
    for res  in results:
        index = res["index"]
        image_name = res["image_name"]
        pred = res["prediction"]
        gt = res["ground_truth"]
        
        is_word_level, is_lower, is_alphanum_only = True, True, False
        language = image_name.split("_")[0]
        if language in non_basic_languages:
            is_word_level = False
        
        
        pdt_token_list = text_normalize_and_tokenize(pred.strip(), is_word_level, is_lower, is_alphanum_only)
        gt_token_list = text_normalize_and_tokenize(gt.strip(), is_word_level, is_lower, is_alphanum_only)
        pdt_tokenized.append(pdt_token_list)
        gt_tokenized.append(gt_token_list)

    eval_result = calculate_metrics(pdt_tokenized, gt_tokenized, is_verbose=False)
    return  eval_result
