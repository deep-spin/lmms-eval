"""
Code adapted from:
https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/Benchmarks/CC-OCR/evaluation/evaluator.py
"""
import re
from collections import Counter

def token_normalize(token_text, is_lower=False, is_alphanum_only=False):
    """
    """
    if is_lower:
        token_text = token_text.lower()
    if is_alphanum_only:
        token_text = re.sub('[^A-Za-z0-9]+', '', token_text)
    return token_text


def text_normalize_and_tokenize(text, is_keep_blank=True, is_lower=True, is_alphanum_only=False):
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r'\s+', ' ', text)
    if not is_keep_blank:
        text = text.replace(" ", "")
    text_tokens = text.split(" ") if is_keep_blank else list(text)
    text_token_normalized = [token_normalize(t, is_lower, is_alphanum_only) for t in text_tokens]
    text_token_normalized = [x for x in text_token_normalized if len(x) > 0]
    return text_token_normalized


def evaluate_single_sample(gts, preds):
    right_num = 0
    gt_counter_info = dict(Counter(gts))
    pdt_counter_info = dict(Counter(preds))
    for gt_token, gt_count in gt_counter_info.items():
        pred_count = pdt_counter_info.get(gt_token, 0)
        right_num += min(gt_count, pred_count)
    return right_num

def calculate_metrics(predictions, groundtruths, is_verbose=False):
    """
    """
    macro_recall_list, macro_precision_list, macro_f1_list = [], [], []
    total_gt_num, total_pred_num, total_right_num = 0, 0, 0
    for pred_tokenized, gt_tokenized in zip(predictions,groundtruths):
        right_num = evaluate_single_sample(gt_tokenized, pred_tokenized)
        total_right_num += right_num
        total_gt_num += len(gt_tokenized)
        total_pred_num += len(pred_tokenized)

        macro_recall = right_num / (len(gt_tokenized) + 1e-9)
        macro_precision = right_num / (len(pred_tokenized) + 1e-9)
        macro_f1 = 2 * macro_recall * macro_precision / (macro_recall + macro_precision + 1e-9)
        macro_recall_list.append(macro_recall)
        macro_precision_list.append(macro_precision)
        macro_f1_list.append(macro_f1)

    # marco
    final_macro_recall = sum(macro_recall_list) / (len(macro_recall_list) + 1e-9)
    final_macro_precision = sum(macro_precision_list) / (len(macro_precision_list) + 1e-9)
    final_macro_f1 = sum(macro_f1_list) / (len(macro_f1_list) + 1e-9)

    # micro
    recall_acc = total_right_num / (total_gt_num + 1e-9)
    preci_acc = total_right_num / (total_pred_num + 1e-9)
    hmean = 2 * recall_acc * preci_acc / (recall_acc + preci_acc + 1e-9)
    vbs_eval_result = {
        'macro_recall': final_macro_recall, 'macro_precision': final_macro_precision, 'macro_f1_score': final_macro_f1,
        'micro_recall': recall_acc, 'micro_precision': preci_acc, 'mirco_f1_score': hmean
    }
    eval_result = vbs_eval_result if is_verbose else {'macro_f1_score': final_macro_f1, 'mirco_f1_score': hmean}
    return eval_result