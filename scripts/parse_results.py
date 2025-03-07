import argparse
import json
import glob
import os
from tabulate import tabulate

def get_results_m3exam(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["m3exam_english"]["m3exam,none"]) * 100
    scores["it"] = float(results["m3exam_italian"]["m3exam,none"]) * 100
    scores["pt"] = float(results["m3exam_portuguese"]["m3exam,none"]) * 100
    scores["zh"] = float(results["m3exam_chinese"]["m3exam,none"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

def get_results_chartqa(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["chartqa"]["relaxed_overall,none"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

"""
We use Exact Match Accuracy for this task.
This is based on the information provided in the origial paper: 
MaXM: Towards Multilingual Visual Question Answering (https://arxiv.org/pdf/2209.05401)
Section 5.1:
Evaluation Metrics. We use Exact Match Accu-
racy as the main evaluation measure for MaXM,
following previous work on VQA
"""
def get_results_maxm(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["maxm_en"]["relaxed_accuracy,none"])
    scores["fr"] = float(results["maxm_fr"]["relaxed_accuracy,none"])
    scores["zh"] = float(results["maxm_zh"]["relaxed_accuracy,none"])
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

def get_results_mme(results, task_name, model_name):
    rows = []
    scores = {}
    mme_cognition_score = float(results['mme']['mme_cognition_score,none'])
    mme_perception_score = float(results['mme']['mme_perception_score,none'])
    scores["en"] = f"{round(mme_cognition_score,2)}/{round(mme_perception_score,2)}"
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

def get_results_mmmu(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["mmmu_val"]["mmmu_acc,none"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

def get_results_ocrbench(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["ocrbench"]["ocrbench_accuracy,none"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

def get_results_scienceqa(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["scienceqa"]["exact_match,none"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

def get_results_textvqa(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["textvqa_val"]["exact_match,none"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

def get_results_xgqa(results, task_name, model_name):
    rows = []
    scores = {}
    scores["de"] = float(results["xgqa_de"]["exact_match,none"]) * 100
    scores["en"] = float(results["xgqa_en"]["exact_match,none"]) * 100
    scores["ko"] = float(results["xgqa_ko"]["exact_match,none"]) * 100
    scores["pt"] = float(results["xgqa_pt"]["exact_match,none"]) * 100
    scores["ru"] = float(results["xgqa_ru"]["exact_match,none"]) * 100
    scores["zh"] = float(results["xgqa_zh"]["exact_match,none"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

# def get_results_xm100(results, task_name, model_name):
#     rows = []
#     scores = {}
#     scores["de"] = float(results["xm100_de"]["xm100_CIDEr,none"]) * 100
#     scores["en"] = float(results["xm100_en"]["xm100_CIDEr,none"]) * 100
#     scores["es"] = float(results["xm100_es"]["xm100_CIDEr,none"]) * 100
#     scores["fr"] = float(results["xm100_fr"]["xm100_CIDEr,none"]) * 100
#     scores["it"] = float(results["xm100_it"]["xm100_CIDEr,none"]) * 100
#     scores["ko"] = float(results["xm100_ko"]["xm100_CIDEr,none"]) * 100
#     scores["nl"] = float(results["xm100_nl"]["xm100_CIDEr,none"]) * 100
#     scores["pt"] = float(results["xm100_pt"]["xm100_CIDEr,none"]) * 100
#     scores["ru"] = float(results["xm100_ru"]["xm100_CIDEr,none"]) * 100
#     scores["zh"] = float(results["xm100_zh"]["xm100_CIDEr,none"]) * 100
    
#     for kk, vv in scores.items():
#         rows.append({
#             "model": model_name,
#             "task": task_name,
#             "language": kk,
#             "score": vv
#         })
#     return rows

def get_results_xmmmu(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["mmmu_English_val"]["mmmu_acc,none"]) * 100
    scores["fr"] = float(results["mmmu_French_val"]["mmmu_acc,none"]) * 100
    scores["pt"] = float(results["mmmu_Portuguese_val"]["mmmu_acc,none"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows


def get_results_ai2d(results, task_name, model_name):
    rows = []
    scores = {}
    scores["en"] = float(results["ai2d"]["exact_match,flexible-extract"]) * 100
    
    for kk, vv in scores.items():
        rows.append({
            "model": model_name,
            "task": task_name,
            "language": kk,
            "score": vv
        })
    return rows

def process_results(data, task_name, model_name):
    results = data["results"]
    results = result_functions[task_name](results, task_name, model_name)
    return results


result_functions = {
    "m3exam": get_results_m3exam,
    "chartqa": get_results_chartqa,
    "maxm": get_results_maxm,
    "mme": get_results_mme,
    "mmmu": get_results_mmmu,
    "ocrbench": get_results_ocrbench,
    "scienceqa": get_results_scienceqa,
    "textvqa": get_results_textvqa,
    "xgqa": get_results_xgqa,
    # "xm100": get_results_xm100,
    "xmmmu": get_results_xmmmu,
    "ai2d": get_results_ai2d,
}


def main():
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument("--input-dir", type=str, help="Path to the input JSON file")
    parser.add_argument("--task", type=str, help="The name of the task", required=True)
    parser.add_argument("--model", type=str, help="The name of the model", required=True)
    args = parser.parse_args()
    
    model_name = args.model.replace("/", "__")
    task = args.task
    
    

    input_dir = os.path.join(args.input_dir, task, model_name)
    input_files = glob.glob(os.path.join(input_dir, "*results.json"))
    if not input_files:
        print(f"No files found in directory {args.input_dir} with pattern *results.json")
    assert len(input_files) == 1, f"Found more than one file in {args.input_dir} with pattern *results.json"
    input_file = input_files[0]

    
    
    with open(input_file, "r") as file:
        data = json.load(file)
        
    processed_data = process_results(data, task_name=args.task, model_name=model_name)
    # print(tabulate(processed_data, headers="keys", tablefmt=","))
    # print(processed_data)
    for row in processed_data:
        print(f"{row['model']},{row['task']},{row['language']},{row['score']}")

if __name__ == "__main__":
    main()
