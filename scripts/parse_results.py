import argparse
import json


def get_results_m3exam(results):
    score_en = results["m3exam_english"]["m3exam,none"]
    results_multi = 0
    results_multi = [
        result["m3exam,none"]
        for task, result in results.items()
        if (task != "m3exam_english" and task != "m3exam")
    ]
    score_multi = sum(results_multi) / len(results_multi)
    return score_en * 100, score_multi * 100


def get_results_marvl(results):
    score_en = results["nlvr2"]["exact_match,none"]
    results_multi = 0
    results_multi = [
        result["exact_match,none"]
        for task, result in results.items()
        if (task != "marvl" and task != "nlvr2")
    ]
    score_multi = sum(results_multi) / len(results_multi)
    return score_en * 100, score_multi * 100


def get_results_maxm(results):
    score_en = results["maxm_en"]["relaxed_accuracy,none"]
    results_multi = 0
    results_multi = [
        result["relaxed_accuracy,none"]
        for task, result in results.items()
        if (task != "maxm" and task != "maxm_en")
    ]
    score_multi = sum(results_multi) / len(results_multi)
    return score_en, score_multi


def get_results_xgqa(results):
    score_en = results["xgqa_en"]["exact_match,none"]
    results_multi = 0
    results_multi = [
        result["exact_match,none"]
        for task, result in results.items()
        if (task != "xgqa" and task != "xgqa_en")
    ]
    score_multi = sum(results_multi) / len(results_multi)
    return score_en * 100, score_multi * 100


def get_results_xm100(results):
    score_en = results["xm100_en"]["xm100_ROUGE_L,none"]
    results_multi = 0
    results_multi = [
        result["xm100_ROUGE_L,none"]
        for task, result in results.items()
        if (task != "xm100" and task != "xm100_en")
    ]
    score_multi = sum(results_multi) / len(results_multi)
    return score_en * 100, score_multi * 100


def get_results_xmmmu(results):
    score_en = results["mmmu_English_val"]["mmmu_acc,none"]
    results_multi = 0
    results_multi = [
        result["mmmu_acc,none"]
        for task, result in results.items()
        if (task != "xmmmu" and task != "mmmu_English_val")
    ]
    score_multi = sum(results_multi) / len(results_multi)
    return score_en * 100, score_multi * 100


def get_results_multilingual_llava_bench(results):
    score_en = results["multilingual_llava_bench_english"]["gpt_eval_llava_all,none"]
    results_multi = 0
    results_multi = [
        result["gpt_eval_llava_all,none"]
        for task, result in results.items()
        if (
            task != "multilingual_llava_bench"
            and task != "multilingual_llava_bench_english"
        )
    ]
    score_multi = sum(results_multi) / len(results_multi)
    return score_en, score_multi


def get_results_ai2d(results):
    score_en = results["ai2d"]["exact_match,flexible-extract"]
    return score_en * 100, None


def get_results_mmmu(results):
    score_en = results["mmmu_val_group_img"]["mmmu_acc,none"]
    return score_en * 100, None


def get_results_textvqa(results):
    score_en = results["textvqa_val"]["exact_match,none"]
    return score_en * 100, None

def get_results_scienceqa(results):
    score_en = results["scienceqa"]["exact_match,none"]
    return score_en * 100, None

def process_results(data, task):
    results = data["results"]
    score_en, score_multi = result_functions[task](results)
    print(f"============== Results for {task} ==============")
    if score_multi:
        print(f"{score_en:.2f}\t{score_multi:.2f}")
    else:
        print(f"{score_en:.2f}")


result_functions = {
    "ai2d": get_results_ai2d,
    "mmmu": get_results_mmmu,
    "textvqa": get_results_textvqa,
    "m3exam": get_results_m3exam,
    "marvl": get_results_marvl,
    "maxm": get_results_maxm,
    "xgqa": get_results_xgqa,
    "xm100": get_results_xm100,
    "xmmmu": get_results_xmmmu,
    "mllava_bench": get_results_multilingual_llava_bench,
    "scienceqa": get_results_scienceqa,
}


def main():
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file")
    parser.add_argument("--task", type=str, help="The name of the task", required=True)
    args = parser.parse_args()

    with open(args.input_file, "r") as file:
        data = json.load(file)

    processed_data = process_results(data, task=args.task)


if __name__ == "__main__":
    main()
