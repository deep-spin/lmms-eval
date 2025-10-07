import json
import argparse
import re 
import os
import numpy as np
from typing import Dict


ENGLISH_TASKS = {
    "marvl":"nlvr2",
    "m3exam":"m3exam_english",
    "maxm":"maxm_en",
    "xm100":"xm100_en",
    "xmmmu":"mmmu_English_val",
    "xgqa":"xgqa_en",
    'cc-ocr-multi-lan':None,
}


def parse_arguments():
    """
    Parse command-line arguments using argparse.
    Returns:
        Namespace: Parsed arguments containing input_file and output_file paths.
    """
    parser = argparse.ArgumentParser(description="Calculate the average metrics from a JSON file.")
    parser.add_argument("--inputfile", type=str, help="Path to the input JSON file.")
    parser.add_argument("--outputdir", type=str, help="Path to the output dir.")
    return parser.parse_args()

def aggregate_results(data):
    
    subtasks_dict = data["group_subtasks"]
    pattern = r"^[a-z0-9_]+(?<!_stderr)(?=,none$)"
    pattern = r"^(?!.*_stderr)[^,]+(?=,none$)"

    subtask_collect_multi_results={maintask_name:{} for maintask_name in subtasks_dict }
    subtask_collect_eng_results={maintask_name:{} for maintask_name in subtasks_dict }
    for maintask_name in subtasks_dict:
        main_task_dict_multi = {}
        main_task_dict_eng = {}
        for subtask in subtasks_dict[maintask_name]:
            if "cc-ocr-multi-lan" in subtask:
                draft_dict = data["results"][subtask]["ocr_results,none"]
                metric_names = list(draft_dict.keys())
                if subtask==ENGLISH_TASKS[maintask_name]:
                    for metric in metric_names:
                        value = draft_dict[f"{metric}"]
                        if metric not in main_task_dict_eng:
                            main_task_dict_eng[metric] = [value]
                        else:
                            main_task_dict_eng[metric].append(value)
                else:
                    for metric in metric_names:
                        value = draft_dict[f"{metric}"]
                        if metric not in main_task_dict_multi:
                            main_task_dict_multi[metric] = [value]
                        else:
                            main_task_dict_multi[metric].append(value)
            else:
                draft_dict = data["results"][subtask]
                metric_names = [re.match(pattern, field).group() for field in draft_dict if re.match(pattern, field)]
                if subtask==ENGLISH_TASKS[maintask_name]:
                    for metric in metric_names:
                        value = draft_dict[f"{metric},none"]
                        if metric not in main_task_dict_eng:
                            main_task_dict_eng[metric] = [value]
                        else:
                            main_task_dict_eng[metric].append(value)
                else:
                    for metric in metric_names:
                        value = draft_dict[f"{metric},none"]
                        if metric not in main_task_dict_multi:
                            main_task_dict_multi[metric] = [value]
                        else:
                            main_task_dict_multi[metric].append(value)
        
        subtask_collect_multi_results[maintask_name] = main_task_dict_multi
        subtask_collect_eng_results[maintask_name] = main_task_dict_eng
    
    aggregated_results_multi = {}
    aggregated_results_eng = {}
    for task_name in subtask_collect_multi_results:
        aggregated_results_multi[task_name] = {metric:np.mean(subtask_collect_multi_results[task_name][metric]) for metric in subtask_collect_multi_results[task_name] if metric!="submission"}
        aggregated_results_eng[task_name] = {metric:np.mean(subtask_collect_eng_results[task_name][metric]) for metric in subtask_collect_eng_results[task_name] if metric!="submission"}

    return aggregated_results_eng, aggregated_results_multi


def main():
    # Parse arguments
    args = parse_arguments()

    # Load input JSON
    with open(args.inputfile, "r") as infile:
        data = json.load(infile)


    results_eng, results_multi = aggregate_results(data)

    # Prepare output directory
    os.makedirs(args.outputdir, exist_ok=True)

    # Write English task results to JSON file
    eng_output_path = os.path.join(args.outputdir, "english_tasks_results.json")
    with open(eng_output_path, "w") as eng_outfile:
        json.dump(results_eng, eng_outfile, indent=4)

    # Write multi-task results to JSON file
    multi_output_path = os.path.join(args.outputdir, "multi_tasks_results.json")
    with open(multi_output_path, "w") as multi_outfile:
        json.dump(results_multi, multi_outfile, indent=4)

    print(f"English task results saved to {eng_output_path}.")
    print(f"Multi-task results saved to {multi_output_path}.")


if __name__ == "__main__":
    main()
