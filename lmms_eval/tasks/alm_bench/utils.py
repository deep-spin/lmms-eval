from PIL import Image


def exact_match(pred, target):
    if pred == target:
        return 1
    else:
        return 0
    
def alm_bench_doc_to_visual(doc):
    image = (doc['file_name']).convert('RGB')
    return [image]


def alm_bench_doc_to_text(doc, lmms_eval_specific_kwargs):
    # Process MCQ question to extract answer, at the moment since we are not filtered the data set, I only process MCQ question types
    question = doc["Translated_Question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs["prompt_format"] == "mcq" and doc["Question_Type"] == "Multiple Choice Questions":
        choices = doc["Translated_Answer"].split("(Options: ")[1].rstrip(")").split(", ")
        choices_str = "\n".join([f"{choice}" for choice in choices])
        print(f"{pre_prompt}{question}\n{choices_str}{post_prompt}")
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    # elif lmms_eval_specific_kwargs["prompt_format"] == "tf" and doc["Question_Type"] == "True False Question":
    #     options = "\n".join(["true","false"])
    #     return f"{pre_prompt}{question}{options}{post_prompt}"
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
        return doc["Translated_Answer"].split(" (Options: ")[0]
    # elif model_specific_target_kwargs == "tf":
    #     return doc["Translated_Answer"]
    else:
        return "NO ANSWER"
