import argparse
import json
from PIL import Image
import os
import io
from loguru import logger

from datasets import load_dataset
from lmms_eval.tasks.ayavisionbench.judge_utils import run_judge,compute_results,parse_judge_responses

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type",choices=["openai", "anthropic","litellm"], type=str, required=True)
    parser.add_argument("--judge_model_name",default="gpt-4o-mini", type=str, required=True)
    parser.add_argument("--judge_prompt_type",default="comparative",choices=["comparative", "direct assessment"], type=str, required=True)
    parser.add_argument("--text_only",default=False, type=bool, required=True,help="if true, the judge model will not see the image.")
    parser.add_argument("--model_output_path", type=str, required=True)
    parser.add_argument("--baseline_output_path", type=str, required=True)
    parser.add_argument("--lp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_judge_parsed_outputs",default=True, type=bool, required=True,help="if true, the judge parsed outputs will be saved.")
    parser.add_argument("--save_judge_full_responses",default=True, type=bool, required=True,help="if true, the judge full responses will be saved.")
    parser.add_argument("--save_judge_results",default=True, type=bool, required=True,help="if true, the judge results will be saved.")
    parser.add_argument("--max_tokens",default=2048, type=int, required=True,help="the max tokens for the judge model.")
    parser.add_argument("--temperature",default=0.0, type=float, required=True,help="the temperature for the judge model.")
    parser.add_argument("--top_p",default=1.0, type=float, required=True,help="the top p for the judge model.")
    parser.add_argument("--tensor_parallel_size",default=1, type=int, required=True,help="the tensor parallel size for the judge model.")
    parser.add_argument("--api_url",default=None, type=str,help="the api url for the judge model.")
    parser.add_argument("--random_ordering",default=False, action="store_true",help="if true the position of the baseline and model outputs will be shuffled randomly.")
    parser.add_argument("--seed",default=None, type=int,help="the seed for the judge model.")

    args = parser.parse_args()
    return args

def pil_to_image_dict(pil_img):
    """
    Convert a PIL Image to an image dict with 'bytes' and 'path' keys.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")  # or "JPEG" if you prefer
    img_bytes = buf.getvalue()
    return {'bytes': img_bytes, 'path': None}

def load_aya_vision_bench_data(language):
    mapping_language_to_dataset_name = {
        "cs": "ces_Latn",
        "eng": "eng_Latn",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "es": "spa_Latn",
        "it": "ita_Latn",
        "ko": "kor_Hang",
        "nl": "nld_Latn",
        "pt": "por_Latn",
        "ru": "rus_Cyrl",
        "zh": "zho_Hans",
    }
    dataset = load_dataset("CohereForAI/AyaVisionBench",name=mapping_language_to_dataset_name[language],split="test")
    logger.info(f"Loaded Aya Vision Bench dataset for language: {language}")
    # dataset = dataset.select(range(5))
    # Extract questions and images from the dataset
    questions = [item["prompt"] for item in dataset]

    images_bytes = []
    questions = []
    for item in dataset:
        questions.append(item["prompt"])
        pil_img = item["image"][0]
        images_bytes.append(pil_to_image_dict(pil_img))
    return dataset,questions,images_bytes

def load_model_outputs(model_outputs_path):
    filtered_responses = []
    with open(model_outputs_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'filtered_resps' in data:
                    # Handle both single string and list of strings cases
                    resps = data['filtered_resps']
                    if isinstance(resps, list):
                        filtered_responses.extend(resps)
                    else:
                        filtered_responses.append(resps)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
                continue
    return filtered_responses

def extract_judge_message(response):
    """
    Extracts the output message content from a judge response object.
    Assumes response.choices[0].message.content exists.
    """
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return None

if __name__ == "__main__":
    
    args = load_args()

    logger.info(f"Running judge for language: {args.lp}.")
    # 1. Load Aya Vision Bench data
    aya_dataset,questions,images_bytes = load_aya_vision_bench_data(args.lp)

    # # 2. Load model outputs
    baseline_model_outputs = load_model_outputs(args.baseline_output_path)
    model_outputs = load_model_outputs(args.model_output_path)
    logger.info(f"Loaded baseline and model outputs for language: {args.lp}")
    
    # 3. Prepare judge_config dictionary
    judge_config = {
        "api_type": args.api_type,
        "judge_model_name": args.judge_model_name,
        "judge_prompt_type": args.judge_prompt_type,
        "text_only": args.text_only,
        # "api_url": os.environ.get("API_URL", None),  # or set as needed
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tensor_parallel_size": args.tensor_parallel_size,
        "api_url": args.api_url
        # Add any other config keys your run_judge expects
    }

    # 4. Run judge
    logger.info(f"Running judge for language: {args.lp}")
    full_responses, position_ordering = run_judge(questions, model_outputs, judge_config, baseline_model_outputs, images_bytes,args.lp,args.random_ordering,args.seed)

    logger.info(f"Judge completed for language: {args.lp}")
    parsed_responses = parse_judge_responses(full_responses, judge_config,position_ordering_list=position_ordering)
    logger.info(f"Parsed responses completed for language: {args.lp}")
    
    # 7. Compute results
    results = compute_results(parsed_responses, judge_config)

    # 5. Save results
    if args.save_judge_parsed_outputs:
        with open(os.path.join(args.output_dir, "judge_results_parsed.json"), "w") as f:
            logger.info(f"Saving parsed responses...")
            json.dump(parsed_responses, f)


    if args.save_judge_results:
        with open(os.path.join(args.output_dir, "judge_results.json"), "w") as f:
            logger.info(f"Saving results...")
            json.dump(results, f)
    

        

    
    