import yaml
from pathlib import Path
import os
import requests
import json
import re
from loguru import logger
from dotenv import load_dotenv
import base64
import io
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import litellm


from lmms_eval.tasks.ayavisionbench.judge_templates import (
    COMPARATIVE_GEN_USER_PROMPT,
    COMPARATIVE_GEN_SYSTEM_PROMPT,
    DIRECT_ASSESSMENT_SYSTEM_PROMPT,
    DIRECT_ASSESSMENT_USER_PROMPT 
)


def img_bytes_to_url(image_dict: dict) -> str:
    """
    Convert image bytes to base64 URL.
    
    Args:
        image_dict: Dictionary containing image bytes
        
    Returns:
        str: base64 URL-encoded JPEG string
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_dict['bytes']))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPEG to buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        
        # Convert to base64
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        # Create data URL
        data_url = f"data:image/jpeg;base64,{base64_image}"
        
        return data_url
        
    except Exception as e:
        logger.error(f"Error converting image bytes to URL: {str(e)}")
        return None

def get_judge_config():
    with open(Path(__file__).parent / "eval_judge_template.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
        config = yaml.safe_load("".join(safe_data))
    return config


def set_prompts(judge_config,questions,preds,baseline_model_outputs=None):
    if judge_config["judge_prompt_type"] == "comparative":
        system_prompt = COMPARATIVE_GEN_SYSTEM_PROMPT
        user_prompt_template = COMPARATIVE_GEN_USER_PROMPT
        prompts = [user_prompt_template.format(question=question,answer_1=base_output,answer_2=pred) for question,pred,base_output in zip(questions,preds,baseline_model_outputs)]
    elif judge_config["judge_prompt_type"] == "direct_assessment":
        system_prompt = DIRECT_ASSESSMENT_SYSTEM_PROMPT
        user_prompt_template = DIRECT_ASSESSMENT_USER_PROMPT
        prompts = [user_prompt_template.format(question=question,answer=pred) for question,pred in zip(questions,preds)]
    else:
        raise ValueError(f"Invalid judge prompt type: {judge_config['judge_prompt_type']}")
    return system_prompt, prompts



def run_judge(
    questions: List[str],
    preds: List[str],
    judge_config: Dict,
    baseline_model_outputs: Optional[List[str]] = None,
    images: Optional[Dict[str, Union[bytes, None]]] = None,
) -> List[str]:
    """
    Run judge evaluation on predictions using LiteLLM.
    
    Args:
        questions: List of questions to evaluate
        preds: List of predicted outputs to evaluate
        judge_config: Configuration dictionary for the judge
        baseline_model_outputs: Optional list of baseline outputs to compare against
        images: Optional dictionary containing image data with format {'bytes': bytes, 'path': None}
        
    Returns:
        List[Dict]: List of parsed responses from the judge
    """
    logger.info(f"Selected judge type: {judge_config['judge_prompt_type']}")
    load_dotenv()
    
    # Set up model configuration
    model = judge_config["judge_model_name"]
    


    if judge_config["api_type"] == "anthropic":
        if os.getenv("ANTHROPIC_API_KEY"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError("No API key found. Please set ANTHROPIC_API_KEY environment variable or define the api_key in the judge_config")
    elif judge_config["api_type"] == "openai":
        if os.getenv("OPENAI_API_KEY"):
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("No API key found. Please set OPENAI_API_KEY environment variable or define the api_key in the judge_config")
    elif judge_config["api_type"] == "litellm":
        if os.getenv("LITELLM_API_KEY"):
            api_key = os.getenv("LITELLM_API_KEY")
        else:
            raise ValueError("No API key found. Please set LITELLM_API_KEY environment variable or define the api_key in the judge_config")
    else:
        raise ValueError("No API key found. Please set LITELLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY environment variable or define the api_key in the judge_config")

    

    # if judge_config["text_only"] == False:
    #     assert litellm.supports_vision(model=model),f"Selected judge model:{model} does not support vision"

    # Get prompts
    system_prompt, user_prompts = set_prompts(judge_config, questions, preds, baseline_model_outputs)
    
    responses = []
    total_items = len(user_prompts)
    logger.info(f"Running judge evaluation for {total_items} items...")
    
    with tqdm(total=total_items, 
              desc="Running judge evaluation", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
              ncols=100) as pbar:
              
        for idx, (image, prompt) in enumerate(zip(images, user_prompts)):
            try:
                # Prepare messages based on whether image is present
                if image is not None or judge_config["text_only"] == False:
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_bytes_to_url(image)}}
                        ]}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]

                # Make the LiteLLM call
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    max_tokens= judge_config["max_tokens"] if "max_tokens" in judge_config else None,
                    temperature=judge_config["temperature"] if "temperature" in judge_config else None,
                    top_p=judge_config["top_p"] if "top_p" in judge_config else None,
                    api_key=api_key,
                    base_url=judge_config.get("api_url")
                )
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error in LiteLLM call: {str(e)}")
                responses.append(None)
            
            pbar.set_postfix({"Current": f"{idx+1}/{total_items}"})
            pbar.update(1)
    return responses

def parse_comparative_response(response: str) -> str:
    """
    Extract verdict pattern from judge response.
    Matches: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], [[B>>A]]
    
    Args:
        response: String containing judge's response
        
    Returns:
        str: Matched verdict pattern or None if no match found
    """
    # Regex pattern to match all possible verdicts
    pattern = r'\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]'
    # Search for pattern in response
    match = re.search(pattern, response)
    # Return matched pattern or None
    return match.group(0) if match else None


def parse_judge_responses(responses,judge_config):
    logger.info(f"Parsing responses for {judge_config['judge_prompt_type']} judge...")
    parsed_responses = []
    for response in responses:
        if response is None:
            parsed_responses.append(None)
            continue
        response_data = response["choices"][0]["message"]["content"]
        if judge_config["judge_prompt_type"] == "comparative":
            parsed_responses.append(parse_comparative_response(response_data))
        elif judge_config["judge_prompt_type"] == "direct_assessment":
            raise NotImplementedError("Direct assessment parsing not implemented yet!")
    return parsed_responses


def compute_results(responses,judge_config):
    logger.info(f"Computing results for {judge_config['judge_prompt_type']} judge...")
    if judge_config["judge_prompt_type"] == "comparative":
        a_better_than_b = 0
        a_significantly_better_than_b = 0
        b_better_than_a = 0
        b_significantly_better_than_a = 0
        a_equal_to_b = 0
        no_answer = 0
        for response in responses:
            if response == "[[A>>B]]":
                a_significantly_better_than_b += 1  
            elif response == "[[A>B]]":
                a_better_than_b += 1
            elif response == "[[B>>A]]":
                b_significantly_better_than_a += 1
            elif response == "[[B>A]]":
                b_better_than_a += 1
            elif response == "[[A=B]]":
                a_equal_to_b += 1
            else:
                no_answer += 1
        results = {
            "baseline_better_than_model": a_better_than_b/len(responses),
            "baseline_significantly_better_than_model": a_significantly_better_than_b/len(responses),
            "model_better_than_baseline": b_better_than_a/len(responses),
            "model_significantly_better_than_baseline": b_significantly_better_than_a/len(responses),
            "model_equal_to_baseline": a_equal_to_b/len(responses),
            "no_answer_matched": no_answer/len(responses)
        }
    elif judge_config["judge_prompt_type"] == "direct_assessment":
        raise NotImplementedError("Direct assessment results computation not implemented yet!")
    else:
        raise ValueError(f"Invalid judge prompt type: {judge_config['judge_prompt_type']}")
    return results