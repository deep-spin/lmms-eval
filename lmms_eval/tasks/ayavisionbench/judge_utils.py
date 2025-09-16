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
import random


from lmms_eval.tasks.ayavisionbench.judge_templates import (
    COMPARATIVE_GEN_USER_PROMPT,
    COMPARATIVE_GEN_SYSTEM_PROMPT,
    COMPARATIVE_SYS_PROMPT_NO_GEN,
    COMPARATIVE_USER_PROMPT_NO_GEN,
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



def set_prompts(judge_config,questions,preds,baseline_model_outputs=None,language="en",random_ordering=False,seed=None):
    if random_ordering:
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(42)
    
    order_flags = []  # True = (baseline first, pred second), False = (pred first, baseline second)

    prompts = []
    if judge_config["judge_prompt_type"] == "comparative_gen":
        system_prompt = COMPARATIVE_GEN_SYSTEM_PROMPT
        user_prompt_template = COMPARATIVE_GEN_USER_PROMPT

        if not random_ordering:
            prompts = [user_prompt_template.format(question=question,answer_1=base_output,answer_2=pred) for question,pred,base_output in zip(questions,preds,baseline_model_outputs)]
        else:
            for question, pred, base_output in zip(questions, preds, baseline_model_outputs):
                if random.random() < 0.5:
                    # baseline first
                    prompts.append(user_prompt_template.format(question=question, answer_1=base_output, answer_2=pred))
                    order_flags.append(True)
                else:
                    # pred first
                    prompts.append(user_prompt_template.format(question=question, answer_1=pred, answer_2=base_output))
                    order_flags.append(False)

    elif judge_config["judge_prompt_type"] == "direct_assessment":
        system_prompt = DIRECT_ASSESSMENT_SYSTEM_PROMPT
        user_prompt_template = DIRECT_ASSESSMENT_USER_PROMPT
        prompts = [user_prompt_template.format(question=question,answer=pred) for question,pred in zip(questions,preds)]
    elif judge_config["judge_prompt_type"] == "comparative":
        system_prompt = COMPARATIVE_SYS_PROMPT_NO_GEN
        user_prompt_template = COMPARATIVE_USER_PROMPT_NO_GEN

        if not random_ordering:
            prompts = [user_prompt_template.format(question=question,completion_a=base_output,completion_b=pred,language=language) for question,pred,base_output in zip(questions,preds,baseline_model_outputs)]
        else:
            for question, pred, base_output in zip(questions, preds, baseline_model_outputs):
                if random.random() < 0.5:
                    # baseline first
                    prompts.append(user_prompt_template.format(question=question, completion_a=base_output, completion_b=pred, language=language))
                    order_flags.append(True)
                else:
                    # pred first
                    prompts.append(user_prompt_template.format(question=question, completion_a=pred, completion_b=base_output, language=language))
                    order_flags.append(False)

    else:
        raise ValueError(f"Invalid judge prompt type: {judge_config['judge_prompt_type']}")
    
    if not random_ordering:
        order_flags = None

    return system_prompt, prompts, order_flags



def run_judge(
    questions: List[str],
    preds: List[str],
    judge_config: Dict,
    baseline_model_outputs: Optional[List[str]] = None,
    images: Optional[Dict[str, Union[bytes, None]]] = None,
    language: str = "en",
    random_ordering: bool = False,
    seed: Optional[int] = None
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
    system_prompt, user_prompts, order_flags = set_prompts(judge_config, questions, preds, baseline_model_outputs,language,random_ordering,seed)
    
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
    return responses, order_flags


def parse_comparative_response(response: str,baseline_first=True) -> str:
    """
    Extract verdict pattern from judge response.
    Matches: [[A>>B]], [[A≫B]], [[A>B]], [[A=B]], [[B>A]], [[B>>A]], [[B≫A]]
    
    Args:
        response: String containing judge's response
        
    Returns:
        str: Matched verdict pattern or None if no match found
    """
    # Regex pattern to match all possible verdicts, supporting ">>" and "≫"
    pattern = r'\[\[(A(?:>>|≫)B|A>B|A=B|B>A|B(?:>>|≫)A)\]\]'
    match = re.search(pattern, response)
    if baseline_first:
        return match.group(0).replace("A","baseline").replace("B","model") if match.group(0) else None
    else:
        return match.group(0).replace("A","model").replace("B","baseline") if match.group(0) else None


def extract_judge_decision(response: str, baseline_first: bool = True) -> str:
    """Extract judge decision from response with simplified pattern matching."""
    pattern = r'\[\[(A(?:>>|≫)B|A>B|A=B|B>A|B(?:>>|≫)A)\]\]'
    match = re.search(pattern, response)
    
    if not match:
        logging.warning(f"No valid judge decision pattern found in response: {response[:100]}...")
        return None
    
    decision = match.group(0)
    if baseline_first:
        return decision.replace("A", "baseline").replace("B", "model")
    else:
        return decision.replace("A", "model").replace("B", "baseline")

def parse_judge_responses(responses,judge_config,position_ordering_list=None):
    logger.info(f"Parsing responses for {judge_config['judge_prompt_type']} judge...")
    parsed_responses = []
    if position_ordering_list is None:
        for idx,response in enumerate(responses):
            if response is None:
                parsed_responses.append("None")
                logger.warning(f"Response is None for {idx}.")
                continue
            response_data = response["choices"][0]["message"]["content"]
            if judge_config["judge_prompt_type"] == "comparative":
                parsed_answer = parse_comparative_response(response_data,baseline_first=True)
                if parsed_answer is None:
                    logger.warning(f"Parsed answer is None for {idx}. Response: {response_data}")
                    parsed_answer = "None"
                parsed_responses.append(parsed_answer)
            elif judge_config["judge_prompt_type"] == "direct_assessment":
                raise NotImplementedError("Direct assessment parsing not implemented yet!")
    else:
        for idx, (response,pos_order) in enumerate(zip(responses,position_ordering_list)):
            if response is None:
                parsed_responses.append("None")
                logger.warning(f"Response is None for {idx}.")
                continue
            response_data = response["choices"][0]["message"]["content"]
            if judge_config["judge_prompt_type"] == "comparative":
                parsed_answer = parse_comparative_response(response_data,baseline_first=pos_order)
                if parsed_answer is None:
                    logger.warning(f"Parsed answer is None for {idx}. Response: {response_data}")
                    parsed_answer = "None"
                parsed_responses.append(parsed_answer)
            elif judge_config["judge_prompt_type"] == "direct_assessment":
                raise ValueError("Direct assessment should not be used with position ordering list!This is used only for comparative judge.")
    return parsed_responses


def compute_results(responses,judge_config):
    logger.info(f"Computing results for {judge_config['judge_prompt_type']} judge...")
    if judge_config["judge_prompt_type"] == "comparative":
        baseline_better_than_model = 0
        baseline_significantly_better_than_model = 0
        model_better_than_baseline = 0
        model_significantly_better_than_baseline = 0
        model_equal_to_baseline = 0
        no_answer = 0
        none_answer = 0
        for response in responses:
            if response == "[[model>>baseline]]" or response == "[[model≫baseline]]":
                model_significantly_better_than_baseline += 1  
            elif response == "[[model>baseline]]":
                model_better_than_baseline += 1
            elif response == "[[baseline>>model]]" or response == "[[baseline≫model]]":
                baseline_significantly_better_than_model += 1
            elif response == "[[baseline>model]]":
                baseline_better_than_model += 1
            elif response == "[[model=baseline]]" or response == "[[baseline=model]]":
                model_equal_to_baseline += 1
            elif response == "None":
                logger.warning(f"None answer for {response}")
                none_answer += 1
            else:
                logger.warning(f"This response is not getting matched: {response}")
                no_answer += 1
        results = {
            "baseline_better_than_model": baseline_better_than_model/len(responses),
            "baseline_significantly_better_than_model": baseline_significantly_better_than_model/len(responses),
            "model_better_than_baseline": model_better_than_baseline/len(responses),
            "model_significantly_better_than_baseline": model_significantly_better_than_baseline/len(responses),
            "model_equal_to_baseline": model_equal_to_baseline/len(responses),
            "none_answer": none_answer/len(responses),
            "no_answer_matched": no_answer/len(responses)
        }
    elif judge_config["judge_prompt_type"] == "direct_assessment":
        raise NotImplementedError("Direct assessment results computation not implemented yet!")
    else:
        raise ValueError(f"Invalid judge prompt type: {judge_config['judge_prompt_type']}")
    return results