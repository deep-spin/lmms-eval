#!/bin/bash

# -------------------------------
# Activate the Python environment
# -------------------------------
env_path=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/venv-lmms-eval-final
source $env_path/bin/activate

# -------------------------------
# Export the API key for OpenAI
# -------------------------------


# -------------------------------
# Judge and API parameters
# -------------------------------
# API_TYPE="litellm"
# JUDGE_MODEL_NAME="litellm_proxy/neulab/claude-3-7-sonnet-20250219"
API_TYPE="openai"
JUDGE_MODEL_NAME="gpt-4o-mini"
JUDGE_PROMPT_TYPE="comparative"
TEXT_ONLY="False"
MAX_TOKENS=2048
TEMPERATURE=0.0
TOP_P=1.0
TENSOR_PARALLEL_SIZE=1
# define the api url for the judge model if 
# API_URL="https://cmu.litellm.ai"
# API_URL="https://api.openai.com/v1"

# -------------------------------
# Define language pairs and models
# -------------------------------
# lps=("eng" "de" "fr" "it" "es")  # List of language codes
lps=("eng")
models=('mistralai/Pixtral-12B-2409' 'CohereForAI/aya-vision-8b')  # List of model names
models=('mistralai/Pixtral-12B-2409')  # List of model names

# Baseline model name (must match directory structure)
BASELINE_MODEL_NAME='utter-project/EuroVLM-9B-Preview' 

# Root directory for all outputs
OUT_ROOT="/mnt/scratch-artemis/manos/data/tower-vision-eval-outputs/tower-vision/aya-vision-bench-gen-final-results/aya-vision-bench-gen/v6"

# -------------------------------
# Main loop over language pairs
# -------------------------------
for lp in "${lps[@]}"; do
    # ------------------------------------------
    # Find the most recent baseline output file
    # ------------------------------------------
    baseline_model_dir="${BASELINE_MODEL_NAME//\//__}"
    baseline_dir="$OUT_ROOT/${baseline_model_dir}"
    BASELINE_OUTPUT_PATH=$(ls -1t "$baseline_dir"/*samples_aya-vision-bench-gen-${lp}.jsonl 2>/dev/null | head -n 1)
    if [ -z "$BASELINE_OUTPUT_PATH" ]; then
        echo "No baseline output file found for $BASELINE_MODEL_NAME and $lp in $baseline_dir, skipping language."
        continue
    fi

    # ------------------------------------------
    # Loop over models for this language
    # ------------------------------------------
    for model in "${models[@]}"; do
        # Convert model name to directory-friendly format
        model_dir="${model//\//__}"
        # ------------------------------------------
        # Find the most recent model output file
        # ------------------------------------------
        model_output_dir="$OUT_ROOT/${model_dir}"
        MODEL_OUTPUT_PATH=$(ls -1t "$model_output_dir"/*samples_aya-vision-bench-gen-${lp}.jsonl 2>/dev/null | head -n 1)
        if [ -z "$MODEL_OUTPUT_PATH" ]; then
            echo "No model output file found for $model ($model_dir) and $lp in $model_output_dir, skipping."
            continue
        fi

        # ------------------------------------------
        # Set output directory for judge results
        # ------------------------------------------
        # Extract the version (filename without extension) from the model output path
        MODEL_OUTPUT_FILENAME=$(basename "$MODEL_OUTPUT_PATH")
        MODEL_OUTPUT_VERSION="${MODEL_OUTPUT_FILENAME%.jsonl}"
        MODEL_OUTPUT_VERSION_CLEAN="${MODEL_OUTPUT_VERSION%_samples_aya-vision-bench-gen-${lp}}"

        JUDGE_MODEL_NAME_REPLACED="${JUDGE_MODEL_NAME//\//__}"
        BASELINE_MODEL_NAME_REPLACED="${BASELINE_MODEL_NAME//\//__}"
        # Create the results directory using the version
        RESULTS_DIR="$OUT_ROOT/${model_dir}/${MODEL_OUTPUT_VERSION_CLEAN}_${model_dir}_vs_${BASELINE_MODEL_NAME_REPLACED}_with_${JUDGE_MODEL_NAME_REPLACED}/${lp}"
        mkdir -p "$RESULTS_DIR"
        

        judge_script_path=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/lmms-eval/lmms_eval/tasks/ayavisionbench/judge_eval.py
        # ------------------------------------------
        # Run the judge evaluation script
        # ------------------------------------------
        echo "Running for model: $model, language: $lp" 

        # Conditionally add --api_url if API_URL is set and not empty
        api_url_arg=""
        if [ ! -z "$API_URL" ]; then
            api_url_arg="--api_url $API_URL"
        fi

        python $judge_script_path \
            --api_type $API_TYPE \
            --judge_model_name $JUDGE_MODEL_NAME \
            --judge_prompt_type $JUDGE_PROMPT_TYPE \
            --text_only $TEXT_ONLY \
            --model_output_path "$MODEL_OUTPUT_PATH" \
            --baseline_output_path "$BASELINE_OUTPUT_PATH" \
            --lp $lp \
            --output_dir $RESULTS_DIR \
            --save_judge_parsed_outputs True \
            --save_judge_full_responses True \
            --save_judge_results True \
            --max_tokens $MAX_TOKENS \
            --temperature $TEMPERATURE \
            --top_p $TOP_P \
            --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
            $api_url_arg

    done
done