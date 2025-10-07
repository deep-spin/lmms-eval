#!/bin/bash

# -------------------------------
# Activate the Python environment
# -------------------------------
env_path=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/venv-lmms-eval
source $env_path/bin/activate

# -------------------------------
# Export the API key for OpenAI
# -------------------------------

# -------------------------------
# Judge and API parameters
# -------------------------------
API_TYPE="openai"
JUDGE_MODEL_NAME="gpt-4o-mini"
JUDGE_PROMPT_TYPE="comparative"
TEXT_ONLY="False"
MAX_TOKENS=2048
TEMPERATURE=0.0
TOP_P=1.0
TENSOR_PARALLEL_SIZE=1

# -------------------------------
# Define language pairs and models
# -------------------------------
lps=("eng")  # List of language codes
models=("Qwen/Qwen2.5-VL-7B-Instruct" )  # List of model names

# Baseline model name (must match directory structure)
BASELINE_MODEL_NAME="Unbabel__lnext-qwen2p5-7b-siglip2-v5"

# Root directory for all outputs
OUT_ROOT="/mnt/scratch-artemis/manos/data/tower-vision-eval-outputs/tower-vision/tower-vision-debug-results"

# -------------------------------
# Main loop over language pairs
# -------------------------------
for lp in "${lps[@]}"; do
    # ------------------------------------------
    # Find the most recent baseline output file
    # ------------------------------------------
    baseline_dir="$OUT_ROOT/aya-vision-bench-gen-${lp}/${BASELINE_MODEL_NAME}"
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
        model_output_dir="$OUT_ROOT/aya-vision-bench-gen-${lp}/${model_dir}"
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

        # Create the results directory using the version
        RESULTS_DIR="$OUT_ROOT/aya-vision-bench-gen-${lp}/${model_dir}/${MODEL_OUTPUT_VERSION_CLEAN}_${model_dir}_vs_${BASELINE_MODEL_NAME}_with_${JUDGE_MODEL_NAME}"
        mkdir -p "$RESULTS_DIR" 

        judge_script_path=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/lmms-eval/lmms_eval/tasks/ayavisionbench/judge_eval.py
        # ------------------------------------------
        # Run the judge evaluation script
        # ------------------------------------------
        echo "Running for model: $model, language: $lp" 

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
            --tensor_parallel_size $TENSOR_PARALLEL_SIZE
    done
done