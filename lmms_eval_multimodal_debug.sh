#!/bin/bash

#SBATCH --job-name=pangea_eval   # Job name
#SBATCH --time=48:00:00         # Run time (hh:mm:ss) 
#SBATCH --gres=gpu:2           # Number of GPUs to be used
#SBATCH --qos=gpu-medium         # QOS to be used
#SBATCH --partition=a6000         # QOS to be used
#SBATCH --mem=500GB         # mem to use
#SBATCH --output=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/%x_%j.out  # Standard output
#SBATCH --error=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/%x_%j.err   # Standard error

module load openjdk
source ~/.bashrc
out_root=/mnt/scratch-artemis/manos/data/tower-vision-eval-outputs/tower-vision/results-tower-vision-iclr-results
# out_root=/mnt/scratch-artemis/manos/data/tower-vision-eval-outputs/tower-vision/aya-vision-bench-gen-final-results
env_path=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/venv-lmms-eval-final
source $env_path/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn


declare -A system_prompts
system_prompts["alm-bench-en"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["alm-bench-de"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["alm-bench-es"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["alm-bench-fr"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["alm-bench-it"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["alm-bench-ko"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["alm-bench-nl"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["alm-bench-pt"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["alm-bench-ru"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
system_prompts["mmmu"]="End your answer with 'Final Answer: <your answer>' where <your answer> in the case of Multiple Choice Questions is strictly the option letter from the given choices only. In the case of open-ended questions it is strictly the answer only in a single word or phrase."
system_prompts["ai2d"]="End your answer with 'Final Answer: <your answer>' where <your answer> is strictly the option letter from the given choices only."
# system_prompts["aya-vision-bench-gen"]="You are given an image and a question. Answer the question based on the image."



declare -A model_args_map
# model_args_map["neulab/Pangea-7B"]="device_map=auto"
# model_args_map["Qwen/Qwen2-VL-7B"]="device_map=auto,use_cache=False"
# model_args_map["Qwen/Qwen2-VL-7B-Instruct"]="device_map=auto,use_cache=False"
# model_args_map["OpenGVLab/InternVL2-8B"]="device_map=auto"
# model_args_map["OpenGVLab/InternVL2-Llama3-76B"]="device_map=auto,trust_remote_code=True"
# model_args_map["Qwen/Qwen2.5-VL-7B-Instruct"]="device_map=auto"

# model_args_map["microsoft/Phi-3-vision-128k-instruct"]="trust_remote_code=True"
# model_args_map["llava-hf/llava-1.5-7b-hf"]="device_map=auto"
# model_args_map["llava-hf/llava-v1.6-mistral-7b-hf"]="device_map=auto"

# model_args_map["Unbabel/qwen2p5-7b-hdr-sft-visionblocks-v0.3-notblocks"]="dtype=auto,trust_remote_code=true"
# model_args_map["Unbabel/qwen2p5-7b-hdr-sft-visionblocks-1102-v1"]="dtype=auto,trust_remote_code=true"
# model_args_map["Unbabel/qwen2p5-14b-hdr-sft-visionblocks-1102-v1"]="dtype=auto,trust_remote_code=true"
# model_args_map["Unbabel/qwen2p5-7b-hdr-sft-visionblocks-1102-v1-midtraining"]="dtype=auto,trust_remote_code=true"
# model_args_map["Unbabel/qwen2p5-7b-hdr-sft-visionblocks-v0.4"]="dtype=auto,trust_remote_code=true"
# model_args_map["Unbabel/lnext-qwen2p5-7b-siglip2-v5"]="device_map=auto"

# model_args_map["allenai/Molmo-7B-D-0924"]="device_map=auto"

# model_args_map["Unbabel/Tower4-Sugarloaf-Vision"]="device_map=auto"
# model_args_map["Unbabel/Tower4-Sugarloaf-Vision-merged"]="device_map=auto"
# model_args_map["Unbabel/lnext-tower4-sugarloaf-siglip2-v6"]="device_map=auto"
# model_args_map["utter-project/EuroVLM-9B-Preview"]="device_map=auto"
# model_args_map["CohereForAI/aya-vision-8b"]="device_map=auto"
# model_args_map["mistralai/Pixtral-12B-2409"]="gpu_memory_utilization=0.8,tokenizer_mode=mistral"
# model_args_map["Qwen/Qwen2.5-VL-7B-Instruct"]="device_map=auto,torch_dtype=auto"

model_args_map["mistralai/Pixtral-12B-2409"]="gpu_memory_utilization=0.8,tokenizer_mode=mistral"
model_args_map["CohereForAI/aya-vision-8b"]="device_map=auto"
model_args_map["allenai/Molmo-7B-D-0924"]="device_map=auto"
model_args_map["neulab/Pangea-7B"]="device_map=auto"
model_args_map["Qwen/Qwen2.5-VL-7B-Instruct"]="device_map=auto"
model_args_map["microsoft/Phi-3-vision-128k-instruct"]="device_map=auto"

model_args_map["utter-project/TowerVision-Plus-2B"]="device_map=auto,dtype=bfloat16"
model_args_map["utter-project/TowerVision-Plus-9B"]="device_map=auto,dtype=bfloat16"
model_args_map["utter-project/TowerVision-4-Anthill-CPT"]="device_map=auto,dtype=bfloat16"

declare -A model_types
# model_types["llava-hf/llava-1.5-7b-hf"]="llava_hf"
# model_types["llava-hf/llava-v1.6-mistral-7b-hf"]="llava_hf"
# model_types["Qwen/Qwen2-VL-7B-Instruct"]="qwen2_vl"
# model_types["neulab/Pangea-7B"]="pangea"
# model_types["Unbabel/lnext-qwen2p5-7b-siglip2-v5"]="llava"
# model_types["Unbabel/lnext-qwen2p5-14bb-siglip2-v5"]="llava"
# model_types["microsoft/Phi-3-vision-128k-instruct"]="phi3v"
# model_types["mistralai/Pixtral-12B-2409"]="pixtral"
# model_types["allenai/Molmo-7B-D-0924"]="molmo_hf"
# model_types["Qwen/Qwen2.5-VL-7B-Instruct"]="qwen2_5_vl"

model_types["mistralai/Pixtral-12B-2409"]="pixtral"
model_types["CohereForAI/aya-vision-8b"]="aya"
model_types["allenai/Molmo-7B-D-0924"]="molmo_hf"
model_types["neulab/Pangea-7B"]="pangea"
model_types["Qwen/Qwen2.5-VL-7B-Instruct"]="qwen2_5_vl"
model_types["microsoft/Phi-3-vision-128k-instruct"]="phi3v"

model_types["utter-project/TowerVision-Plus-2B"]="llava_hf"
model_types["utter-project/TowerVision-Plus-9B"]="llava_hf"
model_types["utter-project/TowerVision-4-Anthill-CPT"]="llava_hf"



# model_types["Unbabel/Tower4-Sugarloaf-Vision"]="llava_v6"
# model_types["Unbabel/Tower4-Sugarloaf-Vision-merged"]="llava_v6"
# model_types["Unbabel/lnext-tower4-sugarloaf-siglip2-v6"]="llava_v6"
# model_types["utter-project/EuroVLM-9B-Preview"]="llava_v6"

# models=('mistralai/Pixtral-12B-2409' 'CohereForAI/aya-vision-8b' 'Qwen/Qwen2.5-VL-7B-Instruct' )
models=( utter-project/TowerVision-Plus-2B   ) # utter-project/TowerVision-4-Anthill-CPT 'utter-project/TowerVision-Plus-9B' 'utter-project/TowerVision-4-Anthill-CPT' 'mistralai/Pixtral-12B-2409' 'CohereForAI/aya-vision-8b'
tasks=('aya-vision-bench-gen' )


for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "--------------------------------------------------------------------------------------"
        echo "Model: $model Task: $task Model Version: $model_version"
    
        model_name=$(echo $model | sed 's/\//__/g')
        results_folder=${out_root}/${task}/${model_version}

        # system_prompt="${system_prompts[$task]}"
        # echo $system_prompt
        
        # if [ $model_version == "v6" ]; then
        #     model_type="${model_types[$model]}"
        #     model_args="pretrained=$model,device_map=auto"
        #     # model_type="vllm"
        #     # model_args="model_version=$model"
        # else
        #     model_args="${model_args_map[$model]}"
        #     model_type="${model_types[$model]}"
        #     system_prompt="${system_prompts[$task]}"
        #     # model_args="pretrained=$model,device_map=auto,add_system_prompt=\"$system_prompt\""
        #     model_args="pretrained=$model,$model_args,add_system_prompt=\"$system_prompt\""
        # fi
        
        
        model_arguments="${model_args_map[$model]}"
        model_type="${model_types[$model]}"
        model_args="pretrained=$model,$model_arguments"

        echo "model_args: $model_args"

        CUDA_VISIBLE_DEVICES=0
        accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
        --model $model_type \
        --model_args $model_args \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $task \
        --output_path $results_folder \
        --verbosity=DEBUG

        # CUDA_LAUNCH_BLOCKING=1 python -m accelerate.commands.launch \
        # --num_processes=1 \
        # -m lmms_eval \
        # --model $model_type \
        # --model_args $model_args \
        # --tasks $task \
        # --batch_size 1 \
        # --log_samples \
        # --log_samples_suffix $task \
        # --output_path $results_folder 
    done
done

# CUDA_VISIBLE_DEVICES=0
#         accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
#         --model $model_type \
#         --model_args $model_args \
#         --tasks $task \
#         --batch_size 1 \
#         --log_samples \
#         --log_samples_suffix $task \
#         --output_path $results_folder \
#         --verbosity=DEBUG


# run for other models
# models=('mistralai/Pixtral-12B-2409' )
# models=('CohereForAI/aya-vision-8b' )
# models=('Unbabel/Tower4-Sugarloaf-Vision' )

# for model in "${models[@]}"; do
#     for task in "${tasks[@]}"; do
#         echo "--------------------------------------------------------------------------------------"
#         echo "Model: $model Task: $task"
    
#         model_name=$(echo $model | sed 's/\//__/g')
#         results_folder=${out_root}/${task}

#         model_type="${model_types[$model]}"
#         model_args="${model_args_map[$model]}"

#         echo "model_args: $model_args"

#         CUDA_VISIBLE_DEVICES=0
#         accelerate launch --num_processes 1 --main_process_port 12345 -m lmms_eval \
#             --model $model_type \
#             --model_args "pretrained=$model,$model_args" \
#             --tasks $task \
#             --batch_size 1 \
#             --log_samples \
#             --log_samples_suffix $task \
#             --output_path $results_folder \
#             --verbosity=DEBUG
#     done
# done