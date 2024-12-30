#!/bin/bash

#SBATCH --job-name=pangea_eval   # Job name
#SBATCH --time=48:00:00         # Run time (hh:mm:ss) 
#SBATCH --gres=gpu:2           # Number of GPUs to be used
#SBATCH --qos=gpu-medium         # QOS to be used
#SBATCH --partition=a6000         # QOS to be used
#SBATCH --mem=500GB         # mem to use
#SBATCH --output=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/%x_%j.out  # Standard output
#SBATCH --error=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/%x_%j.err   # Standard error


source ~/.bashrc

out_root=/mnt/scratch-artemis/manos/data/tower-vision-eval-outputs
env_path=/mnt/data-poseidon/manos/tower-vision-eval/deepspin-lmms-eval/venv-lmms-eval


module load openjdk
# Activate the environment
source $env_path/bin/activate

mkdir -p $out_root/outs3/

# python -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model molmo_hf \
#     --model_args pretrained='allenai/Molmo-7B-D-0924' \
#     --tasks mmmu,mme,scienceqa \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_next \
#     --output_path $out_root/outs1/

python -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model pangea  \
    --model_args pretrained="neulab/Pangea-7B",device_map=auto \
    --tasks cvqa \
    --batch_size 1 \
    --output_path $out_root/outs3/ \
    --log_samples 
    # --log_samples_suffix llava_next \
# --tasks xmmmu,marvl,m3exam,maxm,xgqa \

    # --model_args pretrained="Unbabel/qwen2p5-7b-clip-hdr-sft-v3",add_system_prompt="Answer the questions." \
    # --model_args pretrained="Unbabel/qwen2p5-7b-clip-hdr-sft-v3",add_system_prompt="Answer the questions." \
    # --model_args pretrained="liuhaotian/llava-v1.5-7b"
    #--model_args pretrained="lmms-lab/llama3-llava-next-8b",conv_template=llava_llama_3 \



