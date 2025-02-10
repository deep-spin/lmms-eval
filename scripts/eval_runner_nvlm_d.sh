rdir="/mnt/data/amin/NLP/widn/projects/tower-vision/results"

tagged=false
if [ "$tagged" = true ]; then
    rdir="${rdir}.tagged"
fi

tasks=('ai2d' 'textvqa' 'mmmu')

declare -A tags
tags["ai2d"]="ai2d:"
tags["textvqa"]="text_vqa:"
tags["mmmu"]="figure_qa:"

models=('Unbabel/qwen2p5-7b-hdr-sft-visionblocks-v0.4-tagged')
for model in "${models[@]}"; do
    echo "======================================================================================"
    echo "model: $model"
    for task in "${tasks[@]}"; do
        echo "--------------------------------------------------------------------------------------"
        echo "task: $task"
        model_name=$(echo $model | sed 's/\//__/g')
        results_folder=${rdir}/${task}/${model_name}

        if ls ${results_folder}/*results.json 1> /dev/null 2>&1; then
            echo "Results file already exists for $task"
            continue
        fi

        model_args="${model}",add_system_prompt="\"Answer the questions.\"",device_map=auto

        if [ "$tagged" = true ]; then
            tag="${tags[$task]}"
            model_args=${model_args},tag="${tag}"
        fi
        model_args="$model_args"
        CUDA_VISIBLE_DEVICES=6,7 python -m accelerate.commands.launch \
        --num_processes=2 \
        -m lmms_eval \
        --model nvlm_d \
        --model_args pretrained="$model_args" \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $task \
        --output_path $rdir/$task
    done
done