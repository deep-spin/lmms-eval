out_root=/mnt/scratch-artemis/manos/data/tower-vision-eval-outputs

mkdir -p $out_root/outs1/

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model molmo_hf \
    --model_args pretrained='allenai/Molmo-7B-D-0924' \
    --tasks mmmu,mme,scienceqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_next \
    --output_path $out_root/outs1/

    # --model_args pretrained="Unbabel/qwen2p5-7b-clip-hdr-sft-v3",add_system_prompt="Answer the questions." \
    # --model_args pretrained="Unbabel/qwen2p5-7b-clip-hdr-sft-v3",add_system_prompt="Answer the questions." \
    # --model_args pretrained="liuhaotian/llava-v1.5-7b"
    #--model_args pretrained="lmms-lab/llama3-llava-next-8b",conv_template=llava_llama_3 \



