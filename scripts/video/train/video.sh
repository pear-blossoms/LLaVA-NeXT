#!/bin/bash

# # Set up the data folder
# IMAGE_FOLDER="XXX"
# VIDEO_FOLDER="XXX"
# DATA_YAML="XXX" # e.g exp.yaml

# ############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
# alias python=python3
# ############### Show Envs ####################

# nvidia-smi

# ################ Arnold Jobs ################

# LLM_VERSION="Qwen/Qwen2-7B-Instruct"
# LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
# VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
# VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
# #

# BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# # Stage 2
# PROMPT_VERSION="qwen_1_5"
# MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_to_video_am9"
# PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov-si"
# echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
# echo "MID_RUN_NAME: ${MID_RUN_NAME}"


# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
deepspeed --include localhost:0 --master_port 30000 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /home/panwen.hu/workspace/haokun.lin/checkpoints/LLaVA-Video-7B-Qwen2 \
    --version "qwen_1_5" \
    --data_path /home/panwen.hu/workspace/haokun.lin/data/video/gpt7000.json \
    --image_folder '' \
    --video_folder /home/panwen.hu/workspace/haokun.lin/data/video/extracted_frame_50 \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower "google/siglip-so400m-patch14-384" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name 'LLaVA-Video-7B-Qwen2-7000' \
    --output_dir /home/panwen.hu/workspace/haokun.lin/checkpoints/LLaVA-Video-7B-Qwen2-7000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 50 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0;