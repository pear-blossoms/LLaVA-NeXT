#!/bin/bash

# NAS_REGION="vl-research"
# # set up wandb
# export WANDB_API_KEY=a651c244635bc6f913ab654af3f0eebaecdc9381
# export WANDB_ENTITY=llava-vl
# export WANDB_PROJECT=llava-next
# export WANDB_MODE=online
# export PYTHONWARNINGS="ignore"

# export ACCELERATE_DEBUG_MODE="1"
# export HF_HOME="/mnt/bn/${NAS_REGION}/workspace/.cache/huggingface"
# export HF_TOKEN="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"
# export HF_HUB_ENABLE_HF_TRANSFER="1"

# ############### Prepare Envs #################
# cd /mnt/bn/vl-research/workspace/yhzhang/LLaVA-NeXT/
# python3 -m pip install --upgrade pip
# python3 -m pip install -e ".[train]"

# python3 -m pip install ninja
# python3 -m pip install flash-attn --no-build-isolation
# alias python=python3
# ############### Show Envs ####################

# nvidia-smi
# # 取 worker0 第一个 port
# ports=($(echo $METIS_WORKER_0_PORT | tr ',' ' '))
# port=${ports[0]}
# port_in_cmd="$(echo "${METIS_WORKER_0_PORT:-2222}" | awk -F',' '{print $1}')"

# echo "total workers: ${ARNOLD_WORKER_NUM}"
# echo "cur worker id: ${ARNOLD_ID}"
# echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
# echo "master ip: ${METIS_WORKER_0_HOST}"
# echo "master port: ${port}"
# echo "master port in cmd: ${port_in_cmd}"

# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# # export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=WARN

# PORT=26000
# GPUS="0,1,2,3,4,5,6,7"

# wandb login a651c244635bc6f913ab654af3f0eebaecdc9381
# wandb online

# ################ Arnold Jobs ################

# LLM_VERSION="Qwen/Qwen2-7B-Instruct"
# LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
# VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
# VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# PROMPT_VERSION=plain
# PRETRAIN_DATA_VERSION="blip558k"
# ############### Pretrain ################

# BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
# # echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# # Stage 2
# PROMPT_VERSION="qwen_1_5"
# MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_to_video_am9_aug16_faster"

# PREV_STAGE_CHECKPOINT="/mnt/bn/vl-research/checkpoints/onevision/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mid_to_final_next_2p4m_am9"
# echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
# echo "MID_RUN_NAME: ${MID_RUN_NAME}"




# deepspeed --master_port 30000 llava/train/train_mem.py \
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
# deepspeed --master_port 30000 \
# # "mm_vision_tower,mm_mlp_adapter,mm_language_model" \

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
# /mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/scripts/i18n/scale_llms/next_ov_video_specific_stage_aug12.yaml \
# #/mnt/bn/vl-research/workspace/boli01/projects/LLaVA_Next/scripts/i18n/scale_llms/next_ov_video_specific_stage_july31.yaml
unset CUDA_VISIBLE_DEVICES

deepspeed --include localhost:0,1 --master_port 30000 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /home/panwen.hu/workspace/haokun.lin/checkpoints/LLaVA-Video-7B-Qwen2 \
    --version "qwen_1_5" \
    --data_path /home/panwen.hu/workspace/haokun.lin/data/video/gpt7000_img50.json \
    --image_folder '' \
    --video_folder /home/panwen.hu/workspace/haokun.lin/data/video/extracted_frame_50 \
    --mm_tunable_parts="mm_mlp_adapter, mm_language_model" \
    --vision_tower "google/siglip-so400m-patch14-384" \
    --mm_vision_tower_lr=2e-6 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name 'LLaVA-Video-7B-Qwen2-7000-model_config-qwen' \
    --output_dir /home/panwen.hu/workspace/haokun.lin/checkpoints/LLaVA-Video-7B-Qwen2-7000-model_config-qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
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
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 50 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --faster_token_stride 5 \
# # exit 0;

# --add_faster_video True \
# --faster_token_stride 5 \
# --run_name 'llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_aug16_faster' \
# --mm_tunable_parts="mm_mlp_adapter" \