#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/data1/xuhang/DeepSpeedResults/rft_1.3b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path /data1/xuhang/DeepSpeedResults/sft_run_1.3b/epoch_21 \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 128 \
   --dropout 0.0 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --gradient_checkpointing \
   --num_warmup_steps 0 \
   --seed 1234 \
   --dtype bf16 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
    --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
