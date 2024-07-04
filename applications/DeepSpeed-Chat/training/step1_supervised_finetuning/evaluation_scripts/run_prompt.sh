#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path_baseline ~/data/model/opt-1.3b \
    --model_name_or_path_finetune /data1/xuhang/DeepSpeedResults/sft_run_1.3b/epoch_21
