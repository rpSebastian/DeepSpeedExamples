#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Add the path to the finetuned model
python  rw_eval.py \
    --model_name_or_path /data1/xuhang/hf_hub/model/chat-opt-350m-reward-deepspeed
