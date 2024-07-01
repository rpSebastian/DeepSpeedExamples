#!/bin/bash

deepspeed --bind_cores_to_rank --autotuning run cifar10_deepspeed.py --deepspeed $@
