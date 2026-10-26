#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nomad_train
# 使用 nohup 后台运行，日志输出到 training.log
nohup python train.py -c ./config/nomad.yaml > training.log 2>&1 &
echo "训练已在后台启动，PID: $!"
echo "查看日志: tail -f training.log"
