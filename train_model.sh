#!/bin/bash
set -e  # ← 途中でエラーが出たら止める（安全）

# === 1つ目 ===
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn --beta 10 \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --layer_usage all --group_num 6 --log_cka 
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）








