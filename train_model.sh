#!/bin/bash
set -e  # ← 途中でエラーが出たら止める（安全）


echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn --beta 200 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --group_num 6 --layer_usage all
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）

echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn --beta 200 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --group_num 6 --layer_usage all
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）

echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn --beta 200 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --group_num 6 --layer_usage all
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）
