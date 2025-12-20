#!/bin/bash
# set -e  # ← 途中でエラーが出たら止める（安全）

# how to use:

# # train teacher
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model vgg16_bn_half
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）

# # train student
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill hint --log_cka
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --layer_usage key --log_cka 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --layer_usage key 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --layer_usage key 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn_half -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --layer_usage key 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn_half -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill ckad --layer_usage key 
echo "=== Done ==="

