#!/bin/bash

# how to use:
# # train teacher
# setecho "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model vgg16_bn_half
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意） -e  # ← 途中でエラーが出たら止める（安全）

# # train student
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill hint
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）

# echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill hint
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）

# echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn -c 1 -d 1 -b 1000 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill attention
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）

# echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn -c 1 -d 1 -b 3000 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill similarity
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）



# # CIFAR
# # KD
# python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn -c 1 -d 1 -b 0 --trial 0 --gpu_id 0
# sleep 5  # GPUメモリ開放のため5秒待機（任意）
# # FitNet
# python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill hint --model_s vgg16_bn_half -c 1 -d 1 -b 100 --trial 0 --gpu_id 0
# sleep 5  # GPUメモリ開放のため5秒待機（任意）
# # AT
# python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill attention --model_s vgg16_bn_half -c 1 -d 1 -b 1000 --trial 0 --gpu_id 0
# sleep 5  # GPUメモリ開放のため5秒待機（任意）
# # SP
# python train_student.py --path_t ./save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth --distill similarity --model_s vgg16_bn_half -c 1 -d 1 -b 3000 --trial 0 --gpu_id 0
# sleep 5  # GPUメモリ開放のため5秒待機（任意）



