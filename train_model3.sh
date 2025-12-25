# 1222daytime~
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model vgg16_bn
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model vgg16_bn
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model vgg16_bn
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）


# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill ckad --layer_usage all --group_num 6 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill ckad --layer_usage all --group_num 6 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill ckad --layer_usage all --group_num 6 
# echo "=== Done ==="


# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill hint --layer_usage all --group_num 6 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill hint --layer_usage all --group_num 6 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill hint --layer_usage all --group_num 6 
# echo "=== Done ==="

# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill kd --layer_usage all --group_num 6 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill kd --layer_usage all --group_num 6 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill kd --layer_usage all --group_num 6 
# echo "=== Done ==="

# 1223daytime~
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model resnet14x2
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model resnet14x2
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model resnet14x2
# echo "=== Done ==="
# sleep 5  # GPUメモリ開放のため5秒待機（任意）

# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill kd --layer_usage all --group_num 4 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill kd --layer_usage all --group_num 4 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill kd --layer_usage all --group_num 4 
# echo "=== Done ==="

# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill hint --layer_usage all --group_num 4 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill hint --layer_usage all --group_num 4 
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill hint --layer_usage all --group_num 4
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill hint --layer_usage all --group_num 4
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill hint --layer_usage all --group_num 4
# echo "=== Done ==="

# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage all --group_num 4
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage all --group_num 4
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage all --group_num 4
# echo "=== Done ==="

# 20251225morning~
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 100 \ 
# --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
# --distill ckad --layer_usage all --group_num 8 --log_cka
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 100 \ 
# --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
# --distill ckad --layer_usage all --group_num 8  
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 100 \ 
# --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
# --distill ckad --layer_usage all --group_num 8  
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 100 \ 
# --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
# --distill ckad --layer_usage all --group_num 8  
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 100 \ 
# --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
# --distill ckad --layer_usage all --group_num 8  
# echo "=== Done ==="


# 1221night~
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage all --group_num 6
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage all --group_num 6
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage all --group_num 6
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage all --group_num 6
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage all --group_num 6
# echo "=== Done ==="


# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill ckad --layer_usage key --group_num 6
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model vgg16_bn_half --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
#   --distill ckad --layer_usage key --group_num 6
# echo "=== Done ==="

# 1221night~
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage key --group_num 4
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage key --group_num 4
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage key --group_num 4
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage key --group_num 4
# echo "=== Done ==="
# echo "=== Start ==="
# python train_student.py --dataset cifar100 --model resnet14 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
#   --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
#   --distill ckad --layer_usage key --group_num 4
# echo "=== Done ==="