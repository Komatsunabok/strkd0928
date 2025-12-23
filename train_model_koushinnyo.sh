# fitnets beta ミス
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg19_bn_half  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg19_bn_half  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg19_bn_half  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg19_bn_half  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg19_bn_half  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg19_bn_half  -c 1 -d 1 -b 1 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill hint 
echo "=== Done ==="


# resnet8x2じゃね？
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model resnet8x2
# echo "=== Done ==="
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model resnet8x2
# echo "=== Done ==="
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model resnet8x2
# echo "=== Done ==="
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model resnet8x2
# echo "=== Done ==="
# echo "=== Start ==="
# python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model resnet8x2
# echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill kd --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill kd 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill kd 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill kd 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill kd 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet8x2 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 
echo "=== Done ==="


echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 1 \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill hint 
echo "=== Done ==="


# グループ数ミス
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 -c 1 -d 1 -b 100 --beta_method fixed \ 
--model_t resnet14x2 --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \ 
--distill ckad --layer_usage all --group_num 4 
echo "=== Done ==="


# 教師と生徒逆
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 100 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill ckad --layer_usage key --group_num 6 --log_cka
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 100 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill ckad --layer_usage key --group_num 6  
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 100 \ 
--model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \ 
--distill ckad --layer_usage key --group_num 6  
echo "=== Done ==="
