echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill hint --layer_usage all --group_num 6 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill hint --layer_usage all --group_num 6 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill hint --layer_usage all --group_num 6 
echo "=== Done ==="

echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill kd --layer_usage all --group_num 6 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill kd --layer_usage all --group_num 6 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg16_bn --model_t vgg16_bn -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 \
  --distill kd --layer_usage all --group_num 6 
echo "=== Done ==="