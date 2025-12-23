echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
  --distill hint --layer_usage all --group_num 4 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
  --distill hint --layer_usage all --group_num 4 
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
  --distill hint --layer_usage all --group_num 4
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
  --distill hint --layer_usage all --group_num 4
echo "=== Done ==="
echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet14x2 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110 \
  --distill hint --layer_usage all --group_num 4
echo "=== Done ==="