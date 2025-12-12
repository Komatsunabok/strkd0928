set -e

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t "resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110"  \
  --distill ckad --log_cka
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t "resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110"  \
  --distill ckad 
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t "resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110"  \
  --distill ckad 
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t "resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110"  \
  --distill ckad 
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）

echo "=== Start ==="
python train_student.py --dataset cifar100 --model resnet20 --model_t resnet14x2 -c 1 -d 1 -b 100 --beta_method fixed \
  --model_name_t "resnet14x2-cifar100-trial_0-epochs_240-bs_64-20251128_013110"  \
  --distill ckad 
echo "=== Done ==="
sleep 5  # GPUメモリ開放のため5秒待機（任意）

