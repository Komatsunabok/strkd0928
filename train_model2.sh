
echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg8_bn  -c 1 -d 1 -b 100 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill kd --log_cka
echo "=== Done ==="


echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg8_bn  -c 1 -d 1 -b 100 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill kd 
echo "=== Done ==="


echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg8_bn  -c 1 -d 1 -b 100 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill kd 
echo "=== Done ==="


echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg8_bn  -c 1 -d 1 -b 100 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill kd 
echo "=== Done ==="


echo "=== Start ==="
python train_student.py --dataset cifar100 --model vgg8_bn  -c 1 -d 1 -b 100 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill kd 
echo "=== Done ==="

