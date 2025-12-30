
python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 10 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill ckad --layer_usage all --group_num 6 --log_cka

python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 10 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill ckad --layer_usage all --group_num 6  

python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 50 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill ckad --layer_usage all --group_num 6 --log_cka

python train_student.py --dataset cifar100 --model vgg16_bn_half  -c 1 -d 1 -b 50 --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_240-bs_64-20251014_162623 --distill ckad --layer_usage all --group_num 6  



