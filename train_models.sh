python train_student.py --dataset cifar100 --model vgg8_bn --model_t vgg16_bn --model_name_t vgg16_bn-cifar100-trial_0-epochs_1-bs_64-20251010_120649 --epoch 1

# 20251006
python train_teacher.py --dataset cifar100 --model vgg16_bn --epoch 1

# 20251005
python train_teacher_pytorch.py --dataset cifar100 --model vgg16_bn --pretrained --freeze_layers --epoch 1

# train teacher
# resNet32x4
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model resnet32x4
# resNet8x4
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model resnet8x4
python train_teacher.py --dataset cinic10 --epochs 240 --trial 0 --model resnet8x4

python train_teacher.py --dataset cinic10 --epochs 240 --trial 0 --model vgg8
python train_teacher.py --dataset cinic10 --epochs 240 --trial 0 --model vgg13
# resNet8x4 kd from resNet32x4
python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --dataset cifar10 --model_s resnet8x4 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0
python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cinic10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --dataset cinic10 --model_s resnet8x4 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0

# resNet8x4 SemCKD from resNet32x4
python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --model_s resnet8x4 --distill semckd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0

python train_student.py --path_t save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/resnet32x4_best.pth --model_s resnet8x4 --distill semckd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0

python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --distill ckad --epoch 1 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0

# for 中間発表
# teacher
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13
# student
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 4 --method "mean" --reduction "mean" --grouping "uniform"
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 11 --method "mean" --reduction "mean" --grouping "uniform"
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 4 --method "mean" --reduction "mean" --grouping "proportional"

# teacher
python train_teacher.py --dataset cinic10 --epochs 240 --trial 0 --model vgg13
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cinic10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cinic10 --distill kd --epoch 1 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cinic10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cinic10 --distill ckad --epoch 1 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 4 --method "mean" --reduction "mean" --grouping "uniform"
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cinic10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cinic10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 11 --method "mean" --reduction "mean" --grouping "uniform"
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cinic10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cinic10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 4 --method "mean" --reduction "mean" --grouping "proportional"

python train_teacher.py --dataset cifar100 --epochs 1 --trial 0 --model vgg13
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar100_trial_0_epochs_1_bs_64/vgg13_best.pth --model_s vgg13 --dataset cifar100 --distill kd --epoch 1 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 

python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cinic10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cinic10 --distill kd --epoch 1 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --num_workers 4


20250801
学習スケジュールは何がいいかたしかめる
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --lr_scheduler step --lr_decay_epochs 100,140,180 
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --lr_scheduler step --lr_decay_epochs 100,120,140,160
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --lr_scheduler cosine --lr_decay_epochs 100,140,180 

20250803
学習スケジュールは何がいいかたしかめる
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler step --lr_decay_epochs 150,180,210 オリジナル
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler step --lr_decay_epochs 100,140,180
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler step --lr_decay_epochs 100,120,140,160
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler cosine 

python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer adam --lr_scheduler step --lr_decay_epochs 150,180,210 
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer adam --lr_scheduler step --lr_decay_epochs 100,140,180
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer adam --lr_scheduler step --lr_decay_epochs 100,120,140,160
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer adam --lr_scheduler cosine 

python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler step --lr_decay_epochs 50,100,150,200
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer adam --lr_scheduler cosine --learning_rate 0.01

python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer adam --lr_scheduler constant --learning_rate 0.05
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer adam --lr_scheduler constant --learning_rate 0.01
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler cosine --learning_rate 0.01

いまのところこれが良さそう
初期学習率はどれがいいか
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler cosine --learning_rate 0.005
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler cosine --learning_rate 0.001

# 20250815 再実験
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 4 --method "mean" --reduction "mean" --grouping "uniform"
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 11 --method "mean" --reduction "mean" --grouping "uniform"
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 4 --method "mean" --reduction "mean" --grouping "proportional"

# 20250816 再実験 cifar100で
python train_teacher.py --dataset cifar100 --epochs 240 --trial 0 --model vgg13
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar100-trial_0-epochs_240-bs_64-20250816/vgg13_best.pth --model_s vgg13 --dataset cifar100 --distill kd --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar100-trial_0-epochs_240-bs_64-20250816/vgg13_best.pth --model_s vgg13 --dataset cifar100 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 4 --method "mean" --reduction "mean" --grouping "uniform"
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar100-trial_0-epochs_240-bs_64-20250816/vgg13_best.pth --model_s vgg13 --dataset cifar100 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 11 --method "mean" --reduction "mean" --grouping "uniform"
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar100-trial_0-epochs_240-bs_64-20250816/vgg13_best.pth --model_s vgg13 --dataset cifar100 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --group_num 4 --method "mean" --reduction "mean" --grouping "proportional"

# 20250818 CKAベース損失を変化させる
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --b_method constant --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 200 --b_method constant --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 10 --b_method constant --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --b_method step --b_decay_epochs 120 --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional

# 20250824 KAベース損失をexpで変化させる（初期重みが最終的に0.1倍になる）
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 200 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64_opt_sgd_lr_she_cosine_lr_0.01/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 10 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional

# 20250825 KAベース損失をexpで変化させる（初期重みが最終的に0.1倍になる）cifar100で
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar100-trial_0-epochs_240-bs_64-20250816/vgg13_best.pth --model_s vgg13 --dataset cifar100 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar100-trial_0-epochs_240-bs_64-20250816/vgg13_best.pth --model_s vgg13 --dataset cifar100 --distill ckad --epoch 240 -c 1 -d 1 -b 200 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar100-trial_0-epochs_240-bs_64-20250816/vgg13_best.pth --model_s vgg13 --dataset cifar100 --distill ckad --epoch 240 -c 1 -d 1 -b 10 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional

# 20250903 もろもろ設定しなおして中間発表の再実験
python train_teacher.py --dataset cifar10 --epochs 240 --trial 0 --model vgg13 --optimizer sgd --lr_scheduler cosine --learning_rate 0.01
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar10-trial_0-epochs_240-bs_64-20250903/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill kd --epoch 240 -c 1 -d 1 -b 400 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar10-trial_0-epochs_240-bs_64-20250903/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping uniform
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar10-trial_0-epochs_240-bs_64-20250903/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --b_method exp --trial 0 --gpu_id 0 --group_num 11 --method mean --reduction mean --grouping uniform
python train_student_cka.py --path_t save/teachers/models/vgg13-vanilla-cifar10-trial_0-epochs_240-bs_64-20250903/vgg13_best.pth --model_s vgg13 --dataset cifar10 --distill ckad --epoch 240 -c 1 -d 1 -b 400 --b_method exp --trial 0 --gpu_id 0 --group_num 4 --method mean --reduction mean --grouping proportional
