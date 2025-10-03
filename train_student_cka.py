"""
the general training framework
"""

from __future__ import print_function

import os
import re
import argparse
import time
from datetime import datetime

import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import model_dict
from models.util import ConvReg, SelfA, SRRL, SimKD

from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader,  get_dataloader_sample
from dataset.cinic10 import get_cinic10_dataloaders, get_cinic10_dataloaders_sample

from helper.loops import train_distill as train, validate_vanilla, validate_distill
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate
from helper.cka_mapper import CKAMapper

from distiller_zoo.KD import DistillKL
from distiller_zoo.CKAD import CKADistillLoss

from helper.hooks import register_hooks


split_symbol = '~' if os.name == 'nt' else ':'
# nt:Windows
# posix:Linux, macOS
# java:Jython
split_symbol = '-'

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    # basic
    
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',choices=['step', 'cosine'],help='learning rate scheduler: step or cosine')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'cinic10'], help='dataset')
    parser.add_argument('--model_s', type=str, default='vgg13')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'ckad'])
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('--b_method', type=str, default='constant', choices=['constant', 'step', 'exp'])
    parser.add_argument('--b_decay_epochs', type=str, default='120', help='where to decay b, can be a list')


    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # CKAD
    parser.add_argument('--group_num', type=int, default=4, help='number of groups for CKA')
    parser.add_argument('--method', type=str, default='mean', choices=['mean', 'max', 'min'], help='method for CKA')
    parser.add_argument('--reduction', type=str, default='mean', choices=['sum', 'mean'], help='reduction method for CKA loss')
    parser.add_argument('--grouping', type=str, default='uniform', choices=['uniform', 'proportional'], help='grouping method for student layers')
    
    # multiprocessing
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')
    
    opt = parser.parse_args()

    # set different learning rates for these MobileNet/ShuffleNet models
    if opt.model_s in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/students/models'
    opt.tb_path = './save/students/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if opt.distill == 'kd':
        opt.model_name = 'S_{}-T_{}-{}-{}-r_{}-a_{}-b_{}-b_method_{}-{}-{}'.format(
            opt.model_s, opt.model_t, opt.dataset, opt.distill,
            opt.cls, opt.div, opt.beta, opt.b_method,
            opt.trial, now_str
        )
    elif opt.distill == 'ckad':
        opt.model_name = 'S_{}-T_{}-{}-{}-r_{}-a_{}-b_{}-b_method_{}-{}-Distill_gn-{}-me_{}-red_{}-sgrp_{}-{}'.format(
            opt.model_s, opt.model_t, opt.dataset, opt.distill,
            opt.cls, opt.div, opt.beta, opt.b_method,
            opt.trial,
            opt.group_num, opt.method, opt.reduction, opt.grouping,
            now_str            
        )

    opt.b_decay_epochs = int(opt.b_decay_epochs)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    # アンダースコア or ハイフンで区切る
    segments = re.split(r"[-_]", directory)
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model


best_acc = 0
total_time = time.time()
def main():
    
    opt = parse_option()
    
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)

    # model
    n_cls = {
        'cifar10':10,
        'cifar100': 100,
        'imagenet': 1000,
        'cinic10': 10
    }.get(opt.dataset, None)


    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        multiprocessing_distributed=opt.multiprocessing_distributed)
    elif opt.dataset == 'cinic10':
        train_loader, val_loader = get_cinic10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)
    
    
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    try:
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    model_t.eval()
    model_s.eval()

    # ランダムなデータを使って特徴量を取得する例
    # if opt.dataset == 'cifar100' or opt.dataset == 'cifar10' or opt.dataset == 'cinic10':
    #     data = torch.randn(2, 3, 32, 32)
    # elif opt.dataset == 'imagenet':
    #     data = torch.randn(2, 3, 224, 224)  # get features    
    # feat_t, _ = model_t(data, is_feat=True)
    # feat_s, _ = model_s(data, is_feat=True)

    # 1バッチの画像を使って特徴量を取得
    # 例：cifar10の場合 64枚の画像を使って特徴量を取得
    # for images, labels in train_loader:
    #     feat_t, _ = model_t(images, is_feat=True)
    #     feat_s, _ = model_s(images, is_feat=True)
    #     # ...CKA計算に使う...
    #     break

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    # model_t と model_s に hook を登録
    # 必要な層にのみhookをかける
    # hook を登録した層の情報一覧（インデックス・名前・ハンドル）
    # hooks_t = [
    #     (3, 'block0.1', handle1),
    #     (6, 'block0.4', handle2),
    #     (9, 'block1.1', handle3),
    #     ...
    # ]
    # 実際に出力を記録する人（hookをかけたもののみ）
    # feature_hook_t.outputs = [
    #     tensor1,  # block0.1の出力
    #     tensor2,  # block0.4の出力
    #     tensor3,  # block1.1の出力
    #     ...
    # ]
    hooks_t, feature_hook_t = register_hooks(model_t, (nn.BatchNorm2d, nn.Linear))
    hooks_s, feature_hook_s = register_hooks(model_s, (nn.BatchNorm2d, nn.Linear))

    # dataをモデルに通して特徴量を取得(実際の各層の出力)
    # feat_t = [
    #     torch.Size([2, 128, 8, 8]),
    #     torch.Size([2, 256, 4, 4]),
    #     torch.Size([2, 512, 2, 2]),
    #     ...
    # ]
    for images, labels in train_loader:
        feat_t, _ = model_t(images, is_feat=True)
        feat_s, _ = model_s(images, is_feat=True)
        # ...CKA計算に使う...
        break

    # どの層の出力か確認
    # for i, (idx, name, _) in enumerate(hooks_t):
    #     print(f"{i}: Hooked layer = {name}")
    #     print(f"    Output shape: {feature_hook_t.outputs[i].shape}")

    # KD
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'ckad':
        # 特徴量の形状を取得
        # t_shapesとs_shapesは以下のようなリストになる
        # [
        #     torch.Size([バッチサイズ, チャンネル数1, 高さ1, 幅1]),
        #     torch.Size([バッチサイズ, チャンネル数2, 高さ2, 幅2]),
        # ...
        # ]
        t_shapes = [f.shape for f in feature_hook_t.outputs]
        s_shapes = [f.shape for f in feature_hook_s.outputs]

        # CKAグループ対応モジュール
        cka_mapper = CKAMapper(
            t_shapes=t_shapes, s_shapes=s_shapes, 
            feat_t=feature_hook_t.outputs,
            group_num=opt.group_num, grouping=opt.grouping
        )
        module_list.append(cka_mapper)
        trainable_list.append(cka_mapper)
    
        # CKAベースの蒸留損失関数
        criterion_kd = CKADistillLoss(group_num=opt.group_num, method_inner_group=opt.method, method_inter_group=opt.reduction)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    module_list.append(model_t)
    
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.epochs,
        eta_min=opt.lr_min if hasattr(opt, 'lr_min') else 0
    )
    
    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    if not opt.skip_validation:
        # validate teacher accuracy
        teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt)

        if opt.dali is not None:
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')

    # Early Stoppingのパラメータ
    patience = 100  # 精度が改善しない場合に停止するまでのエポック数
    no_improve_count = 0  # 改善が見られないエポック数をカウント
    train_acc_history = []  # train_accを記録するリスト
    test_acc_history = []
    test_loss_history = []
    lr_history = []
    
    # 学習前にfeature_hook_t.outputsをクリアしておく
    # ※feature_hook_t.outputsはmodel(input)を呼ぶたびにたまり続ける
    feature_hook_t.outputs.clear()
    feature_hook_s.outputs.clear()

    # routine
    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)

        # adjust_learning_rate(epoch, opt, optimizer) # 下部で実行するのでコメントアウト
        print("==> training...")

        time1 = time.time()
        # module_list：訓練時に使用するモデルやモジュールのリスト
        # module_list[0]はstudent model
        # module_list[1]~はCKAMapperなどのCKA関連モジュール
        # module_list[-1](末尾)はteacher model
        # criterion_list：損失関数のリスト
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt,
                                                      feature_hook_t, feature_hook_s)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, opt.gpu, train_acc, train_acc_top5, time2 - time1))
            writer.add_scalar('train_acc', train_acc, epoch)    
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc_top5', train_acc_top5, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print('GPU %d validating' % (opt.gpu))
        test_acc, test_acc_top5, test_loss = validate_distill(val_loader, module_list, criterion_cls, opt)        

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
            
            writer.add_scalar('test_acc', test_acc, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc_top5', test_acc_top5, epoch)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                # no_improve_count = 0  # 改善が見られたのでリセット

                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }
                if opt.distill == 'simkd':
                    state['proj'] = trainable_list[-1].state_dict() 
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
                
                test_merics = {'test_loss': test_loss,
                                'test_acc': test_acc,
                                'test_acc_top5': test_acc_top5,
                                'epoch': epoch}
                
                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)
            # else:
            #     no_improve_count += 1  # 改善が見られない場合カウントを増やす

            # # Early Stoppingの判定
            # if no_improve_count >= patience:
            #     print(f"Early stopping triggered. No improvement for {patience} epochs.")
            #     break
        
        # 記録
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        test_loss_history.append(test_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])

        # optimizarの学習率を更新
        scheduler.step()
    
    # historyを保存
    train_acc_file = os.path.join(opt.save_folder, "train_acc_history.json")
    save_dict_to_json({"train_acc": train_acc_history}, train_acc_file)
    print(f"Train accuracy history saved to {train_acc_file}")

    test_acc_file = os.path.join(opt.save_folder, "test_acc_history.json")
    save_dict_to_json({"test_acc": test_acc_history}, test_acc_file)
    print(f"Train accuracy history saved to {test_acc_file}")

    test_loss_file = os.path.join(opt.save_folder, "test_loss_history.json")
    save_dict_to_json({"test_acc": test_loss_history}, test_loss_file)
    print(f"Train accuracy history saved to {test_loss_file}")


            
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)
        
        # save parameters
        save_state = {k: v for k, v in opt._get_kwargs()}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model_s.parameters())/1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] =  (time.time() - total_time)/3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json") 
        save_dict_to_json(save_state, params_json_path)

if __name__ == '__main__':
    main()
