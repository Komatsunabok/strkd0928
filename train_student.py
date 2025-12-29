"""
the general training framework
"""

from __future__ import print_function

import os
import re
import argparse
import time
from datetime import datetime
import json

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

from models import model_dict
from models.util import ConvReg

from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cinic10 import get_cinic10_dataloaders

from helper.loops import train_distill as train, validate_vanilla, validate_distill
from helper.util import save_dict_to_json, reduce_tensor
from helper.cka_mapper import CKAMapper

# from distiller_zoo.KD import DistillKL
# from distiller_zoo.CKAD import CKADistillLoss
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, CKADistillLoss


from helper.hooks import register_hooks

# ASSIGN CUDA_ID
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    # baisc
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency') # トレーニング中にログ（進捗状況やメトリクス）を出力する頻度（バッチ）
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size') 
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers') # ワーカープロセスの数
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES') # GPUのID（0:最初のGPUを使用）
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr_min', type=float, default=0.0, help='minimum learning rate for cosine annealing')
    
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'cinic10'], help='dataset')
    parser.add_argument('--model', type=str, default='vgg16_bn')
    parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')

    # teacher model
    parser.add_argument('--model_t', type=str, default='vgg16_bn')
    parser.add_argument('--model_name_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'ckad', 'hint', 'attention', 'similarity'])
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=1.0, help='weight balance for other losses')
    parser.add_argument('--beta_method', type=str, default='fixed', choices=['fixed', 'epoch_based'],
                        help='method to adjust beta during training')

    # hint layer
    parser.add_argument('--hint_layer_s', default=-1, type=int, help='student hint layer index (-1: auto middle layer)')
    parser.add_argument('--hint_layer_t', default=-1, type=int, help='teacher hint layer index (-1: auto middle layer)')

    # CKA-based Knowledge Distillation (CKAD)
    parser.add_argument('--group_num_method', type=str, default='custom', choices=['custom', 'auto'],
                        help='method to determine the number of groups for CKA-based loss')
    parser.add_argument('--group_num', type=int, default=6,
                        help='number of groups for CKA-based loss (if group_num_method is custom)')
    parser.add_argument('--layer_usage', type=str, default='all', choices=['all', 'key'],
                        help='which layers to use for loss calculation')
    parser.add_argument('--student_grouping', type=str, default='proportional', choices=['uniform', 'proportional'],
                        help='grouping method for student layers')
    parser.add_argument('--inner_group_aggregation', type=str, default='mean', choices=['sum', 'mean'],
                        help='aggregation method for inner-group CKA loss')
    parser.add_argument('--inter_group_aggregation', type=str, default='mean', choices=['sum', 'mean'],
                        help='aggregation method for inter-group CKA loss')
    parser.add_argument('--log_cka', action='store_true', help='whether to log CKA values during training')

    opt = parser.parse_args()

    # set the path of model and tensorboard
    opt.model_path = os.path.join('save','students', 'models')
    opt.tb_path = os.path.join('save','students', 'tensorboard')

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.model_name = 'S_{}-T_{}-{}-trial_{}-epochs_{}-bs_{}-{}-cls_{}-div_{}-beta_{}-{}'.format(
        opt.model, opt.model_t, opt.dataset, opt.trial, opt.epochs, opt.batch_size,
        opt.distill, opt.cls, opt.div, opt.beta, now_str
    )

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    return opt


def load_teacher(model_name, n_cls, gpu=None, opt=None):
    """Load teacher model using model directory"""
    model_dir = os.path.join("save", "teachers", "models", model_name)

    # --- parameters.json を読み込み ---
    param_path = os.path.join("save", "teachers", "models", model_name, "parameters.json")
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"parameters.json not found in {model_name}")

    with open(param_path, "r") as f:
        params = json.load(f)

    model_name = params["model"]

    # --- モデル構築 ---
    model = model_dict[model_name](num_classes=n_cls)

    # --- .pth ファイルを自動検出 ---
    pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not pth_files:
        raise FileNotFoundError(f"No .pth file found in {model_dir}")
    model_path = os.path.join(model_dir, pth_files[0])  # 最初の1個を使用
    print(f"Loading weights: {pth_files[0]}")

    # --- 重み読み込み ---
    map_location = None if gpu is None else {'cuda:0': f'cuda:{gpu}'}
    state_dict = torch.load(model_path, map_location=map_location)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)

    return model, model_name


def main():
    opt = parse_option()

    # ASSIGN CUDA_ID
    if torch.cuda.is_available() and opt.gpu_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
        print(f"Using GPU: {opt.gpu_id}")
    else:
        print("No valid GPU ID provided or GPU not available. Using CPU.")

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node

    # 単一GPU/CPU用のワーカーを呼ぶ
    main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    best_acc = 0
    total_time = time.time()
    opt.gpu = int(gpu) if gpu and gpu.isdigit() else None
    opt.gpu_id = int(gpu) if gpu and gpu.isdigit() else None    

    # deviceオブジェクトを定義
    device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() and opt.gpu is not None else 'cpu')
    print(f"Using device: {device}")

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))
    else:
        print("No GPU detected. Using CPU for training.")
        
    # dataset
    n_cls = {
        'cifar10':10,
        'cifar100': 100,
        'imagenet': 1000,
        'cinic10': 10
    }.get(opt.dataset, None)

    # dataloader
    print(f"==> Loading dataset: {opt.dataset}...")
    if opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'cinic10':
        train_loader, val_loader = get_cinic10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)
    print("Dataset loaded successfully!")
    
    # teacherをロード
    print(f"==> Loading teacher model from: {opt.model_name_t}...")
    model_t, model_t_name = load_teacher(opt.model_name_t, n_cls, opt.gpu, opt)
    print(f"==> Teacher loaded successfully! (model: {model_t_name})")

    # modelの初期化
    print(f"==> Initializing student model from scratch: {opt.model}...")
    try:
        model_s = model_dict[opt.model](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    model_t.eval()
    model_s.eval()

    # module_list：訓練時に使用するモデルやモジュールのリスト
    # module_list[0]はstudent model
    # module_list[1]~はCKAMapperなどのCKA関連モジュール
    # module_list[-1](末尾)はteacher model
    print("==> Setting up modules and criteria...")
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

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
    # 呼び出すたびに追加され続ける
    # feature_hook_t.outputs = [
    #     tensor1,  # block0.1の出力
    #     tensor2,  # block0.4の出力
    #     tensor3,  # block1.1の出力
    #     ...
    # ]
    print("==> Registering hooks...")
    hooks_s, feature_hook_s = register_hooks(model_s, (nn.BatchNorm2d, nn.Linear))
    hooks_t, feature_hook_t = register_hooks(model_t, (nn.BatchNorm2d, nn.Linear))

    hooks_s, feature_hook_s = register_hooks(model_name=opt.model, model=model_s)
    hooks_t, feature_hook_t = register_hooks(model_name=model_t_name, model=model_t)


    # Conv 用 (Hint用)
    if opt.distill == 'hint':
        hooks_s_conv, feature_hook_s_conv = register_hooks(model_s, (nn.Conv2d,))
        hooks_t_conv, feature_hook_t_conv = register_hooks(model_t, (nn.Conv2d,))

    # dataをモデルに通して特徴量を取得(実際の各層の出力)
    # feat_t = [
    #     torch.Size([2, 128, 8, 8]),
    #     torch.Size([2, 256, 4, 4]),
    #     torch.Size([2, 512, 2, 2]),
    #     ...
    # ]
    # torch.Size([バッチサイズ, チャンネル数, 高さ, 幅])
    for images, labels in train_loader:
        _ = model_s(images, is_feat=False)
        _ = model_t(images, is_feat=False)
        # feat_t_conv, _ = model_t(images, is_feat=True)
        # feat_s_conv, _ = model_s(images, is_feat=True)
        break

    # print("Feature shapes from teacher model:")
    # for i, f in enumerate(feat_t):
    #     print(f"  Layer {i}: {f.shape}")
    # print("Feature shapes from teacher model(conv):")
    # for i, f in enumerate(feat_t_conv):
    #     print(f"  Conv Layer {i}: {f.shape}")

    # # どの層の出力か確認
    # for i, (idx, name, _) in enumerate(hooks_t):
    #     print(f"{i}: Hooked layer = {name}")
    #     print(f"    Output shape: {feature_hook_t.outputs[i].shape}")

    # for i, (idx, name, _) in enumerate(hooks_t_conv):
    #     print(f"{i}: Conv Hooked layer = {name}")
    #     print(f"    Conv Output shape: {feature_hook_t_conv.outputs[i].shape}")

    # KD
    print(f"==> Setting up distillation method: {opt.distill}...")
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
        s_shapes = [f.shape for f in feature_hook_s.outputs]
        t_shapes = [f.shape for f in feature_hook_t.outputs]

        # CKAグループ対応モジュール
        cka_mapper = CKAMapper(
            s_shapes=s_shapes, t_shapes=t_shapes, feat_t=feature_hook_t.outputs,
            group_num_method=opt.group_num_method, group_num=opt.group_num, 
            student_grouping=opt.student_grouping, layer_usage=opt.layer_usage
        )
        module_list.append(cka_mapper)
        trainable_list.append(cka_mapper)

        # グループ情報をoptに保存
        opt.student_grouping = cka_mapper.s_groups
        opt.teacher_grouping = cka_mapper.t_groups
        if opt.layer_usage == 'key':
            opt.student_key_layers = cka_mapper.s_key_layers
            opt.teacher_key_layers = cka_mapper.t_key_layers    
        # student group [[0, 1], [2, 3], [4, 5, 6], [7, 8], [9, 10], [11, 12, 13]]
        # student key layers [1, 3, 5, 8, 10, 12]
        # teacher key layers [1, 3, 5, 8, 10, 12]
    
        # CKAベースの蒸留損失関数
        criterion_kd = CKADistillLoss(
            inner_group_aggregation=opt.inner_group_aggregation,
            inter_group_aggregation=opt.inter_group_aggregation
        )
    elif opt.distill == 'hint':
        # Conv出力だけ抽出
        conv_indices_s = [i for i, f in enumerate(feature_hook_s_conv.outputs) if f.dim() == 4]
        conv_indices_t = [i for i, f in enumerate(feature_hook_t_conv.outputs) if f.dim() == 4]

        print(f"Student conv layer indices: {conv_indices_s}")
        print(f"Teacher conv layer indices: {conv_indices_t}")

        # Hint層を自動選択または明示的に指定
        if opt.hint_layer_s == -1:
            # 自動選択: 中間層
            opt.hint_layer_s = conv_indices_s[len(conv_indices_s) // 2]
        if opt.hint_layer_t == -1:
            # 自動選択: 中間層
            opt.hint_layer_t = conv_indices_t[len(conv_indices_t) // 2]

        print(f"Student hint layer: {opt.hint_layer_s}, shape: {feature_hook_s_conv.outputs[opt.hint_layer_s].shape}")
        print(f"Teacher hint layer: {opt.hint_layer_t}, shape: {feature_hook_t_conv.outputs[opt.hint_layer_t].shape}")

        # HintLossとConvRegを作る（Conv専用Hookの出力を渡す）
        criterion_kd = HintLoss()
        regress_s = ConvReg(
            feature_hook_s_conv.outputs[opt.hint_layer_s].shape,  # ← ここを修正
            feature_hook_t_conv.outputs[opt.hint_layer_t].shape   # ← ここを修正
        )

        module_list.append(regress_s)
        trainable_list.append(regress_s)

    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    else:
        raise NotImplementedError(opt.distill)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    
    # criterion_list：損失関数のリスト
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

    cudnn.benchmark = False
    
    if torch.cuda.is_available():
        module_list = module_list.cuda()
        criterion_list = criterion_list.cuda()
        cudnn.benchmark = True  # optional（速度アップ）

    # tensorboard
    writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    # validate teacher accuracy
    teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt, device)
    print('teacher accuracy: ', teacher_acc)

    # 記録用リスト
    train_acc_history = []
    test_acc_history = []
    test_loss_history = []
    lr_history = []
    train_time_history = []

    train_loss_total_history = []
    train_loss_cls_history = []
    train_loss_div_history = []
    train_loss_kd_history = []

    # ckad 用（epoch ごとに [g1, g2, ...] を入れる）
    train_loss_kd_group_history = []

    
    # 学習前にfeature_hook_t.outputsをクリアしておく
    # ※feature_hook_t.outputsはmodel(input)を呼ぶたびにたまり続ける
    if feature_hook_s is not None:
        feature_hook_s.outputs.clear()    
    if feature_hook_t is not None:         
        feature_hook_t.outputs.clear()

        if opt.distill == 'hint':
            if feature_hook_s_conv is not None:
                feature_hook_s_conv.outputs.clear()            
            if feature_hook_t_conv is not None:
                feature_hook_t_conv.outputs.clear()

    # AMP scaler
    scaler = GradScaler("cuda")
    
    print("autocast enabled:", torch.is_autocast_enabled())



    # routine
    for epoch in range(1, opt.epochs + 1):
        print(f"Starting epoch {epoch}/{opt.epochs}...")

        # train
        print("==> training...")
        time1 = time.time()
        # train_acc, train_acc_top5, train_loss = train(
        #     epoch, train_loader, module_list, criterion_list, optimizer, opt,
        #     feature_hook_t=feature_hook_t,
        #     feature_hook_s=feature_hook_s,
        #     feature_hook_t_conv=feature_hook_t_conv,
        #     feature_hook_s_conv=feature_hook_s_conv,
        #     device=device
        # )
        
        train_log = train(
            epoch, train_loader, module_list, criterion_list, optimizer, opt,
            feature_hook_t=feature_hook_t,
            feature_hook_s=feature_hook_s,
            feature_hook_t_conv=feature_hook_t_conv,
            feature_hook_s_conv=feature_hook_s_conv,
            device=device,
            scaler=scaler
        )
        train_log = train(
            epoch, train_loader, module_list, criterion_list, optimizer, opt,
            feature_hook_t=feature_hook_t,
            feature_hook_s=feature_hook_s,
            feature_hook_t_conv=feature_hook_t_conv if opt.distill == 'hint' else None,
            feature_hook_s_conv=feature_hook_s_conv if opt.distill == 'hint' else None,
            device=device,
            scaler=scaler
        )

        train_acc = train_log["acc1"]
        train_acc_top5 = train_log["acc5"]
        train_loss = train_log["loss_total"]
        loss_total = train_log["loss_total"]
        loss_cls   = train_log["loss_cls"]
        loss_div   = train_log["loss_div"]
        loss_kd    = train_log["loss_kd"]
        loss_kd_group = train_log["loss_kd_group"]  # None or list


        time2 = time.time()
        print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, opt.gpu, train_acc, train_acc_top5, time2 - time1))
        writer.add_scalar('train_acc', train_acc, epoch)    
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc_top5', train_acc_top5, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluate
        print("==> evaluating...")
        test_acc, test_acc_top5, test_loss = validate_distill(val_loader, module_list, criterion_cls, opt, device)        
        print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc_top5', test_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            test_merics = {'test_loss': test_loss,
                            'test_acc': test_acc,
                            'test_acc_top5': test_acc_top5,
                            'epoch': epoch}
            
            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
            print('saving the best model!')
            torch.save(state, save_file)
        
        # 記録
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        test_loss_history.append(test_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])
        epoch_train_time = time2 - time1
        train_time_history.append(epoch_train_time)

        train_loss_total_history.append(loss_total)
        train_loss_cls_history.append(loss_cls)
        train_loss_div_history.append(loss_div)
        train_loss_kd_history.append(loss_kd)

        if loss_kd_group is not None:
            train_loss_kd_group_history.append(loss_kd_group)


        # optimizarの学習率を更新
        scheduler.step()
    
    # historyを保存
    hist_path = os.path.join(opt.save_folder, "training_history.json")
    avg_train_time = sum(train_time_history) / len(train_time_history)
    total_train_time = sum(train_time_history)

    history = {
        "train_acc": train_acc_history,
        "test_acc": test_acc_history,
        "test_loss": test_loss_history,
        "lr": lr_history,
        "train_time_per_epoch_sec": train_time_history,
        "avg_train_time_per_epoch_sec": avg_train_time,
        "total_train_time_sec": total_train_time,

        # 追加
        "train_loss_total": train_loss_total_history,
        "train_loss_cls": train_loss_cls_history,
        "train_loss_div": train_loss_div_history,
        "train_loss_kd": train_loss_kd_history,
    }
    if opt.distill == "ckad":
        history["train_loss_kd_group"] = train_loss_kd_group_history

    save_dict_to_json(history, hist_path)
    print(f"Training history saved to {hist_path}")

    # This best accuracy is only for printing purpose.
    print('best accuracy:', best_acc)
    
    # save parameters
    save_state = {k: v for k, v in opt._get_kwargs()}
    # No. parameters(M)
    num_params = (sum(p.numel() for p in model_s.parameters())/1000000.0)
    save_state['Total params'] = num_params
    state['Total time'] =  float('%.2f' % ((time.time() - total_time) / 3600.0))
    params_json_path = os.path.join(opt.save_folder, "parameters.json") 
    save_dict_to_json(save_state, params_json_path)

if __name__ == '__main__':
    main()
