"""
Training a single model (student or teacher)
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# from models import model_dict, weight_class_dict
from bin.pytorch_models import model_dict, weight_class_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cinic10 import get_cinic10_dataloaders
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate
from helper.loops import train_vanilla as train, validate_vanilla

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

    # pretrained model
    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights if available') # default=False, --pretrainedでTrue
    parser.add_argument('--pretrained_weights', type=str, default='IMAGENET1K_V1',
                        help="Pretrained weight version (e.g. 'IMAGENET1K_V1', 'IMAGENET1K_V2')")
    parser.add_argument('--freeze_layers', action='store_true', 
                        help='Freeze the feature extraction layers and only train the classifier')
    opt = parser.parse_args()

    # set the path of model and tensorboard 
    opt.model_path = './save/teachers/models'
    opt.tb_path = './save/teachers/tensorboard'

    # set the model name    
    # 学習戦略を決定するタグを作成
    if not opt.pretrained:
        strategy_tag = 'scratch'
    else:
        if opt.freeze_layers:
            strategy_tag = 'fe'  # Feature Extraction
        else:
            strategy_tag = 'ft'  # Fine-tuning
    now_str = datetime.now().strftime("%Y%m%d")
    opt.model_name = '{}-{}-{}-trial_{}-epochs_{}-bs_{}-{}'.format(
        opt.model, strategy_tag, opt.dataset, opt.trial, opt.epochs, opt.batch_size, now_str
    )

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def get_model(opt, n_cls):
    """
    公式モデルをロードまたは初期化する関数
    - opt: コマンドライン引数
    - n_cls: データセットのクラス数
    """
    # === 1. 事前学習済みモデルを使う場合 (`--pretrained` が指定された時) ===
    if opt.pretrained:
        print(f"Loading official pretrained model: {opt.model}")

        # ---- (1) torchvisionから学習済み重みをダウンロード ----
        if opt.model not in weight_class_dict:
            raise ValueError(f"No pretrained weights class found for model {opt.model}")

        weight_class = weight_class_dict[opt.model]
        if not hasattr(weight_class, opt.pretrained_weights):
            raise ValueError(f"Weight '{opt.pretrained_weights}' not found in {weight_class}")
        
        pretrained_weights = getattr(weight_class, opt.pretrained_weights)
        model = model_dict[opt.model](weights=pretrained_weights)

        # ---- (2) 必要であれば、特徴抽出層を凍結 ----
        if opt.freeze_layers:
            print("Freezing feature extraction layers...")
            # VGG系のモデルの場合
            if hasattr(model, 'features'):
                for param in model.features.parameters():
                    param.requires_grad = False
            # ResNet系のモデルの場合 (fc層以外を凍結)
            elif hasattr(model, 'layer1'):
                for name, param in model.named_parameters():
                    if not name.startswith('fc.'):
                        param.requires_grad = False

        # ---- (3) データセットに合わせて最終層を新しいものに置き換え ----
        if 'vgg' in opt.model:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_cls)
        elif 'resnet' in opt.model:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, n_cls)
        # 他のモデルアーキテクチャもここに追加可能
        
    # === 2. ゼロから学習させる場合 (scratch) ===
    else:
        print(f"Initializing model from scratch: {opt.model}")
        # `weights=None` で学習済み重みを使わずに初期化
        model = model_dict[opt.model](weights=None, num_classes=n_cls)

    return model

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
  
    print("Initializing model...")

    # model
    n_cls = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'cinic10': 10,
    }.get(opt.dataset, None)
    
    # modelの初期化
    model = get_model(opt, n_cls).to(device) # .to(device) を使う
    
    criterion = nn.CrossEntropyLoss().to(device) # .to(device) を使う

    # 学習させるパラメータを指定する
    params_to_update = []
    print("Parameters to train:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(f"\t{name}")

    # optimizer
    optimizer = optim.SGD(params_to_update,
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)
        
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.epochs,
        eta_min=opt.lr_min if hasattr(opt, 'lr_min') else 0
    )

    cudnn.benchmark = True if torch.cuda.is_available() else False

    # dataloader
    print(f"Loading dataset: {opt.dataset}...")
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'cinic10':
        train_loader, val_loader = get_cinic10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)
    print("Dataset loaded successfully!")

    # tensorboard
    writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    # 記録用リスト
    train_acc_history = []
    test_acc_history = []
    test_loss_history = []
    lr_history = []

    # routine
    for epoch in range(1, opt.epochs + 1):
        print(f"Starting epoch {epoch}/{opt.epochs}...")

        # adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt, device)
        time2 = time.time()

        print(' * Epoch {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, train_acc, train_acc_top5, time2 - time1))
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)

        # train_accを記録
        train_acc_history.append(train_acc)

        test_acc, test_acc_top5, test_loss = validate_vanilla(val_loader, model, criterion, opt, device)

        print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))

        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_acc_top5', test_acc_top5, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': model.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            test_merics = { 'test_loss': float('%.4f' % test_loss),
                            'test_acc': float('%.4f' % test_acc),
                            'test_acc_top5': float('%.4f' % test_acc_top5),
                            'epoch': epoch}
            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
            print('saving the best model!')
            torch.save(state, save_file)

        # 記録
        test_acc_history.append(test_acc)
        test_loss_history.append(test_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])

        # optimizarの学習率を更新
        if scheduler is not None:
            scheduler.step()
            
        print(f"Epoch {epoch} completed. Train Acc: {train_acc:.2f}, Loss: {train_loss:.2f}")

    writer.close()

    # train_accの履歴を保存
    hist_path = os.path.join(opt.save_folder, "training_history.json")
    history = {
        "train_acc": train_acc_history,
        "test_acc": test_acc_history,
        "test_loss": test_loss_history,
        "lr": lr_history,
    }
    save_dict_to_json(history, hist_path)
    print(f"Training history saved to {hist_path}")

    # This best accuracy is only for printing purpose.
    print('best accuracy:', best_acc)

    # save parameters
    state = {k: v for k, v in opt._get_kwargs()}

    # No. parameters(M)
    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    state['Total params'] = num_params
    state['Total time'] =  float('%.2f' % ((time.time() - total_time) / 3600.0))
    params_json_path = os.path.join(opt.save_folder, "parameters.json") 
    save_dict_to_json(state, params_json_path)

if __name__ == '__main__':
    main()