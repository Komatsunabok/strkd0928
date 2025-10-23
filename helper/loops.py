from __future__ import print_function, division
from cProfile import label

import sys
import time
import torch
from .util import AverageMeter, accuracy, reduce_tensor

from cka.LinearCKA import linear_CKA
from .util import safe_flatten_and_mean
from .log_cka import log_cka_matrix

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt, device):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        images, labels = batch_data
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ===================forward=====================
        output = model(images)
        loss = criterion(output, labels)
        # nn.Module(継承している)のforwardメソッドは自動的に呼び出されるので、明示的に呼び出す必要はない
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def validate_vanilla(val_loader, model, criterion, opt, device):
    """validation"""
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            images, labels = batch_data
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, 
                  feature_hook_t, feature_hook_s, device):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()

    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    # 確認のため30エポックごとにCKAを計算
    # if epoch % 30 == 1:
    if epoch < 30:
        model_s.eval()
        with torch.no_grad():
            inputs, _ = next(iter(train_loader))
            inputs = inputs.cuda(0, non_blocking=True)

            feature_hook_s.outputs.clear()            
            feature_hook_t.outputs.clear()

            feat_s, _ = model_s(inputs, is_feat=True)
            feat_t, _ = model_t(inputs, is_feat=True)

            # hook出力はここで取得済み
            feat_s = feature_hook_s.outputs
            feat_t = feature_hook_t.outputs

            cka_matrix = torch.zeros(len(feat_s), len(feat_t))
            for i, s in enumerate(feat_s):
                for j, t in enumerate(feat_t):
                    fs = safe_flatten_and_mean(s)
                    ft = safe_flatten_and_mean(t)
                    cka_matrix[i, j] =  linear_CKA(fs, ft).item()

            log_cka_matrix(epoch, cka_matrix, opt)  # CKAの結果をCSV記録する関数
            print("cka matrix was saved!")

    # training
    model_s.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)

    end = time.time()
    for idx, data in enumerate(train_loader):
        images, labels = data

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ===================forward=====================
        feature_hook_t.outputs.clear()
        feature_hook_s.outputs.clear()

        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad(): # 勾配追跡しない
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t] # テンソルをグラフから切り離して、以降の計算で勾配を計算しないようにする
            # 教師モデルの特徴マップを使っても勾配が更新されないようにする
 
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)
        
        # other kd loss
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'ckad':
            # グループ化（module_list[1]がCKAMapperの場合）
            # 各グループごとにまとめた特徴マップのリスト（リストのリスト）
            # s_group_feats = [
            #     [feat_s[0], feat_s[1]],  # グループ1
            #     [feat_s[2], feat_s[3]],  # グループ2
            #     [feat_s[4], feat_s[5]],  # グループ3
            #     [feat_s[6], feat_s[7]],  # グループ4
            # ]
            s_group_feats, t_group_feats = module_list[1](
                feat_t=feature_hook_t.outputs, feat_s=feature_hook_s.outputs)
            loss_kd = criterion_kd(s_group_feats, t_group_feats)
        else:
            raise NotImplementedError(opt.distill)
        

        b = opt.beta * (0.1 ** (epoch / opt.epochs))  # ベータをエポックに応じて減衰させる
            
        loss = opt.cls * loss_cls + opt.div * loss_div + b * loss_kd
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
    
    # キャッシュをクリア
    # メモリが足りない場合に備えて、解決策になるかもしれない
    # torch.cuda.empty_cache() 

    return top1.avg, top5.avg, losses.avg

def validate_distill(val_loader, module_list, criterion, opt, device):
    """validation"""
    # switch to evaluate mode
    for module in module_list:
        module.eval()
    model_s = module_list[0]
    model_t = module_list[-1]
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            images, labels = batch_data
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # compute output
            output = model_s(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

