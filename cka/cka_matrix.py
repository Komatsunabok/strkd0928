import torch
import torch.nn as nn
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.load_model import load_model
from helper.load_dataset import load_dataset
from helper.hooks import register_hooks
from helper.util import safe_flatten_and_mean
from cka.LinearCKA import linear_CKA

def calc_cka_matrix(feature1_list, feature2_list):
    cka_matrix = torch.zeros(len(feature1_list), len(feature2_list))
    for i, s in enumerate(feature2_list):
        for j, t in enumerate(feature1_list):
            fs = safe_flatten_and_mean(s)
            ft = safe_flatten_and_mean(t)
            cka_matrix[i, j] =  linear_CKA(fs, ft).item()
    return cka_matrix

def main(model1_path, model1_name, model2_path, model2_name, 
                    dataset='cifar10', batch_size=64, num_workers=8,
                    layer_types=None):
    """
    入力：
        model*_path: モデルのパス（ex. ../save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth）
        model*_name: モデルの名前（ex. "VGG13"）
        dataset:
        batch_size:
        num_workers:
        layer_types: CKAで利用する層タイプ（ex. (nn.BatchNorm2d, nn.Linear)）
    """
    # データロード
    _, val_loader = load_dataset(dataset, batch_size, num_workers)
    
    # model
    n_cls = {
        'cifar10':10,
        'cifar100': 100,
        'imagenet': 1000,
        'cinic10': 10
    }.get(dataset, None)

    # モデルをロード
    model1 = load_model(model1_path, model1_name, n_cls)
    model2 = load_model(model2_path, model2_name, n_cls)

    # 推論モードに
    model1.eval()
    model2.eval()

    # hookをかける
    hooks_1, feature_hook_1 = register_hooks(model1, layer_types)
    hooks_2, feature_hook_2 = register_hooks(model2, layer_types)
    
    # 推論
    # 特徴量の取得
    for images, labels in val_loader:
        feat_t, _ = model1(images, is_feat=True)
        feat_s, _ = model2(images, is_feat=True)
        break

    # CKAマトリクスの計算
    cka_matrix = calc_cka_matrix(feature_hook_1.outputs, feature_hook_2.outputs)

    print(cka_matrix)


if __name__ == '__main__':
    main(model1_path="../save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth",
         model1_name="vgg13",
         model2_path="../save/students/models/S-vgg13_T-vgg13_cifar10_ckad_r-1.0_a-1.0_b-400.0_0/vgg13_best.pth",
         model2_name="vgg13",
         layer_types=(nn.BatchNorm2d, nn.Linear))
    
"""
model_dict = {
    'resnet38': resnet38,
    'resnet110': resnet110,
    'resnet116': resnet116,
    'resnet14x2': resnet14x2,
    'resnet38x2': resnet38x2,
    'resnet110x2': resnet110x2,
    'resnet8x4': resnet8x4,
    'resnet14x4': resnet14x4,
    'resnet32x4': resnet32x4,
    'resnet38x4': resnet38x4,
    'vgg8': vgg8_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,  # vgg16 is not defined, using vgg13_bn as a placeholder
    'MobileNetV2': mobile_half,
    'MobileNetV2_1_0': mobile_half_double,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_1_5': ShuffleV2_1_5,
    
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'resnext50_32x4d': resnext50_32x4d,
    'ResNet10x2': wide_resnet10_2,
    'ResNet18x2': wide_resnet18_2,
    'ResNet34x2': wide_resnet34_2,
    'wrn_50_2': wide_resnet50_2,
    
    'MobileNetV2_Imagenet': mobilenet_v2,
    'ShuffleV2_Imagenet': shufflenet_v2_x1_0,
}
"""