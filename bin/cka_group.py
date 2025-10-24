import torch
import torch.nn as nn
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import os
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model_dict
from helper.load_model import load_model
from helper.load_dataset import load_dataset
from helper.hooks import register_hooks
from helper.util import safe_flatten_and_mean
from cka.LinearCKA import linear_CKA
from cka.cka_matrix import calc_cka_matrix

split_symbol = '~' if os.name == 'nt' else ':'

def split_groups_by_cka(feat=None, group_num=4, cka_matrix=None):
    """
    隣接層間のCKA行列（2次元） or feat からグループを分割する。

    引数:
        feat: 特徴マップのリスト
        group_num: 分けたいグループ数
        cka_matrix: CKA行列（形状: [n_layers, n_layers]）

    戻り値:
        groups: [[idx1, idx2, ...], ...] のようなインデックスのリスト
    """
    if cka_matrix is not None:
        # 2次元のCKAマトリクスから、隣接層だけを抜き出す（1次元に）
        n = cka_matrix.shape[0]
        cka_vec = [cka_matrix[i, i + 1] for i in range(n - 1)]
    else:
        assert feat is not None, "cka_matrix か feat のどちらかを指定してください。"
        n = len(feat)
        cka_vec = []
        for i in range(n - 1):
            f1 = safe_flatten_and_mean(feat[i])
            f2 = safe_flatten_and_mean(feat[i + 1])
            cka_val = linear_CKA(f1, f2).item()
            cka_vec.append(cka_val)

    # 初期化：各層を別グループにする
    groups = [[i] for i in range(n)]

    while len(groups) > group_num:
        merge_scores = [cka_vec[i] for i in range(len(groups) - 1)]
        max_idx = np.argmax(merge_scores)
        groups[max_idx] = groups[max_idx] + groups[max_idx + 1]
        del groups[max_idx + 1]
        del cka_vec[max_idx]  # マージしたところのCKAは削除

    print("group split:", groups)
    return groups


def split_groups(num_layers, group_num):
    """
    入力:
        num_layers: 特徴マップの数（= 選択された層の数）
        group_num: グループ数
    出力:
        groups: [[idx1, idx2, ...], ...] の形でインデックスのグループを返す
    """
    groups = []
    base = num_layers // group_num  # 基本のサイズ
    rem = num_layers % group_num    # 余り（等分できなかったぶん）

    start = 0
    for i in range(group_num):
        group_size = base + (1 if i < rem else 0)  # 余りを前のグループに1つずつ配る
        group = list(range(start, start + group_size))
        groups.append(group)
        start += group_size

    return groups

def get_group_boundaries(groups):
    """
    グループリスト（[[0,1], [2,3,4], [5,6]]）から区切りインデックスを返す。
    """
    boundaries = []
    count = 0
    for g in groups[:-1]:  # 最後のグループの後ろには線を引かない
        count += len(g)
        boundaries.append(count)
    return boundaries

def plot_cka_with_groups(cka_array, model1_groups, model2_groups, 
                         model1_name='Model1', model2_name='Model2', img_name=''):
    """
    CKAマトリクスと層のグループ分けを可視化する
    """
    # グループ境界インデックスを取得
    x_bounds = get_group_boundaries(model1_groups)
    y_bounds = get_group_boundaries(model2_groups)

    # ヒートマップ表示
    plt.figure(figsize=(7, 5))
    plt.imshow(cka_array, cmap='inferno', origin='upper')
    plt.colorbar()
    plt.xlabel(f'{model1_name} Layers')
    plt.ylabel(f'{model2_name} Layers')
    plt.title(f'CKA Matrix - {os.path.basename(img_name)}')

    # 縦線（x方向：モデル1）
    for x in x_bounds:
        plt.axvline(x - 0.5, color='white', linestyle='--', linewidth=1.5)

    # 横線（y方向：モデル2）
    for y in y_bounds:
        plt.axhline(y - 0.5, color='white', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.show()

def main(model1_path=None, model1_name="model1", model2_path=None, model2_name="model1", 
         csv_path=None, dataset='cifar10', batch_size=64, num_workers=8,
         layer_types=None, group_num=3):
    """
    入力：
        model*_path: モデルのパス（ex. ../save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth）
        model*_name: モデルの名前（ex. "VGG13"）
        ※大抵
        dataset:
        batch_size:
        num_workers:
        layer_types: CKAで利用する層タイプ（ex. (nn.BatchNorm2d, nn.Linear)）
    """
    feat_t = None
    if csv_path == None:
        # データロード
        data_folder_dir = {
            'cifar10':"../data",
            'cifar100': "../data",
            'imagenet': "../data",
            'cinic10': "../data/cinic-10"
        }.get(dataset, None)
        print("data_folder_dir: ", data_folder_dir)
        _, val_loader = load_dataset(data_folder_dir, dataset, batch_size, num_workers)
        
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
    else:
        cka_matrix = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row
            for row in reader:
                cka_matrix.append([float(value) for value in row])
        cka_matrix = np.array(cka_matrix)

    print(cka_matrix)
    # グループ分け
    group1 = split_groups_by_cka(feat_t, group_num, cka_matrix)
    # group2 = group1

    group2 = split_groups(cka_matrix.shape[0], group_num)

    print("group1: ", group1)
    print("group2: ", group2)

    # CKAマトリクスと層のグループ分けを可視化する
    plot_cka_with_groups(cka_matrix, group1, group2, 
                         model1_name, model2_name, str(group_num)+" groups")



if __name__ == '__main__':
    # main(model1_path="save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth",
    #      model1_name="vgg13",
    #      model2_path="save/students/models/S-vgg13_T-vgg13_cifar10_kd_r-1.0_a-1.0_b-400.0_0/vgg13_best.pth",
    #      model2_name="vgg13",
    #      csv_path="save/cka_logs/cka_log_S-vgg13_T-vgg13_cifar10_ckad_r-1.0_a-1.0_b-400.0_0/cka_epoch_001.csv",
    #      dataset="cifar10", layer_types=(nn.BatchNorm2d, nn.Linear), group_num=4)
    main(model1_path="../save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth",
         model1_name="vgg13",
         model2_path="../save/teachers/models/vgg13_vanilla_cifar10_trial_0_epochs_240_bs_64/vgg13_best.pth",
         model2_name="vgg13",
         dataset="cifar10", layer_types=(nn.BatchNorm2d, nn.Linear), group_num=4)
