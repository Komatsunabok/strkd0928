import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cka.LinearCKA import linear_CKA
from helper.util import safe_flatten_and_mean

class CKAMapper(nn.Module):
    """
    教師・生徒の特徴マップをグループ分けし、対応付けを管理するモジュール
    """
    def __init__(self, s_shapes, t_shapes, feat_t, group_num=4, grouping='proportional'):
        super().__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes
        self.group_num = group_num

        # 与えられたデータすべてを使用
        # =ここに与えられるデータは計算されるべきものだけにする

        # CKAベースのグループ分け
        # 層インデックスをグループ数で分割したリスト
        # 例: s_shapes の長さが8（生徒の対象層が8個）、group_num=4 の場合
        # 　  self.s_groups == [[0, 1], [2, 3], [4, 5], [6, 7]]
        # 与えられた層のみでグループ分け
        # 与えられた層について0から番号をふっていく
        # これはただのインデックスなので、これをもとのモデルに使うと層は取得できてしまう
        # しかしそれは意図した割り当てではないので注意
        self.t_groups = self._split_groups_by_cka(feat_t, group_num)
        if grouping == 'uniform':
            self.s_groups = self._split_groups(len(s_shapes), group_num)
        elif grouping == 'proportional':
            self.s_groups = self._map_groups_by_ratio(len(s_shapes), self.t_groups)
        
        # 各グループのキー層インデックス（グループの中心層）
        self.s_key_layers = self._get_center_indices('student')
        self.t_key_layers = self._get_center_indices('teacher')
        print("student key layers", self.s_key_layers)
        print("teacher key layers", self.t_key_layers)
    
    def forward(self, feat_s, feat_t):
        # 各グループごとに特徴マップリストを返す
        # s_group_feats = [
        #     [feat_s[0], feat_s[1]],  # グループ1
        #     [feat_s[2], feat_s[3]],  # グループ2
        #     [feat_s[4], feat_s[5]],  # グループ3
        #     [feat_s[6], feat_s[7]],  # グループ4
        # ]
        s_group_feats = [[feat_s[i] for i in idxs] for idxs in self.s_groups]
        t_group_feats = [[feat_t[i] for i in idxs] for idxs in self.t_groups]
        return s_group_feats, t_group_feats

    # def forward(self, feat_s, feat_t):
    #     # 各グループのキー層特徴マップのみを返す
    #     # s_key_feats = [feat_s[1], feat_s[3], feat_s[5], feat_s[7], ... ]
    #     s_key_feats = [feat_s[i] for i in self.s_key_layers]
    #     t_key_feats = [feat_t[i] for i in self.t_key_layers]

    #     # s_key_feats shapes [torch.Size([64, 64, 32, 32]), torch.Size([64, 512, 4, 4])]
    #     # t_key_feats shapes [torch.Size([64, 64, 32, 32]), torch.Size([64, 512, 4, 4])]
    #     print("s_key_feats shapes", [f.size() for f in s_key_feats]) #ok
    #     print("t_key_feats shapes", [f.size() for f in t_key_feats]) #ok
        
    #     return s_key_feats, t_key_feats
    
    def _split_groups_by_cka(self, feat, group_num):
        # 1. CKA行列を計算（隣接層のみ）
        n = len(feat)
        cka_mat = np.zeros((n-1,))
        for i in range(n-1):
            f1 = safe_flatten_and_mean(feat[i])
            f2 = safe_flatten_and_mean(feat[i + 1])           
            cka_mat[i] = linear_CKA(f1, f2).item()  # 隣接層間のCKA値
            # 隣接した層でないと同じグループにはならないので、隣接層間のCKA値のみで十分
            # LinearCKAはtorch.Tensorを受け取るので、feat_t[i]とfeat_t[i+1]はtorch.Tensorである必要がある
            # LinearCKAでCKA計算のために２次元テンソルに変形されるので、ここでは４次元テンソルをそのまま渡してもよい

        # 2. CKA値が高い順にグループ化（貪欲法）
        # まず全層を1つずつグループに分ける
        groups = [[i] for i in range(n)]
        # グループ数が指定数になるまで、CKA値が最大の隣接グループをマージ
        while len(groups) > group_num:
            # 隣接グループ間のCKA値を取得
            merge_scores = [cka_mat[groups[i][-1]] for i in range(len(groups)-1)]
            # 最高のCKA値を持つ隣接グループを見つける
            # ある層とその次の層のCKA値を用いて、隣接グループ間のCKA値を計算
            max_idx = np.argmax(merge_scores)
            # マージ
            groups[max_idx] = groups[max_idx] + groups[max_idx+1] # 隣接グループmax_ind+1をmax_idxにマージ
            del groups[max_idx+1] # マージして必要なくなったグループ(max_ind+1)を削除
        
        print("teacher group", groups)
        return groups

    def _split_groups(self, num_layers, group_num):
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

        print("setudent group", groups)
        return groups

    def _map_groups_by_ratio(self, num_layers, teacher_groups):
        # 教師の総層数
        total_teacher_layers = sum(len(g) for g in teacher_groups)
        # 教師のグループごとの割合
        ratios = [len(g) / total_teacher_layers for g in teacher_groups]

        # 生徒側のグループサイズの割当（まずは丸め前）
        raw_sizes = [r * num_layers for r in ratios]

        # 小数点以下での切り上げ・切り下げを考慮して調整
        sizes = [int(round(r)) for r in raw_sizes]

        # 合計が num_layers になるように補正
        diff = sum(sizes) - num_layers
        while diff != 0:
            if diff > 0:
                # 大きすぎるので1引く
                for i in range(len(sizes)):
                    if sizes[i] > 1:
                        sizes[i] -= 1
                        diff -= 1
                        if diff == 0:
                            break
            else:
                # 小さすぎるので1足す
                for i in range(len(sizes)):
                    sizes[i] += 1
                    diff += 1
                    if diff == 0:
                        break

        # 実際のインデックスに基づく生徒グループを作成
        groups = []
        idx = 0
        for size in sizes:
            group = list(range(idx, idx + size))
            groups.append(group)
            idx += size

        print("student group", groups)
        return groups
    
    # 各グループのキー層インデックス（グループの中心をとる）
    def _get_center_indices(self, who):
        center_indices = []
        if who == 'teacher':
            groups = self.t_groups
        elif who == 'student':
            groups = self.s_groups

        for idxs in groups:  # もしくは self.t_groups
            if len(idxs) == 0:
                continue
            center = idxs[len(idxs)//2]
            center_indices.append(center)

        return center_indices
