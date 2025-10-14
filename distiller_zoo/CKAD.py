from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cka.LinearCKA import linear_CKA
from helper.util import safe_flatten_and_mean

class CKADistillLoss(nn.Module):
    """
    CKAベースの蒸留損失関数
    """
    def __init__(self, group_num=4, method_inner_group='mean', method_inter_group='mean'):
        super().__init__()
        self.group_num = group_num
        self.method_inner_group = method_inner_group  # グループ内のCKA計算方法
        self.method_inter_group = method_inter_group  # グループ間のCKA計算方法

    def forward(self, s_group_feats, t_group_feats):
        """
        入力
        s_group_feats, t_group_feats：選択した層の特徴量（なので全部計算に使ってよい）
        """
        # 各グループ間でCKA損失を計算
        inter_group_losses = []
        for s_feats, t_feats in zip(s_group_feats, t_group_feats):
            print("s_feats shapes in CKADistillLoss", [f.size() for f in s_feats])
            print("t_feats shapes in CKADistillLoss", [f.size() for f in t_feats])
            inner_loss = self._calc_inner_group_loss(s_feats, t_feats)
            inter_group_losses.append(inner_loss)

        # グループ間での集約
        total_loss = self._aggregate_inter_group(inter_group_losses)
        return total_loss

    # -------------------------------
    # ▼ グループ内CKA計算部分
    # -------------------------------
    def _calc_inner_group_loss(self, s_feats, t_feats):
        """
        1つのグループ内で、層ごとの特徴マップ間のCKAを計算して平均化
        """
        if self.method_inner_group == 'mean':
            cka_losses = []
            for s in s_feats:
                for t in t_feats:
                    fs = safe_flatten_and_mean(s)
                    ft = safe_flatten_and_mean(t)
                    cka = linear_CKA(fs, ft)
                    cka_losses.append(1 - cka)
            return torch.mean(torch.stack(cka_losses))

        elif self.method_inner_group == 'max':
            # 最大損失を取る（最も類似していないペアを重視）
            cka_losses = []
            for s in s_feats:
                for t in t_feats:
                    fs = safe_flatten_and_mean(s)
                    ft = safe_flatten_and_mean(t)
                    cka = linear_CKA(fs, ft)
                    cka_losses.append(1 - cka)
            return torch.max(torch.stack(cka_losses))

        else:
            raise ValueError(f"Unknown inner group method: {self.method_inner_group}")

    # -------------------------------
    # ▼ グループ間CKA集約部分
    # -------------------------------
    def _aggregate_inter_group(self, inter_group_losses):
        """
        グループ間の損失のまとめ方
        """
        if self.method_inter_group == 'mean':
            return torch.mean(torch.stack(inter_group_losses))
        elif self.method_inter_group == 'sum':
            return torch.sum(torch.stack(inter_group_losses))
        elif self.method_inter_group == 'max':
            return torch.max(torch.stack(inter_group_losses))
        else:
            raise ValueError(f"Unknown inter group method: {self.method_inter_group}")
