# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import MODELS


class Permute(nn.Module):

    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.transpose(-1, -2)


@MODELS.register()
class ConvPyramid(nn.Module):

    def __init__(self, dims, strides):
        super(ConvPyramid, self).__init__()

        self.blocks = nn.ModuleList()
        for s in strides:
            p = int(math.log2(s))
            if p == 0:
                layers = nn.ReLU(inplace=True)
            else:
                layers = nn.Sequential()
                conv_cls = nn.Conv1d if p > 0 else nn.ConvTranspose1d
                for _ in range(abs(p)):
                    layers.extend([
                        Permute(),
                        conv_cls(dims, dims, 2, stride=2),
                        Permute(),
                        nn.LayerNorm(dims),
                        nn.ReLU(inplace=True)
                    ])
            self.blocks.append(layers)

        self.strides = strides

    def forward(self, x, mask, return_mask=False):
        pymid, pymid_msk = [], []

        for s, blk in zip(self.strides, self.blocks):
            if x.size(1) < s:
                continue

            pymid.append(blk(x))

            if return_mask:
                if s > 1:
                    msk = F.max_pool1d(mask.float(), s, stride=s).long()
                elif s < 1:
                    msk = mask.repeat_interleave(int(1 / s), dim=1)
                else:
                    msk = mask
                pymid_msk.append(msk)

        return pymid, pymid_msk


@MODELS.register()
class AdaPooling(nn.Module):

    def __init__(self, dims):
        super(AdaPooling, self).__init__()
        self.att = nn.Linear(dims, 1, bias=False)

    def forward(self, x, mask):
        a = self.att(x) + torch.where(mask.unsqueeze(2) == 1, .0, float('-inf'))
        a = a.softmax(dim=1)
        x = torch.matmul(x.transpose(1, 2), a)
        x = x.squeeze(2).unsqueeze(1)
        return x


@MODELS.register()
class ConvHead(nn.Module):

    def __init__(self, dims, out_dims, kernal_size=3):
        super(ConvHead, self).__init__()

        # yapf:disable
        self.module = nn.Sequential(
            Permute(),
            nn.Conv1d(dims, dims, kernal_size, padding=kernal_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(dims, out_dims, kernal_size, padding=kernal_size // 2),
            Permute())
        # yapf:enable

    def forward(self, x):
        return self.module(x)
