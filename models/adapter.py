# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import torch
import torch.nn as nn
from nncore.nn import MODELS, build_model


@MODELS.register()
class R2Block(nn.Module):

    def __init__(self,
                 dims,
                 in_dims,
                 k=4,
                 dropout=0.5,
                 use_tef=True,
                 pos_cfg=None,
                 tem_cfg=None):
        super(R2Block, self).__init__()

        # yapf:disable
        self.video_map = nn.Sequential(
            nn.LayerNorm((in_dims[0] + 2) if use_tef else in_dims[0]),
            nn.Dropout(dropout),
            nn.Linear((in_dims[0] + 2) if use_tef else in_dims[0], dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dims),
            nn.Dropout(dropout),
            nn.Linear(dims, dims))

        self.query_map = nn.Sequential(
            nn.LayerNorm(in_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(in_dims[1], dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dims),
            nn.Dropout(dropout),
            nn.Linear(dims, dims))
        # yapf:enable

        if k > 1:
            self.gate = nn.Parameter(torch.zeros([k - 1]))

        self.v_map = nn.Linear(dims, dims)
        self.q_map = nn.Linear(dims, dims)
        self.scale = nn.Parameter(torch.zeros([k]))

        self.pos = build_model(pos_cfg, dims=dims)
        self.tem = build_model(tem_cfg, dims=dims)

        self.dims = dims
        self.in_dims = in_dims
        self.k = k
        self.dropout = dropout
        self.use_tef = use_tef

    def forward(self, video_emb, query_emb, video_msk, query_msk):
        video_emb = video_emb[-self.k:]
        query_emb = query_emb[-self.k:]

        _, b, t, p, _ = video_emb.size()

        if self.use_tef:
            tef_s = torch.arange(0, 1, 1 / t, device=video_emb.device)
            tef_e = tef_s + 1.0 / t
            tef = torch.stack((tef_s, tef_e), dim=1)
            tef = tef.unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(self.k, b, 1, p, 1)
            video_emb = torch.cat((video_emb, tef[:, :, :video_emb.size(2)]), dim=-1)

        coll_v, coll_q, last = [], [], None
        for i in range(self.k - 1, -1, -1):
            v_emb = self.video_map(video_emb[i])  # B * T * P * C
            q_emb = self.query_map(query_emb[i])  # B * L * C

            coll_v.append(v_emb[:, :, 0])
            coll_q.append(q_emb)

            v_pool = v_emb.view(b * t, -1, self.dims)  # BT * P * C
            q_pool = q_emb.repeat_interleave(t, dim=0)  # BT * L * C

            v_pool_map = self.v_map(v_pool)  # BT * P * C
            q_pool_map = self.q_map(q_pool)  # BT * L * C

            att = torch.bmm(q_pool_map, v_pool_map.transpose(1, 2)) / self.dims**0.5
            att = att.softmax(-1)  # BT * L * P

            o_pool = torch.bmm(att, v_pool) + q_pool  # BT * L * C
            o_pool = o_pool.amax(dim=1, keepdim=True)  # BT * 1 * C
            v_emb = v_pool[:, 0, None] + o_pool * self.scale[i].tanh()
            v_emb = v_emb.view(b, t, self.dims)  # B * T * C

            if i < self.k - 1:
                gate = self.gate[i].sigmoid()
                v_emb = gate * v_emb + (1 - gate) * last

            v_pe = self.pos(v_emb)
            last = self.tem(v_emb, q_emb, q_pe=v_pe, q_mask=video_msk, k_mask=query_msk)

        return last, q_emb, coll_v, coll_q
