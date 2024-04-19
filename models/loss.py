# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import LOSSES, Parameter, build_loss


@LOSSES.register()
class SampledNCELoss(nn.Module):

    def __init__(self,
                 temperature=0.07,
                 max_scale=100,
                 learnable=False,
                 direction=('row', 'col'),
                 loss_weight=1.0):
        super(SampledNCELoss, self).__init__()

        scale = torch.Tensor([math.log(1 / temperature)])

        if learnable:
            self.scale = Parameter(scale)
        else:
            self.register_buffer('scale', scale)

        self.temperature = temperature
        self.max_scale = max_scale
        self.learnable = learnable
        self.direction = (direction, ) if isinstance(direction, str) else direction
        self.loss_weight = loss_weight

    def extra_repr(self):
        return ('temperature={}, max_scale={}, learnable={}, direction={}, loss_weight={}'
                .format(self.temperature, self.max_scale, self.learnable, self.direction,
                        self.loss_weight))

    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        batch_inds = torch.arange(video_emb.size(0), device=video_emb.device)

        pos_scores = saliency[batch_inds, pos_clip].unsqueeze(-1)
        loss_msk = (saliency <= pos_scores) * video_msk

        scale = self.scale.exp().clamp(max=self.max_scale)
        i_sim = F.cosine_similarity(video_emb, query_emb, dim=-1) * scale
        i_sim = i_sim + torch.where(loss_msk > 0, .0, float('-inf'))

        loss = 0

        if 'row' in self.direction:
            i_met = F.log_softmax(i_sim, dim=1)[batch_inds, pos_clip]
            loss = loss - i_met.sum() / i_met.size(0)

        if 'col' in self.direction:
            j_sim = i_sim.t()
            j_met = F.log_softmax(j_sim, dim=1)[pos_clip, batch_inds]
            loss = loss - j_met.sum() / j_met.size(0)

        loss = loss * self.loss_weight
        return loss


@LOSSES.register()
class BundleLoss(nn.Module):

    def __init__(self,
                 sample_radius=1.5,
                 loss_cls=None,
                 loss_reg=None,
                 loss_sal=None,
                 loss_video_cal=None,
                 loss_layer_cal=None):
        super(BundleLoss, self).__init__()

        self._loss_cls = build_loss(loss_cls)
        self._loss_reg = build_loss(loss_reg)
        self._loss_sal = build_loss(loss_sal)
        self._loss_video_cal = build_loss(loss_video_cal)
        self._loss_layer_cal = build_loss(loss_layer_cal)

        self.sample_radius = sample_radius

    def get_target_single(self, point, gt_bnd, gt_cls):
        num_pts, num_gts = point.size(0), gt_bnd.size(0)

        lens = gt_bnd[:, 1] - gt_bnd[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        gt_seg = gt_bnd[None].expand(num_pts, num_gts, 2)
        s = point[:, 0, None] - gt_seg[:, :, 0]
        e = gt_seg[:, :, 1] - point[:, 0, None]
        r_tgt = torch.stack((s, e), dim=-1)

        if self.sample_radius > 0:
            center = (gt_seg[:, :, 0] + gt_seg[:, :, 1]) / 2
            t_mins = center - point[:, 3, None] * self.sample_radius
            t_maxs = center + point[:, 3, None] * self.sample_radius
            dist_s = point[:, 0, None] - torch.maximum(t_mins, gt_seg[:, :, 0])
            dist_e = torch.minimum(t_maxs, gt_seg[:, :, 1]) - point[:, 0, None]
            center = torch.stack((dist_s, dist_e), dim=-1)
            cls_msk = center.min(-1)[0] >= 0
        else:
            cls_msk = r_tgt.min(-1)[0] >= 0

        reg_dist = r_tgt.max(-1)[0]
        reg_msk = torch.logical_and((reg_dist >= point[:, 1, None]),
                                    (reg_dist <= point[:, 2, None]))

        lens.masked_fill_(cls_msk == 0, float('inf'))
        lens.masked_fill_(reg_msk == 0, float('inf'))
        min_len, min_len_inds = lens.min(dim=1)

        min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)),
                                         (lens < float('inf'))).to(r_tgt.dtype)

        label = F.one_hot(gt_cls[:, 0], 2).to(r_tgt.dtype)
        c_tgt = torch.matmul(min_len_mask, label).clamp(min=0.0, max=1.0)[:, 1]
        r_tgt = r_tgt[range(num_pts), min_len_inds] / point[:, 3, None]

        return c_tgt, r_tgt

    def get_target(self, data):
        cls_tgt, reg_tgt = [], []

        for i in range(data['boundary'].size(0)):
            gt_bnd = data['boundary'][i] * data['fps'][i]
            gt_cls = gt_bnd.new_ones(gt_bnd.size(0), 1).long()

            c_tgt, r_tgt = self.get_target_single(data['point'], gt_bnd, gt_cls)

            cls_tgt.append(c_tgt)
            reg_tgt.append(r_tgt)

        cls_tgt = torch.stack(cls_tgt)
        reg_tgt = torch.stack(reg_tgt)

        return cls_tgt, reg_tgt

    def loss_cls(self, data, output, cls_tgt):
        src = data['out_class'].squeeze(-1)
        msk = torch.cat(data['pymid_msk'], dim=1)

        loss_cls = self._loss_cls(src, cls_tgt, weight=msk, avg_factor=msk.sum())

        output['loss_cls'] = loss_cls
        return output

    def loss_reg(self, data, output, cls_tgt, reg_tgt):
        src = data['out_coord']
        msk = cls_tgt.unsqueeze(2).repeat(1, 1, 2).bool()

        loss_reg = self._loss_reg(src, reg_tgt, weight=msk, avg_factor=msk.sum())

        output['loss_reg'] = loss_reg
        return output

    def loss_sal(self, data, output):
        video_emb = data['video_emb']
        query_emb = data['query_emb']
        video_msk = data['video_msk']

        saliency = data['saliency']
        pos_clip = data['pos_clip'][:, 0]

        output['loss_sal'] = self._loss_sal(video_emb, query_emb, video_msk, saliency,
                                            pos_clip)
        return output

    def loss_cal(self, data, output):
        pos_clip = data['pos_clip'][:, 0]

        batch_inds = torch.arange(pos_clip.size(0), device=pos_clip.device)

        coll_v_emb, coll_q_emb = [], []
        for v_emb, q_emb in zip(data['coll_v'], data['coll_q']):
            v_emb_pos = v_emb[batch_inds, pos_clip]
            q_emb_pos = q_emb[:, 0]

            coll_v_emb.append(v_emb_pos)
            coll_q_emb.append(q_emb_pos)

        v_emb = torch.stack(coll_v_emb)
        q_emb = torch.stack(coll_q_emb)
        output['loss_video_cal'] = self._loss_video_cal(v_emb, q_emb)

        v_emb = torch.stack(coll_v_emb, dim=1)
        q_emb = torch.stack(coll_q_emb, dim=1)
        output['loss_layer_cal'] = self._loss_layer_cal(v_emb, q_emb)

        return output

    def forward(self, data, output):
        if self._loss_reg is not None:
            cls_tgt, reg_tgt = self.get_target(data)
            output = self.loss_reg(data, output, cls_tgt, reg_tgt)
        else:
            cls_tgt = data['saliency']

        if self._loss_cls is not None:
            output = self.loss_cls(data, output, cls_tgt)

        if self._loss_sal is not None:
            output = self.loss_sal(data, output)

        if self._loss_video_cal is not None or self._loss_layer_cal is not None:
            output = self.loss_cal(data, output)

        return output
