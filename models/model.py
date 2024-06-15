# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import math

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import MODELS, build_loss, build_model

from .generator import PointGenerator

_CLIP_ARCHS = {
    'ViT-B/32': (768, 512, 50),
    'ViT-B/16': (768, 512, 197),
    'ViT-L/14': (1024, 768, 50),
    'ViT-L/14-336px': (1024, 768, 577)
}


@MODELS.register()
class R2Tuning(nn.Module):

    def __init__(self,
                 arch='ViT-B/32',
                 init=True,
                 dims=256,
                 strides=(1, 2, 4, 8),
                 buffer_size=1024,
                 max_num_moment=50,
                 merge_cls_sal=True,
                 adapter_cfg=None,
                 pyramid_cfg=None,
                 pooling_cfg=None,
                 class_head_cfg=None,
                 coord_head_cfg=None,
                 loss_cfg=None):
        super(R2Tuning, self).__init__()

        if init:
            self.clip, _ = clip.load(arch, device='cpu')
            for param in self.clip.parameters():
                param.requires_grad = False

        self.cfg = _CLIP_ARCHS[arch]
        self.adapter = build_model(adapter_cfg, dims, self.cfg[:2])
        self.pyramid = build_model(pyramid_cfg, dims, strides)
        self.pooling = build_model(pooling_cfg, dims)

        self.class_head = build_model(class_head_cfg, dims, 1)
        self.coord_head = build_model(coord_head_cfg, dims, 2)

        self.generator = PointGenerator(strides, buffer_size)

        self.coef = nn.Parameter(torch.ones(len(strides)))
        self.loss = build_loss(loss_cfg)

        self.max_num_moment = max_num_moment
        self.merge_cls_sal = merge_cls_sal

    def train(self, mode=True):
        super(R2Tuning, self).train(mode=mode)
        if hasattr(self, 'clip'):
            self.clip.eval()

    @torch.no_grad
    def clip_video_tower(self, video):
        video = video.type(self.clip.dtype)
        video = self.clip.visual.conv1(video)
        video = video.reshape(video.size(0), video.size(1), -1).permute(0, 2, 1)
        c_emb = video.new_zeros(video.size(0), 1, video.size(-1))
        c_emb = self.clip.visual.class_embedding.to(video.dtype) + c_emb
        video = torch.cat((c_emb, video), dim=1)
        video = video + self.clip.visual.positional_embedding.to(video.dtype)
        video = self.clip.visual.ln_pre(video).permute(1, 0, 2)
        emb = [video]
        for blk in self.clip.visual.transformer.resblocks:
            emb.append(blk(emb[-1]))
        video = torch.stack([e.permute(1, 0, 2) for e in emb])
        return video

    @torch.no_grad
    def clip_query_tower(self, query):
        query = self.clip.token_embedding(query).type(self.clip.dtype)
        query = query + self.clip.positional_embedding.type(self.clip.dtype)
        query = query.permute(1, 0, 2)
        emb = [query]
        for blk in self.clip.transformer.resblocks:
            emb.append(blk(emb[-1]))
        query = torch.stack([e.permute(1, 0, 2) for e in emb])
        return query

    def forward(self, data, mode='test'):
        video, query = data['video'], data['query']

        if hasattr(self, 'clip'):
            video_msk = torch.where(video[:, :, 0].isfinite(), 1, 0)
            query_msk = torch.where(query == 0, 0, 1)

            video[~video.isfinite()] = 0

            (b, t), d = video.size()[:2], int(math.sqrt(video.size(2) / 3))
            video = video.view(b * t, 3, d, d)

            video_emb = self.clip_video_tower(video)
            query_emb = self.clip_query_tower(query)

            n, _, p, c = video_emb.size()
            video_emb = video_emb.view(n, b, t, p, c)
        else:
            video_msk = torch.where(video[:, :, 0].isfinite(), 1, 0)
            query_msk = torch.where(query[:, :, 0].isfinite(), 1, 0)

            video[~video.isfinite()] = 0
            query[~query.isfinite()] = 0

            (b, t), l = video.size()[:2], query.size(1)
            video = video.view(b, t, -1, self.cfg[2], self.cfg[0]).permute(2, 0, 1, 3, 4)
            query = query.view(b, l, -1, self.cfg[1]).permute(2, 0, 1, 3)

            video_emb = video.float()
            query_emb = query.float()

        # video_emb: N * B * T * P * C
        # query_emb: N * B * L * C

        video_emb, query_emb, coll_v, coll_q = self.adapter(video_emb, query_emb,
                                                            video_msk, query_msk)

        pymid, pymid_msk = self.pyramid(video_emb, video_msk, return_mask=mode != 'test')
        point = self.generator(pymid)

        with torch.autocast('cuda', enabled=False):
            video_emb = video_emb.float()
            query_emb = self.pooling(query_emb.float(), query_msk)

            out_class = [self.class_head(e.float()) for e in pymid]
            out_class = torch.cat(out_class, dim=1)

            if self.coord_head is not None:
                out_coord = [
                    self.coord_head(e.float()).exp() * self.coef[i]
                    for i, e in enumerate(pymid)
                ]
                out_coord = torch.cat(out_coord, dim=1)
            else:
                out_coord = None

            output = dict(_avg_factor=b)

            if mode != 'test':
                data['coll_v'] = [e.float() for e in coll_v]
                data['coll_q'] = [self.pooling(e.float(), query_msk) for e in coll_q]

                data['point'] = point
                data['video_emb'] = video_emb
                data['query_emb'] = query_emb
                data['video_msk'] = video_msk
                data['pymid_msk'] = pymid_msk
                data['out_class'] = out_class
                data['out_coord'] = out_coord

                output = self.loss(data, output)

            if mode != 'train':
                assert b == 1, 'batch size larger than 1 is not supported for inference'
                out_class = out_class.sigmoid()
                out_score = F.cosine_similarity(video_emb, query_emb, dim=-1)

                output['_out'] = dict(label=data.get('label', [None])[0])

                pyd_shape = [e.size(1) for e in pymid]
                pyd_class = out_class[0, :, 0].split(pyd_shape)

                saliency = []
                for shape, score in zip(pyd_shape, pyd_class):
                    if t >= shape:
                        score = score.repeat_interleave(int(t / shape))
                        postfix = score[-1:].repeat(t - score.size(0))
                        score = torch.cat((score, postfix))
                    else:
                        scale = int(shape / t)
                        score = F.max_pool1d(score.unsqueeze(0), scale, stride=scale)[0]
                    saliency.append(score)

                saliency = torch.stack(saliency).amax(dim=0)

                if self.merge_cls_sal:
                    saliency *= out_score[0]

                output['_out']['saliency'] = saliency

                if self.coord_head is not None:
                    boundary = out_coord[0]
                    boundary[:, 0] *= -1
                    boundary *= point[:, 3, None].repeat(1, 2)
                    boundary += point[:, 0, None].repeat(1, 2)
                    boundary /= data['fps'][0]
                    boundary = torch.cat((boundary, out_class[0]), dim=-1)

                    _, inds = out_class[0, :, 0].sort(descending=True)
                    boundary = boundary[inds[:self.max_num_moment]]

                    output['_out']['boundary'] = boundary

        return output
