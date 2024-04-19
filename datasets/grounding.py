# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import math
import random

import clip
import nncore
import torch
import torchvision.transforms as T
from nncore.dataset import DATASETS, Dataset
from nncore.ops import temporal_iou
from nncore.parallel import DataContainer

from .eval import vtg_eval

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


@DATASETS.register()
class Grounding(Dataset):

    def __init__(self,
                 label_path,
                 video_path=None,
                 cache_path=None,
                 query_path=None,
                 use_cache=False,
                 min_video_len=-1,
                 max_video_len=-1,
                 max_saliency=12,
                 fps=None,
                 unit=None):
        assert fps is not None
        self.pipeline = T.Compose([T.Normalize(_MEAN, _STD)])

        label = nncore.load(label_path)
        self.label = []
        for anno in label:
            if min_video_len > 0 and float(anno['duration']) < min_video_len:
                continue
            if max_video_len > 0 and float(anno['duration']) > max_video_len:
                continue
            if 'tacos' in label_path:
                anno['vid'] = f"{anno['vid']}-cam-002"
            self.label.append(anno)

        self.label_path = label_path
        self.video_path = video_path
        self.cache_path = cache_path
        self.query_path = query_path
        self.use_cache = use_cache
        self.min_video_len = min_video_len
        self.max_video_len = max_video_len
        self.max_saliency = max_saliency
        self.fps = fps
        self.unit = unit

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]

        data = dict()
        data = self.bind_video(label, data)
        data = self.bind_query(label, data)

        if 'relevant_windows' in label:
            data = self.bind_boundary(label, data)

        if 'relevant_windows' in label or 'saliency_scores' in label:
            data = self.bind_saliency(label, data)

        data['label'] = DataContainer(label, cpu_only=True)
        data['fps'] = DataContainer(self.fps, cpu_only=True)

        return data

    def bind_video(self, label, data):
        vid = label['vid']
        if self.use_cache:
            video = nncore.load(nncore.join(self.cache_path, f'{vid}.npy'))
            video = torch.from_numpy(video).view(video.shape[0], -1)
        else:
            video = nncore.load(nncore.join(self.video_path, f'{vid}.npy'))
            video = torch.from_numpy(video).float() / 255
            video = self.pipeline(video).view(video.size(0), -1)
        data['video'] = DataContainer(video, pad_value=float('inf'))
        return data

    def bind_query(self, label, data):
        if self.use_cache:
            qid = label['qid']
            query = nncore.load(nncore.join(self.query_path, f'{qid}.npy'))
            query = torch.from_numpy(query).view(query.shape[0], -1)
        else:
            query = label['query']
            query = clip.tokenize(query, truncate=True)[0]
        data['query'] = DataContainer(query, pad_value=float('inf'))
        return data

    def bind_boundary(self, label, data):
        max_time = data['video'].data.size(0) / self.fps

        boundary = torch.Tensor(label['relevant_windows'])
        boundary = boundary.clamp(min=0, max=max_time)

        inds = boundary[:, 1] - boundary[:, 0] < 0
        for idx in inds.nonzero()[:, 0]:
            boundary[idx] = boundary[idx].roll(1)

        inds = boundary[:, 1] - boundary[:, 0] < 1 / self.fps
        for idx in inds.nonzero()[:, 0]:
            center = boundary[idx].sum() / 2
            center = center.clamp(min=0.5 / self.fps, max=max_time - 0.5 / self.fps)
            boundary[idx, 0] = center - 0.5 / self.fps
            boundary[idx, 1] = center + 0.5 / self.fps

        data['boundary'] = DataContainer(boundary, pad_value=float('inf'))
        return data

    def bind_saliency(self, label, data):
        num_clips = data['video'].data.size(0)

        if 'saliency_scores' in label and 'relevant_clip_ids' in label:
            saliency = torch.zeros(int(label['duration'] * self.fps))
            for idx, score in zip(label['relevant_clip_ids'], label['saliency_scores']):
                saliency[idx] = sum(score) / self.max_saliency
            assert saliency.size(0) >= num_clips and saliency.size(0) - num_clips < 5
            saliency = saliency[:num_clips]
        else:
            boundary_ind = data['boundary'].data * self.fps
            pos_clips = []
            for bnd in boundary_ind.tolist():
                pos_clips += list(range(math.ceil(bnd[0]), math.ceil(bnd[1])))
            assert len(pos_clips) > 0
            saliency = torch.zeros(num_clips)
            saliency[pos_clips] = 1

        pos_clip = random.sample(saliency.nonzero()[:, 0].tolist(), 1)
        pos_clip = torch.LongTensor(pos_clip)

        data['saliency'] = DataContainer(saliency)
        data['pos_clip'] = DataContainer(pos_clip)

        return data

    def evaluate(self,
                 blobs,
                 nms_cfg=dict(type='normal', thres=0.7),
                 dump_template=None,
                 logger=None):
        assert len(blobs) == len(self)

        out = []
        for blob in blobs:
            pred = dict(vid=blob['label']['vid'], qid=blob['label']['qid'])

            if 'boundary' in blob:
                bnd = blob['boundary']

                # 1. clamp regression ranges
                inds = bnd[:, 1] < blob['label']['duration']
                bnd[:, :2] = bnd[:, :2].clamp(min=0, max=blob['label']['duration'])

                # 2. round boundaries to units
                if self.unit is not None and self.unit > 0:
                    bnd[inds, :2] = torch.round(bnd[inds, :2] / self.unit) * self.unit

                # 3. perform nms
                if nms_cfg is not None:
                    assert nms_cfg['type'] in ('normal', 'linear', 'gaussian')

                    for i in range(bnd.size(0)):
                        max_idx = bnd[i:, -1].argmax(dim=0)
                        bnd = nncore.swap_element(bnd, i, max_idx + i)
                        iou = temporal_iou(bnd[i, None, :-1], bnd[i + 1:, :-1])[0]

                        if nms_cfg['type'] == 'normal':
                            bnd[i + 1:, -1][iou >= nms_cfg['thres']] = 0
                        elif nms_cfg['type'] == 'linear':
                            bnd[i + 1:, -1] *= 1 - iou
                        else:
                            bnd[i + 1:, -1] *= (-iou.pow(2) / nms_cfg['sigma']).exp()

                    _, inds = bnd[:, -1].sort(descending=True)
                    bnd = bnd[inds]

                pred['pred_relevant_windows'] = bnd.tolist()

            if 'saliency' in blob:
                pred['pred_saliency_scores'] = blob['saliency'].tolist()

            out.append(pred)

        if dump_template is not None:
            out_path = dump_template.format('val' if 'val' in self.label_path else 'test')
            logger.info(f'Dumping inference outputs to {out_path}...')
            nncore.dump(out, out_path)
            return

        label = nncore.load(self.label_path)
        results = vtg_eval(out, label)['brief']

        return results
