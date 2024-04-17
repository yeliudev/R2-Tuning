# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import random

import clip
import nncore
import torch
import torchvision.transforms as T
from nncore.dataset import DATASETS, Dataset
from nncore.parallel import DataContainer

from .constants import TVSUM_SPLITS

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


@DATASETS.register()
class TVSum(Dataset):

    SPLITS = TVSUM_SPLITS

    def __init__(self,
                 domain,
                 label_path,
                 video_path=None,
                 cache_path=None,
                 query_path=None,
                 use_cache=False):
        self.pipeline = T.Compose([T.Normalize(_MEAN, _STD)])

        self.label = nncore.load(label_path)

        self.vid = {
            k: [s for s in self.SPLITS[domain][k] if s in self.label]
            for k in ('train', 'val')
        }

        self.label_path = label_path
        self.video_path = video_path
        self.cache_path = cache_path
        self.query_path = query_path
        self.use_cache = use_cache

        self.state = 'train'

    def __len__(self):
        return len(self.vid[self.state])

    def __getitem__(self, idx):
        vid = self.vid[self.state][idx]

        data = dict()
        data = self.bind_video(vid, data)
        data = self.bind_query(vid, data)
        data = self.bind_saliency(vid, data)

        data['label'] = DataContainer(self.label[vid], cpu_only=True)
        return data

    def set_state(self, state):
        self.state = 'train' if state == 'train' else 'val'

    def bind_video(self, vid, data):
        if self.use_cache:
            video = nncore.load(nncore.join(self.cache_path, f'{vid}.npy'))
            video = torch.from_numpy(video).view(video.shape[0], -1)
        else:
            video = nncore.load(nncore.join(self.video_path, f'{vid}.npy'))
            video = torch.from_numpy(video).float() / 255
            video = self.pipeline(video).view(video.size(0), -1)
        data['video'] = DataContainer(video, pad_value=float('inf'))
        return data

    def bind_query(self, vid, data):
        if self.use_cache:
            query = nncore.load(nncore.join(self.query_path, f'{vid}.npy'))
            query = torch.from_numpy(query).view(query.shape[0], -1)
        else:
            query = self.label[vid]['title']
            query = clip.tokenize(query, truncate=True)[0]
        data['query'] = DataContainer(query, pad_value=float('inf'))
        return data

    def bind_saliency(self, vid, data):
        saliency = torch.Tensor(self.label[vid]['anno'])
        saliency = (saliency.sum(dim=1) - 20) / 80

        assert abs(data['video'].data.size(0) - saliency.size(0)) < 5
        num_clips = min(data['video'].data.size(0), saliency.size(0))
        data['video']._data = data['video']._data[:num_clips]
        saliency = saliency[:num_clips]

        binary = (saliency > saliency.median()).long()
        pos_clip = random.sample(binary.nonzero()[:, 0].tolist(), 1)
        pos_clip = torch.LongTensor(pos_clip)

        data['saliency'] = DataContainer(saliency)
        data['pos_clip'] = DataContainer(pos_clip)

        return data

    def evaluate(self, blobs, k=5, **kwargs):
        assert len(blobs) == len(self)

        collected = []

        for i in range(20):
            video_ap = []

            for blob in blobs:
                inds = torch.argsort(blob['saliency'], descending=True).cpu()

                label = torch.Tensor(blob['label']['anno'])[:, i]
                label = torch.where(label > label.median(), 1.0, .0)
                label = label[inds].tolist()[:k]

                if (num_gt := sum(label)) == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc

                video_ap.append(ap)

            collected.append(sum(video_ap) / len(video_ap))

        mean_ap = sum(collected) / len(collected)
        results = dict(mAP=round(mean_ap, 5))

        return results
