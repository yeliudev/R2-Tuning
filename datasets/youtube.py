# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import random

import clip
import nncore
import torch
from nncore.dataset import DATASETS
from nncore.parallel import DataContainer

from .constants import YOUTUBE_SPLITS
from .tvsum import TVSum


@DATASETS.register()
class YouTubeHighlights(TVSum):

    SPLITS = YOUTUBE_SPLITS

    def bind_query(self, vid, data):
        if self.use_cache:
            query = nncore.load(nncore.join(self.query_path, f'{vid}.npy'))
            query = torch.from_numpy(query).view(query.shape[0], -1)
        else:
            query = self.label[vid]['info'][5]
            query = clip.tokenize(query, truncate=True)[0]
        data['query'] = DataContainer(query, pad_value=float('inf'))
        return data

    def bind_saliency(self, vid, data):
        saliency = (torch.Tensor(self.label[vid]['match']) + 1) / 2
        assert data['video'].data.size(0) == saliency.size(0)

        pos_clip = random.sample((saliency > 0.5).nonzero()[:, 0].tolist(), 1)
        pos_clip = torch.LongTensor(pos_clip)

        data['saliency'] = DataContainer(saliency)
        data['pos_clip'] = DataContainer(pos_clip)
        return data

    def evaluate(self, blobs, **kwargs):
        assert len(blobs) == len(self)

        collected = []

        for blob in blobs:
            inds = torch.argsort(blob['saliency'], descending=True).cpu()

            label = [1 if s > 0 else 0 for s in blob['label']['match']]
            label = torch.Tensor(label)[inds].tolist()

            if (num_gt := sum(label)) == 0:
                collected.append(0)
                continue

            hits = ap = rec = 0
            prc = 1

            for i, gt in enumerate(label):
                hits += gt

                _rec = hits / num_gt
                _prc = hits / (i + 1)

                ap += (_rec - rec) * (prc + _prc) / 2
                rec, prc = _rec, _prc

            collected.append(ap)

        mean_ap = sum(collected) / len(collected)
        results = dict(mAP=round(mean_ap, 5))

        return results
