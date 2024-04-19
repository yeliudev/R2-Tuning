# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import argparse
import math

import clip
import nncore
import torch
import torch.nn as nn
import torchvision.transforms as T
from nncore.dataset import DATASETS, Dataset
from nncore.engine import build_dataloader
from nncore.nn import MODELS, build_model
from nncore.parallel import DataContainer

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


@DATASETS.register()
class VideoDataset(Dataset):

    def __init__(self, label, frame_dir, anno_path):
        self.pipeline = T.Compose([T.Normalize(_MEAN, _STD)])

        self.label, vids = [], set()
        if isinstance(label, list):
            for sample in label:
                if 'tacos' in anno_path:
                    sample['vid'] = f"{sample['vid']}-cam-002"
                if sample['vid'] not in vids:
                    self.label.append(sample)
                    vids.add(sample['vid'])
        else:
            for vid, sample in label.items():
                if vid not in vids:
                    sample['vid'] = vid
                    self.label.append(sample)
                    vids.add(vid)

        self.frame_dir = frame_dir

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        vid = self.label[idx]['vid']
        video = nncore.load(nncore.join(self.frame_dir, f'{vid}.npy'))
        video = torch.from_numpy(video).float() / 255
        video = self.pipeline(video).view(video.size(0), -1)
        video = DataContainer(video, pad_value=float('inf'))
        label = DataContainer(self.label[idx], cpu_only=True)
        return dict(label=label, video=video)


@DATASETS.register()
class QueryDataset(Dataset):

    def __init__(self, label):
        if isinstance(label, list):
            self.label = label
        else:
            self.label = []
            for vid, sample in label.items():
                sample['qid'] = vid
                self.label.append(sample)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        for key in ('query', 'title', 'info'):
            if key in self.label[idx]:
                query = self.label[idx][key]
                if key == 'info':
                    query = query[5]
                break
        query = clip.tokenize(query, truncate=True)[0]
        query = DataContainer(query)
        label = DataContainer(self.label[idx], cpu_only=True)
        return dict(label=label, query=query)


@MODELS.register()
class CLIP(nn.Module):

    def __init__(self, arch, k):
        super(CLIP, self).__init__()
        self.clip, _ = clip.load(arch, device='cpu')
        self.k = k

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

    def clip_query_tower(self, query):
        query = self.clip.token_embedding(query).type(self.clip.dtype)
        query = query + self.clip.positional_embedding.type(self.clip.dtype)
        query = query.permute(1, 0, 2)
        emb = [query]
        for blk in self.clip.transformer.resblocks:
            emb.append(blk(emb[-1]))
        query = torch.stack([e.permute(1, 0, 2) for e in emb])
        return query

    @torch.no_grad
    def forward(self, data):
        label = data['label']

        if 'video' in data:
            video = data['video']
            video_msk = torch.where(video[:, :, 0].isfinite(), 1, 0)

            video[~video.isfinite()] = 0
            (b, t), d = video.size()[:2], int(math.sqrt(video.size(2) / 3))
            video = video.view(b * t, 3, d, d)

            video_emb = self.clip_video_tower(video)

            n, _, p, c = video_emb.size()
            video_emb = video_emb.view(n, b, t, p, c)

            for i in range(b):
                emb = video_emb[-self.k:, i, :video_msk[i].sum()]
                emb = emb.permute(1, 0, 2, 3).half().cpu().numpy()
                out = f"{self.out_path}/{label[i]['vid']}.npy"
                if not nncore.is_file(out):
                    nncore.dump(emb, out)
        else:
            query = data['query']
            query_msk = torch.where(query == 0, 0, 1)
            query_emb = self.clip_query_tower(query)
            for i in range(query.size(0)):
                emb = query_emb[-self.k:, i, :query_msk[i].sum()]
                emb = emb.permute(1, 0, 2).half().cpu().numpy()
                out = f"{self.out_path}/{label[i]['qid']}.npy"
                if not nncore.is_file(out):
                    nncore.dump(emb, out)

        return dict()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('anno_path', type=str)
    parser.add_argument('frame_dir', type=str)
    parser.add_argument('--video_feat_dir', type=str)
    parser.add_argument('--query_feat_dir', type=str)
    parser.add_argument('--arch', type=str, default='ViT-B/32')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.arch in ('ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14-336px')

    print(f'Loading annotations from {args.anno_path}')
    anno = nncore.load(args.anno_path)
    npys = set(nncore.ls(args.frame_dir, ext='npy'))

    if isinstance(anno, list):
        if 'tacos' in args.anno_path:
            label = [s for s in anno if f"{s['vid']}-cam-002.npy" in npys]
        else:
            label = [s for s in anno if f"{s['vid']}.npy" in npys]
    else:
        label = {k: v for k, v in anno.items() if f'{k}.npy' in npys}

    print(f'Total samples: {len(anno)} Filtered samples: {len(label)}')

    if args.video_feat_dir is None:
        info = f"{args.arch[4:].lower().replace('/', '').replace('-', '_')}_vid_k{args.k}"
        args.video_feat_dir = nncore.join(nncore.dir_name(args.frame_dir), f'clip_{info}')

    if args.query_feat_dir is None:
        info = f"{args.arch[4:].lower().replace('/', '').replace('-', '_')}_txt_k{args.k}"
        args.query_feat_dir = nncore.join(nncore.dir_name(args.frame_dir), f'clip_{info}')

    model = build_model('CLIP', arch=args.arch, k=args.k, dist=False)

    print(f'Extracting video features to {args.video_feat_dir}')

    model.module.out_path = args.video_feat_dir
    dataset = dict(
        type='VideoDataset',
        label=label,
        frame_dir=args.frame_dir,
        anno_path=args.anno_path,
        loader=dict(batch_size=args.batch_size, num_workers=args.workers))

    data_loader = build_dataloader(dataset)
    for data in nncore.ProgressBar(data_loader):
        model(data)

    print(f'Extracting query features to {args.query_feat_dir}')

    model.module.out_path = args.query_feat_dir
    dataset = dict(
        type='QueryDataset',
        label=label,
        loader=dict(batch_size=args.batch_size, num_workers=args.workers))

    data_loader = build_dataloader(dataset)
    for data in nncore.ProgressBar(data_loader):
        model(data)


if __name__ == '__main__':
    main()
