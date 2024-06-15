# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import argparse

import clip
import decord
import nncore
import torch
import numpy as np
import torchvision.transforms.functional as F
from decord import VideoReader
from nncore.engine import load_checkpoint
from nncore.nn import build_model
from tabulate import tabulate

CONFIG = 'configs/qvhighlights/r2_tuning_qvhighlights.py'
WEIGHT = 'https://huggingface.co/yeliudev/R2-Tuning/resolve/main/checkpoints/r2_tuning_qvhighlights-ed516355.pth'  # noqa


def load_video(video_path, cfg):
    decord.bridge.set_bridge('torch')

    vr = VideoReader(video_path)
    stride = vr.get_avg_fps() / cfg.data.val.fps
    fm_idx = [min(round(i), len(vr) - 1) for i in np.arange(0, len(vr), stride).tolist()]
    video = vr.get_batch(fm_idx).permute(0, 3, 1, 2).float() / 255

    size = 336 if '336px' in cfg.model.arch else 224
    h, w = video.size(-2), video.size(-1)
    s = min(h, w)
    x, y = round((h - s) / 2), round((w - s) / 2)
    video = video[..., x:x + s, y:y + s]
    video = F.resize(video, size=(size, size))
    video = F.normalize(video, (0.481, 0.459, 0.408), (0.269, 0.261, 0.276))
    video = video.reshape(video.size(0), -1).unsqueeze(0)

    return video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('query')
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--checkpoint', default=WEIGHT)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)
    cfg.model.init = True

    if args.checkpoint.startswith('http'):
        out_path = args.checkpoint.split('/')[-1]
        if not nncore.is_file(out_path):
            print(f'Downloading checkpoint from {args.checkpoint}...')
            nncore.download(args.checkpoint, progress=True)
        args.checkpoint = out_path

    print(f'Building model from {args.config}...')
    model = build_model(cfg.model, dist=False).eval()

    print(f'Loading checkpoint from {args.checkpoint}...')
    load_checkpoint(model, args.checkpoint)

    print(f'Loading video from {args.video}...')
    video = load_video(args.video, cfg)

    print(f'Query: {args.query}')
    query = clip.tokenize(args.query, truncate=True)

    device = next(model.parameters()).device
    data = dict(video=video.to(device), query=query.to(device), fps=[cfg.data.val.fps])

    with torch.inference_mode():
        pred = model(data)

    print('\nPrediction:')
    tab = [('Start time', 'End time', 'Score')]
    for b in pred['_out']['boundary'][:5].tolist():
        tab.append([round(n, 2) for n in b])
    print(tabulate(tab))


if __name__ == '__main__':
    main()
