# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import argparse
import math
import multiprocessing as mp
from functools import partial

import ffmpeg
import nncore
import numpy as np


def _proc(video_path, frame_dir, size, fps, max_len, anno=None):
    try:
        vid = nncore.pure_name(video_path)

        if anno is not None:
            duration = int(anno[vid]['frames']) / anno[vid]['fps']
            num_anno = len(anno[vid]['match'])
            fps = num_anno / duration

        probe = ffmpeg.probe(video_path)

        info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        w, h = int(info['width']), int(info['height'])

        if w > h:
            w, h = int(w * size / h), size
        else:
            w, h = size, int(h * size / w)

        x, y = int((w - size) / 2), int((h - size) / 2)

        stream = ffmpeg.input(video_path)
        stream = ffmpeg.filter(stream, 'fps', fps)
        stream = ffmpeg.filter(stream, 'scale', w, h)
        stream = ffmpeg.crop(stream, x, y, size, size)
        stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
        out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

        fm = np.frombuffer(out, np.uint8).reshape(-1, size, size, 3)
        fm = fm.transpose(0, 3, 1, 2)
    except Exception as e:
        print(f'Failed to process {video_path}: {e.__class__.__name__}: {e}')
        return

    if max_len > 0:
        for i in range(math.ceil(fm.shape[0] / max_len)):
            seg = fm[i * max_len:(i + 1) * max_len]
            out = nncore.join(frame_dir, f'{vid}_{i}.npy')
            nncore.dump(seg, out)
    else:
        out = nncore.join(frame_dir, f'{vid}.npy')
        nncore.dump(fm, out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=str)
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--frame_dir', type=str)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--fps', type=float, default=0.5)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--chunksize', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    video_paths = nncore.ls(args.video_dir, ext=['mp4', 'mkv', 'avi'], join_path=True)

    if args.anno_path is not None:
        print(f'Loading annotations from {args.anno_path}')
        anno = nncore.load(args.anno_path)
        video_paths = [p for p in video_paths if nncore.pure_name(p) in anno]
    else:
        anno = None

    if args.frame_dir is None:
        name = f'frames_{args.size}_' + (f'{args.fps}fps' if anno is None else 'auto')
        args.frame_dir = nncore.join(nncore.dir_name(args.video_dir), name)

    print(f'Processing {args.video_dir} to {args.frame_dir}')

    proc = partial(
        _proc,
        frame_dir=args.frame_dir,
        size=args.size,
        fps=args.fps,
        max_len=args.max_len,
        anno=anno)

    prog_bar = nncore.ProgressBar(num_tasks=len(video_paths))
    with mp.Pool(args.workers) as pool:
        for _ in pool.imap_unordered(proc, video_paths, chunksize=args.chunksize):
            prog_bar.update()


if __name__ == '__main__':
    main()
