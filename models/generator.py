# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import torch
import torch.nn as nn


class BufferList(nn.Module):

    def __init__(self, buffers):
        super(BufferList, self).__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PointGenerator(nn.Module):

    def __init__(self, buffer_size, strides, offset=False):
        super(PointGenerator, self).__init__()

        reg_range, last = [], 0
        for stride in strides[1:]:
            reg_range.append((last, stride))
            last = stride
        reg_range.append((last, float('inf')))

        self.buffer_size = buffer_size
        self.levels = len(strides)
        self.strides = strides
        self.reg_range = reg_range
        self.offset = offset

        self.buffer_points = self._cache_points()

    def _cache_points(self):
        points_list = []
        for l, stride in enumerate(self.strides):
            reg_range = torch.as_tensor(self.reg_range[l], dtype=torch.float)
            lv_stride = torch.as_tensor(stride, dtype=torch.float)
            points = torch.arange(0, self.buffer_size, stride)[:, None]
            if self.offset:
                points += 0.5 * stride
            reg_range = reg_range[None].repeat(points.size(0), 1)
            lv_stride = lv_stride[None].repeat(points.size(0), 1)
            points_list.append(torch.cat((points, reg_range, lv_stride), dim=1))
        return BufferList(points_list)

    def forward(self, feats):
        pts_list = []
        feat_lens = [feat.size(1) for feat in feats]
        feat_lens += [0] * (len(self.buffer_points) - len(feat_lens))
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            if feat_len == 0:
                continue
            assert feat_len <= buffer_pts.size(0), 'reached max buffer length'
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return torch.cat(pts_list)
