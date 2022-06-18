import torch.nn as nn
import torch.nn.functional as F

import functools
from libs import InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

import torch

class PointToAnchor(nn.Module):
    def __init__(self, n_bbox, in_channels, block_size):
        super(PointToAnchor, self).__init__()
        self.block_size = block_size
        self.point_weight_conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=block_size*block_size, kernel_size=[1, 1], stride=1)
        self.n_bbox = n_bbox
        self.conv_weight = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))
        self.conv_fuse = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))

    def forward(self, feature, rois_feature, feature_size_with_block):
        feature_for_weight = F.relu(self.conv_weight(rois_feature))
        feature_for_fuse = F.relu(self.conv_fuse(rois_feature))
        point_weight = self.point_weight_conv1x1(feature_for_weight)
        N, C, H, W = feature.shape

        # pad = (
        #     0, self.block_size - W % self.block_size,
        #     0, self.block_size - H % self.block_size
        # )
        # feature = F.pad(feature, pad)

        feature = feature.reshape(
            [feature.shape[0], feature.shape[1], feature.shape[2] // self.block_size, self.block_size,
             feature.shape[3] // self.block_size, self.block_size])
        feature = feature.permute([0, 1, 2, 4, 3, 5]).contiguous()

        feature = feature.repeat([self.n_bbox, 1, 1, 1, 1, 1, 1]).permute([1, 0, 2, 3, 4, 5, 6]).contiguous()
        point_weight = point_weight.squeeze(-1).squeeze(-1)
        point_weight = point_weight.reshape([self.n_bbox, N, feature_size_with_block, feature_size_with_block, self.block_size, self.block_size]).permute(
            [1, 0, 2, 3, 4, 5]).contiguous()
        point_weight = point_weight.unsqueeze(dim=2)
        out = torch.sum(feature * point_weight, dim=[-1, -2])
        out = out.permute([1, 0, 3, 4, 2]).contiguous()
        out = out.reshape([out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3], out.shape[4]])
        out = out.unsqueeze(dim=-1).unsqueeze(dim=-1)
        fused = feature_for_fuse + out
        return fused, out


if __name__ == '__main__':
    point_to_anchor = PointToAnchor(n_bbox=3, in_channels=512, block_size=2).cuda()
    feature = torch.randn([1, 512, 97, 97], dtype=torch.float32).cuda()
    rois_feature = torch.randn([7203, 512, 1, 1], dtype=torch.float32).cuda()
    out = point_to_anchor(feature, rois_feature, 49)
    print(out.shape)
