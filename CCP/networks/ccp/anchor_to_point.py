import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from libs import InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class AnchorToPoint(nn.Module):
    def __init__(self, n_bbox, in_channels, block_size):
        super(AnchorToPoint, self).__init__()
        self.box_weight_conv1x1 = nn.Conv2d(in_channels, n_bbox, 1, 1, 0)
        self.box_weight_conv = nn.Conv2d(in_channels * n_bbox,
                                         block_size * block_size * n_bbox, 1, 1, 0)
        self.block_size = block_size
        self.n_bbox = n_bbox
        self.conv_weight = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))
        self.conv_fuse = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))

    def forward(self, feature, rois_feature, feature_size_with_block):
        N, C, H, W = feature.shape

        feature_for_weight = F.relu(self.conv_weight(rois_feature))
        feature_for_fuse = F.relu(self.conv_fuse(rois_feature))

        x = feature_for_weight.reshape(self.n_bbox, N, feature_size_with_block,
                                 feature_size_with_block, C, 1, 1)\
            .permute(1, 2, 3, 0, 4, 5, 6)
        x = x.reshape([N * feature_size_with_block * feature_size_with_block
                          ,self.n_bbox * C, 1, 1])
        box_weight = self.box_weight_conv(x)
        box_weight = box_weight.reshape(N, 1, feature_size_with_block, feature_size_with_block,
                                        self.n_bbox, self.block_size, self.block_size)
        box_weight = box_weight.permute(0, 4, 1, 2, 5, 3, 6)
        H_weight, W_weight = feature.shape[-2:]

        box_weight = box_weight.permute([0, 1, 2, 3, 5, 4, 6]).contiguous()

        feature_for_fuse = feature_for_fuse.reshape([self.n_bbox, N, feature_size_with_block, feature_size_with_block, C]).permute([1, 0, 4, 2, 3]).contiguous()
        outs = box_weight * feature_for_fuse.unsqueeze(5).unsqueeze(6)
        weighted_box = torch.sum(outs, dim=[-1, -2])
        weighted_box = weighted_box.permute([1, 0, 3, 4, 2]).contiguous()
        weighted_box = weighted_box.reshape([weighted_box.shape[0] * weighted_box.shape[1] * weighted_box.shape[2] * weighted_box.shape[3], weighted_box.shape[4]])
        weighted_box = weighted_box.unsqueeze(dim=-1).unsqueeze(dim=-1)

        outs = outs.sum(dim=1)
        outs = outs.permute([0, 1, 2, 4, 3, 5]).reshape(
            [N, C, feature_size_with_block * self.block_size, feature_size_with_block * self.block_size])
        outs = outs[:, :, :H_weight, :W_weight] + feature

        return outs, weighted_box

if __name__ == '__main__':
    import torch
    anchor_to_point = AnchorToPoint(n_bbox=4, in_channels=512, block_size=3).cuda(4)
    feature = torch.randn([2, 512, 97, 97], dtype=torch.float32).cuda(4)
    rois_feature = torch.randn([8712, 512, 1, 1], dtype=torch.float32).cuda(4)
    out = anchor_to_point(feature, rois_feature, 33)
    print(out.shape)
