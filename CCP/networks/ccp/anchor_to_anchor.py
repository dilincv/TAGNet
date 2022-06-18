import torch
import torch.nn as nn
from sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.functional as F
import math
from networks.ccp.box_regress import BoxRegress
from networks.ccp.get_rois_feature import GetRoisFeature
from networks.ccp.get_box_rois import GetBoxRois

BatchNorm2d = SynchronizedBatchNorm2d

class AnchorToAnchor(nn.Module):
    def __init__(self, anchors_size, in_channels, block_size, input_size, pool_shape, alpha):
        super(AnchorToAnchor, self).__init__()
        self.feature_size_with_block = math.ceil(input_size[0] / block_size)
        r = torch.arange(0.5 * (block_size - 1),
                         0.5 * (block_size - 1) + block_size * self.feature_size_with_block,
                         step=block_size)
        x_center = r.repeat(1, self.feature_size_with_block, 1)
        y_center = r.repeat(1, self.feature_size_with_block, 1).transpose(1, 2)
        self.blocks_center = [x_center, y_center]
        self.box_regress = BoxRegress(anchors_size, in_channels, self.feature_size_with_block, 1)
        self.get_rois = GetBoxRois(block_size, anchors_size, alpha)
        self.get_rois_feature = GetRoisFeature(pool_shape)

        self.block_num_total = self.feature_size_with_block * self.feature_size_with_block
        self.box_weight_a2a = nn.Conv2d(in_channels=in_channels, out_channels=self.block_num_total, kernel_size=(1, 1))
        self.n_bbox = len(anchors_size)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))

    def rois_to_rois(self, N, rois_feature_1, rois_feature_2):
        rois_feature_1 = rois_feature_1.reshape(
            [self.n_bbox, N, rois_feature_1.shape[0] // (self.n_bbox * N), rois_feature_1.shape[1],
             rois_feature_1.shape[2], rois_feature_1.shape[3]]
        ).permute([1, 0, 3, 2, 4, 5]).contiguous().squeeze(-1)
        rois_feature_1 = rois_feature_1.reshape(
            [rois_feature_1.shape[0] * rois_feature_1.shape[1] * rois_feature_1.shape[2],
             rois_feature_1.shape[3], rois_feature_1.shape[4]])

        rois_feature_2 = rois_feature_2.reshape(
            [self.n_bbox, N, rois_feature_2.shape[0] // (self.n_bbox * N), rois_feature_2.shape[1],
             rois_feature_2.shape[2], rois_feature_2.shape[3]]
        ).permute([1, 0, 3, 4, 2, 5]).contiguous().squeeze(-1)
        rois_feature_2 = rois_feature_2.reshape(
            [rois_feature_2.shape[0] * rois_feature_2.shape[1] * rois_feature_2.shape[2],
             rois_feature_2.shape[3], rois_feature_2.shape[4]])

        box_relation = torch.matmul(rois_feature_1, rois_feature_2)
        box_relation = F.softmax(box_relation, dim=1)

        box_relation = box_relation.reshape(
            [N * self.n_bbox, box_relation.shape[0] // (self.n_bbox * N), box_relation.shape[1], box_relation.shape[2]]
        )
        rois_feature_1 = rois_feature_1.reshape(
            [N * self.n_bbox, rois_feature_1.shape[0] // (self.n_bbox * N),
             rois_feature_1.shape[1], rois_feature_1.shape[2]]
        )
        rois_feature_2 = rois_feature_2.reshape(
            [N * self.n_bbox, rois_feature_2.shape[0] // (self.n_bbox * N),
             rois_feature_2.shape[1], rois_feature_2.shape[2]]
        ).permute([0, 1, 3, 2]).contiguous()

        out = torch.sum(box_relation * rois_feature_1, dim=2, keepdim=True)
        out = out.permute([0, 1, 3, 2]).contiguous()
        out = rois_feature_2 + out

        out = out.reshape(
            [N, out.shape[0] // N, out.shape[1], out.shape[2], out.shape[3]]
        ).permute([1, 0, 3, 2, 4]).contiguous().unsqueeze(dim=-1)
        out = out.reshape(
            [out.shape[0] * out.shape[1] * out.shape[2], out.shape[3], out.shape[4], out.shape[5]]
        )
        return out

    def forward(self, ori_feature_shape, rois_feature_a, feature_b, rois_feature_c):
        N, _, _, _ = ori_feature_shape

        regress_result = self.box_regress(feature_b)
        rois, regressed_center_point = self.get_rois(regress_result[:, :, 0, :, :], feature_b.shape, self.blocks_center)
        rois_feature_b = self.get_rois_feature(feature_b, rois)

        out = self.rois_to_rois(N, rois_feature_a, rois_feature_b)
        out = self.rois_to_rois(N, out, rois_feature_c)

        return out

if __name__ == '__main__':
    anchor_to_anchor = AnchorToAnchor(3, 512, 49).cuda(4)
    feature = torch.randn([1, 512, 97, 97], dtype=torch.float32).cuda(4)
    rois_feature = torch.randn([49*49*3*1, 512, 1, 1], dtype=torch.float32).cuda(4)
    out = anchor_to_anchor(feature.shape, rois_feature)
    print(out.shape)
