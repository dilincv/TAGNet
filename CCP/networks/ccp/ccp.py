import torch
import torch.nn as nn

from networks.ccp.roi_upsample import RoiUpsample
from prroi_pool import PrRoIPool2D
from sync_batchnorm import SynchronizedBatchNorm2d
import math
from networks.ccp.anchor_to_point import AnchorToPoint
from networks.ccp.point_to_anchor import PointToAnchor
from networks.ccp.level_to_level import LevelToLevel
from networks.ccp.anchor_to_anchor import AnchorToAnchor
from networks.ccp.box_regress import BoxRegress
from networks.ccp.get_box_rois import GetBoxRois
from networks.ccp.get_rois_feature import GetRoisFeature
from utils.utils import print_settings, get_settings
from utils import same_padding_conv
import torch.nn.functional as F

BatchNorm2d = SynchronizedBatchNorm2d

BLOCKS_SIZE_A = 1
BLOCKS_SIZE_B = 24
ANCHORS_SIZE = [[3, 3]]
IN_CHANNELS = 256
POOL_SHAPE = (7, 7)
INPUT_SIZE = (97, 97)
ALPHA = 4
LEVEL = 2
FORWARD_PATH = True
FC_PATH = True


class CCP(nn.Module):
    def __init__(self, block_size_a=BLOCKS_SIZE_A, block_size_b=BLOCKS_SIZE_B, anchors_size=ANCHORS_SIZE, in_channels=IN_CHANNELS, pool_shape=POOL_SHAPE,
                 input_size=INPUT_SIZE, alpha=ALPHA, level=LEVEL, forward_path=FORWARD_PATH, fc_path=FC_PATH):
        super(CCP, self).__init__()
        self.convB = same_padding_conv.Conv2d(IN_CHANNELS, IN_CHANNELS, kernel_size=3)
        self.convC = same_padding_conv.Conv2d(IN_CHANNELS, IN_CHANNELS, kernel_size=3)

        self.n_bbox = len(anchors_size)
        self.block_size_a = block_size_a
        self.block_size_b = block_size_b
        self.anchors_size = anchors_size
        self.pool_shape = pool_shape
        self.pool_height, self.pool_width = pool_shape
        self.input_size = input_size
        self.feature_size_with_block = math.ceil(input_size[0] / block_size_a)
        assert (level >= 1), 'level should >= 1'
        self.level = level
        self.box_regress = BoxRegress(self.anchors_size, in_channels, self.feature_size_with_block, level)
        self.get_rois = GetBoxRois(block_size_a, self.anchors_size, alpha)
        self.get_rois_feature = GetRoisFeature(pool_shape)
        self.pr_roi_pool2d = PrRoIPool2D(self.pool_height, self.pool_width, 1)
        self.avg_pool2d = nn.AvgPool2d(pool_shape)
        self.box_weight_conv1x1 = nn.Conv2d(in_channels, self.n_bbox, 1, 1, 0)

        r = torch.arange(0.5 * (block_size_a - 1), 0.5 * (block_size_a - 1) + block_size_a * self.feature_size_with_block,
                         step=block_size_a)
        x_center = r.repeat(self.n_bbox, self.feature_size_with_block, 1)
        y_center = r.repeat(self.n_bbox, self.feature_size_with_block, 1).transpose(1, 2)
        self.blocks_center = [x_center, y_center]

        self.point_to_anchor = PointToAnchor(self.n_bbox, in_channels, block_size_a)
        for i in range(self.level):
            exec('self.high_to_low_' + str(i) + '=LevelToLevel(in_channels)')
            exec('self.low_to_high_' + str(i) + '=LevelToLevel(in_channels)')

        self.anchor_to_anchor = AnchorToAnchor(anchors_size, in_channels, self.block_size_b, input_size, pool_shape, alpha)
        self.anchor_to_point = AnchorToPoint(self.n_bbox, in_channels, block_size_a)

        self.forward_path = forward_path
        self.fc_path = fc_path

        self.roiUsp = RoiUpsample()

        self.variables_for_vis = {'rois': [], 'rois_center': []}
        print_settings(get_settings(self.__class__.__init__), 'ccp')

    def forward(self, feature_a, vis):
        assert self.input_size == tuple(feature_a.shape[-2:]), 'input_size unmatched!!!'
        feature_b = F.relu(self.convB(feature_a))
        feature_c = F.relu(self.convC(feature_a))

        regress_result = self.box_regress(feature_a)

        all_rois = []
        all_rois_center = []
        self.variables_for_vis = {'rois': [], 'rois_center': []}
        center_point = self.blocks_center
        for i in range(self.level):
            rois, regressed_center_point = self.get_rois(regress_result[:, :, i, :, :], feature_a.shape, center_point)
            all_rois.append(rois)
            all_rois_center.append(regressed_center_point)
            center_point = regressed_center_point
            if vis:
                self.variables_for_vis['rois'].append(rois.cpu())
                self.variables_for_vis['rois_center'].append([regressed_center_point[0].cpu(), regressed_center_point[1].cpu()])

        rois_feature = self.get_rois_feature(feature_a, all_rois[-1])
        rois_feature_usps = []
        if self.forward_path:
            rois_feature, rois_feature_usp = self.point_to_anchor(
                feature_a, self.get_rois_feature(feature_a, all_rois[0]), self.feature_size_with_block
            )
            rois_feature_usps.append(rois_feature_usp)
            for i in range(self.level - 1):
                low_level_rois_feature = rois_feature
                high_level_rois_feature = self.get_rois_feature(feature_a, all_rois[i+1])
                rois_feature, rois_feature_usp = eval('self.low_to_high_' + str(i))(low_level_rois_feature, high_level_rois_feature)
                rois_feature_usps.append(rois_feature_usp)
            feature_b = feature_a + self.roiUsp(feature_a.shape[1:], all_rois_center.copy(), rois_feature_usps)

        if self.fc_path:
            rois_feature = self.anchor_to_anchor(
                feature_a.shape, rois_feature, feature_b, self.get_rois_feature(feature_c, all_rois[-1])
            )

        rois_feature_usps = []
        for i in range(self.level - 1):
            high_level_roi_feature = rois_feature
            low_level_roi_feature = self.get_rois_feature(feature_c, all_rois[-(i+2)])
            rois_feature, rois_feature_usp = eval('self.high_to_low_' + str(i))(high_level_roi_feature, low_level_roi_feature)
            rois_feature_usps.append(rois_feature_usp)

        out, rois_feature_usp = self.anchor_to_point(feature_c, rois_feature, self.feature_size_with_block)
        rois_feature_usps.append(rois_feature_usp)

        out = out + self.roiUsp(out.shape[1:], all_rois_center, rois_feature_usps)

        out = torch.cat([feature_a, out], dim=1)

        return out


if __name__ == '__main__':
    ccp = CCP().cuda()
    x = torch.rand(2, IN_CHANNELS, INPUT_SIZE[0], INPUT_SIZE[1]).cuda()
    x_ccp = ccp(x, False)
    print("ok")

