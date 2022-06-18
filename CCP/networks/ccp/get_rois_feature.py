import torch.nn as nn
from prroi_pool import PrRoIPool2D

class GetRoisFeature(nn.Module):
    def __init__(self, pool_shape):
        super(GetRoisFeature, self).__init__()
        pool_height, pool_width = pool_shape
        self.pr_roi_pool2d = PrRoIPool2D(pool_height, pool_width, 1)
        self.avg_pool2d = nn.AvgPool2d(pool_shape)

    def forward(self, feature, rois):
        rois = rois.permute([1, 0, 2, 3, 4]).contiguous()
        all_rois = rois.reshape([-1, 5])
        pooling_out = self.pr_roi_pool2d(feature, all_rois)
        pooling_out = self.avg_pool2d(pooling_out)
        return pooling_out