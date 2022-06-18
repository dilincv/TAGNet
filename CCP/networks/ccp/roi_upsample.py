import numpy as np
import torch
import torch.nn as nn

class RoiUpsample(nn.Module):
    def __init__(self):
        super(RoiUpsample, self).__init__()

    def roiUpsample(self, feature_shape, rois_center, rois_feature):
        rois_center_floor = rois_center.long()
        rois_center_ceil = rois_center_floor + 1

        pixels = torch.cat([rois_center_floor, rois_center_ceil], dim=1).reshape([-1, 2])
        pixels_x = pixels[:, 0]
        pixels_x = pixels_x.repeat([2, 1]).permute([1, 0]).reshape([1, -1])
        pixels_y = pixels[:, 1]
        pixels_y = pixels_y.reshape([-1, 2]).repeat([1, 2]).reshape([1, -1])
        pixels = torch.cat([pixels_x, pixels_y]).permute([1, 0])
        mask = (pixels[:, 0] < feature_shape[1]) * (pixels[:, 1] < feature_shape[1])
        mask = mask.unsqueeze(dim=0).permute([1, 0])
        pixels = torch.masked_select(pixels, mask)
        pixels = pixels.reshape([-1, 2])

        p = rois_center - rois_center_floor.float()
        x_p = p[:, 0]
        y_p = p[:, 1]
        value_x = x_p ** 2 + (1 - x_p) ** 2
        value_y = y_p ** 2 + (1 - y_p) ** 2
        R_p_g = 0.25 * rois_feature.permute([1, 0])
        R_p_11 = (((1 - x_p) * (1 - y_p)) / (value_x * value_y)) * R_p_g
        R_p_12 = (((1 - x_p) * y_p) / (value_x * value_y)) * R_p_g
        R_p_21 = ((x_p * (1 - y_p)) / (value_x * value_y)) * R_p_g
        R_p_22 = ((x_p * y_p) / (value_x * value_y)) * R_p_g
        pixels_feature = torch.cat([R_p_11, R_p_12, R_p_21, R_p_22]).permute([1, 0]).reshape([-1, feature_shape[0]])
        pixels_feature = torch.masked_select(pixels_feature, mask)
        pixels_feature = pixels_feature.reshape([-1, feature_shape[0]]).cpu()

        empty_feature = torch.zeros([feature_shape[1] * feature_shape[2], feature_shape[0]])
        pixels_x = pixels[:, 0]
        pixels_y = pixels[:, 1]
        indices = (pixels_x * feature_shape[2] + pixels_y).long().cpu()
        empty_feature.index_add_(0, indices, pixels_feature)
        empty_feature = empty_feature.reshape([feature_shape[1], feature_shape[2], feature_shape[0]]).permute([2, 0, 1])

        return empty_feature

    def forward(self, feature_shape, all_rois_center, rois_feature_usps):
        for i in range(len(all_rois_center)):
            all_rois_center[i] = torch.stack(all_rois_center[i])
        all_rois_center = torch.stack(all_rois_center)
        N = all_rois_center.shape[2]
        all_rois_center = all_rois_center.permute([2, 0, 3, 4, 5, 1]).reshape([N, -1, 2])
        rois_feature_usps = torch.stack(rois_feature_usps).squeeze()
        levels = rois_feature_usps.shape[0]
        channels = rois_feature_usps.shape[2]
        rois_feature_usps = rois_feature_usps.reshape([levels, N, -1, channels]).permute([1, 0, 2, 3]).reshape([N, -1, channels])
        fm_batch = []
        for i in range(N):
            fm = self.roiUpsample(feature_shape, all_rois_center[i], rois_feature_usps[i])
            fm_batch.append(fm)
        fm_batch = torch.stack(fm_batch)
        return fm_batch.cuda()

if __name__ == '__main__':
    feature_map = torch.zeros(2, 3, 3)
    rois_center = torch.Tensor([[0.1, 1.1], [0.1, 2.1], [1.1, 0.1], [1.1, 1.1], [1.1, 2.1], [2.1, 1.1]])
    # rois_center = torch.Tensor([(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)])
    rois_feature = torch.Tensor([[3.1, 1.1, 2.2, 3.3], [6.4, 2.5, 1.6, 5.8], [4.1, 8.2, 1.3, 7.4], [5.6, 2.1, 8.6, 3.8], [1.2, 5.4, 8.8, 2.3], [7.8, 6.3, 8.5, 8.7]])
    empty_feature = RoiUpsample().roiUpsample([4, 3, 3], rois_center, rois_feature)
    print(empty_feature)
