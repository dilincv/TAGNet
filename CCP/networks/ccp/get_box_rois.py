import numpy as np
import torch
import torch.nn as nn

class GetBoxRois(nn.Module):
    def __init__(self, block_size, anchors_size, alpha):
        super(GetBoxRois, self).__init__()
        self.anchors_size = np.asarray(anchors_size)
        self.anchors_w = torch.from_numpy(self.anchors_size[:, 0][:, np.newaxis, np.newaxis]).float()
        self.anchors_h = torch.from_numpy(self.anchors_size[:, 1][:, np.newaxis, np.newaxis]).float()
        self.n_anchors = len(anchors_size)
        self.block_size = block_size
        self. alpha = alpha

    @staticmethod
    def rois_add_index(rois):
        N, n_bbox, H, W, _ = rois.shape
        rois = rois.permute([1, 0, 2, 3, 4]).contiguous()
        rois_reshaped = rois.reshape([n_bbox, N * H * W, 4])
        idx_tensor = rois.new_tensor(torch.arange(start=0., end=N * H * W))
        idx_tensor = idx_tensor.reshape([1, N * W * H]) // (H * W)
        idx_tensor = torch.cat([idx_tensor] * n_bbox, dim=0).unsqueeze_(dim=-1)
        new_rois = torch.cat([idx_tensor, rois_reshaped], dim=-1)
        new_rois = new_rois.reshape([n_bbox, N, H, W, 5])
        new_rois = new_rois.permute([1, 0, 2, 3, 4]).contiguous()
        return new_rois

    def calculate_rois(self, regressed_result, ori_feature_shape, center_point):
        N, C, H, W = ori_feature_shape
        _, n_bbx2, H_, W_ = regressed_result.shape
        regressed_result = regressed_result.reshape([N, n_bbx2 // 2, 2, H_, W_])
        x_center, y_center = center_point

        tx = regressed_result[:, :, 0, :, :]
        ty = regressed_result[:, :, 1, :, :]

        anchors_w = tx.new_tensor(self.anchors_w)
        anchors_h = ty.new_tensor(self.anchors_h)
        x_center = tx.new_tensor(x_center)
        y_center = ty.new_tensor(y_center)

        x_center_regressed = x_center + W/self.alpha * torch.sin(tx * np.pi)
        y_center_regressed = y_center + H/self.alpha * torch.sin(ty * np.pi)

        x_center_regressed = torch.stack(
            [torch.clamp(x_center_regressed[:, i, :, :], anchors_w[i, 0, 0] / 2, W - anchors_w[i, 0, 0] / 2)
             for i in range(self.n_anchors)], dim=1)
        y_center_regressed = torch.stack(
            [torch.clamp(y_center_regressed[:, i, :, :], anchors_h[i, 0, 0] / 2, H - anchors_h[i, 0, 0] / 2)
             for i in range(self.n_anchors)], dim=1)

        x_top_left = x_center_regressed - anchors_w / 2
        y_top_left = y_center_regressed - anchors_h / 2
        x_bottom_right = x_center_regressed + anchors_w / 2
        y_bottom_right = y_center_regressed + anchors_h / 2

        out = torch.stack([x_top_left, y_top_left, x_bottom_right, y_bottom_right], dim=-1)

        return out, [x_center_regressed, y_center_regressed]

    def forward(self, regressed_result, ori_feature_shape, center_point):
        rois, regressed_center_point = self.calculate_rois(regressed_result, ori_feature_shape, center_point)
        rois = self.rois_add_index(rois)
        return rois, regressed_center_point
