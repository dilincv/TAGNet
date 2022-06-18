import torch
import torch.nn as nn
from utils import same_padding_conv
import torch.nn.functional as F

class BoxRegress(nn.Module):
    def __init__(self, anchors_size, in_channels, feature_size_with_block, level):
        super(BoxRegress, self).__init__()
        self.n_bbox = len(anchors_size)
        self.conv1x1 = same_padding_conv.Conv2d(in_channels, in_channels//2, kernel_size=[1, 1])
        self.feature_size_with_block = feature_size_with_block
        self.level = level
        self.fc = nn.Linear(in_channels//2, self.feature_size_with_block * self.feature_size_with_block * self.n_bbox * level * 2)
        self.fc.weight = torch.nn.init.normal_(self.fc.weight, std=1)

    def forward(self, feature):
        x = self.conv1x1(feature)
        x = F.avg_pool2d(x, x.shape[-2:])
        x = self.fc(x.squeeze(-1).squeeze(-1))
        x = x.reshape([x.shape[0], self.n_bbox * self.level, 2, self.feature_size_with_block, self.feature_size_with_block])
        x = x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        x = x.reshape([x.shape[0], self.n_bbox, self.level, 2, self.feature_size_with_block, self.feature_size_with_block]).permute([0,1,3,2,4,5]).contiguous()
        x = x.reshape([x.shape[0], self.n_bbox * 2, self.level, self.feature_size_with_block, self.feature_size_with_block])
        return x
