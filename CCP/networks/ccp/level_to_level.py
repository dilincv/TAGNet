import torch.nn as nn
import functools
from libs import InPlaceABNSync
import torch.nn.functional as F
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class LevelToLevel(nn.Module):
    def __init__(self, in_channels):
        super(LevelToLevel, self).__init__()
        self.weight_conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=(1, 1))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1))
    def forward(self, high_level_feature, low_level_feature):
        weight = self.weight_conv1x1(low_level_feature)
        weighted_low = F.relu(self.conv(low_level_feature)) * weight
        out = weighted_low + high_level_feature
        return out, weighted_low

if __name__ == '__main__':
    import torch
    level_to_level = LevelToLevel(512).cuda()
    middle_feature = torch.randn([7203, 512, 1, 1], dtype=torch.float32).cuda()
    end_feature = torch.randn([7203, 512, 1, 1], dtype=torch.float32).cuda()
    out = level_to_level(middle_feature, end_feature)
    print(out.shape)
