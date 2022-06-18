import torch.nn as nn
from torch.nn import functional as F
import torch
from networks.ccp.ccp import CCP
affine_par = True
import functools
from libs import InPlaceABNSync
from utils import same_padding_conv

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
        )
        self.relu = nn.ReLU(inplace=False)
    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change

        self.conv_inductionA = same_padding_conv.Conv2d(512, 512, kernel_size=3)
        self.conv_inductionB = same_padding_conv.Conv2d(512, 256, kernel_size=3)
        self.conv_inductionC = same_padding_conv.Conv2d(512, 512, kernel_size=3)
        self.conv_inductionD = same_padding_conv.Conv2d(512, 256, kernel_size=3)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv_ccp = same_padding_conv.Conv2d(512, 512, kernel_size=3)
        self.conv_res4 = same_padding_conv.Conv2d(1024, 512, kernel_size=3)
        self.conv_res3 = same_padding_conv.Conv2d(512, 512, kernel_size=3)
        self.conv_res2 = same_padding_conv.Conv2d(256, 256, kernel_size=3)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

        self.psp = PSPModule(2048, 256)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.ccp = CCP()
        self.ccp2 = CCP(input_size=(32, 32))


        self.head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.down_conv = same_padding_conv.Conv2d(256, 256, kernel_size=3)
        self.ccp2_conv = same_padding_conv.Conv2d(256, 256, kernel_size=3)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def element_wise_multiplication(self, feature1, feature2):
        assert (feature1.shape[2] <= feature2.shape[2]) and (feature1.shape[3] <= feature2.shape[3]), \
            'feature1.shape should smaller than feature2.shape'
        if (feature1.shape[2] < feature2.shape[2]) and (feature1.shape[3] < feature2.shape[3]):
            feature1 = F.upsample(feature1, feature2.shape[-2:], mode='bilinear', align_corners=True)
        return feature1 * feature2

    def fuse(self, feature1, feature2, deconv=None):
        assert (feature1.shape[2] <= feature2.shape[2]) and (feature1.shape[3] <= feature2.shape[3]), \
            'feature1.shape should smaller than feature2.shape'
        if deconv is not None:
            feature1 = deconv(feature1)
        return feature1 + feature2

    def forward(self, x, vis=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x_pool = self.maxpool(x)
        res2 = self.layer1(x_pool)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        x_dsn = self.dsn(res4)
        res5 = self.layer4(res4)
        x_psp = self.psp(res5)
        x_ccp = self.ccp(x_psp, vis)
        x_ccp2 = self.ccp2(F.max_pool2d(F.relu(self.down_conv(x_psp)), [3, 3]), vis)
        x_ccp = x_ccp + F.upsample(input=x_ccp2, size=x_ccp.shape[2:], mode='bilinear', align_corners=True)

        ##Encoder_decoder Structure
        x_ccp = self.conv_ccp(x_ccp)
        res4 = self.conv_res4(res4)
        res3 = self.conv_res3(res3)
        res2 = self.conv_res2(res2)

        res4_xccp = self.element_wise_multiplication(x_ccp, res4)
        y4 = self.fuse(x_ccp, res4_xccp)
        res3_res4_xccp = self.element_wise_multiplication(self.conv_inductionA(res4_xccp), res3)
        y3 = self.fuse(self.conv_inductionC(y4), res3_res4_xccp)
        res2_res3_res4_xccp = self.element_wise_multiplication(self.conv_inductionB(res3_res4_xccp), res2)
        y2 = self.fuse(self.conv_inductionD(y3), res2_res3_res4_xccp, self.deconv)

        out = self.head(y2)

        return [out, x_dsn]


def Res_Deeplab(num_classes):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model

if __name__ == '__main__':
    model = Res_Deeplab(19)
    model = nn.DataParallel(model, device_ids=[4,5,6,7])
    model.cuda()
    x = torch.Tensor(8, 3, 769, 769).cuda()
    y = model(x).cuda()
