from turtle import forward
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import resnet_backbone
from torchvision.models import resnet34


class imageFeatureExtractor(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(imageFeatureExtractor, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x_show = x.detach().data.cpu().numpy()
        # plt.subplot(121)
        # plt.imshow(np.mean(x_show[0], axis=0))
        x = self.CNN(x)
        # x_show = x.detach().data.cpu().numpy()
        # plt.subplot(122)
        # plt.imshow(np.mean(x_show[0], axis=0))
        # plt.show()
        x = x.view(x.shape[0], -1)
        output = self.fc_block(x)
        return output


class onehotFeatureExtractor(nn.Module):

    def __init__(self):
        super(onehotFeatureExtractor, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output


class actionGenerator(nn.Module):

    def __init__(self):
        super(actionGenerator, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output


class actionGeneratorAngle(nn.Module):

    def __init__(self):
        super(actionGeneratorAngle, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output


class fixedActionGeneratorAngle(nn.Module):

    def __init__(self):
        super(fixedActionGeneratorAngle, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output


class CIMNet(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(CIMNet, self).__init__()
        # self.rgbFeatureExtractor = imageFeatureExtractor(norm_layer)
        # self.rgbFeatureExtractor = resnet50()
        # self.rgbFeatureExtractor.fc = nn.Linear(2048, 512)
        self.rgbFeatureExtractor = resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.commandFeatureExtractor = onehotFeatureExtractor()
        self.fc_block = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )

    def forward(self, x1, x2):
        feature_rgb = self.rgbFeatureExtractor(x1)
        feature_command = self.commandFeatureExtractor(x2)
        feature_concat = torch.cat([feature_rgb, feature_command], dim=-1)
        output = self.fc_block(feature_concat)
        return output


class BranchedCIMNet(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(BranchedCIMNet, self).__init__()
        # self.rgbFeatureExtractor = imageFeatureExtractor(norm_layer)
        # self.rgbFeatureExtractor = resnet50()
        # self.rgbFeatureExtractor.fc = nn.Linear(2048, 512)
        self.rgbFeatureExtractor = resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.commandFeatureExtractor = onehotFeatureExtractor()
        self.actionGenerator_up = actionGenerator()
        self.actionGenerator_left = actionGenerator()
        self.actionGenerator_down = actionGenerator()
        self.actionGenerator_right = actionGenerator()
        self.actionGenerator_straight = actionGenerator()

    def forward(self, x1, x2):
        feature_rgb = self.rgbFeatureExtractor(x1)
        feature_command = self.commandFeatureExtractor(x2)
        feature_concat = torch.cat([feature_rgb, feature_command], dim=-1)
        batch_output = []
        for batch in range(x2.shape[0]):
            if x2[batch, 0].item() > 0.5:
                batch_output.append(self.actionGenerator_up(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 1].item() > 0.5:
                batch_output.append(self.actionGenerator_left(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 2].item() > 0.5:
                batch_output.append(self.actionGenerator_down(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 3].item() > 0.5:
                batch_output.append(self.actionGenerator_right(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 4].item() > 0.5:
                batch_output.append(self.actionGenerator_straight(feature_concat[batch].unsqueeze(0)))
            else:
                raise NotImplementedError()
        output = torch.cat(batch_output, dim=0)
        return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BasicBlockResNet(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, mid_planes=None, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockResNet, self).__init__()
        if not mid_planes:
            mid_planes = planes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, mid_planes, stride)
        self.bn1 = norm_layer(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.downsample =  nn.Sequential(
            conv1x1(inplanes, planes, stride),
            norm_layer(planes),
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class UpResNet(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BasicBlockResNet(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = BasicBlockResNet(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BranchedCIMNetWithDepth(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(BranchedCIMNetWithDepth, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.commandFeatureExtractor = onehotFeatureExtractor()
        self.actionGenerator_up = actionGenerator()
        self.actionGenerator_left = actionGenerator()
        self.actionGenerator_down = actionGenerator()
        self.actionGenerator_right = actionGenerator()
        self.actionGenerator_straight = actionGenerator()
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)

    def forward(self, x1, x2):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
        feature_command = self.commandFeatureExtractor(x2)
        feature_concat = torch.cat([feature_rgb, feature_command], dim=-1)
        batch_output = []
        for batch in range(x2.shape[0]):
            if x2[batch, 0].item() > 0.5:
                batch_output.append(self.actionGenerator_up(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 1].item() > 0.5:
                batch_output.append(self.actionGenerator_left(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 2].item() > 0.5:
                batch_output.append(self.actionGenerator_down(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 3].item() > 0.5:
                batch_output.append(self.actionGenerator_right(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 4].item() > 0.5:
                batch_output.append(self.actionGenerator_straight(feature_concat[batch].unsqueeze(0)))
            else:
                raise NotImplementedError()
        output = torch.cat(batch_output, dim=0)
        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.up(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)
        return output, output_depth


class BranchedCIMNetWithDepthAngle(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(BranchedCIMNetWithDepthAngle, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.commandFeatureExtractor = onehotFeatureExtractor()
        self.actionGenerator_up = actionGeneratorAngle()
        self.actionGenerator_left = actionGeneratorAngle()
        self.actionGenerator_down = actionGeneratorAngle()
        self.actionGenerator_right = actionGeneratorAngle()
        self.actionGenerator_straight = actionGeneratorAngle()
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)

    def forward(self, x1, x2):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
        feature_command = self.commandFeatureExtractor(x2)
        feature_concat = torch.cat([feature_rgb, feature_command], dim=-1)
        batch_output = []
        for batch in range(x2.shape[0]):
            if x2[batch, 0].item() > 0.5:
                batch_output.append(self.actionGenerator_up(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 1].item() > 0.5:
                batch_output.append(self.actionGenerator_left(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 2].item() > 0.5:
                batch_output.append(self.actionGenerator_down(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 3].item() > 0.5:
                batch_output.append(self.actionGenerator_right(feature_concat[batch].unsqueeze(0)))
            elif x2[batch, 4].item() > 0.5:
                batch_output.append(self.actionGenerator_straight(feature_concat[batch].unsqueeze(0)))
            else:
                raise NotImplementedError()
        output = torch.cat(batch_output, dim=0)
        output = F.tanh(output)
        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.up(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)
        return output, output_depth


class fixedBranchedCIMNetWithDepthAngle(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(fixedBranchedCIMNetWithDepthAngle, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        # self.commandFeatureExtractor = onehotFeatureExtractor()
        self.actionGenerator_up = fixedActionGeneratorAngle()
        self.actionGenerator_left = fixedActionGeneratorAngle()
        self.actionGenerator_down = fixedActionGeneratorAngle()
        self.actionGenerator_right = fixedActionGeneratorAngle()
        self.actionGenerator_straight = fixedActionGeneratorAngle()
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)

    def forward(self, x1, x2):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
        # feature_command = self.commandFeatureExtractor(x2)
        # feature_concat = torch.cat([feature_rgb, feature_command], dim=-1)
        batch_output = []
        for batch in range(x2.shape[0]):
            if x2[batch, 0].item() > 0.5:
                batch_output.append(self.actionGenerator_up(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 1].item() > 0.5:
                batch_output.append(self.actionGenerator_left(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 2].item() > 0.5:
                batch_output.append(self.actionGenerator_down(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 3].item() > 0.5:
                batch_output.append(self.actionGenerator_right(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 4].item() > 0.5:
                batch_output.append(self.actionGenerator_straight(feature_rgb[batch].unsqueeze(0)))
            else:
                raise NotImplementedError()
        output = torch.cat(batch_output, dim=0)
        output = F.tanh(output)
        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.up(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)
        return output, output_depth


class fixedBranchedCIMNetWithDepthAngleMultiFrame(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(fixedBranchedCIMNetWithDepthAngleMultiFrame, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True, input_channels=15)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        # self.commandFeatureExtractor = onehotFeatureExtractor()
        self.actionGenerator_up = fixedActionGeneratorAngle()
        self.actionGenerator_left = fixedActionGeneratorAngle()
        self.actionGenerator_down = fixedActionGeneratorAngle()
        self.actionGenerator_right = fixedActionGeneratorAngle()
        self.actionGenerator_straight = fixedActionGeneratorAngle()
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 5)

    def forward(self, x1, x2):
        feature_rgb, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x1)
        # feature_command = self.commandFeatureExtractor(x2)
        # feature_concat = torch.cat([feature_rgb, feature_command], dim=-1)
        batch_output = []
        for batch in range(x2.shape[0]):
            if x2[batch, 0].item() > 0.5:
                batch_output.append(self.actionGenerator_up(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 1].item() > 0.5:
                batch_output.append(self.actionGenerator_left(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 2].item() > 0.5:
                batch_output.append(self.actionGenerator_down(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 3].item() > 0.5:
                batch_output.append(self.actionGenerator_right(feature_rgb[batch].unsqueeze(0)))
            elif x2[batch, 4].item() > 0.5:
                batch_output.append(self.actionGenerator_straight(feature_rgb[batch].unsqueeze(0)))
            else:
                raise NotImplementedError()
        output = torch.cat(batch_output, dim=0)
        output = F.tanh(output)
        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.up(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)
        return output, output_depth


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator_our(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_our, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        # self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks1.weight_init(0, 0.02)
        # self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks2.weight_init(0, 0.02)
        # self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks3.weight_init(0, 0.02)
        # self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks4.weight_init(0, 0.02)
        # self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks5.weight_init(0, 0.02)
        # self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks6.weight_init(0, 0.02)
        # self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks7.weight_init(0, 0.02)
        # self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks8.weight_init(0, 0.02)
        # self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks9.weight_init(0, 0.02)

        self.deconv1_content = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)
        self.deconv3_content = nn.Conv2d(ngf, 27, 7, 1, 0)

        self.deconv1_attention = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        self.deconv3_attention = nn.Conv2d(ngf, 10, 1, 1, 0)
        
        self.tanh = torch.nn.Tanh()
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        # x = self.resnet_blocks1(x)
        # x = self.resnet_blocks2(x)
        # x = self.resnet_blocks3(x)
        # x = self.resnet_blocks4(x)
        # x = self.resnet_blocks5(x)
        # x = self.resnet_blocks6(x)
        # x = self.resnet_blocks7(x)
        # x = self.resnet_blocks8(x)
        # x = self.resnet_blocks9(x)
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        content = self.deconv3_content(x_content)
        image = self.tanh(content)
        image1 = image[:, 0:3, :, :]
        # print(image1.size()) # [1, 3, 256, 256]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]
        # image10 = image[:, 27:30, :, :]

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        # x_attention = F.pad(x_attention, (3, 3, 3, 3), 'reflect')
        # print(x_attention.size()) [1, 64, 256, 256]
        attention = self.deconv3_attention(x_attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]

        attention1 = attention1_.repeat(1, 3, 1, 1)
        # print(attention1.size())
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        # output10 = image10 * attention10
        output10 = input * attention10

        o=output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10

        return o, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, attention1,attention2,attention3, attention4, attention5, attention6, attention7, attention8,attention9,attention10, image1, image2,image3,image4,image5,image6,image7,image8,image9


# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class ResnetGenerator_with_depth(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ResnetGenerator_with_depth, self).__init__()
        self.rgbFeatureExtractor = resnet_backbone.resnet34(pretrained=True)
        self.rgbFeatureExtractor.fc = nn.Linear(512, 512)
        self.rgbDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.rgbDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.rgbDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.rgbDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.rgb_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.rgbDecoder_outc = OutConv(32, 3)
        self.depthDecoder_up1 = UpResNet(512, 256, bilinear=False)
        self.depthDecoder_up2 = UpResNet(256, 128, bilinear=False)
        self.depthDecoder_up3 = UpResNet(128, 64, bilinear=False)
        self.depthDecoder_up4 = UpResNet(128, 64, bilinear=True)
        self.depth_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depthDecoder_outc = OutConv(32, 1)

    def forward(self, x):
        _, f1, f2, f3, f4, f5 = self.rgbFeatureExtractor(x)
        output_rgb = self.rgbDecoder_up1(f5, f4)
        output_rgb = self.rgbDecoder_up2(output_rgb, f3)
        output_rgb = self.rgbDecoder_up3(output_rgb, f2)
        output_rgb = self.rgbDecoder_up4(output_rgb, f1)
        output_rgb = self.rgb_up(output_rgb)
        output_rgb = self.rgbDecoder_outc(output_rgb)
        output_depth = self.depthDecoder_up1(f5, f4)
        output_depth = self.depthDecoder_up2(output_depth, f3)
        output_depth = self.depthDecoder_up3(output_depth, f2)
        output_depth = self.depthDecoder_up4(output_depth, f1)
        output_depth = self.depth_up(output_depth)
        output_depth = self.depthDecoder_outc(output_depth)
        return output_rgb, output_depth
