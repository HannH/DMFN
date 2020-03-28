import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import gauss_kernel
import torchvision.models as models
import numpy as np


class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_BN, self).__init__()
        self.model = nn.Sequential([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels)
            ])

    def forward(self, *input):
        return self.model(*input)


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scale=2):
        super(upsampling, self).__init__()
        assert isinstance(scale, int)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.scale = scale

    def forward(self, x):
        h, w = x.size(2) * self.scale, x.size(3) * self.scale
        xout = self.conv(F.interpolate(input=x, size=(h, w), mode='nearest', align_corners=True))
        return xout


class PureUpsampling(nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super(PureUpsampling, self).__init__()
        assert isinstance(scale, int)
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        h, w = x.size(2) * self.scale, x.size(3) * self.scale
        if self.mode == 'nearest':
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode)
        else:
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode, align_corners=True)
        return xout


class GaussianBlurLayer(nn.Module):
    def __init__(self, size, sigma, in_channels=1, stride=1, pad=1):
        super(GaussianBlurLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.ch = in_channels
        self.stride = stride
        self.pad = nn.ReflectionPad2d(pad)

    def forward(self, x):
        kernel = gauss_kernel(self.size, self.sigma, self.ch, self.ch)
        kernel_tensor = torch.from_numpy(kernel)
        kernel_tensor = kernel_tensor.cuda()
        x = self.pad(x)
        blurred = F.conv2d(x, kernel_tensor, stride=self.stride)
        return blurred


class ConfidenceDrivenMaskLayer(nn.Module):
    def __init__(self, size=65, sigma=1.0 / 40, iters=7):
        super(ConfidenceDrivenMaskLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.iters = iters
        self.propagationLayer = GaussianBlurLayer(size, sigma, pad=32)

    def forward(self, mask):
        # here mask 1 indicates missing pixels and 0 indicates the valid pixels
        init = 1 - mask
        mask_confidence = None
        for i in range(self.iters):
            mask_confidence = self.propagationLayer(init)
            mask_confidence = mask_confidence * mask
            init = mask_confidence + (1 - mask)
        return mask_confidence


class VGG19(nn.Module):
    def __init__(self, pool='max'):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return out


class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        vgg = models.vgg19(pretrained=False)
        state_dict = torch.load(r'F:\python\DeepLearning\hhub\DFBM\inpainting_gmcnn\pytorch\checkpoints\vgg19-dcbb9e9d.pth')
        vgg.load_state_dict(state_dict)
        self.vgg19 = vgg.features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            x = layer(x)
            if isinstance(layer, nn.ReLU) and ri == 1:
                out[name] = x
                if ci == 5:
                    break
        # print([x for x in out])
        return out


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
