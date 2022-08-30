from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import time
import numpy as np

Align_Corners = False


BatchNorm3d = nn.BatchNorm3d
BatchNorm2d = nn.BatchNorm2d



class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
#        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if bn:
            bs = False
        else:
            bs = True
        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=bs, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=bs, **kwargs)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, bn=bn, relu=relu, kernel_size=kernel, stride=2,
                               padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, bn=bn, relu=relu, kernel_size=3, stride=1,
                                   padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, bn=bn, relu=relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

class Correlation(nn.Module):
    def __init__(self, maxdisp):
        super(Correlation, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x, y):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, self.maxdisp, height, width).zero_()
            #            cost = Variable(torch.FloatTensor(x.size()[0], x.size()[1]*2, self.maxdisp,  x.size()[2],  x.size()[3]).zero_(), volatile= not self.training).cuda()
            for i in range(self.maxdisp):
                if i > 0:
                    cost[:, i, :, i:] = (x[:, :, :, i:] *
                                         y[:, :, :, :-i]).mean(dim=1)
                else:
                    cost[:, i, :, :] = (x * y).mean(dim=1)

            cost = cost.contiguous()
        return cost

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.corr2 = nn.Sequential(BasicConv(32, 64, kernel_size=3, stride=1, padding=1),
                                  BasicConv(64, 64, kernel_size=3, stride=1, padding=1))
        self.corr3 = nn.Sequential(BasicConv(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
                                  BasicConv(64, 64, kernel_size=3, stride=1, padding=2, dilation=2))

        self.cat = nn.Sequential(BasicConv(160, 64, kernel_size=3, stride=1, padding=1),
                                 nn.Conv2d(64, 6, kernel_size=1, padding=0, stride=1,
                                           bias=False))

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x
        out1 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)
        Corr2 = self.corr2(x)
        Corr3 = self.corr3(Corr2)

        #output_msfeat = {}
        output_feature = torch.cat((x, Corr2, Corr3), dim=1)
        out = output_feature
        out_cat = self.cat(output_feature)
        output_msfeat = {"gwc_feature": out, "concat_feature": out_cat}

        return out1, output_msfeat

class Unet2D(nn.Module):    
    def __init__(self):
        super(Unet2D, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(64, 64, kernel_size=3, padding=1),
            BasicConv(64, 64, kernel_size=3, padding=1))
        
        self.conv1a = nn.Sequential(
            BasicConv(64, 64, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1))

        self.conv2a = nn.Sequential(
            BasicConv(64, 64, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1))

        self.conv3a = nn.Sequential(
            BasicConv(64, 64, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1))
        
        self.conv4a = BasicConv(64, 64, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(64, 64, deconv=True)
        self.deconv4b = BasicConv(64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv3a = Conv2x(64, 64, deconv=True)
        self.deconv3b = BasicConv(64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv2a = Conv2x(64, 64, deconv=True)
        self.deconv2b = BasicConv(64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv1a = Conv2x(64, 64, deconv=True)
        self.deconv1b = BasicConv(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x    #64

        x = self.conv1a(x)
        rem1 = x    #80

        x = self.conv2a(x)
        rem2 = x    #96

        x = self.conv3a(x)
        rem3 = x

        x = self.conv4a(x)

        x = self.deconv4a(x, rem3)
        x = self.deconv4b(x)

        x = self.deconv3a(x, rem2)
        x = self.deconv3b(x)

        x = self.deconv2a(x, rem1)
        x = self.deconv2b(x)

        x = self.deconv1a(x, rem0)
        x = self.deconv1b(x)

        return x

class Disp(nn.Module):

    def __init__(self, maxdisp=192, keepdim=False):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.keepdim = keepdim
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp,keepdim=self.keepdim)

    def forward(self, x):
        x = self.softmax(x)

        return self.disparity(x)

def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid


def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    assert disp.min() >= 0

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros')
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask

def disp_warp2(right, disp):
        #right B 3 H W
        #disp B H W
        #output B 3 H W
        bs, channels, height, width = right.size()

        mh, mw = torch.meshgrid([torch.arange(0, height, dtype=right.dtype, device=right.device),
                                 torch.arange(0, width, dtype=right.dtype, device=right.device)])  # (H *W)
        mh = mh.reshape(1, height, width).repeat(bs, 1, 1)  # B H W
        mw = mw.reshape(1, height, width).repeat(bs, 1, 1)  # (B, H, W)

        cur_disp_coords_y = mh
        cur_disp_coords_x = mw - disp

        # print("cur_disp", cur_disp, cur_disp.shape if not isinstance(cur_disp, float) else 0)
        # print("cur_disp_coords_x", cur_disp_coords_x, cur_disp_coords_x.shape)

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=3) #(B, H, W, 2)

        warped_img = F.grid_sample(right, grid, mode='bilinear', padding_mode='border')


        return warped_img

class MulFeature(nn.Module):
    def __init__(self, in_channels):
        super(MulFeature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(in_channels, 24, bn=False, kernel_size=3, padding=1),
            BasicConv(24, 24, bn=False, kernel_size=3, padding=1))

        self.conv1a = nn.Sequential(
            BasicConv(24, 32, bn=False, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, bn=False, kernel_size=3, stride=1, padding=1))

        self.conv2a = nn.Sequential(
            BasicConv(32, 48, bn=False, kernel_size=3, stride=2, padding=1),
            BasicConv(48, 48, bn=False, kernel_size=3, stride=1, padding=1))

        self.conv3a = nn.Sequential(
            BasicConv(48, 64, bn=False, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, bn=False, kernel_size=3, stride=1, padding=1))

        self.conv4a = nn.Sequential(
            BasicConv(64, 80, bn=False, kernel_size=3, stride=2, padding=1),
            BasicConv(80, 80, bn=False, kernel_size=3, stride=1, padding=1))


    def forward(self,input):
        rem0 = self.conv_start(input)

        rem1 = self.conv1a(rem0)

        rem2 = self.conv2a(rem1)

        rem3 = self.conv3a(rem2)

        rem4 = self.conv4a(rem3)


        return rem0, rem1, rem2, rem3, rem4



class DCU(nn.Module):  
    def __init__(self):
        super(DCU, self).__init__()

        self.rgbfea = MulFeature(3)
        self.dfea = MulFeature(1)

        self.deconv4a = Conv2x(80, 64, deconv=True, bn=False)
        self.deconv4b = BasicConv(64, 64, bn=False, kernel_size=3, stride=1, padding=1)

        self.deconv3a = Conv2x(64, 48, deconv=True, bn=False)
        self.deconv3b = BasicConv(48, 48, bn=False, kernel_size=3, stride=1, padding=1)

        self.deconv2a = Conv2x(48, 32, deconv=True, bn=False)
        self.deconv2b = BasicConv(32, 32, bn=False, kernel_size=3, stride=1, padding=1)

        self.deconv1a = Conv2x(32, 24, deconv=True, bn=False)
        self.deconv1b = BasicConv(24, 24, bn=False, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, d):

        a = self.rgbfea(rgb)
        b = self.dfea(d)
        c = {}
        c[0] = a[0] + b[0]
        c[1] = a[1] + b[1]
        c[2] = a[2] + b[2]
        c[3] = a[3] + b[3]
        c[4] = a[4] + b[4]


        x = self.deconv4a(c[4],c[3])
        x = self.deconv4b(x)

        x = self.deconv3a(x, c[2])
        x = self.deconv3b(x)

        x = self.deconv2a(x, c[1])
        x = self.deconv2b(x)

        x = self.deconv1a(x, c[0])
        x = self.deconv1b(x)

        return x

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp, keepdim=False):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp
        self.keepdim = keepdim
    #        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)
    def forward(self, x):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)), [1, self.maxdisp, 1, 1])).cuda(),
                            requires_grad=False)
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1, self.keepdim)
        return out

class GetCostVolume(nn.Module):
    def __init__(self):
        super(GetCostVolume, self).__init__()

    def get_warped_feats(self, x, y, disp_range_samples, ndisp):
        bs, channels, height, width = y.size()

        mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
        mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
        mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

        cur_disp_coords_y = mh
        cur_disp_coords_x = mw - disp_range_samples

        # print("cur_disp", cur_disp, cur_disp.shape if not isinstance(cur_disp, float) else 0)
        # print("cur_disp_coords_x", cur_disp_coords_x, cur_disp_coords_x.shape)

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

        y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros').view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)


        # a littel difference, no zeros filling
        x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)
        x_warped = x_warped.transpose(0, 1) #(C, B, D, H, W)
        #x1 = x2 + d >= d
        x_warped[:, mw < disp_range_samples] = 0
        x_warped = x_warped.transpose(0, 1) #(B, C, D, H, W)

        return x_warped, y_warped

    def build_concat_volume(self, x, y, disp_range_samples, ndisp):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()

        concat_cost = x.new().resize_(bs, channels * 2, ndisp, height, width).zero_()  # (B, 2C, D, H, W)

        x_warped, y_warped = self.get_warped_feats(x, y, disp_range_samples, ndisp)
        concat_cost[:, x.size()[1]:, :, :, :] = y_warped
        concat_cost[:, :x.size()[1], :, :, :] = x_warped

        return concat_cost

    def build_gwc_volume(self, x, y, disp_range_samples, ndisp, num_groups):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()

        x_warped, y_warped = self.get_warped_feats(x, y, disp_range_samples, ndisp) #(B, C, D, H, W)

        assert channels % num_groups == 0
        channels_per_group = channels // num_groups
        gwc_cost = (x_warped * y_warped).view([bs, num_groups, channels_per_group, ndisp, height, width]).mean(dim=2)  #(B, G, D, H, W)

        return gwc_cost

    def forward(self, features_left, features_right, disp_range_samples, ndisp, num_groups):
        # bs, channels, height, width = features_left["gwc_feature"].size()
        gwc_volume = self.build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"],
                                           disp_range_samples, ndisp, num_groups)

        concat_volume = self.build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                 disp_range_samples, ndisp)

        volume = torch.cat((gwc_volume, concat_volume), 1)   #(B, C+G, D, H, W)

        return volume

def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


def disparity_regression(x, disp_values):
    assert len(x.shape) == 4
    return torch.sum(x * disp_values, 1, keepdim=False)

class CostAggregation(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(CostAggregation, self).__init__()

        self.dres0 = nn.Sequential(convbn_3d(in_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(base_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(base_channels, base_channels, 3, 1, 1))

        self.dres2 = hourglass(base_channels)

        self.dres3 = hourglass(base_channels)

        self.dres4 = hourglass(base_channels)

        self.classif0 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(base_channels, base_channels, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, cost, FineD, FineH, FineW, disp_range_samples):

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        cost3 = self.classif3(out3)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)

            cost0 = F.upsample(cost0, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)
            cost1 = F.upsample(cost1, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)
            cost2 = F.upsample(cost2, [FineD, FineH, FineW], mode='trilinear',
                               align_corners=Align_Corners)

            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, disp_range_samples)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, disp_range_samples)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, disp_range_samples)

        cost3 = F.upsample(cost3, [FineD, FineH, FineW], mode='trilinear', align_corners=Align_Corners)
        cost3 = torch.squeeze(cost3, 1)
        pred3_prob = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparity_regression(pred3_prob, disp_range_samples)

        if self.training:
            return pred0, pred1, pred2, pred3
        else:
            return pred3

class MyNet(nn.Module):
    def __init__(self, maxdisp=192):
        super(MyNet, self).__init__()

        self.maxdisp = maxdisp
        self.feature = Feature()
        self.left_1 = BasicConv(32, 32, relu=False, kernel_size=3, padding=1)
        self.right_1 = BasicConv(32, 32, relu=False, kernel_size=3, padding=1)
        self.cv2D = Correlation(int(self.maxdisp/3))
        self.unet2d = Unet2D()
        self.disp = Disp(self.maxdisp, keepdim=True)
        self.comb = DCU()

        self.get_cv = GetCostVolume()

        #self.cv3D = GetCostVolume(int(2))
        self.cost_agg = CostAggregation(32,32)
        #self.disp1 = Disp(24,True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, y):
        start_time = time.time()
        outputs = {}
        left = x
        right = y
        x = self.feature(left)
        print('time = {:3f}'.format(time.time() - start_time))
        start_time = time.time()
        x_left = self.left_1(x[0])
        y = self.feature(right)
        y_right = self.right_1(y[0])
        cost = self.cv2D(x_left, y_right)
        cost = self.unet2d(cost)
        cost = torch.unsqueeze(cost,1)
        cost = F.upsample(cost,[192,cost.size()[3]*3,cost.size()[4]*3], mode='trilinear',align_corners=False)
        cost = torch.squeeze(cost,1)
        cost = self.disp(cost)
        cost = torch.squeeze(cost, 1)

        if self.training:
            pred = cost
            outputs_stage = {
                "pred0": cost,
                "pred1": cost,
                "pred2": cost,
                "pred3": cost,
                "pred": pred}
            outputs["stage1"] = outputs_stage
            outputs.update(outputs_stage)

        else:
                pred = cost
                outputs_stage = {
                    "pred3": cost,
                    "pred": pred}
                outputs["stage1"] = outputs_stage

        print('time = {:3f}'.format(time.time() - start_time))
        start_time = time.time()

        cur_disp = cost.detach()
        # warped_right = torch.abs(disp_warp2(right, cur_disp))
        warped_right = disp_warp2(right, cur_disp)

        #cur_disp = cost.unsqueeze(1)
        #warped_right = disp_warp(right, cur_disp)[0]
        error = warped_right - left
        disp_range_samples = self.comb(error,cur_disp.unsqueeze(1))
        #concat1 = torch.cat((error, cur_disp),dim=1)
        
        
        # print('time = {:3f}'.format(time.time() - start_time))
        # start_time = time.time()

        cost = self.get_cv(x[1], y[1],
                           disp_range_samples=F.interpolate((disp_range_samples / 3).unsqueeze(1),
                                                            [24 // 3,
                                                             left.size()[2] // 3,
                                                             left.size()[3] // 3],
                                                            mode='trilinear',
                                                            align_corners=False).squeeze(1),
                           ndisp=24 // 3,  num_groups=20)

        print('time = {:3f}'.format(time.time() - start_time))
        start_time = time.time()
        
        if self.training:
            pred0, pred1, pred2, pred3 = self.cost_agg(cost,24,left.size()[2],left.size()[3],disp_range_samples)
            pred = pred3
            outputs_stage = {
                "pred0": pred0,
                "pred1": pred1,
                "pred2": pred2,
                "pred3": pred3,
                "pred": pred}
            outputs["stage2"] = outputs_stage
            outputs.update(outputs_stage)

        else:
                pred3 = self.cost_agg(cost,
                                                 FineD=24,
                                                 FineH=left.shape[2],
                                                 FineW=left.shape[3],
                                                 disp_range_samples=disp_range_samples)
                pred = pred3
                outputs_stage = {
                    "pred3": pred3,
                    "pred": pred}
                outputs["stage2"] = outputs_stage
        print('time = {:3f}'.format(time.time() - start_time))
        
        return outputs