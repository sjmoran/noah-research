# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

The polynomial ("cubic-20") filter of Sec. 3.2.4, Eq. 6.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

from blocks import ConvBlock, GlobalPoolingBlock, MaxPoolBlock


class CubicFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64, batch_size=1):
        """Initialisation function

        :param block: a block (layer) of the neural network
        :param num_layers:  number of neural network layers
        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CubicFilter, self).__init__()

        self.cubic_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.cubic_layer2 = MaxPoolBlock()
        self.cubic_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer4 = MaxPoolBlock()
        self.cubic_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer6 = MaxPoolBlock()
        self.cubic_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer8 = GlobalPoolingBlock(2)
        self.fc_cubic = torch.nn.Linear(
            num_out_channels, 60)  # cubic
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)

    def get_cubic_mask(self, feat, img):
        """Cubic filter definition

        :param feat: feature map
        :param img:  image
        :returns: cubic scaling map
        :rtype: Tensor

        """
        #######################################################
        ####################### Cubic #########################
        feat_cubic = torch.cat((feat, img), 1)
        feat_cubic = self.upsample(feat_cubic)

        x = self.cubic_layer1(feat_cubic)
        x = self.cubic_layer2(x)
        x = self.cubic_layer3(x)
        x = self.cubic_layer4(x)
        x = self.cubic_layer5(x)
        x = self.cubic_layer6(x)
        x = self.cubic_layer7(x)
        x = self.cubic_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        R = self.fc_cubic(x)

        cubic_mask = torch.zeros_like(img)

        x_axis = torch.arange(
            img.shape[2], device=img.device).view(-1, 1).repeat(1, img.shape[3]) / img.shape[2]
        y_axis = torch.arange(img.shape[3], device=img.device).repeat(
            img.shape[2], 1) / img.shape[3]

        '''
        Cubic for R channel
        '''
        cubic_mask[0, 0, :, :] = R[0, 0] * (x_axis ** 3) + R[0, 1] * (x_axis ** 2) * y_axis + R[0, 2] * (
            x_axis ** 2) * img[0, 0, :, :] + R[0, 3] * (x_axis ** 2) + R[0, 4] * x_axis * (y_axis ** 2) + R[
            0, 5] * x_axis * y_axis * img[0, 0, :, :] \
            + R[0, 6] * x_axis * y_axis + R[0, 7] * x_axis * (img[0, 0, :, :] ** 2) + R[
            0, 8] * x_axis * img[0, 0, :, :] + R[0, 9] * x_axis + R[0, 10] * (
            y_axis ** 3) + R[0, 11] * (y_axis ** 2) * img[0, 0, :, :] \
            + R[0, 12] * (y_axis ** 2) + R[0, 13] * y_axis * (img[0, 0, :, :] ** 2) + R[
            0, 14] * y_axis * img[0, 0, :, :] + R[0, 15] * y_axis + R[0, 16] * (
            img[0, 0, :, :] ** 3) + R[0, 17] * (img[0, 0, :, :] ** 2) \
            + R[0, 18] * \
            img[0, 0, :, :] + R[0, 19]

        '''
        Cubic for G channel
        '''
        cubic_mask[0, 1, :, :] = R[0, 20] * (x_axis ** 3) + R[0, 21] * (x_axis ** 2) * y_axis + R[0, 22] * (
            x_axis ** 2) * img[0, 1, :, :] + R[0, 23] * (x_axis ** 2) + R[0, 24] * x_axis * (y_axis ** 2) + R[
            0, 25] * x_axis * y_axis * img[0, 1, :, :] \
            + R[0, 26] * x_axis * y_axis + R[0, 27] * x_axis * (img[0, 1, :, :] ** 2) + R[
            0, 28] * x_axis * img[0, 1, :, :] + R[0, 29] * x_axis + R[0, 30] * (
            y_axis ** 3) + R[0, 31] * (y_axis ** 2) * img[0, 1, :, :] \
            + R[0, 32] * (y_axis ** 2) + R[0, 33] * y_axis * (img[0, 1, :, :] ** 2) + R[
            0, 34] * y_axis * img[0, 1, :, :] + R[0, 35] * y_axis + R[0, 36] * (
            img[0, 1, :, :] ** 3) + R[0, 37] * (img[0, 1, :, :] ** 2) \
            + R[0, 38] * \
            img[0, 1, :, :] + R[0, 39]

        '''
        Cubic for B channel
        '''
        cubic_mask[0, 2, :, :] = R[0, 40] * (x_axis ** 3) + R[0, 41] * (x_axis ** 2) * y_axis + R[0, 42] * (
            x_axis ** 2) * img[0, 2, :, :] + R[0, 43] * (x_axis ** 2) + R[0, 44] * x_axis * (y_axis ** 2) + R[
            0, 45] * x_axis * y_axis * img[0, 2, :, :] \
            + R[0, 46] * x_axis * y_axis + R[0, 47] * x_axis * (img[0, 2, :, :] ** 2) + R[
            0, 48] * x_axis * img[0, 2, :, :] + R[0, 49] * x_axis + R[0, 50] * (
            y_axis ** 3) + R[0, 51] * (y_axis ** 2) * img[0, 2, :, :] \
            + R[0, 52] * (y_axis ** 2) + R[0, 53] * y_axis * (img[0, 2, :, :] ** 2) + R[
            0, 54] * y_axis * img[0, 2, :, :] + R[0, 55] * y_axis + R[0, 56] * (
            img[0, 2, :, :] ** 3) + R[0, 57] * (img[0, 2, :, :] ** 2) \
            + R[0, 58] * \
            img[0, 2, :, :] + R[0, 59]

        img_cubic = torch.clamp(img + cubic_mask, 0, 1)
        return img_cubic
