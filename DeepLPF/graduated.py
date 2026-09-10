# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

The graduated filter of Sec. 3.2.2, Eqs. 1-3.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

from blocks import ConvBlock, GlobalPoolingBlock, MaxPoolBlock
from filtercommon import BinaryLayer


class GraduatedFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation function for the graduated filter

        :param num_in_channels:  input channels
        :param num_out_channels: output channels
        :returns: N/A
        :rtype: N/A

        """
        super(GraduatedFilter, self).__init__()

        self.graduated_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.graduated_layer2 = MaxPoolBlock()
        self.graduated_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer4 = MaxPoolBlock()
        self.graduated_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer6 = MaxPoolBlock()
        self.graduated_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer8 = GlobalPoolingBlock(2)
        self.fc_graduated = torch.nn.Linear(
            num_out_channels, 24)
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)
        self.bin_layer = BinaryLayer()

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors

        :param cond: condition e.g. <
        :param x_1: Tensor 1
        :param x_2: Tensor 2
        :returns: Boolean comparison result
        :rtype: Tensor

        """
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_inverted_mask(self, factor, invert, d1, d2, max_scale, top_line):
        """ Inverts the graduated filter based on a learnt binary variable 

        :param factor: scale factor
        :param invert: binary indicator variable
        :param d1: distance between top and mid line
        :param d2: distannce between botto and mid line
        :param max_scale: maximum scaling factor possible
        :param top_line:  representation of top line
        :returns: inverted scaling mask
        :rtype: Tensor

        """
        # `invert` is the binarised g_inv of Eq. 3. Selecting the branch with a
        # Python `if` (or, equivalently, with torch.where, whose condition is
        # not differentiable) means g_inv never enters the arithmetic, so no
        # gradient reaches it and the indicator cannot be learned. Evaluate
        # both branches and blend them by the indicator instead: `invert` is
        # exactly 0 or 1, so the forward value is the same branch as before,
        # but the straight-through gradient of BinaryLayer now flows to g_inv.
        if (factor >= 1).all():
            diff = ((factor-1))/2 + 1
            # invert == 1
            grad1_inv = (diff-factor)/d1
            grad2_inv = (1-diff)/d2
            mask_inv = factor+grad1_inv*top_line+grad2_inv*top_line
            # invert != 1
            grad1_non = (diff-factor)/d1
            grad2_non = (factor-diff)/d2
            mask_non = 1+grad1_non*top_line+grad2_non*top_line
            min_val, max_val = 1, max_scale
        else:
            diff = ((1-factor))/2 + factor
            # invert == 1
            grad1_inv = (diff-factor)/d1
            grad2_inv = (1-diff)/d2
            mask_inv = factor+grad1_inv*top_line+grad2_inv*top_line
            # invert != 1
            grad1_non = (diff-1)/d1
            grad2_non = (factor-diff)/d2
            mask_non = 1+grad1_non*top_line+grad2_non*top_line
            min_val, max_val = 0, 1

        weight = invert.to(mask_inv.dtype)
        mask_scale = weight*mask_inv + (1-weight)*mask_non
        mask_scale = torch.clamp(mask_scale, min=min_val, max=max_val)

        mask_scale = torch.clamp(mask_scale.unsqueeze(0), 0, max_scale)
        return mask_scale

    def get_graduated_mask(self, feat, img):
        """ Graduated filter definition

        :param feat: features
        :param img: image
        :returns: scaling map
        :rtype: Tensor

        """
        #######################################################
        ####################### Graduated #####################
        eps = 1e-10

        x_axis = torch.arange(
            img.shape[2], device=img.device).view(-1, 1).repeat(1, img.shape[3]) / img.shape[2]
        y_axis = torch.arange(img.shape[3], device=img.device).repeat(
            img.shape[2], 1) / img.shape[3]

        feat_graduated = torch.cat((feat, img), 1)
        feat_graduated = self.upsample(feat_graduated)

        # The following layers calculate the parameters of the graduated filters that we use for image enhancement
        x = self.graduated_layer1(feat_graduated)
        x = self.graduated_layer2(x)
        x = self.graduated_layer3(x)
        x = self.graduated_layer4(x)
        x = self.graduated_layer5(x)
        x = self.graduated_layer6(x)
        x = self.graduated_layer7(x)
        x = self.graduated_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_graduated(x)

        # Classification values (above or below the line)
        above_or_below_line1 = ((self.bin_layer(G[0, 0]))+1)/2
        above_or_below_line2 = ((self.bin_layer(G[0, 1]))+1)/2
        above_or_below_line3 = ((self.bin_layer(G[0, 2]))+1)/2

        slope1 = G[0, 3].clone()
        slope2 = G[0, 4].clone()
        slope3 = G[0, 5].clone()

        y_axis_dist1 = self.tanh01(G[0, 6]) + eps
        y_axis_dist2 = self.tanh01(G[0, 7]) + eps
        y_axis_dist3 = self.tanh01(G[0, 8]) + eps

        y_axis_dist1 = torch.clamp(self.tanh01(G[0, 9]), y_axis_dist1.data, 1.0)
        y_axis_dist2 = torch.clamp(self.tanh01(G[0, 10]), y_axis_dist2.data, 1.0)
        y_axis_dist3 = torch.clamp(self.tanh01(G[0, 11]), y_axis_dist3.data, 1.0)

        y_axis_dist4= torch.clamp(self.tanh01(G[0, 12]), 0, y_axis_dist1.data)
        y_axis_dist5 = torch.clamp(self.tanh01(G[0, 13]), 0, y_axis_dist2.data)
        y_axis_dist6 = torch.clamp(self.tanh01(G[0, 14]), 0, y_axis_dist3.data)

        # Scales
        max_scale = 2
        min_scale = 0

        scale_factor1 = self.tanh01(G[0, 15]) * max_scale
        scale_factor2 = self.tanh01(G[0, 16]) * max_scale
        scale_factor3 = self.tanh01(G[0, 17]) * max_scale

        scale_factor4 = self.tanh01(G[0, 18]) * max_scale
        scale_factor5 = self.tanh01(G[0, 19]) * max_scale
        scale_factor6 = self.tanh01(G[0, 20]) * max_scale

        scale_factor7 = self.tanh01(G[0, 21]) * max_scale
        scale_factor8 = self.tanh01(G[0, 22]) * max_scale
        scale_factor9= self.tanh01(G[0, 23]) * max_scale

        slope1_angle = torch.atan(slope1)
        slope2_angle = torch.atan(slope2)
        slope3_angle = torch.atan(slope3)

        # Distances between central line and two outer lines 
        d1 = self.tanh01(y_axis_dist1*torch.cos(slope1_angle))
        d2 = self.tanh01(y_axis_dist4*torch.cos(slope1_angle))
        d3 = self.tanh01(y_axis_dist2*torch.cos(slope2_angle))
        d4 = self.tanh01(y_axis_dist5*torch.cos(slope2_angle))
        d5 = self.tanh01(y_axis_dist3*torch.cos(slope3_angle))
        d6 = self.tanh01(y_axis_dist6*torch.cos(slope3_angle))

        top_line1 = self.tanh01(y_axis - (slope1 * x_axis + y_axis_dist1 + d1))
        top_line2 = self.tanh01(y_axis - (slope2 * x_axis + y_axis_dist2 + d3))
        top_line3 = self.tanh01(y_axis - (slope3 * x_axis + y_axis_dist3 + d5))

        '''
        The following are the scale factors for each of the 9 graduated filters
        '''
        mask_scale1 = self.get_inverted_mask(
            scale_factor1, above_or_below_line1, d1, d2, max_scale, top_line1)
        mask_scale2 = self.get_inverted_mask(
            scale_factor2, above_or_below_line1, d1, d2, max_scale, top_line1)
        mask_scale3 = self.get_inverted_mask(
            scale_factor3, above_or_below_line1, d1, d2, max_scale, top_line1)

        mask_scale_1 = torch.cat(
            (mask_scale1, mask_scale2, mask_scale3), dim=0)
        mask_scale_1 = torch.clamp(mask_scale_1.unsqueeze(0), 0, max_scale)

        mask_scale4 = self.get_inverted_mask(
            scale_factor4, above_or_below_line2, d3, d4, max_scale, top_line2)
        mask_scale5 = self.get_inverted_mask(
            scale_factor5, above_or_below_line2, d3, d4, max_scale, top_line2)
        mask_scale6 = self.get_inverted_mask(
            scale_factor6, above_or_below_line2, d3, d4, max_scale, top_line2)

        mask_scale_4 = torch.cat(
            (mask_scale4, mask_scale5, mask_scale6), dim=0)
        mask_scale_4 = torch.clamp(mask_scale_4.unsqueeze(0), 0, max_scale)

        mask_scale7 = self.get_inverted_mask(
            scale_factor7, above_or_below_line3, d5, d6, max_scale, top_line3)
        mask_scale8 = self.get_inverted_mask(
            scale_factor8, above_or_below_line3, d5, d6, max_scale, top_line3)
        mask_scale9 = self.get_inverted_mask(
            scale_factor9, above_or_below_line3, d5, d6, max_scale, top_line3)

        mask_scale_7 = torch.cat(
            (mask_scale7, mask_scale8, mask_scale9), dim=0)
        mask_scale_7 = torch.clamp(mask_scale_7.unsqueeze(0), 0, max_scale)

        mask_scale = torch.clamp(
            mask_scale_1*mask_scale_4*mask_scale_7, 0, max_scale)

        return mask_scale
