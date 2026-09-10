# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

The elliptical filter of Sec. 3.2.3, Eq. 4.
'''
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from blocks import ConvBlock, GlobalPoolingBlock, MaxPoolBlock


class EllipticalFilter(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation function

        :param block: a block (layer) of the neural network
        :param num_layers:  number of neural network layers
        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(EllipticalFilter, self).__init__()

        self.elliptical_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.elliptical_layer2 = MaxPoolBlock()
        self.elliptical_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer4 = MaxPoolBlock()
        self.elliptical_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer6 = MaxPoolBlock()
        self.elliptical_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer8 = GlobalPoolingBlock(2)
        self.fc_elliptical = torch.nn.Linear(
            num_out_channels, 24)  # elliptical
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear',align_corners=False)
        self.dropout = nn.Dropout(0.5)

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

    def get_mask(self, x_axis, y_axis, shift_x=0, shift_y=0, semi_axis_x=1, semi_axis_y=1, alpha=0,
                 scale_factor=2, max_scale=2, eps=1e-7, radius=1):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :returns: scaling mask
        :rtype: Tensor

        """
        # Check whether a point is inside our outside of the ellipse and set the scaling factor accordingly
        ellipse_equation_part1 = (((x_axis - shift_x)*torch.cos(alpha) + (y_axis - shift_y)*torch.sin(alpha)) ** 2) / ((semi_axis_x)**2)
        ellipse_equation_part2 = (((x_axis - shift_x)*torch.sin(alpha) - (y_axis - shift_y)*torch.cos(alpha)) ** 2) / ((semi_axis_y)**2)

        # Set the scaling factors to decay with radius inside the ellipse
        mask_scale = self.where(ellipse_equation_part1+ellipse_equation_part2 < 1,
                                (torch.sqrt((x_axis - shift_x) ** 2 + (y_axis - shift_y) ** 2 + eps) * (1 - scale_factor)) / radius + scale_factor, 1)

        mask_scale = torch.clamp(mask_scale.unsqueeze(0), 0, max_scale)

        return mask_scale

    def get_elliptical_mask(self, feat, img):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :param feat: features from the backbone
        :param img: image
        :returns: elliptical adjustment maps for each channel
        :rtype: Tensor

        """

        # The two eps parameters are used to avoid numerical issues in the learning
        eps2 = 1e-7
        eps1 = 1e-10

        # max_scale is the maximum an ellipse can scale the image R,G,B values by
        max_scale = 2
        min_scale = 0

        feat_elliptical = torch.cat((feat, img), 1)
        feat_elliptical = self.upsample(feat_elliptical)

        # The following layers calculate the parameters of the ellipses that we use for image enhancement
        x = self.elliptical_layer1(feat_elliptical)
        x = self.elliptical_layer2(x)
        x = self.elliptical_layer3(x)
        x = self.elliptical_layer4(x)
        x = self.elliptical_layer5(x)
        x = self.elliptical_layer6(x)
        x = self.elliptical_layer7(x)
        x = self.elliptical_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_elliptical(x)

        # The next code implements a rotated ellipse according to:
        # https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
        
        # Normalised coordinates for x and y-axes, we instantiate the ellipses in these coordinates
        x_axis = torch.arange(
            img.shape[2], device=img.device).view(-1, 1).repeat(1, img.shape[3]) / img.shape[2]
        y_axis = torch.arange(img.shape[3], device=img.device).repeat(
            img.shape[2], 1) / img.shape[3]

        # x coordinate - h position
        right_x = (img.shape[2] - 1) / img.shape[2]
        left_x = 0

        # Centre of ellipse, x-coordinate
        x_coord1 = self.tanh01(G[0, 0]) + eps1
        x_coord2 = self.tanh01(G[0, 1]) + eps1
        x_coord3 = self.tanh01(G[0, 2]) + eps1

        # y coordinate - k coordinate
        right_y = (img.shape[3] - 1) // img.shape[3]
        left_y = 0

        # Centre of ellipse, y-coordinate
        y_coord1 = self.tanh01(G[0, 3]) + eps1 
        y_coord2 = self.tanh01(G[0, 4]) + eps1
        y_coord3 = self.tanh01(G[0, 5]) + eps1

        # a value of ellipse
        a1 = self.tanh01(G[0, 6]) + eps1
        a2 = self.tanh01(G[0, 7]) + eps1
        a3 = self.tanh01(G[0, 8]) + eps1

        # b value
        b1 = self.tanh01(G[0, 9]) + eps1
        b2 = self.tanh01(G[0, 10]) + eps1
        b3 = self.tanh01(G[0, 11]) + eps1

        # A value is angle to the x-axis
        A1 = self.tanh01(G[0, 12]) * math.pi + eps1
        A2 = self.tanh01(G[0, 13]) * math.pi + eps1
        A3 = self.tanh01(G[0, 14]) * math.pi + eps1

        '''
        The following are the scale factors for each of the 9 ellipses
        '''
        scale1 = self.tanh01(G[0, 15]) * max_scale + eps1 
        scale2 = self.tanh01(G[0, 16]) * max_scale + eps1
        scale3 = self.tanh01(G[0, 17]) * max_scale + eps1

        scale4  = self.tanh01(G[0, 18]) * max_scale + eps1
        scale5 = self.tanh01(G[0, 19]) * max_scale + eps1
        scale6 = self.tanh01(G[0, 20]) * max_scale + eps1

        scale7 = self.tanh01(G[0, 21]) * max_scale + eps1
        scale8 = self.tanh01(G[0, 22]) * max_scale + eps1
        scale9 = self.tanh01(G[0, 23]) * max_scale + eps1

        ############ Angle of orientation of the ellipses with respect to the y semi-axis
        angle_1 = torch.acos(torch.clamp((y_axis-y_coord1) / 
            (torch.sqrt((x_axis-x_coord1)**2 + (y_axis-y_coord1)**2 + eps1)), -1+eps2, 1-eps2))-A1

        angle_2 = torch.acos(torch.clamp((y_axis-y_coord2) / 
            (torch.sqrt((x_axis-x_coord2) ** 2 + (y_axis-y_coord2)**2 + eps1)), -1+eps2, 1-eps2))-A2
        
        angle_3 = torch.acos(torch.clamp((y_axis-y_coord3) / 
            (torch.sqrt((x_axis-x_coord3) ** 2 + (y_axis-y_coord3)**2 + eps1)), -1+eps2, 1-eps2))-A3

        ############ Radius of the ellipses
        # https://math.stackexchange.com/questions/432902/how-to-get-the-radius-of-an-ellipse-at-a-specific-angle-by-knowing-its-semi-majo
        radius_1 = (a1*b1)/torch.sqrt((a1**2)*(torch.sin(angle_1)**2)+(b1**2)*(torch.cos(angle_1)**2) + eps1)

        radius_2 = (a2*b2)/torch.sqrt((a2**2)*(torch.sin(angle_2)**2)+(b2**2)*(torch.cos(angle_2)**2) + eps1)

        radius_3 = (a3*b3)/torch.sqrt((a3**2)*(torch.sin(angle_3)**2)+(b3**2)*(torch.cos(angle_3)**2) + eps1)

        
        ############ Scaling factors for the R,G,B channels, here we learn three ellipses
        mask_scale1 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord1, shift_y=y_coord1, semi_axis_x=a1, semi_axis_y=b1, alpha=angle_1, scale_factor=scale1,
                                    radius=radius_1)

        mask_scale2 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord1, shift_y=y_coord1, semi_axis_x=a1, semi_axis_y=b1, alpha=angle_1, scale_factor=scale2,
                                    radius=radius_1)

        mask_scale3 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord1, shift_y=y_coord1, semi_axis_x=a1, semi_axis_y=b1, alpha=angle_1, scale_factor=scale3,
                                    radius=radius_1)

        mask_scale_1 = torch.cat(
            (mask_scale1, mask_scale2, mask_scale3), dim=0)
        mask_scale_1_rad = torch.clamp(mask_scale_1.unsqueeze(0), 0, max_scale)

        ############ Scaling factors for the R,G,B channels, here we learn three ellipses
        mask_scale4 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord2, shift_y=y_coord2, semi_axis_x=a2, semi_axis_y=b2, alpha=angle_2, scale_factor=scale4,
                                    radius=radius_2)

        mask_scale5 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord2, shift_y=y_coord2, semi_axis_x=a2, semi_axis_y=b2, alpha=angle_2, scale_factor=scale5,
                                    radius=radius_2)

        mask_scale6 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord2, shift_y=y_coord2, semi_axis_x=a2, semi_axis_y=b2, alpha=angle_2, scale_factor=scale6,
                                    radius=radius_2)

        mask_scale_4 = torch.cat(
            (mask_scale4, mask_scale5, mask_scale6), dim=0)
        mask_scale_4_rad = torch.clamp(mask_scale_4.unsqueeze(0), 0, max_scale)

        ############ Scaling factors for the R,G,B channels, here we learn three ellipses
        mask_scale7 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord3, shift_y=y_coord3, semi_axis_x=a3, semi_axis_y=b3, alpha=angle_3, scale_factor=scale7,
                                    radius=radius_3)

        mask_scale8 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord3, shift_y=y_coord3, semi_axis_x=a3, semi_axis_y=b3, alpha=angle_3, scale_factor=scale8,
                                    radius=radius_3)

        mask_scale9 = self.get_mask(x_axis, y_axis,
                                    shift_x=x_coord3, shift_y=y_coord3, semi_axis_x=a3, semi_axis_y=b3, alpha=angle_3, scale_factor=scale9,
                                    radius=radius_3)

        mask_scale_7 = torch.cat(
            (mask_scale7, mask_scale8, mask_scale9), dim=0)
        mask_scale_7_rad = torch.clamp(mask_scale_7.unsqueeze(0), 0, max_scale)

        ############ Mix the ellipses together by multiplication
        mask_scale_elliptical = torch.clamp(
            mask_scale_1_rad * mask_scale_4_rad * mask_scale_7_rad, 0, max_scale)

        return mask_scale_elliptical
