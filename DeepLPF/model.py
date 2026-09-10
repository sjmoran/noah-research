# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Authors: Sean Moran (sean.j.moran@gmail.com),
         Pierre Marza (pierre.marza@gmail.com)

Code-to-paper map (section / equation numbers refer to the arXiv version):

* ``unet.UNetModel``             backbone, Sec. 3.1 (its first three output channels
                                 are the backbone-enhanced image Y1)
* ``cubic.CubicFilter``          polynomial ("cubic-20") filter, Sec. 3.2.4, Eq. 6 -> Y2
* ``graduated.GraduatedFilter``  graduated filter, Sec. 3.2.2, Eqs. 1-3, fused per Eq. 7 -> s_g
* ``elliptical.EllipticalFilter`` elliptical filter, Sec. 3.2.3, Eq. 4, fused per Eq. 7 -> s_e
* ``DeepLPFParameterPrediction`` fusion S = 1 + (s_g - 1) + (s_e - 1), Y3 = S * Y2,
                                 Y = Y3 + Y1, Sec. 3.1
* ``losses.DeepLPFLoss``         training loss, Sec. 3.4, Eq. 8

The filter heads live in :mod:`cubic`, :mod:`graduated` and :mod:`elliptical`,
the conv blocks they are built from in :mod:`blocks`, and the loss in
:mod:`losses`. They are re-exported here so that ``model.CubicFilter`` and
``model.DeepLPFLoss`` keep resolving, as they did when every class lived in
this file. The module hierarchy of the network is unchanged, so state dicts
saved before this split load unchanged.
'''
import torch
import torch.nn as nn

import unet
# Re-exported so that model.CubicFilter and model.DeepLPFLoss resolve, as
# they did when every class lived in this file.
from blocks import Block, ConvBlock, GlobalPoolingBlock, MaxPoolBlock  # noqa: F401
from filters import (BinaryLayer, CubicFilter, EllipticalFilter,  # noqa: F401
                     GraduatedFilter, SignSTE)
from losses import DeepLPFLoss  # noqa: F401


class DeepLPFParameterPrediction(nn.Module):
    import torch.nn.functional as F

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation function

        :param num_in_channels:  Number of input feature maps
        :param num_out_channels: Number of output feature maps
        :returns: N/A
        :rtype: N/A

        """
        super(DeepLPFParameterPrediction, self).__init__()
        self.cubic_filter = CubicFilter()
        self.graduated_filter = GraduatedFilter()
        self.elliptical_filter = EllipticalFilter()
      

    def forward(self, x):
        """DeepLPF combined architecture fusing cubic, graduated and elliptical filters

        :param x: forward the data Tensor x through the network
        :returns: Tensor representing the predicted image batch of shape BxCxWxH
        :rtype: Tensor

        """
        x = x.contiguous()  # remove memory holes

        feat = x[:, 3:64, :, :]  # C' = C - 3 backbone features
        img = x[:, 0:3, :, :]    # Y1: backbone-enhanced image

        # Each branch's parameter predictor consumes cat(feat, image) resized
        # to 300x300. The resize is bilinear and per channel, so
        # upsample(cat(a, b)) == cat(upsample(a), upsample(b)) exactly; the
        # feature part is shared by all three branches, and the graduated and
        # elliptical branches share the whole input. Resize each part once
        # instead of five times.
        upsample = self.cubic_filter.upsample
        feat_up = upsample(feat)

        # Single-stream path: Y2 = polynomial filter applied to Y1
        img_cubic = self.cubic_filter.mask_from_input(
            torch.cat((feat_up, upsample(img)), 1), img)

        # Two-stream path: graduated and elliptical scaling maps estimated from Y2
        feat_img_cubic_up = torch.cat((feat_up, upsample(img_cubic)), 1)
        mask_scale_graduated = self.graduated_filter.mask_from_input(
            feat_img_cubic_up, img_cubic)
        mask_scale_elliptical = self.elliptical_filter.mask_from_input(
            feat_img_cubic_up, img_cubic)

        mask_scale_fuse = torch.clamp(
            1.0 + (mask_scale_graduated - 1.0) + (mask_scale_elliptical - 1.0), 0, 2)

        img_fuse = torch.clamp(img_cubic*mask_scale_fuse, 0, 1)
        
        img = torch.clamp(img_fuse+img, 0, 1)
        
        return img


class DeepLPFNet(nn.Module):

    def __init__(self):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(DeepLPFNet, self).__init__()
        self.backbonenet = unet.UNetModel()
        self.deeplpfnet = DeepLPFParameterPrediction()
        
    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: residual image
        :rtype: numpy ndarray

        """
        feat = self.backbonenet(img)
        img = self.deeplpfnet(feat)
        
        return img
