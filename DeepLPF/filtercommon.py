# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Pieces shared by the three filter heads: the straight-through binarisation
of Sec. 3.2.2.
'''
import torch
import torch.nn as nn


class SignSTE(torch.autograd.Function):
    """sign() with a straight-through gradient (Courbariaux et al.)

    The forward pass binarises; the backward pass passes the gradient through
    unchanged except where the input has saturated (|input| > 1), where it is
    zeroed.

    This is implemented as a torch.autograd.Function because that is the only
    place autograd calls a user-defined backward. A backward() defined as a
    plain method on an nn.Module is never called by autograd, so the gradient
    seen by the layer's input would be torch.sign's own gradient, which is zero
    everywhere.

    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input.abs() <= 1).to(grad_output.dtype)


class BinaryLayer(nn.Module):

    def forward(self, input):
        """Forward function for binary layer

        :param input: data
        :returns: sign of data
        :rtype: Tensor

        """
        return SignSTE.apply(input)
