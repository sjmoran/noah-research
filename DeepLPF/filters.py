# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Import surface for the three parametric filter heads and the binarisation
layer, so that a caller can `from filters import CubicFilter` without needing
to know which module each one lives in.
'''
from cubic import CubicFilter  # noqa: F401
from elliptical import EllipticalFilter  # noqa: F401
from filtercommon import BinaryLayer, SignSTE  # noqa: F401
from graduated import GraduatedFilter  # noqa: F401

__all__ = ['BinaryLayer', 'CubicFilter', 'EllipticalFilter',
           'GraduatedFilter', 'SignSTE']
