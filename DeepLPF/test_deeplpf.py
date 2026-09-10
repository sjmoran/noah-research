# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
Tests for the DeepLPF implementation.

Run with:  python -m pytest test_deeplpf.py

These are CPU-only and use synthetic images, so they need no dataset and no
GPU. They cover the pieces where a mistake is silent: the differentiability of
the binarised inversion indicator, the fusion of the two scaling maps, the
MS-SSIM product, the geometry shared by one ellipse's three channel masks, and
the data loader's file matching.
'''
import os

import numpy as np
import pytest
import torch
from PIL import Image

import data
import model
import util


# ---------------------------------------------------------------- binarisation

def test_binary_layer_forward_is_sign():
    layer = model.BinaryLayer()
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert torch.equal(layer(x), torch.sign(x))


def test_binary_layer_passes_gradient_through():
    """The straight-through estimator must actually be called by autograd."""
    layer = model.BinaryLayer()
    x = torch.tensor([-0.5, 0.5], requires_grad=True)
    layer(x).sum().backward()
    # torch.sign's own gradient is zero; the STE's is one inside |x| <= 1.
    assert torch.equal(x.grad, torch.ones(2))


def test_binary_layer_zeroes_saturated_gradient():
    layer = model.BinaryLayer()
    x = torch.tensor([-2.0, 2.0], requires_grad=True)
    layer(x).sum().backward()
    assert torch.equal(x.grad, torch.zeros(2))


# ------------------------------------------------------------ graduated filter

def _reference_inverted_mask(factor, invert, d1, d2, max_scale, top_line):
    """The original branch-selecting implementation, for a forward comparison."""
    if (invert == 1).all():
        if (factor >= 1).all():
            diff = ((factor - 1)) / 2 + 1
            mask = torch.clamp(factor + ((diff - factor) / d1) * top_line
                               + ((1 - diff) / d2) * top_line, min=1, max=max_scale)
        else:
            diff = ((1 - factor)) / 2 + factor
            mask = torch.clamp(factor + ((diff - factor) / d1) * top_line
                               + ((1 - diff) / d2) * top_line, min=0, max=1)
    else:
        if (factor >= 1).all():
            diff = ((factor - 1)) / 2 + 1
            mask = torch.clamp(1 + ((diff - factor) / d1) * top_line
                               + ((factor - diff) / d2) * top_line, min=1, max=max_scale)
        else:
            diff = ((1 - factor)) / 2 + factor
            mask = torch.clamp(1 + ((diff - 1) / d1) * top_line
                               + ((factor - diff) / d2) * top_line, min=0, max=1)
    return torch.clamp(mask.unsqueeze(0), 0, max_scale)


@pytest.mark.parametrize("invert_value", [0.0, 1.0])
@pytest.mark.parametrize("factor_value", [0.4, 1.6])
def test_inverted_mask_matches_branching_implementation(invert_value, factor_value):
    torch.manual_seed(0)
    filt = model.GraduatedFilter()
    factor = torch.tensor(factor_value)
    invert = torch.tensor(invert_value)
    d1, d2 = torch.tensor(0.4), torch.tensor(0.3)
    top_line = torch.rand(8, 8)

    got = filt.get_inverted_mask(factor, invert, d1, d2, 2, top_line)
    want = _reference_inverted_mask(factor, invert, d1, d2, 2, top_line)
    assert torch.allclose(got, want, atol=0, rtol=0)


def test_inversion_indicator_receives_gradient():
    """A gradient must reach the logit the indicator is binarised from."""
    torch.manual_seed(0)
    filt = model.GraduatedFilter()
    logit = torch.tensor(0.7, requires_grad=True)
    invert = (filt.bin_layer(logit) + 1) / 2
    mask = filt.get_inverted_mask(torch.tensor(1.5), invert,
                                  torch.tensor(0.4), torch.tensor(0.3), 2,
                                  torch.rand(8, 8))
    mask.sum().backward()
    assert logit.grad is not None and logit.grad.abs() > 0


# ----------------------------------------------------------- elliptical filter

def test_ellipse_channel_masks_share_their_geometry():
    """Each ellipse's three channel masks must differ only in scale factor.

    The nine (instance, channel) masks are produced by one batched get_mask
    call. The geometry arguments carry an instance dimension and no channel
    dimension, so all three channels of an instance necessarily see the same
    ellipse; only scale_factor is per channel. Assert exactly that shape
    contract, which is what a per-channel geometry argument would violate.
    """
    torch.manual_seed(0)
    filt = model.EllipticalFilter()
    calls = []
    real_get_mask = filt.get_mask

    def recording_get_mask(x_axis, y_axis, **kwargs):
        calls.append(kwargs)
        return real_get_mask(x_axis, y_axis, **kwargs)

    filt.get_mask = recording_get_mask
    filt.get_elliptical_mask(torch.rand(1, 61, 32, 32), torch.rand(1, 3, 32, 32))

    assert len(calls) == 1
    kwargs = calls[0]
    # (batch, instance, channel, H, W): geometry is broadcast over channels.
    for key in ("shift_x", "shift_y", "semi_axis_x", "semi_axis_y"):
        assert kwargs[key].shape == (1, 3, 1, 1, 1), key
    assert kwargs["scale_factor"].shape == (1, 3, 3, 1, 1)
    # The semi-axes belong to the same ellipse index as the centre.
    assert kwargs["semi_axis_y"].shape[1] == kwargs["shift_x"].shape[1]


def test_ellipse_geometry_is_per_instance_not_per_channel():
    """Two channels of one instance must produce the same mask when their
    scale factors are equal, which fails if they use different semi-axes."""
    torch.manual_seed(0)
    filt = model.EllipticalFilter()
    grid_h = grid_w = 16
    x_axis = torch.arange(grid_h).view(-1, 1).repeat(1, grid_w) / grid_h
    y_axis = torch.arange(grid_w).repeat(grid_h, 1) / grid_w

    geom = dict(shift_x=torch.rand(1, 3, 1, 1, 1),
                shift_y=torch.rand(1, 3, 1, 1, 1),
                semi_axis_x=torch.rand(1, 3, 1, 1, 1) + 0.1,
                semi_axis_y=torch.rand(1, 3, 1, 1, 1) + 0.1,
                alpha=torch.rand(1, 3, 1, grid_h, grid_w),
                radius=torch.rand(1, 3, 1, 1, 1) + 0.1)
    scale = torch.full((1, 3, 3, 1, 1), 1.5)
    mask = filt.get_mask(x_axis, y_axis, scale_factor=scale, **geom)
    for instance in range(3):
        assert torch.equal(mask[0, instance, 0], mask[0, instance, 1])
        assert torch.equal(mask[0, instance, 0], mask[0, instance, 2])


# ------------------------------------------------------------------- the fuse

def test_two_neutral_branches_fuse_to_neutral():
    """S = 1 + (s_g - 1) + (s_e - 1) leaves a neutral image untouched."""
    torch.manual_seed(0)
    head = model.DeepLPFParameterPrediction()
    neutral = torch.ones(1, 3, 16, 16)
    head.graduated_filter.mask_from_input = lambda feat, img: neutral
    head.elliptical_filter.mask_from_input = lambda feat, img: neutral
    cubic = torch.full((1, 3, 16, 16), 0.25)
    head.cubic_filter.mask_from_input = lambda feat, img: cubic

    x = torch.zeros(1, 64, 16, 16)
    out = head(x)
    # img (Y1) is zero here, so the output is exactly the fused Y3 = S * Y2.
    assert torch.allclose(out, cubic)


# ------------------------------------------------------------------- MS-SSIM

def test_msssim_of_identical_images_is_one():
    loss = model.DeepLPFLoss()
    img = torch.rand(1, 1, 64, 64)
    assert loss.compute_msssim(img, img).item() == pytest.approx(1.0, abs=1e-4)


def test_msssim_matches_the_reference_grouping():
    torch.manual_seed(0)
    loss = model.DeepLPFLoss()
    a, b = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)

    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    x, y = a.clone(), b.clone()
    ssims, mcs = [], []
    for _ in range(weights.numel()):
        ssim_val, cs = loss.compute_ssim(x, y)
        ssims.append(ssim_val)
        mcs.append(cs)
        x = torch.nn.functional.avg_pool2d(x, (2, 2))
        y = torch.nn.functional.avg_pool2d(y, (2, 2))
    ssims = (torch.stack(ssims) + 1) / 2
    mcs = (torch.stack(mcs) + 1) / 2
    want = torch.prod(mcs[:-1] ** weights[:-1]) * (ssims[-1] ** weights[-1])

    assert loss.compute_msssim(a, b).item() == pytest.approx(want.item(), rel=1e-5)


# ------------------------------------------------------- colour space, metrics

def test_rgb_to_lab_against_skimage():
    from skimage.color import rgb2lab
    torch.manual_seed(0)
    img = torch.rand(3, 16, 16)
    got = util.ImageProcessing.rgb_to_lab(img.clone())

    # rgb_to_lab returns CxHxW with each channel rescaled to [0, 1].
    got = got.permute(1, 2, 0).detach().numpy()
    got = np.stack([got[..., 0] * 100.0,
                    (got[..., 1] * 2 - 1) * 110.0,
                    (got[..., 2] * 2 - 1) * 110.0], axis=-1)
    want = rgb2lab(img.permute(1, 2, 0).numpy())
    assert np.allclose(got, want, atol=1e-2)


def test_compute_psnr_against_skimage():
    from skimage.metrics import peak_signal_noise_ratio
    rng = np.random.default_rng(0)
    a = rng.random((1, 3, 32, 32)).astype(np.float32)
    b = rng.random((1, 3, 32, 32)).astype(np.float32)
    got = util.ImageProcessing.compute_psnr(a, b, 1.0)
    want = peak_signal_noise_ratio(a[0], b[0], data_range=1.0)
    assert got == pytest.approx(want, rel=1e-4)


# ---------------------------------------------------------------- data loading

def _write_image(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)


def test_ids_match_filenames_with_and_without_a_dash(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    _write_image(str(tmp_path / "input" / "a0001-jmac_DSC1459.png"), pixels)
    _write_image(str(tmp_path / "output" / "a0001-jmac_DSC1459.png"), pixels)
    _write_image(str(tmp_path / "input" / "sunset.png"), pixels)
    _write_image(str(tmp_path / "output" / "sunset.png"), pixels)
    ids = tmp_path / "ids.txt"
    ids.write_text("a0001\nsunset\n")

    loader = data.Adobe5kDataLoader(str(tmp_path), str(ids))
    assert len(loader.load_data()) == 2


def test_missing_target_is_reported_but_optional(tmp_path):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    _write_image(str(tmp_path / "input" / "sunset.png"), pixels)
    ids = tmp_path / "ids.txt"
    ids.write_text("sunset\n")

    with pytest.raises(FileNotFoundError):
        data.Adobe5kDataLoader(str(tmp_path), str(ids)).load_data()

    loaded = data.Adobe5kDataLoader(str(tmp_path), str(ids)).load_data(
        require_output=False)
    assert loaded[0]['output_img'] is None

    dataset = data.Dataset(data_dict=loaded, normaliser=255, is_inference=True)
    sample = dataset[0]
    assert 'output_img' not in sample
    assert sample['input_img'].shape == (3, 8, 8)


def test_no_matching_ids_raises(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("nothing_here\n")
    with pytest.raises(FileNotFoundError):
        data.Adobe5kDataLoader(str(tmp_path), str(ids)).load_data()


@pytest.mark.parametrize("mode,shape", [("L", (8, 8)), ("RGBA", (8, 8, 4))])
def test_greyscale_and_rgba_images_load_as_three_channels(tmp_path, mode, shape):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, shape, dtype=np.uint8)
    path = str(tmp_path / ("img_%s.png" % mode))
    Image.fromarray(pixels, mode=mode).save(path)
    img = util.ImageProcessing.load_image(path, normaliser=255)
    assert img.shape == (8, 8, 3)


# ------------------------------------------------------------------ end to end

def test_forward_pass_runs_on_cpu():
    torch.manual_seed(0)
    net = model.DeepLPFNet()
    net.eval()
    img = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = net(img)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


def test_backward_pass_produces_finite_gradients():
    torch.manual_seed(0)
    net = model.DeepLPFNet()
    criterion = model.DeepLPFLoss(ssim_window_size=5)
    img = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    loss = criterion(torch.clamp(net(img), 0, 1), target)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_get_device_returns_a_device():
    assert isinstance(util.get_device(), torch.device)
