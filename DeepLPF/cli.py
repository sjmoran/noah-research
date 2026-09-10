# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
"""Command-line arguments for training and inference.

Split out of ``main.py``: the flag definitions are most of what a reader
scrolls past, so they sit apart from what the program does with them.
"""
import argparse


def build_parser():
    """Build the argument parser for :func:`main.main`.

    :returns: the parser, with every training and inference flag registered
    :rtype: argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser(
        description="Train the DeepLPF neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, required=False,
        help="Number of epochs (default 100000)", default=100000)
    parser.add_argument(
        "--valid_every", type=int, required=False,
        help="Number of epochs after which to compute validation accuracy",
        default=25)
    parser.add_argument(
        "--checkpoint_filepath", required=False,
        help="Location of checkpoint file", default=None)
    parser.add_argument(
        "--inference_img_dirpath", required=False,
        help="Directory containing images to run through a saved DeepLPF model instance",
        default=None)
    parser.add_argument(
        "--training_img_dirpath", required=False,
        help="Directory containing images to train a DeepLPF model instance",
        default="/home/sjm213/adobe5k/adobe5k/")
    parser.add_argument(
        "--inference_img_list_path", required=False,
        help="Plain text file containing the names of the images to inference")
    parser.add_argument(
        "--train_img_list_path", required=False,
        help="Plain text file containing the names of the training images")
    parser.add_argument(
        "--valid_img_list_path", required=False,
        help="Plain text file containing the names of the validation images")
    parser.add_argument(
        "--test_img_list_path", required=False,
        help="Plain text file containing the names of the test images")

    parser.add_argument(
        "--checkpoint_every", type=int, required=False, default=None,
        help="Also save a checkpoint every N validation rounds regardless of "
             "whether validation improved. Saving only on improvement leaves "
             "late training almost unsampled, so there is nothing to average "
             "over and no way to estimate run-to-run variability.")
    parser.add_argument(
        "--seed", type=int, required=False, default=None,
        help="Seed for torch, numpy and Python RNGs. Without it every run "
             "starts from a different initialisation and shuffle order, so "
             "arms of an ablation differ by luck as well as by the change "
             "under test. Set it for any comparison between runs.")

    return parser
