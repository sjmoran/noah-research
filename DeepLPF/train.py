# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
"""The training loop: data, optimiser, epochs, evaluation and checkpoints."""
import logging
import os

import torch
import torch.optim as optim

import metric
import model
from data import Adobe5kDataLoader, Dataset


def run_training(args, device, log_dirpath, writer):
    """Train a DeepLPF model and checkpoint it whenever validation PSNR improves.

    :param args: parsed command-line arguments
    :param device: device to train on
    :param log_dirpath: directory for checkpoints, logs and evaluation output
    :param writer: TensorBoard summary writer
    :returns: N/A

    """
    num_epoch = args.num_epoch
    BATCH_SIZE = 1
    valid_every = args.valid_every
    training_img_dirpath = args.training_img_dirpath
    train_img_list_path = args.train_img_list_path
    valid_img_list_path = args.valid_img_list_path
    test_img_list_path = args.test_img_list_path


    training_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                             img_ids_filepath=train_img_list_path)
    training_data_dict = training_data_loader.load_data()

    training_dataset = Dataset(data_dict=training_data_dict, normaliser=1,
                               is_valid=False)

    validation_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                           img_ids_filepath=valid_img_list_path)
    validation_data_dict = validation_data_loader.load_data()
    validation_dataset = Dataset(data_dict=validation_data_dict, normaliser=1, is_valid=True)

    testing_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                        img_ids_filepath=test_img_list_path)
    testing_data_dict = testing_data_loader.load_data()
    testing_dataset = Dataset(data_dict=testing_data_dict, normaliser=1,is_valid=True)

    # num_workers is part of the seeded recipe: the per-worker RNG that
    # draws the augmentation flips is seeded from (base_seed + worker_id),
    # so changing the worker count changes which image gets which flip.
    # pin_memory only changes where the host copy lives. Workers are off by
    # default because an exception raised while loading an image is otherwise
    # re-raised from the worker with the original traceback lost.
    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=0, pin_memory=True)
    # Evaluation runs at batch size 1 so per-image metrics are reported/saved.
    testing_data_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False,
                                                  num_workers=0)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1,
                                                     shuffle=False,
                                                     num_workers=0)
    net = model.DeepLPFNet()
    if args.checkpoint_filepath is not None:
        # Fine-tuning from a saved model. This flag used to be read only on the
        # inference path, so passing it here silently trained from scratch and
        # the run looked identical to one that had never seen the checkpoint.
        net.load_state_dict(torch.load(args.checkpoint_filepath,
                                       map_location='cpu'))
        logging.info('Initialised the network from ' + args.checkpoint_filepath
                     + ' (weights only; the optimiser starts fresh)')
    net.to(device)

    logging.info('######### Network created #########')
    logging.info('Architecture:\n' + str(net))

    # Paper Eq. 8 / Sec. 4.1: L1 in CIELab + 1e-3 * (1 - MS-SSIM) on the L channel.
    criterion = model.DeepLPFLoss(ssim_window_size=5)

    '''
    The following objects allow for evaluation of a model on the testing and validation splits of a dataset
    '''
    validation_evaluator = metric.Evaluator(
        criterion, validation_data_loader, "valid", log_dirpath)
    testing_evaluator = metric.Evaluator(
        criterion, testing_data_loader, "test", log_dirpath)

    # Adam with lr 1e-4, as in the paper's implementation details (Sec. 4.1).
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

    best_valid_psnr = 0.0

    optimizer.zero_grad()
    net.train()

    total_examples = 0  # running count over all epochs; x-axis of the per-batch loss curve

    for epoch in range(num_epoch):

        # Train loss
        examples = 0.0
        running_loss = 0.0

        for batch_num, data in enumerate(training_data_loader, 0):

            input_img_batch = data['input_img'].to(device, non_blocking=True)
            gt_img_batch = data['output_img'].to(device, non_blocking=True)

            # Forward, loss, backward, Adam.
            net_img_batch = torch.clamp(net(input_img_batch), 0.0, 1.0)
            loss = criterion(net_img_batch, gt_img_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # One device-to-host sync per step, not two.
            loss_value = loss.item()
            running_loss += loss_value
            examples += BATCH_SIZE
            total_examples+=BATCH_SIZE

            writer.add_scalar('Loss/train', loss_value, total_examples)

        logging.info('[%d] train loss: %.15f' %
                     (epoch + 1, running_loss / examples))
        writer.add_scalar('Loss/train_smooth', running_loss / examples, epoch + 1)

        if (epoch + 1) % valid_every == 0:

            logging.info("Evaluating model on validation and test dataset")

            valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(
                net, epoch)
            test_loss, test_psnr, test_ssim = testing_evaluator.evaluate(
                net, epoch)

            # Checkpoint whenever validation PSNR improves (model selection is on the validation split).
            if valid_psnr > best_valid_psnr:

                # Evaluator.evaluate returns plain floats, so use them
                # directly. This read valid_loss.tolist()[0], which worked
                # only while running_loss accumulated 1-element tensors.
                snapshot_name = 'deeplpf_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(
                    valid_psnr, valid_loss, test_psnr, test_loss, epoch)
                logging.info(
                    "Validation PSNR has increased. Saving the more accurate model to file: " + snapshot_name)

                best_valid_psnr = valid_psnr
                torch.save(net.state_dict(), os.path.join(log_dirpath, snapshot_name))

            # Unconditional schedule, independent of the selection rule
            # above. The best-validation checkpoint is an argmax over
            # near-tied values, so it carries selection noise of the same
            # order as the effects being compared; a fixed schedule gives
            # a population to average over and to quote a spread from.
            if args.checkpoint_every and \
                    (epoch + 1) % (valid_every * args.checkpoint_every) == 0:
                sched_name = 'sched_validpsnr_{}_testpsnr_{}_epoch_{}_model.pt'.format(
                    valid_psnr, test_psnr, epoch)
                torch.save(net.state_dict(), os.path.join(log_dirpath, sched_name))

            net.train()

    # A final pass over the test split, unless the last epoch was already an
    # evaluation epoch, in which case this would score the same weights twice.
    # It is labelled with the epoch it actually ran at: passing epoch=0 wrote
    # the finished model's per-image scores into test_per_image.csv under epoch
    # 1, where anyone reading that file takes them for the first epoch's.
    if num_epoch % valid_every != 0:
        testing_evaluator.evaluate(net, epoch=num_epoch - 1)

    snapshot_prefix = os.path.join(log_dirpath, 'deep_lpf')
    snapshot_path = snapshot_prefix + "_" + str(num_epoch)
    torch.save(net.state_dict(), snapshot_path)
