#!/usr/bin/env python3

# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from datetime import datetime

import GPUtil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from hierarchical.datasets.utils import dataset_creator
from hierarchical.eval import evaluate
from hierarchical.loss_functions.utils import loss_creator
from hierarchical.utils.general import batch_to_device, create_model, load_model, make_np, save_best_model, seed
from hierarchical.utils.plotting import plot_anchors, plt_all2

# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn', force=True)
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def main(exp_dict):
    exp_name = exp_dict["experiment_name"]
    output_dir = exp_dict["output_dir"]
    params = exp_dict["parameters"]

    # Training params
    epochs = int(params["epochs"])
    batch_size = int(params["batch_size"])
    permitted_violations = int(params["permitted_violations"])
    model_save_dir = os.path.join(output_dir, params["model_save_dir"])
    autoclip = params.get("autoclip_percentile", None)
    os.makedirs(model_save_dir, exist_ok=True)
    report_first_epochs = 3
    report_per_iter = 10
    usr_seed = int(params.get("seed", 588))
    seed(usr_seed)

    # Optimizer params, adam
    alpha = float(params["alpha"])
    beta1 = float(params["beta1"])
    beta2 = float(params["beta2"])
    eps = float(params["eps"])
    lr_decay = float(params["lr_decay"])

    # Set up logging
    logfile = os.path.join(output_dir, "{}".format(exp_name))
    logfile += f"_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}.log"
    print(f"Logfile: {logfile}")
    logger.add(sink=logfile, level="DEBUG", format="{time} | {level} | {message}")
    logger.info("Running Experiment")
    [logger.info(f"{k} = {v}") for k, v in exp_dict.items() if not isinstance(v, dict)]
    [logger.info(f"{k} = {v}") for k, v in params.items()]
    writer = SummaryWriter(log_dir=output_dir)
    writer.add_text("experiment_dict", str(exp_dict), -1)
    [writer.add_text(k, str(v), -1) for k, v in exp_dict.items() if not isinstance(v, dict)]
    [writer.add_text(k, str(v), -1) for k, v in params.items()]

    # Check to see which device to use
    if torch.cuda.is_available():
        avail_gpu = GPUtil.getAvailable(
            order="first", limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[]
        )[0]
        device = torch.device(f"cuda:{avail_gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using {device}")

    _, target, t_kwargs = loss_creator(params)

    # Make datasets
    tr_ds = dataset_creator(params, "tr", target, t_kwargs)
    cv_ds = dataset_creator(params, "cv", target, t_kwargs)

    # Make data loaders
    n_workers = params["num_workers"]
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    cv_loader = torch.utils.data.DataLoader(cv_ds, batch_size=batch_size, num_workers=n_workers)

    # Make the model and optimizer
    model = create_model(params).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=alpha, betas=(beta1, beta2), eps=eps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=permitted_violations - 1, factor=lr_decay, verbose=True
    )

    # Make losses
    loss = nn.L1Loss()

    # Setup for main training loop
    iterations = 0
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    epoch = 0
    train_start = datetime.now()
    train_loss = np.inf
    validation_loss = np.inf
    grad_history = []

    # Main training loop
    logger.info("Starting training...")
    logger.info("")
    for epoch in range(epoch, epochs):
        logger.info("")
        logger.info("~" * 75)

        # Training iterations
        epoch_start = datetime.now()
        train_return = train_loop(
            model,
            loss,
            optimizer,
            tr_loader,
            device,
            autoclip,
            False,
            epoch,
            epochs,
            writer,
            plots_dir,
            iterations,
            report_first_epochs,
            report_per_iter,
            grad_history,
        )
        iterations, mean_train_loss, grad_history = train_return

        # Validation iterations
        with torch.no_grad():
            _, mean_val_loss, _ = train_loop(
                model,
                loss,
                optimizer,
                cv_loader,
                device,
                None,
                True,
                epoch,
                epochs,
                writer,
                plots_dir,
                iterations,
                report_first_epochs,
                report_per_iter,
                grad_history,
            )

        # Log everything
        epoch_end = datetime.now()
        logger.info("~" * 50)
        logger.info(
            f"Finished epoch {epoch:3d} "
            f"({iterations:6d} iters):   "
            f"mean_train_loss={mean_train_loss:.10e}   "
            f"mean_val_loss={mean_val_loss:.10e}."
        )
        logger.info(f"Epoch took {epoch_end - epoch_start}. " f"Total elapsed time is {epoch_end - train_start}.")

        # ~~~~~~ Training curriculum ~~~~~~~

        # Check this before we step so that we're checking against the prev best
        if not scheduler.is_better(mean_val_loss, scheduler.best):
            logger.info(
                f"Validation loss worse. " f"{scheduler.num_bad_epochs+1} of {scheduler.patience+1} " f"violations."
            )
        else:
            save_best_model(model, epoch, train_loss, validation_loss, model_save_dir)
            logger.info("Model successful. Saved best model.")

        # step
        bad_epochs = scheduler.num_bad_epochs
        scheduler.step(mean_val_loss, epoch)

        # Reload model if val loss not decreasing
        if bad_epochs >= scheduler.patience:
            model, epoch, train_loss, validation_loss = load_model(params, device, model_save_dir)
            logger.info("Model stagnated. Reloading previous best model.")

    logger.info(f"Finished training. Took {datetime.now() - train_start}.")
    logger.info("Training script finished.")

    evaluate(exp_dict, writer)

    writer.close()
    logger.info(f"Experiment script finished. See results at {output_dir}")


def train_loop(
    model,
    loss_fn,
    optimizer,
    loader,
    device,
    autoclip_percentile,
    is_val,
    epoch,
    epochs,
    writer,
    plot_dir,
    iterations,
    report_first_epochs,
    report_per_iter,
    grad_history,
):
    all_losses = []
    get_arrs = lambda a: np.array([make_np(l) for l in a])
    if is_val:
        label = "Validation"
        model.eval()
    else:
        label = "Train"
        model.train()

    all_leaf_loss = []
    for idx, batch in enumerate(loader):
        feature_dict, target_dict = batch_to_device(batch, device)
        model_output = model(feature_dict)

        if isinstance(model_output, tuple):
            est_mask, anchor = model_output
        else:
            est_mask = model_output
            anchor = None

        gt_mask = target_dict["mask"]

        # plot periodically
        if should_plot(idx, epoch, epochs):
            do_plotting(batch, est_mask, anchor, label, 6, writer, epoch, plot_dir)

        if "leaf_zeroed" in target_dict:
            zero_mask = np.ones(est_mask.shape)
            for b, should_zero in enumerate(target_dict["leaf_zeroed"]):
                if should_zero:
                    zero_mask[b, :, :, 0] *= 0
            zero_mask = torch.from_numpy(zero_mask).to(device)
            est_mask = torch.mul(est_mask, zero_mask)
            gt_mask = torch.mul(gt_mask, zero_mask)

        loss = loss_fn(est_mask, gt_mask)
        with torch.no_grad():
            leaf_loss = loss_fn(est_mask[..., 0], gt_mask[..., 0]).detach()
            all_leaf_loss.append(leaf_loss)

        # Save losses
        all_losses.append(loss)

        if not is_val:
            # do optimizer step
            optimizer.zero_grad()
            loss.backward()
            if autoclip_percentile:
                obs_grad_norm = _get_grad_norm(model)
                grad_history.append(obs_grad_norm)
                clip_value = np.percentile(grad_history, autoclip_percentile)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            if epoch <= report_first_epochs and idx % report_per_iter == 0:
                logger.info(f"epoch: {epoch:3d}   " f"iterations: {iterations:6d}   " f"train_loss={loss:.10e}")

        iterations += 1
    mean_loss = get_arrs(all_losses).mean()
    writer.add_scalar(f"{label}/Mean Loss", mean_loss, epoch)

    mean_leaf_loss = get_arrs(all_leaf_loss).mean()
    writer.add_scalar(f"{label}/Mean Loss of Leaves", mean_leaf_loss, epoch)
    return iterations, mean_loss, grad_history


def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def do_plotting(batch, est_mask, anchor, name, num, writer, epoch, plots_dir):
    feature_dict, target_dict = batch
    gt_mask = target_dict["mask"]
    logger.info(f"Making {name} plots...")
    figlist = []
    for b in range(num):
        figlist.append(
            plt_all2(
                feature_dict["query"],
                gt_mask,
                est_mask,
                feature_dict["mix"],
                plots_dir,
                epoch,
                b,
                name,
                qry_id=feature_dict["query_id"][b],
            )
        )
    writer.add_images(f"{name}/examples", np.array(figlist)[:, :, :, 0:3], epoch, dataformats="NHWC")
    if anchor is not None:
        anchor_img = plot_anchors(anchor[:num], plots_dir, epoch)
        writer.add_image(f"{name}/Anchors", anchor_img[:, :, 0:3], epoch, dataformats="HWC")
    logger.info(f"Finished {name} plots...")


def should_plot(idx, epoch, epochs):
    return idx == 0 and (epoch % 5 == 0 or epoch == epochs - 1)


if __name__ == "__main__":

    main()
