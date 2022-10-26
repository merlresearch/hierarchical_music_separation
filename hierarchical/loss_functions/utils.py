# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect

import torch.nn as nn

import hierarchical.loss_functions.general
import hierarchical.loss_functions.targets

from .general import maskinf_loss
from .targets import anchor_dan_target


def get_loss_func_by_name(name):
    return _get_func_by_name(name, hierarchical.loss_functions.general)


def get_target_by_name(name):
    if name.lower() == "ma":
        return hierarchical.loss_functions.targets.mask_approximation
    elif name.lower() == "msa":
        return hierarchical.loss_functions.targets.magnitude_spectrogram_approx
    elif name.lower() == "psa":
        return hierarchical.loss_functions.targets.phase_spectrogram_approx

    return _get_func_by_name(name, hierarchical.loss_functions.targets)


def _get_func_by_name(name, module):
    return dict(inspect.getmembers(module, inspect.isfunction))[name]


def loss_creator(params):
    loss_type = params["loss_type"]
    if loss_type == "mask_inference":
        loss_func = maskinf_loss
        mask_loss = params["mask_loss"]
        permutation_invariant = params["permutation_invariant"]
        loss_kwargs = {"loss_type": mask_loss, "permutation_search": permutation_invariant}

        target_type = params["target_type"]
        target_func = get_target_by_name(target_type)
        target_kwargs = {"rmax": params["target_trunc_max"]}

    elif loss_type in ["qbe_anchor_dan", "anchor_dan"]:
        loss_func = nn.L1Loss()
        mask_loss = params["mask_loss"]
        loss_kwargs = {"loss_type": mask_loss}

        target_func = anchor_dan_target
        target_kwargs = {
            "target_type": params["target_type"],
            "rmax": params["target_trunc_max"],
        }

    else:
        raise ValueError("Loss function of type {} not currently supported".format(loss_type))

    return loss_func, target_func, target_kwargs
