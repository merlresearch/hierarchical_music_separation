#!/usr/bin/env python3
# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import itertools

import torch
import torch.nn.functional as F


def _sanitize_stack(var):
    if type(var) is list:
        return F.vstack(var)
    return var


def _sanitize_pad(var):
    if type(var) is list:
        return F.pad_sequence(var)
    return var


def l1(estimates, targets, normalize=True, axis=None, **kwargs):
    estimates, targets = _sanitize_pad(estimates), _sanitize_pad(targets)
    if normalize:
        return torch.mean(torch.abs(estimates - targets), dim=axis)
    else:
        return torch.sum(torch.abs(estimates - targets), dim=axis)


def l2square(estimates, targets, normalize=True, axis=None, **kwargs):
    estimates, targets = _sanitize_pad(estimates), _sanitize_pad(targets)
    if normalize:
        return torch.mean(torch.pow(estimates - targets, 2), dim=axis)
    else:
        return torch.sum(torch.pow(estimates - targets, 2), dim=axis)


def mask_applier(mask_estimates, mix_spectrograms):
    xp = mask_estimates[0].xp
    est_spectrograms = [
        mask * xp.broadcast_to(xp.expand_dims(mix, -1), mask.shape)
        for mix, mask in zip(mix_spectrograms, mask_estimates)
    ]
    return est_spectrograms


def _sanitize_loss(loss_type):
    if type(loss_type) is str:
        from .utils import get_loss_func_by_name

        loss_func = get_loss_func_by_name(loss_type)
    elif callable(loss_type):
        loss_func = loss_type
    else:
        raise ValueError(
            "Parameter loss_type is expected to be a string (choose from l1 / l2square)"
            "or a callable function! Got type {}!".format(type(loss_type))
        )
    return loss_func


def maskinf_loss(mask_estimates, target_dict, loss_type="l1", permutation_search=False, normalize=True):
    target_tf = target_dict["mask"]
    mix_stft = target_dict["mix_spec"] if "mix_spec" in target_dict else None
    if mix_stft:
        est_masks = mask_applier(mask_estimates, mix_stft)
    else:
        est_masks = mask_estimates

    if permutation_search:
        loss_mask = find_best_signal_permutation(
            est_masks, target_tf, loss_type=loss_type, permutation_axis=3, normalize_loss=normalize
        )
    else:
        f_mask_loss = _sanitize_loss(loss_type)
        loss_mask = f_mask_loss(est_masks, target_tf, normalize=normalize)

    return loss_mask


def find_best_signal_permutation(estimates, targets, loss_type="l1", permutation_axis=0, normalize_loss=False):
    """
    Performs permutation invariant training (PIT); permutes every utterances by source to find the lowest
    possible loss.
    Args:
        normalize_loss:
        estimates: shape = [n_src, batch, time_step]
        targets: shape = [n_src, batch, time_step]
        loss_type (str, or callable): Either a string

        permutation_axis:

    Returns:

    """

    loss_func = _sanitize_loss(loss_type)
    if isinstance(estimates, list):
        xp = estimates[0].xp
        normalizer = xp.sum(xp.array([e.size for e in estimates]))
    else:
        xp = estimates.xp
        normalizer = estimates.size
    estimates, targets = _sanitize_pad(estimates), _sanitize_pad(targets)

    num_clusters = estimates.shape[permutation_axis]
    # sum over all but the first (batch) dimension for permutation search
    sum_axes = tuple(range(1, len(estimates.shape)))
    loss_perms = []
    for p in itertools.permutations(range(num_clusters)):
        loss = loss_func(
            F.permutate(estimates, xp.array(p), axis=permutation_axis), targets, axis=sum_axes, normalize=False
        )
        loss_perms.append(loss)

    loss_perms = F.stack(loss_perms, axis=1)
    min_loss_perm = torch.min(loss_perms, axis=1)

    if normalize_loss:
        loss = torch.sum(min_loss_perm) / normalizer
    else:
        # some models (e.g. TASNet) return a normalized loss
        # so we only need to normalize over batch size
        loss = torch.mean(min_loss_perm)

    return loss


class LossFuncWrapper(object):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*args, *self.args, **kwargs, **self.kwargs)
