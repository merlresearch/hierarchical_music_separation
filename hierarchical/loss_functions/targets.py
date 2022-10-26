# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

EPSILON = 1e-16


def irm(srcs, eps=EPSILON):
    """
    Ideal ratio mask
    :param srcs: Complex valued np.array of shape [n_time, n_freq, n_srcs]
    :param eps: epsilon (float)
    :return:
    """
    n_time, n_freq, n_srcs = srcs.shape
    all_srcs = np.sum(np.abs(srcs) + eps, axis=2, keepdims=True)
    all_srcs = np.tile(all_srcs, (1, 1, n_srcs))
    return np.abs(srcs) / all_srcs


def ibm(srcs, eps=EPSILON):
    """
    Ideal Binary Mask

    Trick from here: https://stackoverflow.com/a/44606706
    :param srcs: Complex valued np.array of shape [n_time, n_freq, n_srcs]
    :param eps:
    :return:
    """
    masks = irm(srcs, eps)
    amax = np.argmax(masks, axis=2)
    i, j = np.indices(amax.shape)
    result = np.zeros_like(masks)
    result[i, j, amax] = 1
    return result


def wiener_like(srcs, eps=EPSILON):
    """
    Wiener-like mask

    :param srcs: Complex valued np.array of shape [n_time, n_freq, n_srcs]
    :param eps:
    :return:
    """
    srcs_mag = np.square(np.abs(srcs))
    all_srcs = np.sum(srcs_mag, axis=2, keepdims=True) + eps
    all_srcs = np.broadcast_to(all_srcs, srcs_mag.shape)
    return srcs_mag / all_srcs


def iam(mix_stft, srcs_stft, rmax=np.inf, eps=EPSILON):
    """
    Ideal amplitude mask

    :param mix_stft: Complex valued np.array of shape [n_time, n_freq]
    :param srcs_stft: Complex valued np.array of shape [n_time, n_freq, n_srcs]
    :param rmax:
    :param eps:
    :return:
    """
    mix_mag, srcs_mag = np.abs(mix_stft), np.abs(srcs_stft)
    mix_mag = np.broadcast_to(np.expand_dims(mix_mag, 2), srcs_mag.shape)
    iam = srcs_mag / (mix_mag + eps)
    return np.fmin(iam, rmax)


def psf(mix_stft, srcs_stft, rmin=-np.inf, rmax=np.inf, eps=EPSILON):
    """
    Phase sensitive filter

    :param mix_stft: Complex valued np.array of shape [n_time, n_freq]
    :param srcs_stft: Complex valued np.array of shape [n_time, n_freq, n_srcs]
    :param rmin:
    :param rmax:
    :param eps:
    :return:
    """
    mix_stft = np.broadcast_to(np.expand_dims(mix_stft, 2), srcs_stft.shape)
    theta = np.angle(srcs_stft / (mix_stft + eps))
    mix_mag, srcs_mag = np.abs(mix_stft), np.abs(srcs_stft)
    filt = np.cos(theta) * (srcs_mag / (mix_mag + eps))
    return np.fmin(np.fmax(filt, rmin), rmax)


def mask_approximation(mix_stft, srcs_stft, mask_type=None, rmax=1.0, rmin=0.0, eps=EPSILON):
    if mask_type == "ibm":
        all_masks = ibm(srcs_stft, eps)
    elif mask_type == "irm":
        all_masks = irm(srcs_stft, eps)
    elif mask_type == "iam":
        all_masks = iam(mix_stft, srcs_stft, rmax=rmax, eps=eps)
    elif mask_type == "psf":
        all_masks = psf(mix_stft, srcs_stft, rmin=rmin, rmax=rmax, eps=eps)
    elif mask_type == "wf":
        all_masks = wiener_like(srcs_stft, eps)
    else:
        raise ValueError("Unknown mask type: {}!".format(mask_type))

    return {"mask": all_masks}


def magnitude_spectrogram_approx(mix_stft, srcs_stft, rmax=1):
    n_time, n_freq, n_srcs = srcs_stft.shape
    mix_mag, srcs_mag = np.abs(mix_stft), np.abs(srcs_stft)
    mix_mag_extend = np.tile(np.expand_dims(mix_mag, 2), (1, 1, n_srcs))
    result = np.fmin(mix_mag_extend * rmax, srcs_mag)
    return {"mask": result, "mix_spec": mix_mag}


def phase_spectrogram_approx(mix_stft, srcs_stft, rmax=1):
    n_time, n_freq, n_srcs = srcs_stft.shape
    mix_stft = np.tile(np.expand_dims(mix_stft, 2), (1, 1, n_srcs))
    mix_mag, mix_phase = np.abs(mix_stft), np.angle(mix_stft)
    srcs_mag, srcs_phase = np.abs(srcs_stft), np.angle(srcs_stft)
    result = np.fmax(0.0, np.fmin(rmax * mix_mag, srcs_mag * np.cos(srcs_phase - mix_phase)))
    return {"mask": result, "mix_spec": mix_mag[:, :, 0]}


def dpcl_mag_weights(mix_stft):
    n_time, n_freq = mix_stft.shape
    mag = np.abs(mix_stft)
    return np.sqrt((mag / np.sum(mag)) * n_time * n_freq)


def dpcl_binary_weights(srcs_stft, threshold_db=-40.0):
    # n_time, n_freq, n_srcs = srcs_stft.shape
    srcs_db = 20 * np.log10(np.abs(srcs_stft) + EPSILON)
    max_db = np.max(srcs_db, axis=(0, 1), keepdims=True)
    any_src_above = np.any((srcs_db - max_db) > threshold_db, axis=2)
    return any_src_above.astype(np.float32)


def input_feature(mix_stft, input_feat, whiten_mean=None, whiten_std=None):
    if input_feat == "mag":
        result = np.abs(mix_stft)
    elif input_feat == "stft" or input_feat == "wvfm":
        result = mix_stft
    elif input_feat == "pow":
        result = np.square(mix_stft)
    elif input_feat == "logmag":
        result = np.log(np.abs(mix_stft) + 1e-20)
    elif input_feat == "db":
        # Converts a STFT to dB
        # This is adapted from librosa's power_to_db()
        ref = 1.0
        amin = 1e-10
        top_db = 80.0
        log_spec = 10.0 * np.log10(np.maximum(amin, np.square(mix_stft)))
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref))

        result = np.maximum(log_spec, log_spec.max() - top_db)
    else:
        raise ValueError("Unknown mixture feature type: {}!".format(input_feat))

    result = result - whiten_mean if whiten_mean is not None else result
    result = result / whiten_std if whiten_std is not None else result
    return result


def anchor_dan_target(mix_stft, srcs_stft, target_type, rmax=1, **target_func_kwargs):
    # Lazy load to prevent circular imports
    from hierarchical.loss_functions.utils import get_target_by_name

    targ_func = get_target_by_name(target_type)
    target_dict = targ_func(mix_stft, srcs_stft, rmax=rmax, **target_func_kwargs)
    target_dict["target_presence"] = dpcl_binary_weights(srcs_stft)

    return target_dict
