# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import scipy.fftpack as scifft
import scipy.signal
import torch


def sqrt_hann(N):
    """Implement a sqrt-Hann window"""
    return np.sqrt(scipy.signal.hann(N, sym=False))


def get_num_fft_bins(fft_size):
    return fft_size // 2 + 1


def _get_window(win_type, length):
    if win_type == "sqrt_hann":
        return np.sqrt(scipy.signal.hann(length, sym=False))
    else:
        return scipy.signal.get_window(win_type, length)


def do_stft(wav, sample_rate=16000, n_fft=1024, hop_len=256, win_type="sqrt_hann", return_times=False):
    times, stft_ = e_stft(wav, n_fft, hop_len, win_type, sample_rate)
    stft_ = stft_.astype("complex64")
    if return_times:
        return times, stft_
    else:
        return stft_


def do_istft(stft, sample_rate=16000, n_fft=1024, hop_len=256, win_type="sqrt_hann"):
    """

    :return:
    """
    del sample_rate
    return e_istft(stft, n_fft, hop_len, win_type)


def compute_sdr(estimated_signal, reference_signals, source_idx, scaling=True):
    references_projection = reference_signals.T @ reference_signals
    source = reference_signals[:, source_idx]
    scale = (source @ estimated_signal) / references_projection[source_idx, source_idx] if scaling else 1

    e_true = scale * source
    e_res = estimated_signal - e_true

    signal = (e_true**2).sum()
    noise = (e_res**2).sum()
    SDR = 10 * np.log10(signal / noise)

    references_onto_residual = np.dot(reference_signals.transpose(), e_res)
    b = np.linalg.solve(references_projection, references_onto_residual)

    e_interf = np.dot(reference_signals, b)
    e_artif = e_res - e_interf

    SIR = 10 * np.log10(signal / (e_interf**2).sum())
    SAR = 10 * np.log10(signal / (e_artif**2).sum())
    return SDR, SIR, SAR


def compute_measures(estimated_signal, reference_signals, j, scaling=True):
    Rss = np.dot(reference_signals.transpose(), reference_signals)
    this_s = reference_signals[:, j]

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss[j, j]
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * np.log10(Sss / Snn)

    # Get the SIR
    Rsr = np.dot(reference_signals.transpose(), e_res)
    b = np.linalg.solve(Rss, Rsr)

    e_interf = np.dot(reference_signals, b)
    e_artif = e_res - e_interf

    SIR = 10 * np.log10(Sss / (e_interf**2).sum())
    SAR = 10 * np.log10(Sss / (e_artif**2).sum())

    return SDR, SIR, SAR


def _do_si_sdr(estimate, reference, mixture, scaling):
    noise_ref = mixture - reference
    mag_y_ref = np.abs(reference)
    mag_snr = 10 * np.log10((mag_y_ref**2).sum() / ((np.abs(estimate) - mag_y_ref) ** 2).sum())
    ref = np.stack([reference, noise_ref]).T

    def whiten(sig, ax):
        return sig - sig.mean(axis=ax)

    est = whiten(estimate, 0)
    ref = whiten(ref, 0)

    return compute_measures(est, ref, 0, scaling=scaling) + (mag_snr,)


def apply_mask(mix_stft, mask, should_normalize=False):
    """

    :param mix_stft:
    :param mask:
    :return:
    """
    mag, phase = np.abs(mix_stft), np.angle(mix_stft)
    if should_normalize:
        mask /= mag + 1e-12
    src_mag = mag * mask
    src_stft = src_mag * np.exp(1j * phase)
    return src_stft


def detach(arr):
    if isinstance(arr, torch.Tensor):
        return arr.cpu().detach().numpy()
    return arr


def get_output(mix_stft, masks, idx=0, sample_rate=16000, n_fft=1024, hop_len=256, win_type="sqrt_hann"):
    """

    masks have shape [batch==1, n_time, n_fft, n_src]

    :param mix_stft:
    :param masks:
    :return:
    """
    batch, n_time, _, n_src = masks.shape
    mix_stft = detach(mix_stft)
    masks = detach(masks)

    src_masks = [masks[idx, ..., s] for s in range(n_src)]
    src_stfts = [apply_mask(mix_stft, m) for m in src_masks]
    src_wavs = [do_istft(s.T, sample_rate, n_fft, hop_len, win_type) for s in src_stfts]
    return src_wavs


def e_stft(
    signal,
    window_length,
    hop_length,
    window_type,
    sample_rate=16000,
    n_fft_bins=None,
    remove_reflection=True,
    remove_padding=False,
):
    """ """
    if n_fft_bins is None:
        n_fft_bins = window_length

    window_type = "sqrt_hann" if window_type is None else window_type
    window = _get_window(window_type, window_length)

    orig_signal_length = len(signal)
    signal, num_blocks = _add_zero_padding(signal, window_length, hop_length)
    sig_zero_pad = len(signal)
    # figure out size of output stft
    stft_bins = n_fft_bins // 2 + 1 if remove_reflection else n_fft_bins  # only want just over half of each fft

    # this is where we do the stft calculation
    stft = np.zeros((num_blocks, stft_bins), dtype=complex)
    for hop in range(num_blocks):
        start = hop * hop_length
        end = start + window_length
        unwindowed_signal = signal[start:end]
        windowed_signal = np.multiply(unwindowed_signal, window)
        fft = scifft.fft(windowed_signal, n=n_fft_bins)
        stft[
            hop,
        ] = fft[:stft_bins]

    # reshape the 2d array, so it's (n_fft, n_hops).
    stft = stft.T
    stft = _remove_stft_padding(stft, orig_signal_length, window_length, hop_length) if remove_padding else stft

    time_vector = np.array(list(range(stft.shape[1])))
    hop_in_secs = hop_length / (1.0 * sample_rate)
    time_vector = np.multiply(hop_in_secs, time_vector)

    return time_vector, stft


def _add_zero_padding(signal, window_length, hop_length):
    """

    Args:
        signal:
        window_length:
        hop_length:
    Returns:
    """
    original_signal_length = len(signal)
    overlap = window_length - hop_length
    num_blocks = np.ceil(len(signal) / hop_length)

    if overlap >= hop_length:  # Hop is less than 50% of window length
        overlap_hop_ratio = np.ceil(overlap / hop_length)

        before = int(overlap_hop_ratio * hop_length)
        after = int((num_blocks * hop_length + overlap) - original_signal_length)

        signal = np.pad(signal, (before, after), "constant", constant_values=(0, 0))
        extra = overlap

    else:
        after = int((num_blocks * hop_length + overlap) - original_signal_length)
        signal = np.pad(signal, (hop_length, after), "constant", constant_values=(0, 0))
        extra = window_length

    num_blocks = int(np.ceil((len(signal) - extra) / hop_length))
    num_blocks += 1 if overlap == 0 else 0  # if no overlap, then we need to get another hop at the end

    return signal, num_blocks


def _remove_stft_padding(stft, original_signal_length, window_length, hop_length):
    """

    Args:
        stft:
        original_signal_length:
        window_length:
        hop_length:

    Returns:

    """
    overlap = window_length - hop_length
    first = int(np.ceil(overlap / hop_length))
    num_col = int(np.ceil((original_signal_length - window_length) / hop_length))
    stft_cut = stft[:, first : first + num_col]
    return stft_cut


def e_istft(stft, window_length, hop_length, window_type, reconstruct_reflection=True, remove_padding=True):

    n_hops = stft.shape[1]
    overlap = window_length - hop_length
    signal_length = (n_hops * hop_length) + overlap
    signal = np.zeros(signal_length)

    norm_window = np.zeros(signal_length)
    window = _get_window(window_type, window_length)

    # Add reflection back
    stft = _add_reflection(stft) if reconstruct_reflection else stft

    for n in range(n_hops):
        start = n * hop_length
        end = start + window_length
        inv_sig_temp = np.real(scifft.ifft(stft[:, n])) * window
        signal[start:end] += inv_sig_temp[:window_length]
        norm_window[start:end] = norm_window[start:end] + window**2

    norm_window[norm_window == 0.0] = 1e-12  # Prevent dividing by zero
    signal_norm = signal / norm_window

    # remove zero-padding
    if remove_padding:
        if overlap >= hop_length:
            ovp_hop_ratio = int(np.ceil(overlap / hop_length))
            start = ovp_hop_ratio * hop_length
            end = signal_length - overlap

            signal_norm = signal_norm[start:end]

        else:
            signal_norm = signal_norm[hop_length:]

    return signal_norm


def _add_reflection(matrix):
    reflection = matrix[-2:0:-1, :]
    reflection = reflection.conj()
    return np.vstack((matrix, reflection))
