#!/usr/bin/env python3
# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import os
import random
from collections import OrderedDict

import h5py
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from hierarchical.loss_functions.utils import get_target_by_name
from hierarchical.utils import signal_lib

from .hierarchy_utils import HierarchyLevel, sanitize_
from .prep_utils import read_chunks
from .saliency import INST_NAME, LEVEL, START, TRACK_ID, write_saliency_file_slakh


class HierarchicalDataset(Dataset):
    """ """

    def __init__(
        self,
        input_dir,
        saliency_file,
        iter_per_epoch,
        stft_window_length,
        stft_hop_length,
        target_type,
        hierarchy_file,
        input_feature_="logmag",
        whiten_param_file=None,
        frame_size=None,
        frame_return="rand",
        target_kwargs=None,
        inst=None,
        nonsalient_ratio=0.05,
        query_target_different=True,
        query_presence_thresh=-60.0,
        chunk_dur=10.0,
        chunk_hop_ratio=0.25,
        saliency_thresh=-60.0,
        saliency_threads=64,
        window_type="sqrt_hann",
        dtype="complex64",
        eval_mode=False,
    ):

        self.frame_size = frame_size
        self.frame_return = frame_return
        self.window_type = window_type
        self.window_length = stft_window_length
        self.hop_length = stft_hop_length
        self.sample_rate = None  # gets set when we read the file
        self.eval_mode = eval_mode

        self.all_chunks_per_time = {}

        if not np.issubdtype(dtype, np.complexfloating):
            raise AssertionError("dtype must be complex! Got {}".format(dtype))
        self.dtype = dtype

        self.input_feature = input_feature_
        if target_type is not None:
            if callable(target_type):
                self.target = target_type
            elif isinstance(target_type, str):
                self.target = get_target_by_name(target_type)
            else:
                raise ValueError("Target must be str or function!")
            self.target_kwargs = target_kwargs if target_kwargs is not None else {}

        if whiten_param_file is None:
            self.whiten_mean = None
            self.whiten_std = None
        else:
            with h5py.File(whiten_param_file, "r") as stats:
                self.whiten_mean = np.array(stats["mean"])
                self.whiten_std = np.array(stats["std"])

        self.iterations_per_epoch = iter_per_epoch

        if not os.path.exists(saliency_file):
            level_list = [l for l in range(len(HierarchyLevel))]  # we always want to write all levels
            logger.info("Saliency file not found. Writing a new one...")
            write_saliency_file_slakh(
                input_dir,
                saliency_file,
                chunk_dur,
                chunk_hop_ratio,
                saliency_thresh,
                hierarchy_file,
                level_list,
                saliency_threads,
            )
        else:
            logger.info("Found saliency file. Will not rewrite existing saliency file.")
        self.output_saliency_file = saliency_file

        self.nonsalient_ratio = nonsalient_ratio
        self.query_presence_thresh = query_presence_thresh
        self.query_target_different = query_target_different

        self.inst_dict = OrderedDict()
        self.inst = None
        if inst is not None:
            self.inst = inst if isinstance(inst, list) else [inst]
            self.inst = [sanitize_(i) for i in self.inst]

    @staticmethod
    def _make_chid(chunk):
        return f"{chunk[TRACK_ID]}_{chunk[START]}"

    @staticmethod
    def _readable_chid(chunk):
        return f"{chunk[TRACK_ID]}. {chunk[INST_NAME]} ({chunk[LEVEL]})"

    def __len__(self):
        return self.iterations_per_epoch

    def _do_stft(self, signal):
        return signal_lib.do_stft(signal, self.sample_rate, self.window_length, self.hop_length, self.window_type).T

    def _determine_target_chunk(self, *args):
        pass

    def _frame_return(self, stft):
        # `stft` has shape [n_time, n_freq]
        frames, freq = stft.shape
        if (self.frame_size is not None) and (frames > self.frame_size):

            if self.frame_return == "rand":
                # Returns a random frame
                start_frame = np.random.randint(0, frames - self.frame_size)
                end_frame = start_frame + self.frame_size

                stft = stft[start_frame:end_frame, :]
            elif self.frame_return == "all":
                pass
            else:
                raise ValueError("Unknown frame_style: {}!".format(self.frame_return))
        return stft

    def _prepare_chunk(self, chunk, num_other=-1):
        chunk_id = self._make_chid(chunk)
        all_chunks = copy.deepcopy(self.all_chunks_per_time[chunk_id])

        if num_other > 0:
            all_chunks.remove(chunk)
            all_chunks = random.sample(all_chunks, num_other)
            all_chunks.append(chunk)

        mix, target, other, sr = read_chunks(chunk, all_chunks)
        self.sample_rate = sr
        target_stft = np.expand_dims(self._frame_return(self._do_stft(target)), -1)
        mix_stft = self._frame_return(self._do_stft(mix))
        return mix_stft, target_stft
