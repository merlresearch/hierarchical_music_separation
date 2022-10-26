#!/usr/bin/env python3
# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import os

import numpy as np
import soundfile as sf

from .saliency import END, SRC_PATHS, START


def _read_chunk(chunk_info):
    src_paths = chunk_info[SRC_PATHS]
    start, end = chunk_info[START], chunk_info[END]
    chunks, sr = list(zip(*[sf.read(p, start=start, stop=end) for p in src_paths if os.path.isfile(p)]))
    sr = set(sr)
    assert len(sr) == 1
    sr = list(sr)[0]
    chunk = np.sum(chunks, axis=0)
    return chunk, sr


def read_chunks(target_chunk, all_chunks):
    target, sr = _read_chunk(target_chunk)
    other_chunks = copy.copy(all_chunks)
    other_chunks.remove(target_chunk)
    other = np.sum([_read_chunk(o)[0] for o in other_chunks], axis=0)

    mix = np.sum([target, other], axis=0)

    return mix, target, other, sr
