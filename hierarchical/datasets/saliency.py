#!/usr/bin/env python3
# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import os
from multiprocessing.dummy import Pool as ThreadPool

import librosa
import numpy as np
import soundfile as sf
import yaml
from loguru import logger

from .hierarchy_utils import _invert_hierarchy_patches, get_hierarchy_patches

SRC_PATHS = "src_paths"
INST_NAME = "inst_name"
START = "start"
END = "end"
TRACK_ID = "track_id"
IS_SAL = "is_salient"
LEVEL = "level"


def find_salient_starts(audio, duration_sec, hop_ratio, sr, threshold_db, return_nonsalient=False):
    # finds frames in the audio where the RMS is above a dB threshold
    dur = int(sr * duration_sec)
    hop_dur = int(dur * hop_ratio)
    threshold = np.power(10.0, threshold_db / 20.0)
    rms = librosa.feature.rms(audio, frame_length=dur, hop_length=hop_dur)[0, :]
    loud = np.squeeze(np.argwhere(rms > threshold))
    fr = lambda t: np.atleast_1d(librosa.frames_to_samples(t, hop_length=hop_dur))
    if return_nonsalient:
        soft = np.squeeze(np.argwhere(rms < threshold))
        return fr(loud), fr(soft)
    return fr(loud)


def write_saliency_file_slakh(
    input_dir, output_saliency_file, dur, hr, thresh, hierarchy_filename, level_list, threads=1
):
    logger.info(f"Making saliency file at {output_saliency_file}")
    track_dirs = sorted(
        [
            track_dir
            for track_dir in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, track_dir))
            and "metadata.yaml" in os.listdir(os.path.join(input_dir, track_dir))
        ]
    )
    all_mix_recipes_dict = get_hierarchy_patches(hierarchy_filename, level_list)
    inv_hierarchies = _invert_hierarchy_patches(all_mix_recipes_dict)

    def submixes_and_saliency(track_dir):
        """
        This function will:
        - Look at all stems in a track dir for one track
        - Collect all stems that belong to a src for every hierarchy level
        - Find the salient and nonsalient sections
        - Put all the relevant info in al list of dictionaries
        that
        :param track_dir:
        :return:
        """

        # load metadata
        input_srcs_dir = os.path.join(input_dir, track_dir)
        src_metadata = yaml.load(open(os.path.join(input_srcs_dir, "metadata.yaml"), "r"), Loader=yaml.FullLoader)
        stem_metadata = src_metadata["stems"]
        stem_path = os.path.join(input_srcs_dir, "stems")
        sr = None

        mix_recipes = {level_name: {} for level_name in inv_hierarchies.keys()}
        for src_name, src_info in stem_metadata.items():
            src_path = os.path.join(stem_path, f"{src_name}.wav")
            if not os.path.isfile(src_path):
                continue
            src, sr = sf.read(src_path)
            src_patch = src_info["plugin_name"]
            for level_name in inv_hierarchies.keys():
                src_category = inv_hierarchies[level_name][src_patch]
                if src_category not in mix_recipes[level_name]:
                    mix_recipes[level_name][src_category] = {"src_wavs": [], "src_path": []}
                mix_recipes[level_name][src_category]["src_wavs"].append(src)
                mix_recipes[level_name][src_category]["src_path"].append(src_path)

        path_and_times = []

        for level, cur_recipe in mix_recipes.items():

            for inst_name, src_list in cur_recipe.items():
                try:
                    src_wavs = src_list["src_wavs"]
                    inst_wav = np.sum(src_wavs, axis=0)

                    # Find salient clips
                    salient_starts, nonsalient_starts = find_salient_starts(
                        inst_wav, dur, hr, sr=sr, threshold_db=thresh, return_nonsalient=True
                    )

                    src_paths = src_list["src_path"]
                    inst_name = inst_name.replace("/", "_").replace(" ", "_")  # Sanitize name

                    def make_list(all_starts, is_salient):
                        dict_result = []
                        all_ends = all_starts + dur * sr
                        try:
                            for start, end in zip(*(all_starts, all_ends)):
                                if end > len(inst_wav):
                                    continue
                                dict_result.append(
                                    {
                                        SRC_PATHS: src_paths,
                                        START: int(start),
                                        END: int(end),
                                        TRACK_ID: track_dir,
                                        INST_NAME: inst_name,
                                        IS_SAL: is_salient,
                                        LEVEL: level,
                                    }
                                )

                        except:
                            pass

                        return dict_result

                    path_and_times += make_list(salient_starts, True)
                    path_and_times += make_list(nonsalient_starts, False)

                except ValueError as e:
                    logger.warning(f"Couldn't read {track_dir}")
                    logger.warning(f"Error: {e}")
                    return []

        return path_and_times

    logger.info("Calculating saliencies. This might take some time....")
    if threads == 1:
        result = [submixes_and_saliency(t) for t in track_dirs]
    else:
        pool = ThreadPool(threads)
        result = pool.map(submixes_and_saliency, track_dirs)
    saliency_data = [item for sublist in result for item in sublist]

    logger.info("Finished calculating saliencies. Writing file.")
    os.makedirs(os.path.dirname(output_saliency_file), exist_ok=True)
    with open(output_saliency_file, "w") as f:
        json.dump(saliency_data, f, indent=4)

    logger.info(f"Finished writing saliency file at {output_saliency_file}")
