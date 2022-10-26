# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from hierarchical.datasets import multi_level, single_level


def _split_name(split):
    if split in ["tr", "train"]:
        sp = "train"
    elif split in ["cv", "val", "validation"]:
        sp = "val"
    elif split in ["tt", "test"]:
        sp = "test"
    else:
        raise ValueError(f"Unknown split type: {split}.")
    return sp


def dataset_creator(params, split_name, target, t_kwargs):
    split_name = _split_name(split_name)

    is_cv_or_tt = split_name in ["val", "test"]
    is_tt = split_name is "test"
    is_a = params.get("is_a", False)

    saliency_key = f"{split_name}_saliency_file"
    saliency_file = params[saliency_key]
    base_dir = os.path.join(params["slakh_base_dir"], split_name)
    chunk_duration = params.get("chunk_duration", 10.0)
    chunk_hop_ratio = params.get("chunk_hop_ratio", 0.25)
    saliency_threshold = params.get("saliency_threshold", -30.0)
    saliency_threads = params.get("saliency_threads", 4)
    hierarchy_file = params["hierarchy_file"]
    instruments = params.get("instruments", None)
    qbe = params.get("qbe", True)
    nonsalient_ratio = params.get("nonsalient_ratio", 0.0) if qbe else 0.0
    query_presence_thresh = params.get("query_presence_thresh", -60.0)
    hierarchy_levels = params.get("hierarchy_levels", None)
    input_feature = params["input_feature"]
    frame_return = params.get("frame_return", "rand")
    fix_dataset = params.get("fix_dataset", False)
    fix_dataset = True if is_cv_or_tt else fix_dataset

    stft_win = params["fft_frame_size"]
    stft_hop = params["fft_hop"]

    iter_per_epoch = params[f"{split_name}_iterations"]
    frames_per_sample = params["frames_per_sample"]

    use_multilevel = isinstance(hierarchy_levels, list) and len(hierarchy_levels) > 1

    if True:
        parent_ratio = params.get("parent_nonsalient_ratio", 0.0) if qbe else 0.0
        hc = params.get("target_hierarchy_constraint", None)
        zero_pct = params.get("zeroed_leaf_percent", -1)
        zero_pct = -1 if is_cv_or_tt else zero_pct  # only zero during training
        return_level = params.get("return_level", None)
        fix_dataset = False if is_a else fix_dataset
        dataset = multi_level.MultiLevelDataset(
            base_dir,
            saliency_file,
            iter_per_epoch,
            stft_win,
            stft_hop,
            target,
            hierarchy_file,
            hierarchy_levels,
            frame_return=frame_return,
            input_feature_=input_feature,
            frame_size=frames_per_sample,
            target_kwargs=t_kwargs,
            nonsalient_ratio=nonsalient_ratio,
            parent_nonsalient_ratio=parent_ratio,
            query_presence_thresh=query_presence_thresh,
            hierarchical_constraint=hc,
            qbe=qbe,
            inst=instruments,
            eval_mode=is_tt,
            batch_fixed=fix_dataset,
            zeroed_leaf_percent=zero_pct,
            return_level=return_level,
            is_a=is_a,
        )
        dataset.batch_fixed = True if is_a else dataset.batch_fixed
        remove_zeros = params.get("remove_zeros", False)
        if remove_zeros:
            dataset.remove_zeros()
    else:
        n_other_src = params.get("num_other_sources", -1)
        query_target_different = params.get("query_target_different", False)
        dataset = single_level.SingleLevelDataset(
            base_dir,
            saliency_file,
            iter_per_epoch,
            stft_win,
            stft_hop,
            target,
            hierarchy_file,
            input_feature_=input_feature,
            frame_size=frames_per_sample,
            target_kwargs=t_kwargs,
            nonsalient_ratio=nonsalient_ratio,
            query_target_different=query_target_different,
            query_presence_thresh=query_presence_thresh,
            inst=instruments,
            lvl=hierarchy_levels,
            chunk_dur=chunk_duration,
            chunk_hop_ratio=chunk_hop_ratio,
            saliency_thresh=saliency_threshold,
            saliency_threads=saliency_threads,
            frame_return=frame_return,
            qbe=qbe,
            n_other_srcs=n_other_src,
            eval_mode=is_tt,
            batch_fixed=fix_dataset,
        )

    return dataset
