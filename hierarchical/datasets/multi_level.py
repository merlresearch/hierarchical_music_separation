#!/usr/bin/env python3
# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import random
from collections import OrderedDict

import librosa
import librosa.display
import numpy as np
from loguru import logger
from tqdm import tqdm

from hierarchical.loss_functions.targets import dpcl_binary_weights, input_feature

from .dataset_base import HierarchicalDataset
from .hierarchy_utils import NSAL, SAL, HierarchyLevel, flatten
from .saliency import INST_NAME, IS_SAL, LEVEL, START, TRACK_ID

SIB = "sibling"
NSIB = "no_sibling"
MIN_SALIENT_INST = 1000
prune = False


class MultiLevelDataset(HierarchicalDataset):
    """
     Hierarchical dataset class for multiple levels.
     Has the option to do query-by-example

    Provides a mixture, isolated sources from that mixture,
    and an example of an isolated instrument not in the mixture.

    hierarchy_levels should be a list with values like: [0, 1, 2]
        (0 is highest/root level, 2 is lowest/leaf level)

    """

    def __init__(
        self,
        slakh_dir,
        saliency_file,
        iter_per_epoch,
        stft_window_length,
        stft_hop_length,
        target_type,
        hierarchy_def_file,
        hierarchy_levels,
        frame_return="rand",
        target_kwargs=None,
        query_target_different=True,
        nonsalient_ratio=0.05,
        parent_nonsalient_ratio=0.01,
        query_presence_thresh=-60.0,
        hierarchical_constraint=False,
        qbe=True,
        inst=None,
        eval_mode=False,
        batch_fixed=False,
        saliency_threads=64,
        input_feature_="logmag",
        frame_size=None,
        zeroed_leaf_percent=-1,
        return_level=None,
        is_a=False,
    ):

        super().__init__(
            slakh_dir,
            saliency_file,
            iter_per_epoch,
            stft_window_length,
            stft_hop_length,
            target_type,
            hierarchy_def_file,
            frame_return=frame_return,
            target_kwargs=target_kwargs,
            nonsalient_ratio=nonsalient_ratio,
            inst=inst,
            eval_mode=eval_mode,
            query_target_different=query_target_different,
            saliency_threads=saliency_threads,
            input_feature_=input_feature_,
            frame_size=frame_size,
        )

        logger.info(f"Setting up multilevel dataset with saliency file {self.output_saliency_file}")
        self.nonsalient_ratio = nonsalient_ratio
        self.parent_nonsalient_ratio = parent_nonsalient_ratio
        self.query_presence_thresh = query_presence_thresh
        self.hierarchical_constraint = hierarchical_constraint
        self.qbe = qbe
        self.salient_chunks = []
        self.batch_fixed = batch_fixed
        self.zeroed_leaf_percent = zeroed_leaf_percent
        self.is_a = is_a

        self.hierarchy_def = json.load(open(hierarchy_def_file, "r"))
        if not hierarchy_levels:
            raise ValueError("No hierarchy_levels provided!")
        if len(hierarchy_levels) > len(HierarchyLevel):
            raise ValueError("Too big bro... too big...")
        self.hierarchy_levels = [HierarchyLevel(h) for h in sorted(hierarchy_levels)]
        self.hierarchy_flattened = flatten(self.hierarchy_def, self.hierarchy_levels)
        self.all_inst_chunks_per_time = {}

        if isinstance(return_level, int):
            self.return_level = sorted(hierarchy_levels)[::-1].index(return_level)
        elif return_level == "rand" or return_level is None:
            self.return_level = return_level
        else:
            raise ValueError(f"Unknown value for return_level: {return_level}")

        self.salient_leaf_chunks = []
        self.nonsalient_leaf_chunks = []

        if self.inst is None:
            # nothing was specified, so use everything
            self.inst = [k for k, v in self.hierarchy_flattened.items() if v["is_leaf"]]

        if not all(self.hierarchy_flattened[i]["is_leaf"] for i in self.inst):
            non_leaves = [i for i in self.inst if not self.hierarchy_flattened[i]["is_leaf"]]

        self.inst_leaf_chunks = {i: {SAL: [], NSAL: []} for i in self.inst}

        self._populate_id_lists()

        self.all_leaves_dict = {
            SAL: self._populate_sibling_dicts(self.salient_leaf_chunks),
            NSAL: self._populate_sibling_dicts(self.nonsalient_leaf_chunks),
        }

        self.all_inst_leaves_dict = {
            i: {
                SAL: self._populate_sibling_dicts(self.inst_leaf_chunks[i][SAL]),
                NSAL: self._populate_sibling_dicts(self.inst_leaf_chunks[i][NSAL]),
            }
            for i in self.inst
        }
        if prune:
            if eval_mode:
                # no need to prune, just make sure we don't infinite loop
                self.all_inst_leaves_dict = {
                    k: v
                    for k, v in self.all_inst_leaves_dict.items()
                    if len(set([c["track_id"] for c in v[SAL][SIB]])) > 1
                }
            else:
                # prune values that have low salient sibling numbers
                # so we don't overfit during training
                self.all_inst_leaves_dict = {
                    k: v for k, v in self.all_inst_leaves_dict.items() if len(v[SAL][SIB]) > MIN_SALIENT_INST
                }
        #
        self.inst_leaf_chunks = {
            k: v for k, v in self.inst_leaf_chunks.items() if k in self.all_inst_leaves_dict.keys()
        }
        self.inst = list(self.all_inst_leaves_dict.keys())
        if inst is None and len(self.all_inst_leaves_dict) < 5:
            logger.warning(
                f"Less than 5 instruments have {MIN_SALIENT_INST} "
                f"salient chunks with salient siblings. \n"
                f"These constitute most of the training set."
            )

        self.batch_ids = {}
        if batch_fixed:
            logger.info("Query/Target selection is fixed. Determining IDs...")
            for i in range(iter_per_epoch):
                query_chunk = self._select_query_chunk()
                target_chunk = self._determine_target_chunk(query_chunk)
                self.batch_ids[i] = {"query": query_chunk, "target": target_chunk}

            logger.info("Batch determined.")

        logger.info("Finished dataset setup")

    def remove_zeros(self):
        logger.info("Removing all Targets with leaf_zeroed=True")
        new_ids = {}
        for i, val in self.batch_ids.items():
            if not val["target"]["leaf_zeroed"]:
                new_ids[i] = val

        bf_len = len(self.batch_ids)
        self.batch_ids = new_ids
        logger.info(f"Removed leaf_zeroed targets. Original len={bf_len}, New len={len(self)}")

    @staticmethod
    def _make_chid(chunk):
        # encodes track, start time, and level
        # all chunks along same level of hierarchy have same ID
        return f"{chunk[TRACK_ID]}_{chunk[START]}_{chunk[LEVEL]}"

    def _make_inst_chid(self, chunk):
        # encodes track, start time, and root instrument
        # all chunks down the same path of hierarchy have same ID
        root = self.hierarchy_flattened[chunk[INST_NAME]]["root"]
        return f"{chunk[TRACK_ID]}_{chunk[START]}_{root}"

    def _populate_id_lists(self):
        logger.info("Reading saliency file (may take a few seconds)...")
        with open(self.output_saliency_file, "r") as f:
            saliency_data = json.load(f)

        logger.info("Parsing saliency file...")

        # cache this for faster checking within the loop (~25,000x speedup!)
        valid_instruments = list(self.hierarchy_flattened.keys())
        for chunk in tqdm(saliency_data):

            salient = chunk[IS_SAL]
            this_inst = chunk[INST_NAME]

            if this_inst not in valid_instruments:
                # This node is not in our hierarchy, skip it...
                continue

            # Keep all of tracks, start times, and instruments along levels together
            # later we will gather all chunks to make the mixes
            chunk_id = self._make_chid(chunk)
            if chunk_id not in self.all_chunks_per_time:
                self.all_chunks_per_time[chunk_id] = []
            self.all_chunks_per_time[chunk_id].append(chunk)

            # Keep all of tracks, start times, and root instruments together
            # this helps determine siblings
            chunk_id = self._make_inst_chid(chunk)
            if chunk_id not in self.all_inst_chunks_per_time:
                self.all_inst_chunks_per_time[chunk_id] = []
            self.all_inst_chunks_per_time[chunk_id].append(chunk)

            if salient:
                self.salient_chunks.append(chunk)
                if self.hierarchy_flattened[this_inst]["is_leaf"]:
                    self.salient_leaf_chunks.append(chunk)

                    if this_inst in self.inst:
                        self.inst_leaf_chunks[this_inst][SAL].append(chunk)

            else:
                if self.hierarchy_flattened[this_inst]["is_leaf"]:
                    self.nonsalient_leaf_chunks.append(chunk)

                    if this_inst in self.inst:
                        self.inst_leaf_chunks[this_inst][NSAL].append(chunk)

    def _populate_sibling_dicts(self, leaf_chunks):
        sibling_dict = {SIB: [], NSIB: []}
        for chunk in leaf_chunks:
            cur_inst = chunk[INST_NAME]

            chunk_id = self._make_inst_chid(chunk)
            family = self.all_inst_chunks_per_time[chunk_id]
            has_siblings = False
            for node_chunk in family:
                if node_chunk == chunk:
                    # this is me
                    continue

                salient = node_chunk[IS_SAL]
                node_inst = node_chunk[INST_NAME]

                if node_inst in self.hierarchy_flattened[cur_inst]["siblings"] and salient:
                    # we have a sibling
                    has_siblings = True

            # assert parent_chunk
            sib = SIB if has_siblings else NSIB
            sibling_dict[sib].append(chunk)

        return sibling_dict

    @staticmethod
    def _parse_id(id_):
        track_id, start, end, salient, inst = id_.split("_")
        salient = salient == "True"
        inst = inst[1:-1]  # remove the parens
        return track_id, start, salient, inst

    def __len__(self):
        if len(self.batch_ids) > 0:
            return len(self.batch_ids)
        return self.iterations_per_epoch

    def prepare_all_levels(self, target_chunk):

        # prepare mixture first
        mix, _ = self._prepare_chunk(target_chunk)
        mix_feat = input_feature(mix, self.input_feature, whiten_mean=self.whiten_mean, whiten_std=self.whiten_std)

        # find all of the target's relatives
        chunk_id = self._make_inst_chid(target_chunk)
        family = self.all_inst_chunks_per_time[chunk_id]

        # order the family by level
        family_lvls = OrderedDict((n.name.lower(), {}) for n in self.hierarchy_levels[::-1])
        for node in family:
            family_lvls[node[LEVEL]][node[INST_NAME]] = node

        # loop through level and prepare each chunk along the hierarchy
        prepped_lvl_masks = OrderedDict((k, {}) for k in family_lvls.keys())
        prepped_lvl_stfts = OrderedDict((k, {}) for k in family_lvls.keys())
        for lvl, chunk_dict in family_lvls.items():
            for inst_name, chunk in chunk_dict.items():
                if chunk[IS_SAL]:
                    _, target_stft = self._prepare_chunk(chunk)  # ignore mix
                    prepped_lvl_stfts[lvl][chunk[INST_NAME]] = target_stft

                    # TODO: make self.target() just return a mask?
                    target_mask = self.target(mix, target_stft, **self.target_kwargs)["mask"]
                    prepped_lvl_masks[lvl][chunk[INST_NAME]] = target_mask

        # ------- make the stacked target masks -------
        # first we get the leaf mask, because that is always first
        leaf_mask = prepped_lvl_masks[target_chunk[LEVEL]][target_chunk[INST_NAME]]
        result = [leaf_mask]
        stfts = [prepped_lvl_stfts[target_chunk[LEVEL]][target_chunk[INST_NAME]]]
        names = [self._readable_chid(target_chunk)]

        # next get a list of the parents and traverse up the tree
        parent_names = self.hierarchy_flattened[target_chunk[INST_NAME]]["parents"][::-1]
        for name in parent_names:
            level = self.hierarchy_flattened[name]["level"]
            parent_stft = prepped_lvl_stfts[level][name]
            names.append(self._readable_chid(family_lvls[level][name]))

            # the parent masks depend on our hierarchical constraint
            if self.hierarchical_constraint == "max":
                # this is [leaf, max(leaf, parent), ...]
                parent_mask = prepped_lvl_masks[level][name]
                parent_mask = np.maximum(leaf_mask, parent_mask)

            elif self.hierarchical_constraint == "sum":
                # this is [leaf, sum(leaf1, leaf2, etc), ...]
                parent_mask = np.sum(np.dstack(prepped_lvl_masks[level].values()), axis=-1, keepdims=True)

            elif self.hierarchical_constraint == "maxsum":
                # this is [leaf, max(sum(leaf1, leaf2, etc), parent), ...]
                leaf_sums = np.sum(np.dstack(prepped_lvl_masks[level].values()), axis=-1, keepdims=True)
                parent = prepped_lvl_masks[level][name]
                parent_mask = np.maximum(leaf_sums, parent)

            elif not self.hierarchical_constraint:
                parent_mask = prepped_lvl_masks[level][name]

            else:
                raise ValueError(f"Unknown hierarchical_constraint: " f"{self.hierarchical_constraint}")

            # append the calculated parent mask,
            # and set it as the leaf for the next loop
            result.append(parent_mask)
            stfts.append(parent_stft)
            leaf_mask = parent_mask

        feat_dict = {"mix": mix_feat}

        zeroed = target_chunk.get("leaf_zeroed", False)
        if self.return_level == "rand":
            choice = np.random.randint(len(result))
            masks = result[choice]
            names = names[choice]
            stfts = [stfts[choice]]
            feat_dict["knob"] = len(result) - choice - 1  # invert the knob so largest is leaf
        elif self.return_level is not None:
            masks = result[self.return_level]
            names = names[self.return_level]
            stfts = [stfts[self.return_level]]
        else:
            masks = np.dstack(result)

        target_dict = {"mask": masks, "target_id": names, "leaf_zeroed": zeroed}

        if self.eval_mode:
            split = lambda x: {"abs": np.abs(x), "ang": np.angle(x)}
            feat_dict["mix_stft"] = split(mix)
            target_dict["target_stft"] = split(np.dstack(stfts))

        return feat_dict, target_dict

    def plot(self, mix_feat, masks, target_chunk, out_dir=None):
        import matplotlib.pyplot as plt

        n_masks = masks.shape[-1]
        n_rows = n_masks + 1
        height = 3 * n_rows

        mask_titles = [f"Level '{target_chunk[LEVEL]}', {target_chunk[INST_NAME]}"]
        parents = self.hierarchy_flattened[target_chunk[INST_NAME]]["parents"][::-1]
        mask_titles.extend(f"Level '{self.hierarchy_flattened[p][LEVEL]}', " f"{p}" for p in parents)

        max_bin = 128  #
        plt.close("all")
        fig, ax = plt.subplots(nrows=n_rows, figsize=(5, height), sharex=True)
        ax[0].imshow(mix_feat[:, :max_bin].T, aspect="auto")
        ax[0].invert_yaxis()
        ax[0].set_title(f"Mixture, hierarchical constraint={self.hierarchical_constraint}")
        pr = lambda m: librosa.amplitude_to_db(m)
        for i in range(n_masks):
            ax[i + 1].imshow(pr(masks[:, :max_bin, i].T), aspect="auto")
            ax[i + 1].invert_yaxis()
            ax[i + 1].set_title(mask_titles[i])

        plt.axis("tight")
        if not out_dir:
            plt.show()
        else:
            lvls = "".join(str(l.value) for l in self.hierarchy_levels)
            filename = (
                f"hier_masks_lvls{lvls}_hc-{self.hierarchical_constraint}" f"_{self._make_inst_chid(target_chunk)}.png"
            )
            out_file = f"{out_dir}/{filename}"
            plt.savefig(out_file)

    def _fixed_batch(self, itm):
        if itm in self.batch_ids.keys():
            query_chunk = self.batch_ids[itm]["query"]
            target_chunk = self.batch_ids[itm]["target"]
        else:
            query_chunk = self._select_query_chunk()
            target_chunk = self._determine_target_chunk(query_chunk)
            self.batch_ids[itm] = {"query": query_chunk, "target": target_chunk}

        return query_chunk, target_chunk

    def _determine_target_chunk(self, query_chunk):
        query_inst = query_chunk[INST_NAME]
        query_track = query_chunk[TRACK_ID]

        leaf_choices = None
        while not leaf_choices:
            # decide if we're using a non-salient target
            leaf_salient = np.random.rand() > self.nonsalient_ratio
            parent_salient = np.random.rand() > self.parent_nonsalient_ratio

            leaf_sal = SAL if leaf_salient else NSAL
            par_sal = SIB if parent_salient else NSIB

            # choose which list we are going to pick from,
            # sometimes we pick an empty
            leaf_choices = self.all_inst_leaves_dict[query_inst][leaf_sal][par_sal]
        target_chunk = random.choice(leaf_choices)

        if self.query_target_different:
            # make sure they're from different tracks
            while target_chunk[TRACK_ID] == query_track:
                target_chunk = random.choice(leaf_choices)
        else:
            # make sure they're not the exact same thing
            while target_chunk == query_chunk:
                target_chunk = random.choice(leaf_choices)

        # determine whether the leaf is flagged for zeroing
        target_chunk["leaf_zeroed"] = np.random.rand() < self.zeroed_leaf_percent

        return target_chunk

    def _select_query_chunk(self):
        if self.inst:
            options = self.inst_leaf_chunks[random.choice(self.inst)][SAL]
            while len(options) == 0:
                options = self.inst_leaf_chunks[random.choice(self.inst)][SAL]
            return random.choice(options)
        else:
            return random.choice(self.salient_leaf_chunks)

    def _choose_qry_tar(self, itm):
        # Determine query first (always salient), get info about it
        if self.batch_fixed:
            query_chunk, target_chunk = self._fixed_batch(itm)
        else:
            query_chunk = self._select_query_chunk()
            target_chunk = self._determine_target_chunk(query_chunk)

        return query_chunk, target_chunk

    def __getitem__(self, item):

        query_chunk, target_chunk = self._choose_qry_tar(item)

        if self.is_a:

            def clr(d):
                if "leaf_zeroed" in d:
                    d.pop("leaf_zeroed")
                return d

            query_chunk = clr(query_chunk)
            target_chunk = clr(target_chunk)

        # `mix` has shape [n_time, n_freq]
        # `mask` in src_dict has shape [n_time, n_freq, n_lvls]
        feature_dict, target_dict = self.prepare_all_levels(target_chunk)

        if self.qbe:
            # `query_instrument` has shape [n_time, n_freq, n_src]
            _, query = self._prepare_chunk(query_chunk)  # dont care about mix
            query_presence = dpcl_binary_weights(query, self.query_presence_thresh)
            query_feat = input_feature(query, self.input_feature)

            # query is always 1 src, so these both need to have shape [n_time, n_freq]
            feature_dict["presence"] = np.squeeze(query_presence)
            feature_dict["query"] = np.squeeze(query_feat)
            feature_dict["query_id"] = self._readable_chid(query_chunk)

        return feature_dict, target_dict
