# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import random

import numpy as np
from loguru import logger
from tqdm import tqdm

from hierarchical.loss_functions.targets import dpcl_binary_weights, input_feature

from .dataset_base import HierarchicalDataset
from .hierarchy_utils import NSAL, SAL, HierarchyLevel, flatten
from .saliency import INST_NAME, IS_SAL, LEVEL, START, TRACK_ID


class SingleLevelDataset(HierarchicalDataset):
    """
    Query-by-Example dataset class.

    Provides a mixture, isolated sources from that mixture,
    and an example of an isolated instrument not in the mixture.

    This class goes directly from waveform on disk to the network.
    """

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
        nonsalient_ratio=0.05,
        query_target_different=True,
        query_presence_thresh=-60.0,
        query_is_target=False,
        inst=None,
        lvl=None,
        batch_fixed=False,
        chunk_dur=10.0,
        chunk_hop_ratio=0.25,
        saliency_thresh=-40.0,
        saliency_threads=64,
        window_type="sqrt_hann",
        dtype="complex64",
        qbe=True,
        n_other_srcs=-1,
        eval_mode=False,
    ):

        super().__init__(
            input_dir,
            saliency_file,
            iter_per_epoch,
            stft_window_length,
            stft_hop_length,
            target_type,
            hierarchy_file,
            input_feature_=input_feature_,
            whiten_param_file=whiten_param_file,
            frame_size=frame_size,
            frame_return=frame_return,
            target_kwargs=target_kwargs,
            nonsalient_ratio=nonsalient_ratio,
            query_target_different=query_target_different,
            query_presence_thresh=query_presence_thresh,
            chunk_dur=chunk_dur,
            chunk_hop_ratio=chunk_hop_ratio,
            saliency_thresh=saliency_thresh,
            saliency_threads=saliency_threads,
            window_type=window_type,
            dtype=dtype,
            inst=inst,
            eval_mode=eval_mode,
        )

        self.query_is_target = query_is_target
        self.salient_chunks = []
        self.nonsalient_chunks = []
        self.qbe = qbe
        self.n_other_src = n_other_srcs
        self.batch_fixed = batch_fixed

        self.qry_insts = None
        if lvl is not None:
            hier = json.load(open(hierarchy_file, "r"))
            flattened = flatten(hier, [HierarchyLevel(lvl)])
            self.qry_insts = list(flattened.keys())

        self._populate_lists()

        self.batch_ids = {}
        if batch_fixed:
            logger.info("Query/Target selection is fixed. Determining IDs...")
            for i in range(iter_per_epoch):
                query_chunk = self._select_salient_chunk()
                target_chunk = self._determine_target_chunk(query_chunk[TRACK_ID], query_chunk[INST_NAME])
                self.batch_ids[i] = {"query": query_chunk, "target": target_chunk}

            logger.info("Batch determined.")

    def print(self):
        logger.info("")
        logger.info(f"Unique instruments in {self.output_saliency_file}")
        for inst, i_dict in self.inst_dict.items():
            for s, i_list in i_dict.items():
                logger.info(f"\t{inst+',':30} {s+':':12} {len(i_list):7d}")
        logger.info("")

        for inst in self.inst:
            logger.info(f"Stats for selected query: '{inst}'")
            logger.info(f"\t{inst+',':30} {'salient:':12} " f"{len(self.inst_dict[inst][SAL]):7d}")
            logger.info(f"\t{inst+',':30} {'nonsalient:':12} " f"{len(self.inst_dict[inst][NSAL]):7d}")
            t_len = len(self.inst_dict[inst][SAL]) + len(self.inst_dict[inst][NSAL])
            logger.info(f"\tTotal: {t_len}")
            logger.info("")

    @staticmethod
    def _make_chid(chunk):
        return f"{chunk[TRACK_ID]}_{chunk[START]}_{chunk[LEVEL]}"

    def _populate_lists(self):
        logger.info("Reading saliency file (may take a few seconds)...")
        with open(self.output_saliency_file, "r") as f:
            saliency_data = json.load(f)

        logger.info("Parsing saliency file...")

        for chunk in tqdm(saliency_data):
            salient = chunk[IS_SAL]
            inst = chunk[INST_NAME]

            chunk_id = self._make_chid(chunk)
            if chunk_id not in self.all_chunks_per_time:
                self.all_chunks_per_time[chunk_id] = []
            self.all_chunks_per_time[chunk_id].append(chunk)

            if self.qry_insts and inst not in self.qry_insts:
                continue

            # list of all salient IDs (these are queries)
            if salient:
                self.salient_chunks.append(chunk)
            else:
                self.nonsalient_chunks.append(chunk)  # not used right now...

            # Make a dict of lists separated by instrument type and saliency
            if inst not in self.inst_dict:
                self.inst_dict[inst] = {SAL: [], NSAL: []}

            if salient:
                self.inst_dict[inst][SAL].append(chunk)
            else:
                self.inst_dict[inst][NSAL].append(chunk)

    def __len__(self):
        return self.iterations_per_epoch

    def _determine_target_chunk(self, query_track_num, query_inst):
        # decide if we're using a non-salient target
        is_salient = np.random.rand() > self.nonsalient_ratio
        is_sal = SAL if is_salient else NSAL

        # get the list of all other tracks with this inst
        tracks_list = self.inst_dict[query_inst][is_sal]

        if not tracks_list:
            tracks_list = self.inst_dict[query_inst][SAL if not is_salient else NSAL]

        if self.query_target_different:
            # Select a random segment from a different track
            while True:
                target_chunk = tracks_list[np.random.randint(len(tracks_list))]
                target_track = target_chunk[TRACK_ID]
                if query_track_num != target_track:
                    break
        else:
            # Any track is fine, don't assert that they're from different tracks
            target_chunk = tracks_list[np.random.randint(len(tracks_list))]

        return target_chunk

    def _select_salient_chunk(self):
        if not self.inst:
            return random.choice(self.salient_chunks)
        else:
            # make a combined list of salient chunks from the chosen instruments
            combined_insts = []
            for inst in self.inst:
                combined_insts += self.inst_dict[inst][SAL]
            return random.choice(combined_insts)

    def _fixed_batch(self, itm):
        if itm in self.batch_ids.keys():
            query_chunk = self.batch_ids[itm]["query"]
            target_chunk = self.batch_ids[itm]["target"]
        else:
            # Code for not reusing queries:
            #
            # used_queries = [self._make_inst_chid(i['query']) for i in self.batch_ids.values()]
            # while True:
            #     query_chunk = self._select_salient_chunk()
            #     if self._make_inst_chid(query_chunk) not in used_queries:
            #         break
            logger.info(f"Unknown item {itm}")
            query_chunk = self._select_salient_chunk()
            target_chunk = self._determine_target_chunk(query_chunk[TRACK_ID], query_chunk[INST_NAME])
            self.batch_ids[itm] = {"query": query_chunk, "target": target_chunk}

        return query_chunk, target_chunk

    def _choose_qry_tar(self, itm):
        # Determine query first (always salient), get info about it
        if self.batch_fixed:
            query_chunk, target_chunk = self._fixed_batch(itm)
        else:
            query_chunk = self._select_salient_chunk()
            target_chunk = self._determine_target_chunk(query_chunk[TRACK_ID], query_chunk[INST_NAME])

        return query_chunk, target_chunk

    def __getitem__(self, item):

        query_chunk, target_chunk = self._choose_qry_tar(item)

        # `mix` has shape [n_time, n_freq], `srcs` has shape [n_time, n_freq, n_src] (n_src==1)
        mix, target = self._prepare_chunk(target_chunk, self.n_other_src)

        mix_feat = input_feature(mix, self.input_feature, whiten_mean=self.whiten_mean, whiten_std=self.whiten_std)
        target_dict = self.target(mix, target, **self.target_kwargs)
        target_dict["target_id"] = [self._readable_chid(target_chunk)]

        feature_dict = {"mix": mix_feat}

        if self.qbe:
            # `query` has shape [n_time, n_freq, n_src]
            _, query = self._prepare_chunk(query_chunk)
            query_feat = input_feature(query, self.input_feature)
            query_presence = dpcl_binary_weights(query, self.query_presence_thresh)

            # query is always 1 src, so these both need to have shape [n_time, n_freq]
            feature_dict["presence"] = np.squeeze(query_presence)
            feature_dict["query"] = np.squeeze(query_feat)
            feature_dict["query_id"] = self._readable_chid(query_chunk)

        if self.eval_mode:
            split = lambda x: {"abs": np.abs(x), "ang": np.angle(x)}
            feature_dict["mix_stft"] = split(mix)
            target_dict["target_stft"] = split(target)

        return feature_dict, target_dict
