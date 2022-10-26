# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from hierarchical.nets.blocks import EmbeddingLayer, MaskLayer
from hierarchical.nets.hierarchy_constraint import HierarchicalConstraint


class QBE_ADAN(nn.Module):
    """
    This network is inspired anchor DAN, but uses separate networks to create
    the anchors in the embedding space, and the masks. The anchor net makes
    the embedding space and calculates the anchors, then concatenates the anchor
    onto the frequency dimension of the input mixture, which is fed into the
    second network to create the mask.

    """

    def __init__(
        self,
        mask_n_layers,
        mask_hidden_units,
        qry_n_layers,
        qry_hidden_units,
        qry_embed_dim,
        n_freq_bins,
        n_levels,
        dropout=0.0,
        net_arch="blstm",
        mask_nonlinearity=torch.sigmoid,
        hierarchical_constraint=False,
        knob_strategy=None,
        cat_levels=None,
    ):
        super().__init__()
        logger.info("Using QBE DAN")

        bidir = True if net_arch.lower() == "blstm" else False
        output_dim = mask_hidden_units * 2 if bidir else mask_hidden_units
        self.hc = HierarchicalConstraint() if hierarchical_constraint else None
        self.hc = self.hc if n_levels > 1 else None
        self.knob_strategy = knob_strategy
        self.n_levels = n_levels

        self.mix_lstm = nn.LSTM(
            n_freq_bins + qry_embed_dim,
            mask_hidden_units,
            mask_n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidir,
        )
        if knob_strategy == "cat":
            self.cat_levels = cat_levels
            qry_input = n_freq_bins + cat_levels
        else:
            qry_input = n_freq_bins
            self.cat_levels = None
        self.qry_lstm = nn.LSTM(
            qry_input, qry_hidden_units, qry_n_layers, batch_first=True, dropout=dropout, bidirectional=bidir
        )

        self.query_embed_layer = EmbeddingLayer(output_dim, qry_embed_dim, n_freq_bins, None, False)
        self.mask_layer = MaskLayer(output_dim, n_levels, n_freq_bins, mask_nonlinearity)

        for lstm in [self.qry_lstm]:
            for name, param in lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    nn.init.xavier_normal_(param)

            for names in lstm._all_weights:
                for name in filter(lambda nm: "bias" in nm, names):
                    bias = getattr(lstm, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

    def __call__(self, x):

        query = x["query"]
        mix = x["mix"]
        presence = x["presence"]
        knob = x.get("knob", None)
        batch, n_time, n_fft = query.shape

        # Get anchor from a query
        if knob is not None and self.knob_strategy == "cat":
            knob = F.one_hot(knob, num_classes=self.cat_levels)
            knob = knob.type(query.type()).unsqueeze(1).expand(-1, n_time, -1)
            query = torch.cat((query, knob), dim=-1)

        qry_out = self.qry_lstm(query)[0]
        qry_embeddings = self.query_embed_layer(qry_out)
        presence = presence.view(batch, -1).unsqueeze(-1)
        query_anchor = torch.sum(presence * qry_embeddings, dim=1) / torch.sum(presence, dim=1)
        query_anchor = F.normalize(query_anchor, dim=-1)

        if knob is not None and self.knob_strategy == "add":
            qa = query_anchor + query_anchor.min().abs()  # min == 0.0
            qa = qa / qa.max()  # max == 1.0
            query_anchor = (qa.T + knob).T

        query_anchor = query_anchor.unsqueeze(1).expand(-1, n_time, -1)
        mix_and_qry_anchor = torch.cat((mix, query_anchor), dim=-1)

        mix_out = self.mix_lstm(mix_and_qry_anchor)[0]
        mask = self.mask_layer(mix_out).view(batch, n_time, n_fft, self.n_levels)

        if self.hc is not None:
            mask = self.hc(mask)

        return mask, query_anchor[:, 0, :]
