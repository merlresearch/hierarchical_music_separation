# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch.nn as nn

from hierarchical.nets.blocks import BatchNorm, Embedding, RecurrentStack
from hierarchical.nets.hierarchy_constraint import HierarchicalConstraint


class MaskInferenceNussl(nn.Module):
    def __init__(
        self,
        n_fft,
        n_srcs,
        hidden_size,
        num_layers,
        bidirectional,
        dropout,
        hierarchical_constraint=False,
        rnn_type="lstm",
        activation="sigmoid",
    ):
        super(MaskInferenceNussl, self).__init__()
        head_input = 2 * hidden_size if bidirectional else hidden_size
        self.hc = HierarchicalConstraint() if hierarchical_constraint else None
        self.hc = self.hc if n_srcs > 1 else None

        self.batch_norm = BatchNorm()
        self.lstm = RecurrentStack(n_fft, hidden_size, num_layers, bidirectional, dropout, rnn_type)
        self.mask_head = Embedding(n_fft, 1, head_input, n_srcs, activation)

    def forward(self, data):
        """
        Expecting input data shape:
            (batch, times, fft)
        :param data:
        :return:

        """
        if isinstance(data, dict):
            data = data["mix"]

        data = self.batch_norm(data)
        lstm_out = self.lstm(data)
        mask = self.mask_head(lstm_out)

        mask = mask.squeeze(-2)

        if self.hc is not None:
            mask = self.hc(mask)

        return mask
