# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.nn as nn


class HierarchicalConstraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masks):
        """
        Expects shape:
            (batch, frames, fft, n_levels/srcs)
        :param masks:
        :return:
        """
        n_levels = masks.shape[-1]
        leaf_mask = masks[:, :, :, 0]

        result_masks = [leaf_mask]
        for mask_lvl in range(1, n_levels):
            parent_mask = masks[:, :, :, mask_lvl]
            parent_mask = torch.max(leaf_mask, parent_mask)
            result_masks.append(leaf_mask)
            leaf_mask = parent_mask

        return torch.stack(result_masks, dim=-1)
