# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random

import numpy as np
import torch

from hierarchical.nets.mask_inference import MaskInferenceNussl
from hierarchical.nets.qbe_dan import QBE_ADAN


def create_model(params):
    model_name = params["model_name"].lower()
    model_name = "".join([i for i in model_name if i.isalpha()])
    n_fft = int(get_n_fft_bins(params["fft_frame_size"]))
    hierarchy_levels = params.get("hierarchy_levels", None)
    hc = params.get("model_hierarchical_constraint", False)
    n_levels = 1 if hierarchy_levels is None else len(hierarchy_levels)
    return_level = params.get("return_level", None)
    n_levels = 1 if return_level is not None else n_levels

    n_layers = int(params["n_layers"])
    n_hidden = int(params["n_hidden"])
    dropout = float(params["dropout"])

    if model_name == "qbeadan":
        qry_n_layers = int(params["qry_n_layers"])
        qry_hidden_units = int(params["qry_hidden_units"])
        qry_embed_dim = int(params["qry_embed_dim"])
        knob_strategy = params.get("knob_strategy", None)
        return QBE_ADAN(
            n_layers,
            n_hidden,
            qry_n_layers,
            qry_hidden_units,
            qry_embed_dim,
            n_fft,
            n_levels,
            dropout=dropout,
            hierarchical_constraint=hc,
            knob_strategy=knob_strategy,
            cat_levels=len(hierarchy_levels),
        )

    elif model_name == "maskinf":
        return MaskInferenceNussl(n_fft, n_levels, n_hidden, n_layers, True, dropout, hierarchical_constraint=hc)

    else:
        raise ValueError("IDK WTF U THINK UR DOIN")


def make_np(arr):
    if isinstance(arr, torch.Tensor):
        return arr.cpu().detach().numpy()
    return arr


def get_n_fft_bins(n_fft):
    return n_fft // 2 + 1


def batch_to_device(batch, device):
    new_batch = []
    for b in batch:
        new_dict = {}
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                new_dict[k] = v.to(device=device)
            else:
                new_dict[k] = v
        new_batch.append(new_dict)
    return new_batch


def seed(s, set_cudnn=False):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_best_model(model, epoch, tr_loss, cv_loss, model_save_dir):
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "tr_loss": tr_loss, "cv_loss": cv_loss},
        os.path.join(model_save_dir, "best_model.pyt"),
    )


def load_model(params, device, model_save_dir, map_loc=None):
    model = create_model(params).to(device=device)

    checkpoint = torch.load(os.path.join(model_save_dir, "best_model.pyt"), map_location=map_loc)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    try:
        tr_loss = checkpoint["tr_loss"]
        cv_loss = checkpoint["cv_loss"]
    except:
        tr_loss, cv_loss = None, None

    return model, epoch, tr_loss, cv_loss
