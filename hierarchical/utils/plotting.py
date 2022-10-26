# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from hierarchical.utils.general import make_np


def make_if_empty(arr):
    arr = make_np(arr)
    if arr is None:
        return np.zeros((100, 100, 100, 100))
    return arr


def plt_all2(query, gt_mask, est_mask, mix_feat, path, e, n, tag, qry_id=None):
    get = lambda m: m.cpu().detach().numpy()
    est_mask = np.log(get(est_mask) + 1e-12)
    gt_mask = np.log(get(gt_mask) + 1e-12)
    mix_feat = get(mix_feat)
    if query is not None:
        query = get(query)
    else:
        query = np.zeros_like(gt_mask)

    prep = lambda m: m[n, :, :].T

    fig = plot_example(prep(query), prep(gt_mask), prep(est_mask), prep(mix_feat), path, e, n, tag, qry_id=qry_id)
    canvas = FigureCanvas(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    img = np.fromstring(s, np.uint8).reshape((height, width, 4))
    return 255 - img


def plot_example(query, gt_mask, est_mask, mix_feat, path, e, idx, tag, qry_id=None):
    plt.close("all")

    n_levels = est_mask.shape[0]
    n_rows = n_levels + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(6 * n_rows, 16), facecolor=[0.98, 0.98, 0.98])

    im = axes[0, 0].imshow(mix_feat, aspect="auto")
    axes[0, 0].set_title("Mixture")
    fig.colorbar(im, ax=axes[0, 0])
    axes[0, 0].invert_yaxis()

    im = axes[0, 1].imshow(query, aspect="auto")
    qry_title = "Query" if qry_id is None else f"Query: {qry_id}"
    axes[0, 1].set_title(qry_title)
    fig.colorbar(im, ax=axes[0, 1])
    axes[0, 1].invert_yaxis()

    lvls = [l for l in range(n_levels)][::-1]
    for l in range(n_levels):
        im = axes[l + 1, 0].imshow(gt_mask[lvls[l], :, :], aspect="auto")
        axes[l + 1, 0].set_title(f"GT Mask, Level {l}")
        fig.colorbar(im, ax=axes[l + 1, 0])
        axes[l + 1, 0].invert_yaxis()

        im = axes[l + 1, 1].imshow(est_mask[lvls[l], :, :], aspect="auto")
        axes[l + 1, 1].set_title(f"Estimated Mask, Level {l}")
        fig.colorbar(im, ax=axes[l + 1, 1])
        axes[l + 1, 1].invert_yaxis()

    plt.savefig(f"{path}/output_{tag}_epoch{e}_idx{idx}.png")

    return fig


def plot_1d(vals, path, name):
    plt.close("all")
    plt.plot(vals)
    plt.xlabel("Iterations")
    plt.ylabel(name)
    plt.title(f"{name} value per iteration")
    plt.savefig(f"{path}/{name}.png")


def _plt_gt(gt_spec, gt_midi, n, path):
    plt.close("all")
    fig, axes = plt.subplots(ncols=2, figsize=(12, 8))
    axes[0].imshow(gt_spec, aspect="auto")
    axes[0].set_title("GT Spec")
    axes[1].imshow(gt_midi, aspect="auto")
    axes[1].set_title("GT MIDI")
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    fig.suptitle(f"idx: {n}")
    plt.savefig(f"{path}/gt_{n}.png")


def plt_gt(batch, n, path):
    get = lambda m: m.cpu().detach().numpy()
    midi = get(batch["midi"])[n, :, :, :]
    midi = np.sum(midi.astype(int), axis=0)
    spec = get(batch["mix_feat"])[n, :, :]
    _plt_gt(spec, midi, n, path)


def plot_anchors(anchors, path, e):
    plt.close("all")
    anchors = anchors.cpu().detach().numpy()
    n_examples = anchors.shape[0]
    n_dim = anchors.shape[1]

    x = np.linspace(1, n_dim, n_dim)
    nudge = 1 / (n_examples + 2)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    fig, ax = plt.subplots(figsize=(18, 9), facecolor=[0.98, 0.98, 0.98])
    for i in range(1, n_examples + 1):
        x_ = x + nudge * i
        markers, stemlines, baseline = plt.stem(
            x_, anchors[i - 1, :], use_line_collection=True, markerfmt=" ", label=f"idx {i}"
        )
        plt.setp(baseline, color="black", linewidth=0.5)
        plt.setp(markers, marker=".", markersize=5, markeredgecolor=colors[i], markeredgewidth=2)
        plt.setp(stemlines, linestyle="-", color=colors[i], linewidth=1.5)

    plt.title(f"Anchor examples - epoch {e}")
    ax.set_xticks(np.linspace(min(x), len(x) + 1, len(x) + 1), minor=False)
    ax.xaxis.grid(True, which="major")
    plt.legend()
    plt.savefig(f"{path}/anchors_epoch{e}.png")
    canvas = FigureCanvas(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    img = np.fromstring(s, np.uint8).reshape((height, width, 4))
    return img
