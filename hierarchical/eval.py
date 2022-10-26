#!/usr/bin/env python3

# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
import os
from multiprocessing.dummy import Pool as ThreadPool

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from hierarchical.datasets.utils import dataset_creator
from hierarchical.loss_functions.utils import loss_creator
from hierarchical.utils.general import batch_to_device, load_model, seed
from hierarchical.utils.signal_lib import _do_si_sdr, detach, do_istft, get_output


def plot_srcs(gt_srcs, est_srcs, out_file=None):
    n_src = est_srcs.shape[-1]
    n_rows = n_src
    height = 3 * n_rows
    plt.close("all")
    fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(9, height), sharex=True)
    for i in range(n_src):
        ax[i, 0].specgram(gt_srcs[:, i], Fs=16000)
        ax[i, 0].set_title("GT Spec")
        ax[i, 1].specgram(est_srcs[:, i], Fs=16000)
        ax[i, 1].set_title("EST Spec")
    if not out_file:
        plt.show()
    else:
        plt.savefig(out_file)


def parse_name(name):
    track, src_name, src_level = name.split()
    track = track.replace(".", "")
    src_level = src_level[1:-1]
    return track, src_name, src_level


def do_sep_eval(mix_stft, gt_stfts, est_masks, src_names, stft_params, id_, scaling=True, audio_dir=None, sr=16000):
    est_srcs = np.vstack(get_output(mix_stft, est_masks, **stft_params))
    n_srcs = len(src_names)
    gt_srcs = np.vstack([do_istft(gt_stfts[0, ..., i].T, **stft_params) for i in range(n_srcs)])
    mix_wav = do_istft(mix_stft.T, **stft_params)

    if audio_dir:
        out_dir = os.path.join(audio_dir, str(id_))
        logger.info(f"Writing output files at {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        sf.write(os.path.join(out_dir, "mix.wav"), mix_wav, sr)
        est_dir = os.path.join(out_dir, "estimated_sources")
        os.makedirs(est_dir, exist_ok=True)
        gt_dir = os.path.join(out_dir, "gt_sources")
        os.makedirs(gt_dir, exist_ok=True)

        for i, src in enumerate(src_names):
            sf.write(os.path.join(est_dir, f"{src}.wav"), est_srcs[i, :], sr)
            sf.write(os.path.join(gt_dir, f"{src}.wav"), gt_srcs[i, :], sr)

    scores = []
    for i, name in enumerate(src_names):
        noisy_sdr, noisy_sir, noisy_sar, _ = _do_si_sdr(mix_wav, gt_srcs[i, :], mix_wav, scaling)
        sdr, sir, sar, _ = _do_si_sdr(est_srcs[i, :], gt_srcs[i, :], mix_wav, scaling)
        if isinstance(name, tuple):
            name = name[0]
        track, src_name, src_level = parse_name(name)
        r = {
            "sdr": sdr,
            "sar": sar,
            "sir": sir,
            "noisy_sdr": noisy_sdr,
            "noisy_sar": noisy_sar,
            "noisy_sir": noisy_sir,
            "src": name,
            "track": track,
            "src_name": src_name,
            "level": src_level,
        }
        scores.append(r)

    return scores


def save_anchors(results_dir, qry_id, anchor):
    anchor = detach(anchor).squeeze()
    qry_id = qry_id[0]
    filename = qry_id.replace(" ", "_").replace(".", "")
    outfile = os.path.join(results_dir, "anchor_data", f"{filename}.json")
    track, src_name, src_level = parse_name(qry_id)
    anchor_dict = {
        "query_id": qry_id,
        "anchor": list(anchor.astype(float)),
        "track": track,
        "src_name": src_name,
        "level": src_level,
    }

    with open(outfile, "w") as f:
        json.dump(anchor_dict, f, indent=4)


def do_one_eval(batch, model, device, id_, stft_params, results_dir, scaling=True, audio_dir=None, sr=16000):
    try:
        logger.info(f"Starting ID {id_}")

        feature_dict, target_dict = batch_to_device(batch, device)
        model_output = model(feature_dict)

        if isinstance(model_output, tuple):
            est_mask, anchor = model_output

            # save the anchors for later analysis
            save_anchors(results_dir, feature_dict["query_id"], anchor)

        else:
            est_mask = model_output
            anchor = None

        join = lambda x: detach(x["abs"]) * np.exp(1j * detach(x["ang"]))
        mix = np.squeeze(join(feature_dict["mix_stft"]))
        gt_srcs = join(target_dict["target_stft"])
        names = target_dict["target_id"]
        res = do_sep_eval(mix, gt_srcs, est_mask, names, stft_params, id_, scaling, audio_dir, sr)
        return res
    except Exception as e:
        logger.info(f"Issue with {id_}: {e} Skipping...")
        return None


def evaluate(exp_dict, writer=None):
    logger.info("Setting up Evaluation")
    exp_name = exp_dict["experiment_name"]
    exp_id = exp_dict["id"]
    output_dir = exp_dict["output_dir"]
    params = exp_dict["parameters"]
    threads = params.get("tt_threads", 1)
    n_fft = params["fft_frame_size"]
    hop_len = params["fft_hop"]
    win_type = "sqrt_hann"
    stft_params = {"n_fft": n_fft, "hop_len": hop_len, "win_type": win_type}
    output_audio = params.get("output_audio", False)
    sr = params["sample_rate"]
    debug = params.get("debug", False)

    if debug:
        logger.info("Running in debug mode")
        threads = 1

    seed(params["seed"])

    if torch.cuda.is_available():
        avail_gpu = GPUtil.getAvailable(
            order="first", limit=1, maxLoad=0.75, maxMemory=0.75, includeNan=False, excludeID=[], excludeUUID=[]
        )[0]
        device = torch.device(f"cuda:{avail_gpu}")
    else:
        device = torch.device("cpu")

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    audio_dir = os.path.join(results_dir, "audio") if output_audio else None
    os.makedirs(os.path.join(results_dir, "anchor_data"), exist_ok=True)

    logger.info("Loading model")
    model_save_dir = os.path.join(output_dir, params["model_save_dir"])

    model, _, _, _ = load_model(params, device, model_save_dir, map_loc=device)
    logger.info("Model loaded successfully")

    logger.info("Setting up dataset")
    _, target, t_kwargs = loss_creator(params)
    evalset = dataset_creator(params, "test", target, t_kwargs)

    eval_batch_file = params.get("test_subset_saliency_file", None)
    if os.path.isfile(eval_batch_file):
        logger.info("Ignore that stuff about determining the IDs...")
        logger.info(f"Loading test chunks from {eval_batch_file}")
        batch_ids = json.load(open(eval_batch_file))
        evalset.batch_ids = {int(k): v for k, v in batch_ids.items()}
    else:
        logger.info("Saving Eval set metadata (chunks, etc) for reproducibility")
        with open(eval_batch_file, "w") as f:
            json.dump(evalset.batch_ids, f)
            logger.info(f"Wrote {eval_batch_file}")

    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=1)
    logger.info("Dataset set up")

    logger.info(f"Beginning Eval loop with {threads} threads...")
    pool = ThreadPool(threads)
    results = []
    with torch.no_grad():
        model.eval()

        def eval_wrapper(i, b):
            return do_one_eval(b, model, device, i, stft_params, results_dir, True, audio_dir, sr)

        if threads == 1:
            for i, b in enumerate(eval_loader):
                results.append(eval_wrapper(i, b))
                if debug and i > 20:
                    logger.info("Debug mode on. Finishing early...")
                    break
        else:
            results += pool.starmap(eval_wrapper, enumerate(eval_loader))
        res = []
        for sblst in results:
            if sblst is not None:
                for it in sblst:
                    if it is not None:
                        res.append(it)
        results = res

    df = pd.DataFrame(results)
    raw_csv_path = os.path.join(results_dir, "raw_results.csv")
    df.to_csv(raw_csv_path)
    logger.info(f"Saved all results to {raw_csv_path}")

    df_ = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    if len(df) != len(df_):
        logger.warning(
            f"{len(df) - len(df_)} examples had results in [inf, -inf, " f"nan]. Removing from statistics summary..."
        )
        df = df_

    metrics = [k for k in df.keys() if any(m in k for m in ["sdr", "sar", "sir"])]
    metric_stats = {f"{k} mean": df[k].mean() for k in metrics}
    metric_stats.update({f"{k} median": df[k].median() for k in metrics})
    metric_stats.update({f"{k} std": df[k].std() for k in metrics})

    logger.info("~~~~~~~~~~~ Evaluation Results Summary ~~~~~~~~~~~")
    logger.info("")
    logger.info("           ++++++ Overall results ++++++")
    logger.info(f"{'Measure':19} {'Mean':7} {'Median':9}  Std")
    for k in metrics:
        logger.info(f"{k:14} =  " f"{df[k].mean():7.3f}, " f"{df[k].median():7.3f}, " f"{df[k].std():7.3f}")
    logger.info(f"Used {len(df)} of {len(eval_loader)} possible.")

    logger.info("")
    logger.info("           ++++++  Level results  ++++++")
    logger.info("")
    for l in df.level.unique():
        logger.info(f"      \\ \\ \\ {l}, n = {len(df[df.level == l])}  / / /")
        logger.info(f"{'Measure':19} {'Mean':7} {'Median':9}  Std")
        for k in metrics:
            logger.info(
                f"{k:14} =  "
                f"{df[df.level == l][k].mean():7.3f}, "
                f"{df[df.level == l][k].median():7.3f}, "
                f"{df[df.level == l][k].std():7.3f}"
            )
        logger.info("")

    logger.info("")
    logger.info("           ++++++ Per Instrument results  ++++++")
    logger.info("")
    for s in df.src_name.unique():
        logger.info(f"      \\ \\ \\ {s}, n = {len(df[df.src_name == s])}  / / /")
        logger.info(f"{'Measure':19} {'Mean':7} {'Median':9}  Std")
        for k in metrics:
            logger.info(
                f"{k:14} =  "
                f"{df[df.src_name == s][k].mean():7.3f}, "
                f"{df[df.src_name == s][k].median():7.3f}, "
                f"{df[df.src_name == s][k].std():7.3f}"
            )
        logger.info("")

    stats_path = os.path.join(results_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Evaluation statistics for model {exp_name} (ID {exp_id})\n")
        f.write("\n")

        f.write(f"{'Measure':19} {'Mean':7} {'Median':9}  Std\n")
        for k in metrics:
            f.write(f"{k:14} =  " f"{df[k].mean():7.3f}, " f"{df[k].median():7.3f}, " f"{df[k].std():7.3f}\n")

        f.write(f"\nUsed {len(df)} of {len(eval_loader)} possible files.\n")

        f.write("\n\n")
        f.write("           ++++++  Level results  ++++++\n")
        for l in df.level.unique():
            f.write(f"      \\ \\ \\ {l}, n = {len(df[df.level == l])}  / / /\n")
            f.write(f"{'Measure':19} {'Mean':7} {'Median':9}  Std\n")
            for k in metrics:
                f.write(
                    f"{k:14} =  "
                    f"{df[df.level == l][k].mean():7.3f}, "
                    f"{df[df.level == l][k].median():7.3f}, "
                    f"{df[df.level == l][k].std():7.3f}\n"
                )
            f.write("\n")

        f.write("\n\n")
        f.write("           ++++++ Per Instrument results  ++++++\n")
        for s in df.src_name.unique():
            f.write(f"      \\ \\ \\ {s}, n = {len(df[df.src_name == s])}  / / /\n")
            f.write(f"{'Measure':19} {'Mean':7} {'Median':9}  Std\n")
            for k in metrics:
                f.write(
                    f"{k:14} =  "
                    f"{df[df.src_name == s][k].mean():7.3f}, "
                    f"{df[df.src_name == s][k].median():7.3f}, "
                    f"{df[df.src_name == s][k].std():7.3f}\n"
                )
            f.write("\n")
    logger.info(f"All results saved at {raw_csv_path}")
    logger.info(f"Summary of results at {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", "-e", type=str, help="Experiment yaml file")
    args = parser.parse_args()
    exp_file_path = args.exp
    exp_dict = yaml.load(open(os.path.join(exp_file_path), "r"), Loader=yaml.FullLoader)
    evaluate(exp_dict)
