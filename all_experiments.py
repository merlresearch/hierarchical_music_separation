# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, help="Base directory containing experiments folders")
    parser.add_argument("--outfile", "-o", type=str, default=None, required=False, help="Output File")
    args = parser.parse_args()
    base_dir = args.dir
    outfile = args.outfile

    exp_dirs = os.listdir(base_dir)

    all_exps = []
    all_ids = []
    for exp in exp_dirs:
        param_file = os.path.join(base_dir, exp, "exp_def.yaml")
        exp_dict = yaml.load(open(os.path.join(param_file), "r"), Loader=yaml.FullLoader)
        all_ids.append(exp)
        params = exp_dict["parameters"]
        for k, v in params.items():
            if isinstance(v, str):
                if os.path.isfile(v) or os.path.isdir(v):
                    params[k] = os.path.basename(v)

        params["experiment_name"] = exp_dict["experiment_name"]
        params["main_script"] = exp_dict["main_script"]
        params["start_time"] = exp_dict["start_time"]
        all_exps.append(params)

    df = pd.DataFrame(data=all_exps, index=all_ids)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", -1)
    pd.options.display.width = None
    if not outfile:
        print(df)
    else:
        with open(outfile, "w") as f:
            f.write(str(df))


if __name__ == "__main__":
    main()
