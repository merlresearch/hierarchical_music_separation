# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import yaml


def main(experiments_dir, dry_run=False):
    if dry_run:
        print("Doing dry run")
    else:
        print("Updating yaml files")
    paths = yaml.load(open("common_variables.yaml"), Loader=yaml.SafeLoader)

    for exp_yaml in Path(experiments_dir).rglob("*.yaml"):
        update_paths(exp_yaml, paths, dry_run)

    if dry_run:
        print("\nThis was a dry run; I did not change any values.")
        print("Rerun this script without the -d flag to change paths.")


def update_paths(experiment_file, new_paths, dry_run=False):
    print(f"\nWorking on '{experiment_file}'")
    exp_defs = yaml.load(open(experiment_file), Loader=yaml.SafeLoader)

    try:
        for path_key, path in new_paths.items():

            if path_key in exp_defs.keys():
                exp_defs[path_key] = path
                print(f"Set '{path_key:20}' to '{path}'")

            if path_key in exp_defs["parameters"].keys():
                exp_defs["parameters"][path_key] = path
                print(f"Set '{path_key:20}' to '{path}'")

    except Exception as e:
        print(f"Error with {experiment_file}: {e}")
        return

    if not dry_run:
        with open(experiment_file, "w") as f:
            yaml.dump(exp_defs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", "-e", type=str, help="Experiment files directory")
    parser.add_argument("--dry-run", "-d", action="store_false", help="Dry run, if false")
    args = parser.parse_args()
    main(args.exp_dir, args.dry_run)
