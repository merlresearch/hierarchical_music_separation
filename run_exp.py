# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import datetime
import importlib.util
import os
import re
import shutil
import uuid

import yaml

pattern = re.compile("[\W_]+")


def parse_args(args_list):
    arg_names = [pattern.sub("", s) for s in args_list[::2]]
    result = {}
    for i, k in enumerate(arg_names):
        result[k] = _sanitize_value(args_list[2 * i + 1])
    return result


def _sanitize_value(v):
    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        pass

    if v.lower() in ["null", "none"]:
        return None

    if isinstance(v, str):
        if v.lower() == "false":
            return False
        if v.lower() == "true":
            return True

    return v


def _update_dict(dict_, param, val):
    for k in dict_.keys():
        if isinstance(k, str):
            key = pattern.sub("", k)
            if param == key:
                dict_[k] = _sanitize_value(val)
    return dict_


def update_parameters(exp_dict, params_dict):
    sanitized_exp_keys = {pattern.sub("", key): key for key in exp_dict.keys()}
    for param_key, param_val in params_dict.items():
        if param_key in sanitized_exp_keys.keys():
            exp_dict[sanitized_exp_keys[param_key]] = param_val
    return exp_dict


def run_experiment(code_dir, main_script, experiment_dict):
    script_path = os.path.join(code_dir, main_script)
    module_name = f"{os.path.splitext(main_script)[0]}.main"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)
    experiment_module.main(experiment_dict)


def main():
    """
    This script replaces values in the yaml experiment definition with ones provided
    on the command line. It makes a new json file with a random GUID name which
    will get moved and renamed by the experiment script.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", "-e", type=str, help="Experiment yaml file")
    parser.add_argument("--debug", "-d", action="store_true")
    exp, exp_args = parser.parse_known_args()
    exp_file_path = exp.exp
    debug = exp.debug

    if not exp_file_path:
        raise ParameterError("Must have an experiment definition `--exp-def`!")
    exp_dict = yaml.load(open(os.path.join(exp_file_path), "r"), Loader=yaml.FullLoader)

    output_base_dir = exp_dict["output_dir"]

    if len(exp_args) > 0:
        try:
            new_args = parse_args(exp_args)
        except IndexError:
            raise ParameterError("Cannot parse input parameters!")
        new_dict = update_parameters(exp_dict["parameters"], new_args)
        exp_dict["parameters"] = new_dict

    # Create an output directory
    exp_id = str(uuid.uuid4().hex)
    output_dir = os.path.join(output_base_dir, exp_id)
    code_dir = exp_dict["code_dir"]
    exp_dict["id"] = exp_id
    exp_dict["debug"] = True

    if not debug:
        shutil.copytree(code_dir, output_dir)  # This makes a new dir

        # Update the json to have new output directory
        exp_dict["output_dir"] = output_dir
        exp_dict["start_time"] = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

        new_path = os.path.join(output_dir, "exp_def.yaml")
        yaml.dump(exp_dict, open(new_path, "w"))
        exp_dict["debug"] = False

    # Run the experiment
    main_script = exp_dict["main_script"]
    run_experiment(code_dir, main_script, exp_dict)


class ParameterError(Exception):
    pass


if __name__ == "__main__":
    main()
