<!--
Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Hierarchical Musical Instrument Separation

Source code for the ISMIR 2020 paper, **Hierarchical Musical Instrument Classification**
by Ethan Manilow, Gordon Wichern, and Jonathan Le Roux.

[Please click here to read the paper.](https://archives.ismir.net/ismir2020/paper/000105.pdf)

If you use any part of this code for your work, we ask that you include the following citation:

    @InProceedings{Manilow2020ISMIR10,
      author =	 {Manilow, Ethan and Wichern, Gordon and {Le Roux}, Jonathan},
      title =	 {Hierarchical Musical Source Separation},
      booktitle =	 {Proc. International Society for Music Information Retrieval (ISMIR) Conference},
      year =	 2020,
      month =	 oct
    }


## Table of contents

1. [Environment Setup](#environment-setup)
2. [General Experiment Setup](#general-experiment-setup)
3. [Running Experiments](#running-experiments)
4. [Experiment Definitions for the ISMIR 2020 Paper](#experiment-definitions-for-the-ismir-2020-paper)
5. [Hierarchy Levels](#hierarchy-levels)
6. [Batch updating experiment definitions](#batch-updating-the-experiment-definitions)
7. [What you'll need to update in the experiment definitions](#what-youll-need-to-update-in-the-experiment-definitions)
8. [The first run](#the-first-run)
9. [Training gotchas](#training-gotchas)
10. [License](#license)

## Environment Setup

The code has been tested using `python 3.6`. Necessary dependencies can be installed using the included `requirements.txt`:

```bash
 pip install -r requirements.txt
 ```

## General Experiment Setup

Download the  [Slakh2100 dataset](http://www.slakh.com/) if you haven't already. To replicate the experiments in the paper use `splits_v2` and resample to 16 kHz.

Set up a `yaml` file. The yaml file needs 4 things:
- `output_dir`: Directory where your results go (will be made)
- `code_dir`: Path to `hierarchical/tesbed` (I use an absolute path)
- `main_script`: The name of your script that runs your training (including `.py`).
- `parameters`: A dict with all of the parameters you need to train.

Your main script must have a function called `main(experiment_params)`, where
`experiment_params` is a dict containing the everything you need, stored in the `parameters`
entry of your yaml file above.

## Running Experiments

Run an experiment like so:

```bash

python run_exp.py -e hierarchical/experiment_defs/maskinf_cortex.yaml

```

Flags:
`--exp` or `-e`: Path to the experiment definition. Yaml file defined as above.

Any unknown flags will be added to the `parameters` dict, so you can change experiment settings
from the command line. If a parameter exists in the yaml file, it will be overwritten with the
cmd line flag. There is no spelling or collision check, so new items are added to the dict and
existing items are overwritten.


**What this script does:**

This script will make a copy of the `code_dir` into a new folder alongside the log file, best model,
experiment definition, and `poetry.lock` file (which has the dependencies frozen) such that every
experiment is completely reproducible. The new folder's name will be a UUID within the `output_dir`
directory.


## Experiment Definitions for the ISMIR 2020 Paper

The experiment definition files are yaml files that have all of the params for each
experiment. All of the exp def files for this paper are in the `paper_experiments/` dir
in this repo. They are organized as follows:

- `a_experiments_guitar/`: Single-level Single-Source networks for Guitar. `maskinf_lvl*.yaml` are the first row in
    Table 3 of the paper, training on the full dataset. The other two (with a `XXpct_removed` tag) are
    for training a regular mask inference network with 50% or 90% of the data missing
    (unreported in the paper).
- `b_experiments_guitar`: Multi-level Single-Source networks for Guitar.
    `maskinf_nohier_lvl123.yaml` and `maskinf_yeshier_lvl123.yaml` are for Table 2
    of the paper. `maskinf_nohier_lvl123.yaml` is also used for the second row
    of Table 3.
- `c_experiments_qbe`: Single-level Multi-Source networks via QBE. Used for the third
    row in Table 3.
- `d_experiments_qbe`: Multi-level Multi-Source networks via QBE. Used for the forth
    row in Table 3.
- `e_experiments_guitar` / `e_experiments_qbe`: Experiments for removing leaf/all data
    for values in Table 4.


[See the paper for more details.](https://archives.ismir.net/ismir2020/paper/000105.pdf)

## Hierarchy levels

There are 4 usable hierarchy levels as of this writing. They probably need to be renamed, but
the code works right now, and I don't want to break it. They are all defined
by the following numbers:

    MIX = 0  # the full mixture, don't use this
    SUPER_DUPER_CLASS = 1
    SUPER_CLASS = 2
    INST_CLASS = 3
    INST_NAME = 4


Here are the definitions:

- `SUPER_DUPER_CLASS`: clusters of similar instrument types (`strings_guitar_bass`,
`keyboards_electric_keyboards`, `winds_voices`, and `percussion`)
- `SUPER_CLASS`: One instrument type ('all guitars' or 'all strings')
- `INST_CLASS`: A class of instruments like 'clean guitars', or 'effected guitars'
- `INST_NAME`: The most specific type, 'acoustic guitar', 'clean electric guitar'



### Batch updating the experiment definitions

If you need to change a value in the experiments def en masse (like the data paths),
use the provided script `update_exp_defs.py`. It uses the file `common_variables.yaml`
and replaces values of all the experiment defs in a directory. For example, lets say
my `common_varialbes.yaml` looks like this:

    # path to slakh (split into train, val, test subdirs)
    slakh_base_dir: "/media/hdd_8tb/slakh2100_16k_split"

    # path to hierarchy file
    hierarchy_file: "/home/emanilow/hierarchical/testbed/hierarchy_defs/medleydb_hierarchy_mod2.json"

When I run `update_exp_defs.py`, it will replace `slakh_base_dir` in every experiment
definition in a directory with the value above. You can add any key from inside your experiment
definition.

Run `update_exp_defs.py` like this:

    python update_exp_defs.py -e path/to/experiments/dir -d

The `-d` flag is the "dry run" flag. Run it with that flag to see the results, and remove
it to change the yaml for real.

### What you'll need to update in the experiment definitions

All paths (all absolute paths):

- `code_dir`: Path to `testbed` directory of this repo
- `slakh_base_dir`: Base dir of Slakh with `train`/`val`/`test` subdirectories
- `hierarchy_file`: Path to `testbed/hierarchical_defs/ismir_hierarchy.json`
- `train_saliency_file`: Path to a saliency file for the training split (will be created)
- `val_saliency_file`: Path to a saliency file for the validation split (will be created)
- `test_saliency_file`: Path to a saliency file for the test split (will be created)
- `test_subset_saliency_file`: Path to a saliency file for the test subset (will be created)

All multithreading/processing things:

- `saliency_threads`: num processes to make the saliency files (only needs to happen once for all
 experiments)
- `num_workers`: number of workers used for on the fly mixing in the dataloaders
- `tt_threads`: num of processes for doing evaluation

## The first run

Before you can run anything you'll need to make a saliency file. This is a giant json
file that contains saliency data for every possible submix for every track. You will have to make
once for the train/val/test splits. This will take a while (I didn't tqdm this), and you
can specify how many threads/processes you want to use to do this (`saliency_threads` in
your experiment def). Once this is done, you can use it for every experiment in this directory.


## Training gotchas

At the top of `train.py` there are a few lines of commented out code that controls how
torch handles its multiprocessing resources. I've had some issues on one machine,
so uncomment those and see if they help.


## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:
```
Copyright (c) 2020-2022 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```
