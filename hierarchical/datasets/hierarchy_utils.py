#!/usr/bin/env python3
# Copyright (C) 2020-2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from collections import deque
from enum import Enum

import pandas as pd

SAL = "salient"
NSAL = "nonsalient"


class OldHierarchyLevel(Enum):
    # NOT USED
    SUPER_CLASS = 0
    INST_CLASS = 1
    INST_NAME = 2
    PATCH = 3
    NOTE = 4


class HierarchyLevel(Enum):
    MIX = 0
    SUPER_DUPER_CLASS = 1
    SUPER_CLASS = 2
    INST_CLASS = 3
    INST_NAME = 4
    PATCH = 5
    NOTE = 6


def sanitize_(inst):
    return inst.replace(" ", "_")


def depth(d):
    queue = deque([(id(d), d, 0)])
    memo = set()
    level = None
    while queue:
        id_, o, level = queue.popleft()
        if id_ in memo:
            continue
        memo.add(id_)
        if isinstance(o, dict):
            queue += ((id(v), v, level + 1) for v in o.values())
    return level


def flatten_dict(d, sep="/"):
    """
    Nice trick to flatten a nested dict.

    from here: https://stackoverflow.com/a/41801708
    :param d:
    :param sep:
    :return:
    """
    return pd.io.json.json_normalize(d, sep=sep).to_dict(orient="records")[0]


def level_title(l):
    return HierarchyLevel(l).name.lower()


def determine_offset(hierarchy_file_path):
    if "medleydb_hierarchy" in hierarchy_file_path:
        return 1
    else:
        return 0


def get_hierarchy_patches(hierarchy_file_path, level_list):
    """
    output is a dict that looks like this:
    {
    'level_name0': {
        'inst0_0' : [plugin0_0_0, plugin0_0_1, ..., pluginN],
        'inst0_1' : [plugin0_1_1, plugin0_1_2, ..., pluginN],
        },
    'level_name1': {
        'inst1_0' : [plugin1_0_0, plugin1_0_1, ..., pluginN],
        'inst1_1' : [plugin1_1_1, plugin1_1_2, ..., pluginN],
        },

    }


    :param hierarchy_file_path: Path to hierarchy json file
    :param level_list: List of ints
    :return:
    """
    hier = json.load(open(hierarchy_file_path, "r"))
    offset = determine_offset(hierarchy_file_path)
    lvl_dicts = {level_title(l + offset): {} for l in range(depth(hier))}

    sep = "|"
    hier_flattened = flatten_dict(hier, sep=sep)
    for k, v in hier_flattened.items():
        levels = k.split(sep)
        for lvl, lvl_name in enumerate(levels):
            lvl += offset
            if lvl_name not in lvl_dicts[level_title(lvl)]:
                lvl_dicts[level_title(lvl)][lvl_name] = []
            lvl_dicts[level_title(lvl)][lvl_name].extend(v)

    target_levels = [level_title(l) for l in level_list]
    result = {k: v for k, v in lvl_dicts.items() if k in target_levels}
    return result


def _invert_hierarchy_patches(hierarchies):
    """
    Inverts a dictionary returned from get_hierarchy_patches()
    such that the patch_name is the key and the level_name is the value.

    This makes it easy to figure out the level_name if you know the
    patch_name.

    :param hierarchies: Output dictionary of get_hierarchy_patches()
    :return:
        output is a dict that looks like this:
        {
        'level_name0': {
            'plugin0_0_0' : 'inst0_0',
            'plugin0_0_1' : 'inst0_0',
            'plugin0_1_1' : 'inst0_1'
            },
        'level_name1': {
            'plugin1_0_0' : 'inst1_0',
            'plugin1_0_1' : 'inst1_0',
            'plugin1_1_1' : 'inst1_1'
            },

        }
    """
    inv_hierarchies = {}
    for h_name, h_dict in hierarchies.items():
        if h_name not in inv_hierarchies:
            inv_hierarchies[h_name] = {}
        for level_name, inst_list in h_dict.items():
            for inst in inst_list:
                inv_hierarchies[h_name][inst] = level_name

    return inv_hierarchies


def flatten(hier, hierarchy_levels):
    """
    Not the most elegant way to do this, but it works....
    :param hier:
    :return:
    """
    levels = sorted([l.value for l in hierarchy_levels])
    result = {}
    boxed_hier = [i.split("|") for i in flatten_dict(hier, sep="|").keys()]
    boxed_hier = [[b[l] for l in levels] for b in boxed_hier]
    for path in boxed_hier:
        for j, node in enumerate(path):
            node_name = sanitize_(node)
            if node_name not in result:
                is_leaf = j == len(path) - 1
                parents = [sanitize_(n) for idx, n in enumerate(path) if idx < j]
                root = path[0]
                children = [] if is_leaf else list(set([sanitize_(p[j + 1]) for p in boxed_hier if p[j] == node]))
                level = level_title(levels[j])
                if parents:
                    par = parents[0]
                    siblings = [sanitize_(p[j]) for p in boxed_hier if p[0] == par and p[j] != node]
                else:
                    siblings = [sanitize_(p[j]) for p in boxed_hier if p[j] != node]
                siblings = sorted(list(set(siblings)))
                result[node_name] = {
                    "is_leaf": is_leaf,
                    "parents": parents,
                    "root": root,
                    "children": children,
                    "level": level,
                    "siblings": siblings,
                }

    return result
