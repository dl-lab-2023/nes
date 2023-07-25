import json
import os
import re
from argparse import Namespace
from pathlib import Path

from matplotlib import pyplot as plt


def setup(args: Namespace):
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'axes.titlesize': 50})


def load_ensemble_stats(args: Namespace) -> dict[int, [dict[str, dict]]]:
    """
    usage: load_ensemble_stats(args)[233088]["hp"]
    :param args: used for args.ensemble_stats_path
    :returns: dictionary of ensemble statistics first sorted by task-id and then by method (hp, nas,...)
    """

    if args.load_cluster_json:
        return _load_ensemble_stats_cluster_json(args)
    return _load_from_filesystem(args)


def _load_from_filesystem(args: Namespace):
    ensembles_by_taskid_and_method = {}
    path = args.ensemble_stats_path
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # root='.../ensemble_stats/task_<task-id>_appendix'
            taskid = int(re.findall(r'\d+', root)[-1])
            appendix = root.split('_')[-1]
            with open(file_path, 'r') as json_file:
                if taskid not in ensembles_by_taskid_and_method.keys():
                    ensembles_by_taskid_and_method[taskid] = {}
                ensembles_by_taskid_and_method[taskid][appendix] = json.load(json_file)
    return ensembles_by_taskid_and_method


def _load_ensemble_stats_cluster_json(args: Namespace):
    file_path = args.load_cluster_json_path
    ensembles_by_taskid_and_method = {}
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        for key in data.keys():
            taskid = int(re.findall(r'\d+', key)[-1])
            appendix = key.replace(f'task_{taskid}', '')
            if appendix == '':
                continue  # skip weird entries
            appendix = appendix[1:]  # remove leading underscore
            if taskid not in ensembles_by_taskid_and_method.keys():
                ensembles_by_taskid_and_method[taskid] = {}
            ensembles_by_taskid_and_method[taskid][appendix] = data[key]
    return ensembles_by_taskid_and_method
