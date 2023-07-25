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


def load_acc_stats(args: Namespace) -> dict[int, [dict[str, dict]]]:
    """
    usage: load_ensemble_stats(args)["hp-ensemble"][220338"]
    :param args: used for args.ensemble_stats_path
    :returns: dictionary of statistics first sorted by method and then by task-id
    """

    ensembles = _load_ensemble_stats_json(args)
    baselearners = _load_best_baselearners_from_filesystem(args)
    combined_dict = {}
    for key in ensembles.keys():
        combined_dict[key] = ensembles[key]
    for key in baselearners.keys():
        combined_dict[key] = baselearners[key]
    return combined_dict


def _load_best_baselearners_from_filesystem(args: Namespace):
    bl_by_task_appendix = {}
    path = args.saved_model_path
    for root, directories, files in os.walk(path):
        for file in files:
            if file != "train_performance.json":
                continue
            taskid = int(re.findall(r'\d+', root)[-2])  # last is seed
            appendix = root.split("/")[-2].replace(f'task_{taskid}_', '')
            appendix = f"{appendix}-baselearner"
            with open(os.path.join(root, file), 'r') as json_file:
                acc = float(json.load(json_file)["evaluation"]["acc"])
            if appendix not in bl_by_task_appendix:
                bl_by_task_appendix[appendix] = {}
            if taskid not in bl_by_task_appendix[appendix]:
                bl_by_task_appendix[appendix][taskid] = 0
            bl_by_task_appendix[appendix][taskid] = max(bl_by_task_appendix[appendix][taskid], acc)
    return bl_by_task_appendix


def _load_ensemble_stats_json(args: Namespace):
    file_path = args.load_cluster_json_path
    ensembles_by_method_and_taskid = {}
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        for key in data.keys():
            taskid = int(re.findall(r'\d+', key)[-1])
            appendix = key.replace(f'task_{taskid}_', '')
            appendix = f"{appendix}-ensemble"
            if appendix not in ensembles_by_method_and_taskid.keys():
                ensembles_by_method_and_taskid[appendix] = {}
            ensembles_by_method_and_taskid[appendix][taskid] = data[key]['evaluation']['acc']
    return ensembles_by_method_and_taskid
