import json
import os
from argparse import ArgumentParser, Namespace
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from own_pipeline.util import enable_logging


def load_ensemble_stats(args: Namespace):
    ensembles_by_taskid = {}

    path = args.ensemble_stats_path
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # root='.../ensemble_stats/task_<task-id>_appendix'
            taskid = int(re.findall(r'\d+', root)[-1])
            appendix = root.split('_')[-1]
            with open(file_path, 'r') as json_file:
                if taskid not in ensembles_by_taskid.keys():
                    ensembles_by_taskid[taskid] = {}
                ensembles_by_taskid[taskid][appendix] = json.load(json_file)

    return ensembles_by_taskid


def create_ranking_bar_plot(args: Namespace):
    ensemble_stats = load_ensemble_stats(args)

    # create ranking
    rank = {search_mode: 0 for search_mode in ['hp', 'nas', 'initweights']}
    for taskid in ensemble_stats.keys():
        best_acc = 0
        best_mode = ''
        for search_mode in ensemble_stats[taskid].keys():
            acc = ensemble_stats[taskid][search_mode]['evaluation']['acc']
            if acc > best_acc:
                best_acc = acc
                best_mode = search_mode
        rank[best_mode] += 1

    # create plotting
    fig = plt.figure(dpi=args.dpi, figsize=(12.8, 9.6))
    plt.title('Ranking of different Methods', x=0.4)

    x = range(len(rank.keys()))
    bar_colors = ['tab:red', 'tab:purple', 'tab:blue', 'tab:green', 'tab:orange']
    plt.barh(x, rank.values(), color=bar_colors)
    plt.xticks(range(max(rank.values()) + 1))
    plt.yticks(range(len(rank.keys())), labels=rank.keys())
    plt.xlabel('Rank')

    plt.tight_layout()
    save_path = os.path.join(args.save_path, "ranking_bar_plot.jpg")
    fig.savefig(save_path, dpi=fig.dpi)


def setup(args: Namespace):
    Path(args.save_path).mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'axes.titlesize': 50})


if __name__ == '__main__':
    enable_logging()
    argParser = ArgumentParser()
    argParser.add_argument(
        "--ensemble_stats_path", type=str,
        default='/home/jonas/workspace/nes/own_pipeline/ensemble_stats',
        help='absolute path to ensemble_stats, e.g. "/workspace/nes/ensemble_stats"'
    )
    argParser.add_argument(
        "--save_path", type=str,
        default="plots",
        help="path where plots are saved"
    )
    argParser.add_argument(
        "--dpi", type=int,
        default=400,
        help='dpi used for plots'
    )
    args = argParser.parse_args()

    setup(args)
    create_ranking_bar_plot(args)
