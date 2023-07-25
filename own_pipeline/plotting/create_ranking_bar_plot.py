import os
from argparse import Namespace

import matplotlib.pyplot as plt

from own_pipeline.plotting.shared import load_ensemble_stats


def create_ranking_bar_plot(args: Namespace):
    ensemble_stats = load_ensemble_stats(args)

    # create ranking
    rank = {search_mode: 0 for search_mode in args.search_modes}
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
    plt.yticks(range(len(rank.keys())), labels=rank.keys())
    plt.xlabel('Rank')

    plt.tight_layout()
    save_path = os.path.join(args.save_path, "ranking_bar_plot.jpg")
    fig.savefig(save_path, dpi=fig.dpi)
