import os
from argparse import Namespace

import matplotlib.pyplot as plt


def create_ranking_bar_plot(args: Namespace, acc_stats: dict):
    # swap directory-dimensions
    data = {}
    for method in acc_stats.keys():
        for taskid in acc_stats[method].keys():
            if taskid not in data:
                data[taskid] = {}
            data[taskid][method] = acc_stats[method][taskid]

    rank = {method: 0 for method in acc_stats.keys()}
    for taskid in data.keys():
        best_acc = 0
        best_mode = ''
        for search_mode in data[taskid].keys():
            acc = data[taskid][search_mode]
            if acc > best_acc:
                best_acc = acc
                best_mode = search_mode
        rank[best_mode] += 1

    # create plotting
    fig = plt.figure(dpi=args.dpi, figsize=(12.8, 9.6))
    plt.title('Ranking of Methods', x=0.4)

    x = range(len(rank.keys()))
    bar_colors = ['tab:orange', 'tab:red', 'tab:purple', 'tab:blue', 'tab:green']
    plt.barh(x, rank.values(), color=bar_colors)
    plt.yticks(range(len(rank.keys())), labels=rank.keys())
    plt.xlabel('Rank')

    plt.tight_layout()
    save_path = os.path.join(args.save_path, "ranking_bar_plot.jpg")
    fig.savefig(save_path, dpi=fig.dpi)
