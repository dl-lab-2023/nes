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
        best_mode = ['']
        for search_mode in data[taskid].keys():
            acc = data[taskid][search_mode]
            if acc == best_acc:
                best_mode.append(search_mode)
            if acc > best_acc:
                best_acc = acc
                best_mode = [search_mode]
        for bm in best_mode:
            rank[bm] += 1
    rank_percentage = {method: 100 * rank[method] / sum(rank.values()) for method in rank.keys()}

    # create plotting
    fig = plt.figure(dpi=args.dpi, figsize=(20, 10))
    plt.title('Ranking of Methods')

    x = range(len(rank.keys()))
    bar_colors = ['tab:orange', 'tab:red', 'tab:purple', 'tab:blue', 'tab:green', 'tab:olive']
    plt.bar(x, rank_percentage.values(), color=bar_colors)
    plt.xticks(range(len(rank.keys())), labels=rank.keys())
    plt.ylabel('% of datasets the method\nhad the best accuracy')

    plt.tight_layout()
    save_path = os.path.join(args.save_path, "ranking_bar_plot.jpg")
    fig.savefig(save_path, dpi=fig.dpi)
