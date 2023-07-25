import os
from argparse import Namespace

from matplotlib import pyplot as plt


def box_plot_accuracy(args: Namespace, stats: dict):
    data = {}
    for method in stats.keys():
        all_data = []
        for task in stats[method].keys():
            all_data.append(stats[method][task])
        data[method] = all_data

    fig = plt.figure(dpi=args.dpi, figsize=(20, 10))
    plt.title("Accuracy of Ensembles")
    box_plot_data = [data[s] for s in data.keys()]
    bplot = plt.boxplot(
        box_plot_data,
        positions=range(2, (len(data.keys()) + 1) * 2, 2),
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"edgecolor": "white", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5}
    )
    colors = ['tab:orange', 'tab:red', 'tab:purple', 'tab:blue', 'tab:green', 'tab:olive']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel("percentage points")
    plt.xticks(range(2, (len(data.keys()) + 1) * 2, 2), labels=data.keys())
    plt.tight_layout()
    save_path = os.path.join(args.save_path, "box_plot_accuracy.jpg")
    fig.savefig(save_path, dpi=fig.dpi)
