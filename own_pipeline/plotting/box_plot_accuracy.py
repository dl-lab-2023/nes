import os
from argparse import Namespace

from matplotlib import pyplot as plt

from own_pipeline.plotting.shared import load_acc_stats


def box_plot_accuracy(args: Namespace):
    stats = load_acc_stats(args)

    # save acc_ensemble - acc_avg_baselearner in dict for each search_mode
    data = {}
    for method in stats.keys():
        all_data = []
        for task in stats[method].keys():
            all_data.append(stats[method][task])
        data[method] = all_data

    fig = plt.figure(dpi=args.dpi, figsize=(12.8, 9.6))
    plt.title("Accuracy of Ensembles")
    box_plot_data = [data[s] for s in data.keys()]
    plt.boxplot(
        box_plot_data,
        positions=range(2, (len(data.keys()) + 1) * 2, 2),
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 1.5},
        whiskerprops={"color": "C0", "linewidth": 1.5},
        capprops={"color": "C0", "linewidth": 1.5}
    )
    plt.ylabel("percentage points")
    plt.xticks(range(2, (len(data.keys()) + 1) * 2, 2), labels=data.keys())
    plt.tight_layout()
    save_path = os.path.join(args.save_path, "box_plot_accuracy.jpg")
    fig.savefig(save_path, dpi=fig.dpi)
