import os
from argparse import Namespace

from matplotlib import pyplot as plt

from own_pipeline.plotting.shared import load_ensemble_stats


def box_plot_ensemble_improving_over_avg_baselearner(args: Namespace):
    ensemble_stats = load_ensemble_stats(args)

    # save acc_ensemble - acc_avg_baselearner in dict for each search_mode
    data = {search_mode: [] for search_mode in args.search_modes}
    for taskid in ensemble_stats.keys():
        for search_mode in ensemble_stats[taskid].keys():
            acc_ensemble = ensemble_stats[taskid][search_mode]["evaluation"]["acc"]
            acc_avg_baselearner = ensemble_stats[taskid][search_mode]["evaluation_avg_baselearner"]["acc"]
            difference = acc_ensemble - acc_avg_baselearner
            data[search_mode].append(difference)

    fig = plt.figure(dpi=args.dpi, figsize=(12.8, 9.6))
    plt.title("Accuracy Gain of Ensemble\ncompared to Avg. Baselearner\nover all tasks", x=0.4)
    box_plot_data = [data[s] for s in args.search_modes]
    plt.boxplot(
        box_plot_data,
        positions=[2, 4, 6],
        widths=1.5,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 1.5},
        whiskerprops={"color": "C0", "linewidth": 1.5},
        capprops={"color": "C0", "linewidth": 1.5}
    )
    plt.ylabel("percentage points")
    plt.xticks([2, 4, 6], labels=data.keys())
    plt.tight_layout()
    save_path = os.path.join(args.save_path, "box_plot_ensemble_improving_over_avg_baselearner.jpg")
    fig.savefig(save_path, dpi=fig.dpi)
