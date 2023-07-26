import json
import os
from argparse import Namespace

import matplotlib.pyplot as plt

from own_pipeline.plotting.shared import name


def color(method: str) -> str:
    if method == 'hp':
        return 'orange'
    if method == 'nas':
        return 'purple'
    if method == 'initweights':
        return 'green'


def create_fig_6(args: Namespace):
    path = args.multi_ensemble_stats_dir
    taskid = args.taskid

    evaluations = {}
    for root, directories, files in os.walk(path):
        for file in files:
            if f'task_{taskid}_' in root:
                num_baselearners = int(root.split('_')[-1])
                method = root.split('_')[-2]
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as json_file:
                    print(f"loading file {json_file}")
                    data = json.load(json_file)
                if num_baselearners not in evaluations:
                    evaluations[num_baselearners] = {}
                evaluations[num_baselearners][method] = data
    x = evaluations.keys()
    y_data = {}
    for num_baselearners in evaluations.keys():
        for method in evaluations[num_baselearners].keys():
            if method not in y_data:
                y_data[method] = {"acc": [], "err1": [], "err2": []}
            acc = evaluations[num_baselearners][method]["evaluation"]["acc"]
            err = evaluations[num_baselearners][method]["evaluation"]["ece"]
            y_data[method]["acc"].append(acc)
            y_data[method]["err1"].append(acc + err)
            y_data[method]["err2"].append(acc - err)

    fig = plt.figure(dpi=args.dpi, figsize=(13, 7))
    for method in y_data.keys():
        plt.plot(x, y_data[method]["acc"], 'o-', linewidth=4, label=name(method), color=color(method))
        plt.fill_between(x, y_data[method]["err1"], y_data[method]["err2"], color=color(method), alpha=0.1)

    plt.title(f"Dataset openml-{taskid}")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(args.save_path, "fig6.jpg")
    fig.savefig(save_path, dpi=fig.dpi)
