import json
import logging
import os
import re
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
    for root, directories, files in os.walk(path, ):
        for file in files:
            if f'task_{taskid}_' in root:
                num_baselearners = re.findall(r'\d+', file)[-1]
                num_baselearners = int(num_baselearners)
                method = root.split('_')[-1]
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as json_file:
                    logging.info(f"loading file {json_file}")
                    data = json.load(json_file)
                if num_baselearners not in evaluations:
                    evaluations[num_baselearners] = {}
                evaluations[num_baselearners][method] = data
    x = list(range(20, 280, 20))
    y_data = {}
    for num_baselearners in range(20, 261, 20):
        for method in evaluations[num_baselearners].keys():
            if method not in y_data:
                y_data[method] = {"acc": [], "err1": [], "err2": []}
            acc = evaluations[num_baselearners][method]["evaluation"]["acc"] * 100
            err = evaluations[num_baselearners][method]["evaluation"]["ece"]
            y_data[method]["acc"].append(acc)
            y_data[method]["err1"].append(acc + err)
            y_data[method]["err2"].append(acc - err)
    fig = plt.figure(dpi=args.dpi, figsize=(13, 7))
    for method in y_data.keys():
        plt.plot(x, y_data[method]["acc"], 'o-', linewidth=4, label=name(method), color=color(method))
        # plt.fill_between(x, y_data[method]["err1"], y_data[method]["err2"], color=color(method), alpha=0.1)

    plt.title(f"Dataset openml-{taskid}")
    plt.legend()
    plt.ylabel('Accuracy in %')
    plt.xlabel('Amount of Base Learners')
    plt.tight_layout()
    save_path = os.path.join(args.save_path, f"fig6_{taskid}.jpg")
    fig.savefig(save_path, dpi=fig.dpi)
