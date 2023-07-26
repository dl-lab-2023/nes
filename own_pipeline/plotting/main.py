from argparse import ArgumentParser

from own_pipeline.plotting.box_plot_accuracy import box_plot_accuracy
from own_pipeline.plotting.create_fig_6 import create_fig_6
from own_pipeline.plotting.create_ranking_bar_plot import create_ranking_bar_plot
from own_pipeline.plotting.shared import setup, load_acc_stats
from own_pipeline.util import enable_logging

if __name__ == '__main__':
    enable_logging()
    argParser = ArgumentParser()
    argParser.add_argument(
        "--ensemble_stats_path", type=str,
        default='/home/jonas/workspace/nes/own_pipeline/ensemble_stats',
        help='absolute path to ensemble_stats, e.g. "/workspace/nes/ensemble_stats"'
    )
    argParser.add_argument(
        "--saved_model_path", type=str,
        default='/home/jonas/workspace/nes/saved_model',
        help='absolute path to saved_model, e.g. "/workspace/nes/saved_model"'
    )
    argParser.add_argument(
        "--save_path", type=str,
        default="plots",
        help="path where plots are saved"
    )
    argParser.add_argument(
        "--dpi", type=int,
        default=1000,
        help='dpi used for plots'
    )
    argParser.add_argument(
        "--figure_6_step_size", type=int,
        default=25,
        help="increment of base-learner-amount for ensemble at each step"
    )
    argParser.add_argument(
        "--search_modes",
        default=['hp', 'nas', 'initweights']
    )
    argParser.add_argument(
        "--load_cluster_json", type=bool,
        default=True,
        help="set to load cluster json"
    )
    argParser.add_argument(
        "--load_cluster_json_path", type=str,
        default="/home/jonas/workspace/nes/own_pipeline/plotting/plots/ensemble_stats.json",
        help="pass absolute path of cluster-json"
    )
    argParser.add_argument(
        "--multi_ensemble_stats_dir", type=str,
        default="/home/jonas/workspace/nes/multi_ensemble_stats"
    )
    argParser.add_argument(
        "--taskid", type=int,
        default=233088
    )
    args = argParser.parse_args()

    setup(args)

    print("load_acc_stats")
    stats = load_acc_stats(args)

    print("box_plot_accuracy")
    box_plot_accuracy(args, stats)

    print("create_ranking_bar_plot")
    create_ranking_bar_plot(args, stats)

    # print("create_fig_6")
    # create_fig_6(args)
