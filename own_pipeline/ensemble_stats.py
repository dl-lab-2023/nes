import argparse
import os
import json


def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return {
            'evaluation': data['evaluation'],
            'evaluation_avg_baselearner': data['evaluation_avg_baselearner'],
            'evaluation_oracle': data['evaluation_oracle'],
        }


def generate_statistics(base_directory):
    stats = {}
    for root, _, files in os.walk(base_directory):
        task_name = root.split('/')[-1]
        for file_name in files:
            if file_name.startswith('ensemble_') and file_name.endswith('_baselearners_performance.json'):
                file_path = os.path.join(root, file_name)
                stats[task_name] = process_json_file(file_path)

    return stats


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--ensemble_stats_dir",
        type=str,
        required=True,
        help="the directory of the ensemble statsd result")
    args = argParser.parse_args()

    stats = generate_statistics(args.ensemble_stats_dir)
    stats_str = json.dumps(stats, indent=4, sort_keys=True)

    print(f"stats: {stats_str}")

    f = open(f"{args.ensemble_stats_dir}/ensemble_stats.json", "w")
    f.write(stats_str)
    f.close()
