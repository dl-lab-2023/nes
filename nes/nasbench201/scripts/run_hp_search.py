import argparse
import warnings

warnings.filterwarnings('ignore')

# TODO add code following the guideline/change guideline
"""
This file is used to run a Random-Search for Hyperparameters for a given architecture.
The flow is as follows:

- read architecture
- search
    - sample hp-config
    - train (and evaluate?) model
- ensemble
    - create ensemble with previous trained models
    - evaluate ensemble performance
- evaluate
    - plot ?!
"""


# TODO copy/paster from run_nes_rs.py stuff that is needed
def hp_search(_args: argparse.Namespace):
    print("run hp search")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO add arguments to run hp_search
    args = parser.parse_args()

    hp_search(args)
