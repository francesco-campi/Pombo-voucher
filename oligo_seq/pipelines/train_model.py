import argparse
import yaml

import torch
from torch import nn
from torch.utils import data
import optuna
from oligo_seq.database import *
from oligo_seq.models import *


def main():

    #########################
    # read in out arguments #
    #########################

    parser = argparse.ArgumentParser(
        prog="Artificial Dataset",
        usage="generate_artificial_dataset [options]",
        description=main.__doc__,
    )
    parser.add_argument("-c", "--config", help="path to the configuration file", default="config/generate_artificial_dataset.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)


if __name__ == "__main__":
    main()