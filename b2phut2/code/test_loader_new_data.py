from __future__ import print_function
# Import comet_ml in the top of your file

import argparse
import datetime
import os
import pprint
import random
import sys

import torch

from utils.config import cfg, cfg_from_file
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p
from train import parse_args, load_config, fix_seed

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

if __name__ == '__main__':
    print("pytorch version {}".format(torch.__version__))
    # Load the config file

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: {}".format(device))


    # Prepare data
    (train_loader,
     valid_loader,
     test_from_train_loader) = prepare_dataloaders(
        dataset_split='train',
        dataset_path='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293/train/',
        metadata_filename='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293/train/avenue_train_metadata_split.pkl',
        batch_size=2,
        sample_size=1,
        valid_split=0.1,
        test_split=0.1,
        num_worker=1)

    img = next(iter(train_loader))
    print("img", img)


