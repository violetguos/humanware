from __future__ import print_function
# Import comet_ml in the top of your file
from comet_ml import Experiment

import argparse
import datetime
import os
import pprint
import random
import sys
from pathlib import Path
from shutil import copyfile

import dateutil.tz
import numpy as np
import torch

from models.modular.classifiers.length_classifier import LengthClassifier
from models.modular.classifiers.number_classifier import NumberClassifier
from models.modular.modular_svnh_classifier import ModularSVNHClassifier
from models.resnet import ResNet34
from trainer.trainers.base_trainer import BaseTrainer
from trainer.trainers.lr_scheduler_trainer import LRSchedulerTrainer
from utils.config import cfg, cfg_from_file
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p
from train import parse_args, load_config


import cv2  # for debugging and viz

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

# these functions were written by B2T2 but they defined it in such as way that I can't import them 
# copied and pasted


# modify this function if you want to change the model


def instantiate_model(hyper_params):
    return ModularSVNHClassifier(cfg.MODEL,
                                 feature_transformation=ResNet34(
                                     hyper_params["FEATURES_OUTPUT_SIZE"]),
                                 length_classifier=LengthClassifier(cfg.MODEL,
                                                                    hyper_params["FEATURES_OUTPUT_SIZE"]),
                                 number_classifier=NumberClassifier,
                                 hyper_params=hyper_params)

# modify this function if you want to change the trainer


def instantiate_trainer(model, model_optimizer, hyper_params):
    # changed to BaseTrainer for base case testing with integration of bbox
    # json
    return BaseTrainer(model, model_optimizer, cfg, train_loader, valid_loader, test_from_train_loader,
                       device, cfg.OUTPUT_DIR, hyper_params=hyper_params,
                       max_patience=cfg.TRAIN.MAX_PATIENCE)

# modify this function if you want to change the optimizer


def instantiate_optimizer(model, hyper_params):
    return torch.optim.SGD(model.parameters(), lr=hyper_params["LR"],
                           momentum=hyper_params["MOM"],
                           weight_decay=float(hyper_params["WEIGHT_DECAY"]))


if __name__ == '__main__':
    args = parse_args()
    print("pytorch version {}".format(torch.__version__))
    # Load the config file
    load_config(args)
    print("pytorch version {}".format(torch.__version__))
    # Load the config file

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: {}".format(device))

    # Prepare data
    (train_loader,
     valid_loader,
     test_from_train_loader) = prepare_dataloaders(
        dataset_split=cfg.TRAIN.DATASET_SPLIT,
        # TODO: change hardcoded strings
        dataset_path=cfg.INPUT_DIR,
        metadata_filename=cfg.METADATA_FILENAME,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sample_size=cfg.TRAIN.SAMPLE_SIZE,
        valid_split=cfg.TRAIN.VALID_SPLIT,
        test_split=cfg.TRAIN.TEST_SPLIT,
        num_worker=cfg.TRAIN.NUM_WORKER)
    print("Start training from ", cfg.INPUT_DIR)
    current_hyper_params_dict = cfg.HYPER_PARAMS.INITIAL_VALUES  # read from yml file

    model = instantiate_model(current_hyper_params_dict)

    model_optimizer = instantiate_optimizer(model, current_hyper_params_dict)

    current_trainer = instantiate_trainer(
        model, model_optimizer, current_hyper_params_dict)

    # only training for now, debugging
    current_trainer.fit(current_hyper_params_dict)
