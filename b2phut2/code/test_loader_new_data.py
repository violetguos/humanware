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


import cv2 # for debugging and viz

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

# modify this function if you want to change the model
def instantiate_model(hyper_params):
    return ModularSVNHClassifier(cfg.MODEL,
                                 feature_transformation=ResNet34(hyper_params["FEATURES_OUTPUT_SIZE"]),
                                 length_classifier=LengthClassifier(cfg.MODEL,
                                                                    hyper_params["FEATURES_OUTPUT_SIZE"]),
                                 number_classifier=NumberClassifier,
                                 hyper_params=hyper_params)

# modify this function if you want to change the trainer
def instantiate_trainer(model, model_optimizer, hyper_params):
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


    current_hyper_params_dict = cfg.HYPER_PARAMS.INITIAL_VALUES

    model = instantiate_model(current_hyper_params_dict)

    model_optimizer = instantiate_optimizer(model, current_hyper_params_dict)

    current_trainer = instantiate_trainer(model, model_optimizer, current_hyper_params_dict)

    # if model_dict is not None:
    #     model.load_state_dict(model_dict["model_state_dict"])
    #     model_optimizer.load_state_dict(model_dict["optimizer_state_dict"])
    #     current_trainer.load_state_dict(model_dict)
    current_trainer.train(current_hyper_params_dict)


