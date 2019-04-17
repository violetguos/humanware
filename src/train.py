from __future__ import print_function

import argparse
import datetime
import os
import pprint
import random
import sys
from shutil import copyfile
from pathlib import Path

import dateutil.tz
import numpy as np
import torch

from models.modular.classifiers.length_classifier import LengthClassifier
from models.modular.classifiers.number_classifier import NumberClassifier
from models.modular.modular_svnh_classifier import ModularSVNHClassifier
from models.resnet import ResNet34
from trainer.trainers.lr_scheduler_trainer import BaseTrainer
from utils.config import cfg, cfg_from_file
from utils.dataloader import prepare_dataloaders
from utils.misc import mkdir_p

dir_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "./."))
sys.path.append(dir_path)


def parse_args():
    """
    Parser for the arguments.

    Returns
    ----------
    args : obj
        The arguments.

    """
    parser = argparse.ArgumentParser(description="Train a CNN network")
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="""optional config file,
                             e.g. config/base_config.yml""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="""continue previous model training,
                              e.g. config/base_config.yml""",
    )
    parser.add_argument(
        "--metadata_filename",
        type=str,
        default="data/SVHN/train_metadata.pkl",
        help="""metadata_filename will be the absolute
                                path to the directory to be used for
                                training.""",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/SVHN/train/",
        help="""dataset_dir will be the absolute path
                                to the directory to be used for
                                training""",
    )
    parser.add_argument(
        "--valid_metadata_filename",
        type=str,
        required=True,
        help="""metadata_filename will be the absolute
                                path to the directory to be used for
                                training.""",
    )

    parser.add_argument(
        "--valid_dataset_dir",
        type=str,
        required=True,
        help="""dataset_dir will be the absolute path
                                to the directory to be used for
                                training""",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="""results_dir will be the absolute
                        path to a directory where the output of
                        your training will be saved.""",
    )
    args = parser.parse_args()
    return args


def load_config(args):
    """
    Load the config .yml file.

    """

    if args.cfg is None:
        raise Exception("No config file specified.")

    cfg_from_file(args.cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("timestamp: {}".format(timestamp))

    cfg.TIMESTAMP = timestamp
    cfg.INPUT_DIR = args.dataset_dir
    cfg.METADATA_FILENAME = args.metadata_filename
    cfg.OUTPUT_DIR = os.path.join(
        args.results_dir,
        "%s_%s_%s" % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp),
    )

    mkdir_p(cfg.OUTPUT_DIR)
    copyfile(args.cfg, os.path.join(cfg.OUTPUT_DIR, "config.yml"))

    print("Data dir: {}".format(cfg.INPUT_DIR))
    print("Output dir: {}".format(cfg.OUTPUT_DIR))

    print("Using config:")
    pprint.pprint(cfg)


def fix_seed(seed):
    """
    Fix the seed.

    Parameters
    ----------
    seed: int
        The seed to use.

    """
    print("pytorch/random seed: {}".format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    print("pytorch version {}".format(torch.__version__))
    # Load the config file
    load_config(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: {}".format(device))

    # Make the results reproductible
    seed = cfg.SEED
    model_dict = None
    if args.model is not None:
        model_dict = torch.load(args.model, map_location=device)
        seed = model_dict["seed"]
    fix_seed(seed)

    # Prepare data
    (train_loader, valid_loader) = prepare_dataloaders(
        dataset_split=cfg.TRAIN.DATASET_SPLIT,
        dataset_path=cfg.INPUT_DIR,
        metadata_filename=cfg.METADATA_FILENAME,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sample_size=cfg.TRAIN.SAMPLE_SIZE,
        valid_split=cfg.TRAIN.VALID_SPLIT,
        test_split=cfg.TRAIN.TEST_SPLIT,
        valid_metadata_filename=args.valid_metadata_filename,
        valid_dataset_dir=args.valid_dataset_dir,
        num_worker=cfg.TRAIN.NUM_WORKER,
    )
    print("Start training from ", cfg.INPUT_DIR)
    hyper_param_search_state = None
    
    if args.model is not None:
        model_filename = Path(args.model)
        print("\nLoading model from", model_filename.absolute())
        model = torch.load(model_filename, map_location=device)
        hyper_param_search_state = model_dict["hyper_param_search_state"]

    def instantiate_model(hyper_params):
        # modify this function if you want to change the model

        return ModularSVNHClassifier(
            cfg.MODEL,
            feature_transformation=ResNet34(
                hyper_params["FEATURES_OUTPUT_SIZE"]
            ),
            length_classifier=LengthClassifier(
                cfg.MODEL, hyper_params["FEATURES_OUTPUT_SIZE"]
            ),
            number_classifier=NumberClassifier,
            hyper_params=hyper_params,
        )

    def instantiate_trainer(model, model_optimizer, hyper_params):
        # modify this function if you want to change the trainer

        return BaseTrainer(
            model,
            model_optimizer,
            cfg,
            train_loader,
            valid_loader,
            device,
            cfg.OUTPUT_DIR,
            hyper_params=hyper_params,
            max_patience=cfg.TRAIN.MAX_PATIENCE,
        )

    def instantiate_optimizer(model, hyper_params):
        # modify this function if you want to change the optimizer
        return torch.optim.Adam(
            model.parameters(),
            lr=hyper_params["LR"]    
        )
    current_hyper_params_dict = cfg.HYPER_PARAMS.INITIAL_VALUES

    model = instantiate_model(current_hyper_params_dict)

    model_optimizer = instantiate_optimizer(model, current_hyper_params_dict)

    current_trainer = instantiate_trainer(
        model, model_optimizer, current_hyper_params_dict
    )
    if model_dict is not None:
        model.load_state_dict(model_dict["model_state_dict"])
        model_optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        current_trainer.load_state_dict(model_dict)

    current_trainer.fit(current_hyper_params_dict)
