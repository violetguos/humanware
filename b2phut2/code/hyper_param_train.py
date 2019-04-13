from __future__ import print_function

import os
import sys
from pathlib import Path

import torch
from skopt import Optimizer, utils

from models.modular.classifiers.length_classifier import LengthClassifier
from models.modular.classifiers.number_classifier import NumberClassifier
from models.modular.modular_svnh_classifier import ModularSVNHClassifier
from models.resnet import ResNet34
from train import parse_args, load_config, fix_seed
from trainer.hyper_params_searcher import HyperParamsSearcher
from trainer.trainers.base_trainer import BaseTrainer

from utils.config import cfg
from utils.dataloader import prepare_dataloaders

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

if __name__ == '__main__':
    args = parse_args()
    print("pytorch version {}".format(torch.__version__))
    # Load the config file
    load_config(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: {}".format(device))

    # Make the results reproducible
    seed = cfg.SEED
    model_dict = None
    if args.model is not None:
        model_dict = torch.load(args.model, map_location=device)
        seed = model_dict["seed"]
    fix_seed(seed)

    # Prepare data
    (train_loader,
     valid_loader,
     test_from_train_loader) = prepare_dataloaders(
        dataset_split=cfg.TRAIN.DATASET_SPLIT,
        dataset_path=cfg.INPUT_DIR,
        metadata_filename=cfg.METADATA_FILENAME,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sample_size=cfg.TRAIN.SAMPLE_SIZE,
        valid_split=cfg.TRAIN.VALID_SPLIT,
        test_split=cfg.TRAIN.TEST_SPLIT,
        extra_metadata_filename=args.extra_metadata_filename,
        extra_dataset_dir=args.extra_dataset_dir,
        num_worker=cfg.TRAIN.NUM_WORKER)

    hyper_param_search_state = None
    if args.model is not None:
        model_filename = Path(args.model)
        print("\nLoading model from", model_filename.absolute())
        model = torch.load(model_filename, map_location=device)
        hyper_param_search_state = model_dict["hyper_params"]

    dimensions_dict_as_dimensions_list = utils.dimensions_aslist(cfg.HYPER_PARAMS.SPACE)
    initial_hyper_params = utils.point_aslist(cfg.HYPER_PARAMS.SPACE, cfg.HYPER_PARAMS.INITIAL_VALUES)

    hyperparam_optimizer = Optimizer(dimensions_dict_as_dimensions_list, "GP", acq_optimizer="auto")

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

    hyper_param_searcher = HyperParamsSearcher(model_creation_function=instantiate_model,
                                               trainer_creation_function=instantiate_trainer,
                                               optimizer_creation_function=instantiate_optimizer,
                                               minimizer=hyperparam_optimizer,
                                               n_calls=cfg.HYPER_PARAMS.N_CALLS,
                                               space=cfg.HYPER_PARAMS.SPACE)
    if model_dict is not None:
        # load
        hyper_param_searcher.load_state_dict(model_dict)
    hyper_param_searcher.search(initial_hyper_params)
