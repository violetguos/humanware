from __future__ import print_function
import os
import sys
from utils.config import cfg
import torch

from models.modular.classifiers.length_classifier import LengthClassifier
from models.modular.classifiers.number_classifier import NumberClassifier
from models.modular.modular_svnh_classifier import ModularSVNHClassifier
from models.resnet import ResNet50
from trainer.trainers.lr_scheduler_trainer import LRSchedulerTrainer
from utils.dataloader import prepare_dataloaders
from train import parse_args, load_config, fix_seed


dir_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "./."))
sys.path.append(dir_path)


def instantiate_model(hyper_params):
    '''
    modify this function if you want to change the model
    '''
    return ModularSVNHClassifier(
        cfg.MODEL,
        feature_transformation=ResNet50(hyper_params["FEATURES_OUTPUT_SIZE"]),
        length_classifier=LengthClassifier(
            cfg.MODEL, hyper_params["FEATURES_OUTPUT_SIZE"]
        ),
        number_classifier=NumberClassifier,
        hyper_params=hyper_params,
    )


def instantiate_trainer(model, model_optimizer, hyper_params):
    '''
    modify this function if you want to change the trainer
    '''

    return LRSchedulerTrainer(
        model,
        model_optimizer,
        cfg,
        train_loader,
        valid_loader,
        test_from_train_loader,
        device,
        cfg.OUTPUT_DIR,
        hyper_params=hyper_params,
        max_patience=cfg.TRAIN.MAX_PATIENCE,
    )


def instantiate_optimizer(model, hyper_params):
    # modify this function if you want to change the optimizer
    return torch.optim.SGD(
        model.parameters(),
        lr=hyper_params["LR"],
        momentum=hyper_params["MOM"],
        weight_decay=float(hyper_params["WEIGHT_DECAY"]),
    )


if __name__ == "__main__":
    args = parse_args()
    print("pytorch version {}".format(torch.__version__))
    # Load the config file
    load_config(args)
    # Load the config file

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
    (train_loader, valid_loader, test_from_train_loader) = prepare_dataloaders(
        dataset_split=cfg.TRAIN.DATASET_SPLIT,
        dataset_path=cfg.INPUT_DIR,
        metadata_filename=cfg.METADATA_FILENAME,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sample_size=cfg.TRAIN.SAMPLE_SIZE,
        valid_split=cfg.TRAIN.VALID_SPLIT,
        test_split=cfg.TRAIN.TEST_SPLIT,
        num_worker=cfg.TRAIN.NUM_WORKER,
    )
    print("Start training from ", cfg.INPUT_DIR)

    current_hyper_params_dict = (
        cfg.HYPER_PARAMS.INITIAL_VALUES
    )  # read from yml file

    model = instantiate_model(current_hyper_params_dict)

    model_optimizer = instantiate_optimizer(model, current_hyper_params_dict)

    current_trainer = instantiate_trainer(
        model, model_optimizer, current_hyper_params_dict
    )

    if model_dict is not None:
        model.load_state_dict(model_dict["model_state_dict"])
        model_optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        current_trainer.load_state_dict(model_dict)

    # begin hack for supporting checkpoint model re-load and retrain
    if args.model is not None:
        current_trainer.epoch = 0  # reset to retrain from a checkpoint
    current_trainer.fit(current_hyper_params_dict)
