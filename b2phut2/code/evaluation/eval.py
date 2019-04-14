import argparse
from pathlib import Path

import numpy as np
import torch

import time

import random

import sys

sys.path.append("../")
from utils.dataloader import prepare_dataloaders
from models.modular.classifiers.length_classifier import LengthClassifier
from models.modular.classifiers.number_classifier import NumberClassifier
from models.modular.modular_svnh_classifier import ModularSVNHClassifier
from models.resnet import ResNet34
from trainer.performance_evaluator import PerformanceEvaluator
from trainer.stats_recorder import StatsRecorder
from utils.config import cfg_from_file, cfg


def eval_model(
    dataset_dir,
    metadata_filename,
    model_filename,
    batch_size=32,
    sample_size=-1,
):
    """
    Validation loop.

    Parameters
    ----------
    dataset_dir : str
        Directory with all the images.
    metadata_filename : str
        Absolute path to the metadata pickle file.
    model_filename : str
        path/filename where to save the model.
    batch_size : int
        Mini-batch size.
    sample_size : int
        Number of elements to use as sample size,
        for debugging purposes only. If -1, use all samples.

    Returns
    -------
    y_pred : ndarray
        Prediction of the model.

    """

    seed = 1234

    print("pytorch/random seed: {}".format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_split = "test"
    # dataset_split = 'train'

    if dataset_split is "test":
        test_loader = prepare_dataloaders(
            dataset_split=dataset_split,
            dataset_path=dataset_dir,
            metadata_filename=metadata_filename,
            batch_size=batch_size,
            sample_size=sample_size,
            num_worker=0,
        )
    elif dataset_split is "train":
        train_loader, valid_loader, test_loader = prepare_dataloaders(
            dataset_split=dataset_split,
            dataset_path=dataset_dir,
            metadata_filename=metadata_filename,
            batch_size=batch_size,
            sample_size=sample_size,
            num_worker=0,
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # TODO: MODIFIY THIS!!
    cfg_from_file(
        "../../../saved_models/ELEM_AI_b2_base_trainer_PR7/config.yml"
    )

    # Load best model
    model_dict = torch.load(model_filename, map_location=device)

    current_hyper_params_dict = model_dict["hyper_params"]
    model = ModularSVNHClassifier(
        cfg.MODEL,
        feature_transformation=ResNet34(
            current_hyper_params_dict["FEATURES_OUTPUT_SIZE"]
        ),
        length_classifier=LengthClassifier(
            cfg.MODEL, current_hyper_params_dict["FEATURES_OUTPUT_SIZE"]
        ),
        number_classifier=NumberClassifier,
        hyper_params=current_hyper_params_dict,
    )

    model.load_state_dict(model_dict["model_state_dict"])

    since = time.time()
    model = model.to(device)

    print("# Testing Model ... #")

    stats = StatsRecorder()

    performance_evaluator = PerformanceEvaluator(test_loader)

    y_pred, y_true = performance_evaluator.evaluate(
        model, device, stats, mode="test"
    )

    test_accuracy = stats.test_best_accuracy

    print("===============================")
    print("\n\nTest Set Accuracy: {}".format(test_accuracy))

    time_elapsed = time.time() - since

    print(
        "\n\nTesting complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return y_pred


if __name__ == "__main__":
    # DO NOT MODIFY THIS SECTION#
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_filename", type=str, default="")
    # metadata_filename will be the absolute path to the directory to be used for
    # evaluation.

    parser.add_argument("--dataset_dir", type=str, default="")
    # dataset_dir will be the absolute path to the directory to be used for
    # evaluation.

    parser.add_argument("--results_dir", type=str, default="")
    # results_dir will be the absolute path to a directory where the output of
    # your inference will be saved.

    args = parser.parse_args()
    metadata_filename = args.metadata_filename
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir

    #########################################
    # MODIFY THIS SECTION #
    # Put your group name here
    group_name = "b2phut3"

    # TODO: MODIFY THIS!!!
    model_filename = (
        "../../../saved_models/ELEM_AI_b2_base_trainer_PR7/checkpoint_0.57.pth"
    )
    # model_filename should be the absolute path on shared disk to your
    # best model. You need to ensure that they are available to evaluators on
    # Helios.

    #################################

    # DO NOT MODIFY THIS SECTION #
    print("\nEvaluating results ... ")
    y_pred = eval_model(dataset_dir, metadata_filename, model_filename)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = Path(results_dir) / (group_name + "_eval_pred.txt")

    print("\nSaving results to ", results_fname.absolute())
    np.savetxt(results_fname, y_pred, fmt="%.1f")
    #########################################
