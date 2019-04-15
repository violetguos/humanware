import os

from comet_ml import Experiment
from comet_ml import OfflineExperiment

import time
from abc import ABC, abstractmethod

import torch

from trainer.performance_evaluator import PerformanceEvaluator
import copy

from trainer.stats_recorder import StatsRecorder
import numpy as np

from trainer.summary_writer import TBSummaryWriter


class AbstractTrainer(ABC):
    """Abstract class that fits the given model"""

    def __init__(
        self,
        model,
        optimizer,
        cfg,
        train_loader,
        valid_loader,
        test_loader,
        device,
        output_dir,
        hyper_params,
        max_patience=5,
    ):
        """
        :param model: pytorch model
        :param optimizer: pytorch optimizaer
        :param cfg: config instance
        :param train_loader: train data loader
        :param valid_loade: valid data laoder
        :param device: gpu device used (ex: cuda:0)
        :param output_dir: output directory where the model and the results
            will be located
        :param hyper_params: hyper parameters
        :param max_patience: max number of iteration without seeing
            improvement in accuracy
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.stats = StatsRecorder()
        self.performance_evaluator = PerformanceEvaluator(self.valid_loader)
        self.valid_evaluator = PerformanceEvaluator(self.valid_loader)
        self.test_evaluator = PerformanceEvaluator(self.test_loader)
        self.output_dir = output_dir
        self.best_model = None
        self.hyper_params = hyper_params
        self.max_patience = max_patience
        self.current_patience = 0
        self.epoch = 0
        self.comet_ml_experiment = None
        self.last_checkpoint_filename = None

    def initialize_cometml_experiment(self, hyper_params):
        """
        Initialize the comet_ml experiment (only if enabled in config file)
        :param hyper_params: current hyper parameters dictionary
        :return:
        """
        if (
            self.comet_ml_experiment is None
            and self.cfg.COMET_ML_UPLOAD is True
        ):
            # Create an experiment
            self.comet_ml_experiment = Experiment(
                api_key=os.environ["COMET_API_KEY"],
                project_name="general",
                workspace="proguoram",
            )
            if self.comet_ml_experiment.disabled is True:
                # There is problably no internet (in the cluster for example)
                # So we create a offline experiment
                self.comet_ml_experiment = OfflineExperiment(
                    workspace="proguoram",
                    project_name="general",
                    offline_directory=self.output_dir,
                )
            self.comet_ml_experiment.log_parameters(hyper_params)

    def fit(self, current_hyper_params, hyper_param_search_state=None):
        """
        Fit function applied train, val, test to all models.
        Each train/val/test may differe for each model
        I/O is the same, so wrap them with this method and do logging
        """
        self.initialize_cometml_experiment(current_hyper_params)
        print("# Start training #")
        since = time.time()

        summary_writer = TBSummaryWriter(self.output_dir, current_hyper_params)

        for epoch in range(self.epoch, self.cfg.TRAIN.NUM_EPOCHS, 1):
            self.train(current_hyper_params)
            self.validate(self.model)
            self.epoch = epoch
            print(
                "\nEpoch: {}/{}".format(epoch + 1, self.cfg.TRAIN.NUM_EPOCHS)
            )
            self.stats.print_last_epoch_stats()
            summary_writer.add_stats(self.stats, epoch)
            if self.cfg.COMET_ML_UPLOAD is True:
                self.stats.upload_to_comet_ml(self.comet_ml_experiment, epoch)
            if self.early_stopping_check(self.model, hyper_param_search_state):
                break
        time_elapsed = time.time() - since
        self.add_plots_summary(summary_writer)
        print(
            "\n\nTraining complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        # manually uncomment in training
        # comment this out in hyper_param_train
        print("Force save after final epoch")
        model_filename = self.output_dir + "/checkpoint_{}_ep_{}.pth".format(
            self.stats.valid_best_accuracy, self.epoch
        )
        self.save_current_best_model(model_filename, hyper_param_search_state)

    @classmethod
    @abstractmethod
    def train(self, current_hyper_params):
        """
        Abstract method for the training
        :param current_hyper_params: current hyper parameters dictionary
        """
        pass

    def validate(self, model):
        """
        Validate the model
        :param model: pytorch model
        """
        self.valid_evaluator.evaluate(
            model, self.device, self.stats, mode="valid"
        )

    def test(self, model):
        """
        Test the model
        :param model: pytorch model
        """
        model = model.to(self.device)
        self.test_evaluator.evaluate(
            model, self.device, self.stats, mode="test"
        )

    def early_stopping_check(self, model, hyper_param_search_state=None):
        """
        Early stop check
        :param model: pytorch model
        :param current_hyper_params: current hyper parameters dictionary
        :return: True if need to stop. False if continue the training
        """
        last_accuracy_computed = self.stats.valid_accuracies[-1]
        if last_accuracy_computed > self.stats.valid_best_accuracy:
            self.stats.valid_best_accuracy = last_accuracy_computed
            self.best_model = copy.deepcopy(model)
            print("Checkpointing new model...")
            model_filename = self.output_dir + "/checkpoint_{}.pth".format(
                self.stats.valid_best_accuracy
            )
            self.save_current_best_model(
                model_filename, hyper_param_search_state
            )
            if self.last_checkpoint_filename is not None:
                os.remove(self.last_checkpoint_filename)
            self.last_checkpoint_filename = model_filename
            self.current_patience = 0
        else:
            self.current_patience += 1
            if self.current_patience > self.max_patience:
                return True
        return False

    def compute_loss(
        self, length_logits, digits_logits, length_labels, digits_labels
    ):
        """
        Multi loss computing function
        :param length_logits: length logits tensor (N x 7)
        :param digits_logits: digits legits tensor (N x 5 x 10)
        :param length_labels: length labels (N x 5 x 1)
        :param digits_labels: length labels tensor (N x 1)
        :return: loss tensor value
        """
        loss = torch.nn.functional.cross_entropy(length_logits, length_labels)
        for i in range(digits_labels.shape[1]):
            loss = loss + torch.nn.functional.cross_entropy(
                digits_logits[i], digits_labels[:, i], ignore_index=-1
            )
        return loss

    def load_state_dict(self, state_dict):
        """
        Loads the previous state of the trainer
        Should be overriden in the children classes if needed (see LRSchedulerTrainer for an example)
        :param state_dict: state dictionary
        """
        self.epoch = state_dict["epoch"]
        self.stats = state_dict["stats"]
        self.current_patience = state_dict["current_patience"]
        self.best_model = self.model

    def get_state_dict(self, hyper_param_search_state=None):
        """
         Gets the current state of the trainer
         Should be overriden in the children classes if needed (see LRSchedulerTrainer for an example)
         :param hyper_param_search_state: hyper param search state if we are doing an hyper params serach
         (None by default)
        :return state_dict
         """
        seed = np.random.get_state()[1][0]
        return {
            "epoch": self.epoch + 1,
            "model_state_dict": self.best_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": self.stats,
            "seed": seed,
            "current_patience": self.current_patience,
            "hyper_param_search_state": hyper_param_search_state,
            "hyper_params": self.hyper_params,
        }

    def save_current_best_model(self, out_path, hyper_param_search_state=None):
        """
        Saves the current best model
        :param out_path: output path string
        :param hyper_param_search_state: hyper param search state if we are doing an hyper params serach
        (None by default)
        """
        state_dict = self.get_state_dict(hyper_param_search_state)
        torch.save(state_dict, out_path)
        print("Model saved!")

    def _train_batch(self, batch):
        """
        Basic batch method (called by children class normally
        :param batch: batch
        :return: loss tensor value
        """
        inputs, targets = batch["image"], batch["target"]

        inputs = inputs.to(self.device)
        targets = targets.long().to(self.device)
        target_ndigits = targets[:, 0]

        # Zero the gradient buffer
        self.optimizer.zero_grad()

        # Forward
        pred_length, pred_sequences = self.model(inputs)

        # For each digit predicted
        target_digit = targets[:, 1:]
        loss = self.compute_loss(
            pred_length, pred_sequences, target_ndigits, target_digit
        )
        # Backward
        loss.backward()
        # Optimize
        self.optimizer.step()

        return loss

    def add_plots_summary(self, summary_writer):
        """
        Add plotting values for tensor board
        :param summary_writer: Summary writer object from tensor board
        """
        # plot loss curves
        loss_dict = {
            "Train loss": self.stats.train_loss_history,
            "Valid loss": self.stats.valid_losses,
        }
        axis_labels = {"x": "Epochs", "y": "Loss"}
        summary_writer.plot_curves(loss_dict, "Learning curves", axis_labels)

        # plot accuracy curves
        acc_dict = {
            "Valid accuracy": self.stats.valid_accuracies,
            "Length accuracy": self.stats.length_accuracy,
        }
        axis_labels = {"x": "Epochs", "y": "Accuracy"}
        summary_writer.plot_curves(acc_dict, "Accuracy curves", axis_labels)
