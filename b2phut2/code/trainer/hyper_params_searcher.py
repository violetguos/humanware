import json
import os

import numpy
import skopt
import torch

from utils.memory_management_utils import dump_tensors


class HyperParamsSearcher:
    """
    Class that can do various type of hyperparameter search (Bayesian for example) based on the skopt library
    """

    def __init__(self, model_creation_function, trainer_creation_function, optimizer_creation_function, minimizer,
                 n_calls, space):
        """
        :param model_creation_function: pointer to the model instancing function
        :param trainer_creation_function: pointer to the trainer instancing function
        :param optimizer_creation_function: pointer to the optimizer instancing function
        :param minimizer: skopt minimizer instance
        :param n_calls: number of iteration of hyper parameter search
        :param space: Hyper params dimension space (see the config for an example)
        """
        self.model_creation_lambda = model_creation_function
        self.trainer_creation_lambda = trainer_creation_function
        self.optimizer_creation_lambda = optimizer_creation_function
        self.n_calls = n_calls
        self.search_space = space
        self.current_trainer = None
        self.best_score = 0
        self.start_iteration = 0
        self.minimizer = minimizer
        self.has_loaded_state = False
        self.state_dict = None

    def load_state_dict(self, state_dict):
        """
        Load hyper param state
        :param state_dict: hyper param state dictionary
        """
        self.state_dict = state_dict
        self.has_loaded_state = True
        self.start_iteration = state_dict["hyper_param_search_state"]["current_iteration"]
        self.minimizer = state_dict["hyper_param_search_state"]["minimizer"]
        self.best_score = state_dict["hyper_param_search_state"]["best_score_overall"]

    def get_state_dict(self, current_iteration):
        """
        Get the hyper param state dictionary
        :param current_iteration: current iteration in the hyper parameter search
        :return: hyper param state dictionary
        """
        return {
            "current_iteration": current_iteration,
            "minimizer": self.minimizer,
            "best_score_overall": self.best_score
        }

    def _objective(self, current_hyper_params_dict, hyper_param_state):
        """
        Objective function for the bayesian search. We return the best accuracy from the model
        :param current_hyper_params_dict: current hyper parameters dictionary
        :param hyper_param_state: hyper param state dictionary
        :return: best accuracy from the model
        """
        model = self.model_creation_lambda(current_hyper_params_dict)

        model_optimizer = self.optimizer_creation_lambda(model, current_hyper_params_dict)

        self.current_trainer = self.trainer_creation_lambda(model, model_optimizer, current_hyper_params_dict)
        if self.has_loaded_state is True:
            self.has_loaded_state = False
            # We continue the previous iteration
            model.load_state_dict(self.state_dict["model_state_dict"])
            model_optimizer.load_state_dict(self.state_dict["optimizer_state_dict"])
            self.current_trainer.load_state_dict(self.state_dict)

        self.current_trainer.fit(current_hyper_params_dict, hyper_param_state)
        return self.current_trainer.stats.valid_best_accuracy #! type error from origanl code .item()

    def search(self, initial_hyper_params=None):
        """
        Start the hyper parameters search
        :param initial_hyper_params: (optional) initial hyper parameters values from the config files
        """
        print("******************************")
        print("in search start_iteration", self.start_iteration)
        for i in range(self.start_iteration, self.n_calls, 1):
            print("\n\n[Hyper Param Iteration {}]".format(i + 1))
            if i == 0 and initial_hyper_params is not None:
                # We use the init values for the parameter list
                current_hyper_params_list = initial_hyper_params
                current_hyper_params_dict = skopt.utils.point_asdict(self.search_space, current_hyper_params_list)
            else:
                if self.has_loaded_state is True:
                    current_hyper_params_dict = self.state_dict["hyper_params"]
                    current_hyper_params_list = skopt.utils.point_aslist(self.search_space, current_hyper_params_dict)
                else:
                    current_hyper_params_list = self.minimizer.ask()
                    current_hyper_params_dict = skopt.utils.point_asdict(self.search_space, current_hyper_params_list)

            print("Current hyper-parameters : {}".format(current_hyper_params_dict))
            hyper_param_state = self.get_state_dict(i)
            f_val = self._objective(current_hyper_params_dict, hyper_param_state)
            print("f_val", f_val)
            self.minimizer.tell(current_hyper_params_list, f_val)
            print("Score = {}".format(f_val))
            if self.best_score < f_val:
                self.best_score = f_val
                print("New overall best score: {}!".format(f_val))
                self.save_overall_current_best_model(current_hyper_params_dict)

            del self.current_trainer
            torch.cuda.empty_cache()
            dump_tensors()

        print("The hyper-paramaters search has correctly ended with a best valid score of {}".format(self.best_score))

    def save_overall_current_best_model(self, current_hyper_params_dict):
        """
        Saves the overall best model so far
        :param current_hyper_params_dict: current hyper parameters dictionary
        """
        print('Saving model ...')
        model_filename = os.path.join(self.current_trainer.output_dir, 'best_model_{}.pth'.format(self.best_score))
        self.current_trainer.save_current_best_model(model_filename)
        print('Best model saved to :', model_filename)

        # Saving best hyper-params
        with open(os.path.join(self.current_trainer.output_dir, 'best_hyper_params.json'), 'w') as fp:
            if current_hyper_params_dict is not None:
                for key, value in current_hyper_params_dict.items():
                    # We need to convert to native because the json package doesnt support numpy types
                    if isinstance(value, numpy.generic):
                        current_hyper_params_dict[key] = value.item()
            json.dump(current_hyper_params_dict, fp)
