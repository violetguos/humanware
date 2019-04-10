from torch import optim
from trainer.trainers.base_trainer import BaseTrainer


class LRSchedulerTrainer(BaseTrainer):
    """
    Trainer with learning rate scheduler
    """
    def __init__(self, model, optimizer, cfg, train_loader, valid_loader, test_loader, device, output_dir, hyper_params,
                 max_patience):
        """
        :param model: pytorch model
        :param optimizer: pytorch optimizaer
        :param cfg: config instance
        :param train_loader: train data loader
        :param valid_loade: valid data laoder
        :param device: gpu device used (ex: cuda:0)
        :param output_dir: output directory where the model and the results will be located
        :param hyper_params: hyper parameters
        :param max_patience: max number of iteration without seeing improvement in accuracy
        """
        super(LRSchedulerTrainer, self).__init__(model, optimizer, cfg, train_loader, valid_loader, test_loader, device,
                                                 output_dir,
                                                 hyper_params, max_patience)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=cfg.TRAIN.LR_SCHEDULER_PARAMS.MILESTONES,
                                                           gamma=cfg.TRAIN.LR_SCHEDULER_PARAMS.GAMMA)

    def train(self, current_hyper_params):
        """
        Train method
        :param current_hyper_params: current hyper parameters dictionary
        """
        self.lr_scheduler.step()
        super().train(current_hyper_params)

    def load_state_dict(self, state_dict):
        """
        Loads the previous state of the trainer
        Adds the lr scheduler state
        :param state_dict: state dictionary
        """
        super().load_state_dict(state_dict)
        if "lr_scheduler" in state_dict:
            self.lr_scheduler = state_dict["lr_scheduler"]

    def get_state_dict(self, hyper_param_search_state=None):
        """
        Gets the current state of the trainer
        Adds the lr scheduler state
        :param hyper_param_search_state: hyper param search state if we are doing an hyper params serach
        (None by default)
        :return state_dict
        """
        parent_state_dict = super().get_state_dict(hyper_param_search_state)
        parent_state_dict["lr_scheduler"] = self.lr_scheduler
        return parent_state_dict
