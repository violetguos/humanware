from tqdm import tqdm

from trainer.trainers.abstract_trainer import AbstractTrainer


class BaseTrainer(AbstractTrainer):
    """
    BaseTrainer class that fits the given model
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
        super(BaseTrainer, self).__init__(model, optimizer, cfg, train_loader, valid_loader, test_loader, device,
                                          output_dir,
                                          hyper_params, max_patience)

    def train(self, current_hyper_params):
        """
        Method for the training
        :param current_hyper_params: current hyper parameters dictionary
        """
        train_loss = 0
        train_n_iter = 0
        # Set model to train mode
        self.model.train()
        # Iterate over train data
        print("Iterating over training data...")
        for i, batch in enumerate(tqdm(self.train_loader)):
            loss = self._train_batch(batch)
            # Statistics
            train_loss += loss.item()
            train_n_iter += 1
        self.stats.train_loss_history.append(train_loss / train_n_iter)
