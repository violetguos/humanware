from tqdm import tqdm

from trainer.trainers.abstract_trainer import AbstractTrainer


class PoterhsuTrainer(AbstractTrainer):
    """
     Trainer based on Pterhsu implementation. Github link : https://github.com/potterhsu/SVHNClassifier-PyTorch
     """

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
        max_patience,
    ):
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
        super(PoterhsuTrainer, self).__init__(
            model,
            optimizer,
            cfg,
            train_loader,
            valid_loader,
            test_loader,
            device,
            output_dir,
            hyper_params,
            max_patience,
        )
        self.step = 0

    def _adjust_learning_rate(self, step, initial_lr, decay_steps, decay_rate):
        """
        Decay the learning rate over the number of optimization step
        :param step: current step number (number of .step from the optimizer for example)
        :param initial_lr: initial learning rate
        :param decay_steps: decay steps value
        :param decay_rate: decay rate value
        :return:
        """
        lr = initial_lr * (decay_rate ** (step // decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

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
            # Adjust the learning rate
            lr = self._adjust_learning_rate(
                step=self.step,
                initial_lr=current_hyper_params["LR"],
                decay_steps=current_hyper_params["DECAY_STEPS"],
                decay_rate=current_hyper_params["DECAY_RATE"],
            )
            loss = self._train_batch(batch)
            # Statistics
            train_loss += loss.item()
            train_n_iter += 1
            self.step += 1
        print("Current lr: {}".format(lr))
        print("Current number of steps: {}".format(self.step))

        self.stats.train_loss_history.append(train_loss / train_n_iter)
