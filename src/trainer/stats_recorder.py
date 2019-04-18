class StatsRecorder:
    """
    Object that wraps all the stats related values. They are saved in the model states, so you can load them again.
    """

    def __init__(self):
        self.train_loss_history = []
        self.test_best_accuracy = 0.0
        self.valid_best_accuracy = 0.0
        self.valid_losses = []
        self.valid_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.length_accuracy = []
        self.digits_accuracy = []
        self.train_accuracies = []

    def print_last_epoch_stats(self):
        print("\tTrain Loss: {:.4f}".format(self.train_loss_history[-1]))
        print("\tValid Loss: {:.4f}".format(self.valid_losses[-1]))
        print("\tTrain Accuracy: {:.4f}".format(self.train_accuracies[-1]))
        print("\tValid Accuracy: {:.4f}".format(self.valid_accuracies[-1]))
        print("\tLength Accuracy: {:.4f}".format(self.length_accuracy[-1]))
        print("\tDigits Accuracy: {}".format(self.digits_accuracy[-1]))

    def upload_to_comet_ml(self, experiment, epoch):
        """
        Upload to comet_ml the experiment. Only if the flag COMET_ML_UPLOAD is set to true
        :param experiment: comet ml experiment object
        :param epoch: current epoch number
        :return:
        """
        experiment.log_metric(
            "Train loss", self.train_loss_history[-1], step=epoch
        )
        experiment.log_metric("Valid loss", self.valid_losses[-1], step=epoch)
        experiment.log_metric(
            "Valid accuracy", self.valid_accuracies[-1], step=epoch
        )
        experiment.log_metric(
            "Length accuracy", self.length_accuracy[-1], step=epoch
        )
        experiment.log_metric(
            "First digit accuracy",
            self.digits_accuracy[-1][0].item(),
            step=epoch,
        )
        experiment.log_metric(
            "Second digit accuracy",
            self.digits_accuracy[-1][1].item(),
            step=epoch,
        )
        experiment.log_metric(
            "Third digit accuracy",
            self.digits_accuracy[-1][2].item(),
            step=epoch,
        )
        experiment.log_metric(
            "Fourth digit accuracy",
            self.digits_accuracy[-1][3].item(),
            step=epoch,
        )
        experiment.log_metric(
            "Fifth digit accuracy",
            self.digits_accuracy[-1][4].item(),
            step=epoch,
        )
