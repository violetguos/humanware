import datetime
import dateutil.tz
from pathlib import Path
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


class TBSummaryWriter:
    """
    Tensor board summary writer class.
    """

    def __init__(self, output_dir, current_hyper_params):
        """
        create summary writer object
        :param output_dir: output directory to save the results
        :param current_hyper_params: current hyper parameters dictionary
        """
        output_dir = Path(output_dir)
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.log_path = str(output_dir / "summary" / ("_%s" % (timestamp)))
        self.writer = SummaryWriter(self.log_path)
        self.output_dir = output_dir
        self.add_hyper_params(current_hyper_params)

    def add_scalar(self, tag, value, epoch):
        """
        Adds new scalar values to save
        :param tag: name of the value
        :param value: value
        :param epoch: current epoch number
        """
        self.writer.add_scalar(tag, value, epoch)

    def add_scalars(self, tag, value_dict, epoch):
        """
        Adds dictionary of scalar
        :param tag: name of the value
        :param value_dict: Dictionary of scalar
        :param epoch: current epoch number
        """
        self.writer.add_scalars(tag, value_dict, epoch)

    def add_hyper_params(self, hyperparams):
        """
        Saves hyper params dictionary
        :param hyperparams: hyper params dictionary
        :return:
        """
        for key in hyperparams:
            self.add_scalar(key, hyperparams[key], 0)

    def add_stats(self, stats, epoch):
        """
        Adds StatsRecorder object into the summary write
        :param stats: StatsRecorder object
        :param epoch: current epoch
        """
        self.add_scalars(
            "Loss",
            {
                "Train loss": stats.train_loss_history[-1],
                "Valid loss": stats.valid_losses[-1],
            },
            epoch,
        )
        self.add_scalar("Train Accuracy", stats.train_accuracies[-1], epoch)
        self.add_scalar("Valid Accuracy", stats.valid_accuracies[-1], epoch)
        self.add_scalar("Length Accuracy", stats.length_accuracy[-1], epoch)
        self.add_scalars(
            "Digits Accuracy",
            {
                "digit 0": stats.digits_accuracy[-1][0],
                "digit 1": stats.digits_accuracy[-1][1],
                "digit 2": stats.digits_accuracy[-1][2],
                "digit 3": stats.digits_accuracy[-1][3],
                "digit 4": stats.digits_accuracy[-1][4],
            },
            epoch,
        )

    def plot_curves(self, data_dict, title, axis_labels):
        """
        Plot the curves based on data given
        :param data_dict: Data dictionary
        :param title: Title of the plot
        :param axis_labels: Labels of plot (dictionary with x and y entry)
        :return:
        """
        for label in data_dict:
            plt.plot(data_dict[label], label=label)
        plt.xlabel(axis_labels["x"])
        plt.ylabel(axis_labels["y"])
        plt.legend()
        # replace blank spaces in the file names, useful for latex
        title.replace(" ", "_")
        plt.savefig(self.log_path + title + ".png")
        self.writer.add_figure(title, plt.gcf(), close=True)
