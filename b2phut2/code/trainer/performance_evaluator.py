import torch
from tqdm import tqdm


class PerformanceEvaluator(object):
    """
      Class that wraps the evaluation
    """

    def __init__(self, eval_set):
        """
           :param eval_set: test or valid dataloader
        """
        self.loader = eval_set

    def evaluate(
        self, model, device, stats_recorder, mode="valid", include_length=True
    ):
        """
        Evaluate the model, the result are saved in the stats recorder object
        :param model: pytorch model
        :param device: device used (ex: cuda:0)
        :param stats_recorder:
        :param mode: valid or test. In valid mode, we compute the accuracy faster than in test mode, because in test
         mode, it returns the list of predicted number.
        :param include_length: Set it to true for true accuracy. For comparison purpose only.
        :return:
        """
        model = model.eval()
        with torch.no_grad():
            num_correct = 0
            valid_n_sample = 0
            valid_total_loss = 0

            length_accuracy = 0
            digits_accuracy = torch.zeros(5).float().to(device)
            no_digits_count = torch.zeros(5).float().to(device)
            total_digits_count = torch.zeros(5).float().to(device)

            y_true = []
            y_pred = []
            for batch_index, batch in enumerate(tqdm(self.loader)):
                # get the inputs
                inputs, targets = batch["image"], batch["target"]

                inputs = inputs.to(device)
                length_target = targets[:, 0].long().to(device)
                digits_target = targets[:, 1:].long().to(device)

                length_logits, digits_logits = model(inputs)
                length_predictions = length_logits.max(1)[1] + 3
                digits_predictions = [
                    digit_logits.max(1)[1] for digit_logits in digits_logits
                ]

                # We first check the sequence length
                is_output_correct = length_predictions.eq(length_target)

                length_accuracy += is_output_correct.sum()

                if include_length is False:
                    is_output_correct = (
                        torch.ones(length_target.shape[0]).byte().to(device)
                    )

                loss = torch.nn.functional.cross_entropy(
                    length_logits, length_target - 3
                )
                # We then check the digits output
                for i in range(digits_target.shape[1]):
                    is_digit_correct = digits_predictions[i].eq(
                        digits_target[:, i]
                    )
                    is_digit_not_there = digits_target[:, i] == -1
                    digits_accuracy[i] += is_digit_correct.float().sum()
                    no_digits_count[i] += is_digit_not_there.float().sum()
                    total_digits_count[i] += (digits_target[:, i] != -1).sum()
                    is_output_correct &= is_digit_correct + is_digit_not_there
                    loss = loss + torch.nn.functional.cross_entropy(
                        digits_logits[i], digits_target[:, i], ignore_index=-1
                    )
                if mode == "test":
                    for sample_idx in range(digits_target.shape[0]):
                        number_predicted = 0
                        number_true = 0
                        for i in range(length_predictions[sample_idx]):
                            number_predicted += digits_predictions[i][
                                sample_idx
                            ] * 10 ** (
                                (length_predictions[sample_idx] - 1) - i
                            )
                        for i in range(digits_target.shape[1]):
                            if digits_target[sample_idx][i] != -1:
                                number_true += digits_target[sample_idx][
                                    i
                                ] * 10 ** ((length_target[sample_idx] - 1) - i)
                        y_pred.append(int(number_predicted.item()))
                        y_true.append(int(number_true.item()))

                num_correct += is_output_correct.sum()
                valid_n_sample += length_target.size(0)
                valid_total_loss += loss.item()

            valid_accuracy = float(num_correct.item()) / valid_n_sample

            valid_loss = valid_total_loss / (batch_index + 1)

            if mode == "valid":
                stats_recorder.length_accuracy.append(
                    length_accuracy.float() / valid_n_sample
                )
                stats_recorder.digits_accuracy.append(
                    digits_accuracy.data.div(total_digits_count.data)
                )

                stats_recorder.valid_accuracies.append(valid_accuracy)
                stats_recorder.valid_losses.append(valid_loss)
            elif mode == "test":
                stats_recorder.test_best_accuracy = valid_accuracy
                stats_recorder.test_losses.append(valid_loss)
            return y_pred, y_true
