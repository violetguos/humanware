from torch import nn


class LengthClassifier(nn.Module):
    """
       Sequence length classifier
    """

    def __init__(self, model_config, feature_output_size):
        """
        :param model_config: model config from config file
        :param feature_output_size: output size (will be 7 in this case)
        """
        super().__init__()
        self.seq_linear = nn.Sequential(
            nn.Linear(in_features=feature_output_size,
                      out_features=model_config.LENGTH_CLASSIFIER.NUM_CLASSES),
        )

    def forward(self, x):
        return self.seq_linear(x)
