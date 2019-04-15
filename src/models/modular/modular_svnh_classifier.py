from torch import nn


class ModularSVNHClassifier(nn.Module):
    """
    Modular SVHN classifier, so we can easily change the feature transformation part independently from the
    classification task
    """

    def __init__(
        self,
        model_config,
        feature_transformation: nn.Module,
        length_classifier: nn.Module,
        number_classifier: nn.Module,
        hyper_params=None,
    ):
        """
          :param model_config: model config (see config file)
          :param feature_transformation: feature transformation model (see train.py for an example)
          :param length_classifier: length classifier model (see train.py for an example)
          :param number_classifier: digits classifier model (see train.py for an example)
          :param hyper_params: hyper params dictionary
          """
        super().__init__()
        self.model_config = model_config
        self.feature_transformation = feature_transformation
        self.length_classifier = length_classifier

        self.number_classifiers = nn.ModuleList(
            [
                number_classifier(
                    model_config, hyper_params["FEATURES_OUTPUT_SIZE"]
                )
                for _ in range(
                    model_config.NUMBER_CLASSIFIER.MAX_SEQUENCE_LENGTH
                )
            ]
        )

    def forward(self, x):
        out = self.feature_transformation(x)
        length = self.length_classifier(out)
        sequences = [classifier(out) for classifier in self.number_classifiers]
        return length, sequences
