"""Define models and generator functions which receives params as parameter, then add model to available models"""
from torch import nn


class PoterhsuTransformator(nn.Module):
    """
    This features extractor is based on a project on github that attempted to solves the same problem.
    Github link : https://github.com/potterhsu/SVHNClassifier-PyTorch
    """

    def __init__(self, num_classes):
        super(PoterhsuTransformator, self).__init__()

        hidden1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=48, kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(
                in_channels=48, out_channels=64, kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=160, kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(
                in_channels=160, out_channels=192, kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(
                in_channels=192, out_channels=192, kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(
                in_channels=192, out_channels=192, kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(
                in_channels=192, out_channels=192, kernel_size=5, padding=2
            ),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden9 = nn.Sequential(nn.Linear(192 * 7 * 7, num_classes), nn.ReLU())
        hidden10 = nn.Sequential(
            nn.Linear(num_classes, num_classes), nn.ReLU()
        )

        self._features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
        )
        self._classifier = nn.Sequential(hidden9, hidden10)

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 192 * 7 * 7)
        return self._classifier(x)
