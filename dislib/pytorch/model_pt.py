import torch.nn as nn
import torch


class TestsNetworkCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(TestsNetworkCNN, self).__init__()
        self.neural_network_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=1,
                      kernel_size=(1, 1), stride=(1, 1), padding=1),)
        self.dense_neural_network_layers = nn.Sequential(
            nn.Linear(in_features=9, out_features=4),
            nn.Linear(in_features=4, out_features=num_classes))

    def forward(self, x):
        x = self.neural_network_layers(x)
        x = torch.flatten(x, 1)
        return self.dense_neural_network_layers(x)


class TestsNetwork(nn.Module):
    def __init__(self, features, num_classes):
        super(TestsNetwork, self).__init__()
        self.dense_neural_network_layers = nn.Sequential(
            nn.Linear(in_features=features, out_features=4),
            nn.Linear(in_features=4, out_features=num_classes))

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.dense_neural_network_layers(x)


class MNIST_Network(nn.Module):
    def __init__(self, features, num_classes):
        super(MNIST_Network, self).__init__()
        self.dense_neural_network_layers = nn.Sequential(
            nn.Linear(in_features=features, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_classes))

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.dense_neural_network_layers(x)


class VGG19(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(VGG19, self).__init__()
        self.neural_network_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.dense_neural_network_layers = nn.Sequential(
            nn.Linear(in_features=32768, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes))

    def forward(self, x):
        x = self.neural_network_layers(x)
        x = torch.flatten(x, 1)
        return self.dense_neural_network_layers(x)
