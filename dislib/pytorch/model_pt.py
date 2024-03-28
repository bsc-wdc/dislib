import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import flatten
import torch.nn as nn
from torch import flatten

class Imagenette_Red_Big(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Imagenette_Red_Big, self).__init__()
        self.neural_network_layers = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=128,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=128, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        )
        self.dense_neural_network_layers = nn.Sequential(
        nn.Linear(in_features=200704, out_features=300),
        nn.ReLU(),
        nn.Linear(in_features=300, out_features=300),
        nn.ReLU(),
        nn.Linear(in_features=300, out_features=num_classes))
    def forward(self, x):
        x = self.neural_network_layers(x)
        x = torch.flatten(x, 1)
        return self.dense_neural_network_layers(x)


class Imagenette_Red(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Imagenette_Red, self).__init__()
        self.neural_network_layers = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=128,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=128, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        )
        self.dense_neural_network_layers = nn.Sequential(
        nn.Linear(in_features=200704, out_features=200),
        nn.ReLU(),
        nn.Linear(in_features=200, out_features=200),
        nn.ReLU(),
        nn.Linear(in_features=200, out_features=num_classes))
    def forward(self, x):
        x = self.neural_network_layers(x)
        x = torch.flatten(x, 1)
        return self.dense_neural_network_layers(x)

class VGG19_Imagenette(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(VGG19_Imagenette, self).__init__()
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
        nn.Linear(in_features=25088, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=num_classes))
    def forward(self, x):
        x = self.neural_network_layers(x)
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


class VGG16(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(VGG16, self).__init__()
        #self.activation = F.relu()
        self.neural_network_layers = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
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
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512,
                               kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU()
        )
        self.dense_neural_network_layers = nn.Sequential(
        nn.Linear(in_features=512, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=num_classes))
    def forward(self, x):
        x = self.neural_network_layers(x)
        x = flatten(x, 1)
        return self.dense_neural_network_layers(x)

class VGG16_Small(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(VGG16_Small, self).__init__()
        #self.activation = F.relu()
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
        #nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=128, out_channels=128,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=128, out_channels=256,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        #nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
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
        )
        self.dense_neural_network_layers = nn.Sequential(
        nn.Linear(in_features=8192, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=num_classes))
    def forward(self, x):
        x = self.neural_network_layers(x)
        x = flatten(x, 1)
        return self.dense_neural_network_layers(x)


class VGGSmall(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(VGGSmall, self).__init__()
        #self.activation = nn.ReLU()
        self.neural_network_layers = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )
        self.dense_neural_network_layers = nn.Sequential(
        nn.Linear(in_features=512, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=num_classes))
    def forward(self, x):
        x = self.neural_network_layers(x)
        x = flatten(x, 1)
        return self.dense_neural_network_layers(x)


