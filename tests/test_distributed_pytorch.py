import unittest
import torch
import torch.optim as optim
import torch.nn as nn
import dislib as ds
from dislib.pytorch.encapsulated_functions_distributed import \
    EncapsulatedFunctionsDistributedPytorch
from dislib.pytorch.model_pt import TestsNetwork, TestsNetworkCNN


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def set_weights_neural_network(model, weights):
    j = 0
    if hasattr(model, 'neural_network_layers'):
        len_nn = len(model.neural_network_layers)
        for i in range(len_nn):
            if hasattr(model.neural_network_layers[i], 'weight'):
                model.neural_network_layers[i].weight = \
                    nn.Parameter(weights.neural_network_layers[i].weight)
                j += 1
                model.neural_network_layers[i].bias = \
                    nn.Parameter(weights.neural_network_layers[i].bias)
                j += 1
    if hasattr(model, 'dense_neural_network_layers'):
        len_nn = len(model.dense_neural_network_layers)
        aux_j = 0
        for i in range(len_nn):
            if hasattr(model.dense_neural_network_layers[i], 'weight'):
                model.dense_neural_network_layers[i].weight = \
                    nn.Parameter(weights.dense_neural_network_layers[i].weight)
                aux_j += 1
                model.dense_neural_network_layers[i].bias = \
                    nn.Parameter(weights.dense_neural_network_layers[i].bias)
                aux_j += 1
    return model


class TensorPytorchDistributed(unittest.TestCase):
    def test_synchronous_training(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:100] = torch.ones([100, 1, 1, 1])
        y[:100] = 0
        x[100:200] = torch.ones([100, 1, 1, 1]) * 2
        y[100:200] = 1
        x[200:300] = torch.ones([100, 1, 1, 1])*4
        y[200:300] = 2
        x[300:400] = torch.ones([100, 1, 1, 1])*8
        y[300:400] = 3
        x[400:500] = torch.ones([100, 1, 1, 1]) * 16
        y[400:500] = 4
        x[500:600] = torch.ones([100, 1, 1, 1]) * 32
        y[500:600] = 5
        x[600:700] = torch.ones([100, 1, 1, 1]) * 64
        y[600:700] = 6
        x[700:800] = torch.ones([100, 1, 1, 1]) * 128
        y[700:800] = 7
        x[800:900] = torch.ones([100, 1, 1, 1]) * 256
        y[800:900] = 8
        x[900:] = torch.ones([100, 1, 1, 1]) * 512
        y[900:] = 9
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        x_tensor_original = ds.from_pt_tensor(tensor=x,
                                              shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        y_tensor_original = ds.from_pt_tensor(tensor=y,
                                              shape=(2, 2))
        model = TestsNetwork(1, 10)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.002, "momentum": 0.9}
        encaps_function.build(model, optimizer,
                              criterion, optimizer_parameters,
                              num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_synchronous_with_GPU(x_tensor,
                                     y_tensor, 10, 64)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor_original.collect()
        x_tensor = x_tensor_original.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        # Check predictions are better than random
        self.assertTrue((running_accuracy / total) > 0.1)

    def test_synchronous_training_cnn(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:500] = torch.ones([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = torch.ones([500, 1, 1, 1]) * 32
        y[500:] = 1
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        x_tensor_original = ds.from_pt_tensor(tensor=x,
                                              shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        y_tensor_original = ds.from_pt_tensor(tensor=y,
                                              shape=(2, 2))
        model = TestsNetworkCNN(1, 2)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.001, "momentum": 0.9}
        encaps_function.build(model, optimizer,
                              criterion, optimizer_parameters,
                              num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_synchronous_with_GPU(x_tensor, y_tensor, 10, 6)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor_original.collect()
        x_tensor = x_tensor_original.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        self.assertTrue((running_accuracy / total) == 1)

    def test_synchronous_shuffle_training_cnn(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:500] = torch.ones([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = torch.ones([500, 1, 1, 1]) * 32
        y[500:] = 1
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor_original = ds.from_pt_tensor(tensor=x,
                                              shape=(2, 2))
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        y_tensor_original = ds.from_pt_tensor(tensor=y,
                                              shape=(2, 2))
        model = TestsNetworkCNN(1, 2)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.001, "momentum": 0.9}
        encaps_function.build(model, optimizer,
                              criterion, optimizer_parameters,
                              num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_synchronous_shuffle_with_GPU(x_tensor, y_tensor, 10, 4)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor_original.collect()
        x_tensor = x_tensor_original.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        self.assertTrue((running_accuracy / total) == 1)

    def test_synchronous_shuffle_every_n_epochs_with_GPU_cnn(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:500] = torch.ones([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = torch.ones([500, 1, 1, 1]) * 32
        y[500:] = 1
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor_original = ds.from_pt_tensor(tensor=x,
                                              shape=(2, 2))
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor_original = ds.from_pt_tensor(tensor=y,
                                              shape=(2, 2))
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        model = TestsNetworkCNN(1, 2)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.001, "momentum": 0.9}
        encaps_function.build(model, optimizer,
                              criterion, optimizer_parameters,
                              num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_synchronous_shuffle_every_n_epochs_with_GPU(x_tensor,
                                                            y_tensor, 10, 4,
                                                            n_epocs_sync=2)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor_original.collect()
        x_tensor = x_tensor_original.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        self.assertTrue((running_accuracy / total) == 1)

    def test_synchronous_every_n_epochs_with_GPU_cnn(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:500] = torch.ones([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = torch.ones([500, 1, 1, 1]) * 32
        y[500:] = 1
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        model = TestsNetworkCNN(1, 2)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.001, "momentum": 0.9}
        encaps_function.build(model, optimizer,
                              criterion, optimizer_parameters,
                              num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_synchronous_every_n_epochs_with_GPU(x_tensor, y_tensor, 10, 6)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor.collect()
        x_tensor = x_tensor.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        self.assertTrue((running_accuracy / total) == 1)

    def test_synchronous_with_GPU_cnn(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:500] = torch.ones([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = torch.ones([500, 1, 1, 1]) * 32
        y[500:] = 1
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        model = TestsNetworkCNN(1, 2)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.001, "momentum": 0.9}
        encaps_function.build(model, optimizer,
                              criterion, optimizer_parameters,
                              num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_asynchronous_with_GPU(x_tensor, y_tensor, 10, 6)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor.collect()
        x_tensor = x_tensor.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        self.assertTrue((running_accuracy / total) == 1)

    def test_asynchronous_shuffle_with_GPU_cnn(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:500] = torch.ones([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = torch.ones([500, 1, 1, 1]) * 32
        y[500:] = 1
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor_original = ds.from_pt_tensor(tensor=x,
                                              shape=(2, 2))
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor_original = ds.from_pt_tensor(tensor=y,
                                              shape=(2, 2))
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        model = TestsNetworkCNN(1, 2)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.001, "momentum": 0.9}
        encaps_function.build(
            model, optimizer, criterion,
            optimizer_parameters, num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_asynchronous_shuffle_with_GPU(x_tensor, y_tensor, 10, 6)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor_original.collect()
        x_tensor = x_tensor_original.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        self.assertTrue((running_accuracy / total) == 1)

    def test_asynchronous_n_epochs_with_GPU_cnn(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:500] = torch.ones([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = torch.ones([500, 1, 1, 1]) * 32
        y[500:] = 1
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        model = TestsNetworkCNN(1, 2)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.001, "momentum": 0.9}
        encaps_function.build(
            model, optimizer, criterion,
            optimizer_parameters, num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_asynchronous_n_epochs_with_GPU(x_tensor, y_tensor, 10, 6,
                                               n_epocs_sync=2)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor.collect()
        x_tensor = x_tensor.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        self.assertTrue((running_accuracy / total) == 1)

    def test_asynchronous_shuffle_n_epochs_with_GPU_cnn(self):
        x = torch.ones([1000, 1, 1, 1])
        y = torch.ones([1000, 1])
        x[:500] = torch.ones([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = torch.ones([500, 1, 1, 1]) * 32
        y[500:] = 1
        indices = torch.randperm(x.size()[0])
        x = x[indices]
        y = y[indices]
        x_tensor_original = ds.from_pt_tensor(tensor=x,
                                              shape=(2, 2))
        x_tensor = ds.from_pt_tensor(tensor=x,
                                     shape=(2, 2))
        y = torch.nn.functional.one_hot(y.long(), num_classes=-1)
        y = y[:, 0, :].double()
        y_tensor_original = ds.from_pt_tensor(tensor=y,
                                              shape=(2, 2))
        y_tensor = ds.from_pt_tensor(tensor=y,
                                     shape=(2, 2))
        model = TestsNetworkCNN(1, 2)
        model.apply(init_weights)
        encaps_function = EncapsulatedFunctionsDistributedPytorch(
            num_workers=1)
        criterion = nn.CrossEntropyLoss
        optimizer = optim.SGD
        optimizer_parameters = {"lr": 0.001, "momentum": 0.9}
        encaps_function.build(
            model, optimizer,
            criterion, optimizer_parameters,
            num_gpu=1, num_nodes=1)
        trained_weights = encaps_function.\
            fit_asynchronous_shuffle_n_epochs_with_GPU(x_tensor,
                                                       y_tensor, 10, 6,
                                                       n_epocs_sync=2)
        model = set_weights_neural_network(model, trained_weights)
        model.eval()
        total = 0
        running_accuracy = 0
        y_tensor = y_tensor_original.collect()
        x_tensor = x_tensor_original.collect()
        for tensor, labels in zip(x_tensor, y_tensor):
            for images, in_labels in zip(tensor, labels):
                outputs = in_labels
                predicted_outputs = model(images.float())
                preds, predicted = torch.max(predicted_outputs, 1)
                outs, outputs = torch.max(outputs, 1)
                total += predicted_outputs.shape[0]
                running_accuracy += (predicted == outputs).sum().item()
        self.assertTrue((running_accuracy / total) == 1)
