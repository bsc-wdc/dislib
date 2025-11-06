import os
from pycompss.api.task import task
from pycompss.api.parameter import IN, COMMUTATIVE
from pycompss.api.constraint import constraint
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def assign_parameters(model, model_parameters):
    if hasattr(model, 'neural_network_layers'):
        len_nn = len(model.neural_network_layers)
        for i in range(len_nn):
            if hasattr(model_parameters.neural_network_layers[i],
                       'weight'):
                model.neural_network_layers[i].weight = \
                    nn.Parameter(
                        model_parameters.neural_network_layers[i].
                        weight.float())
            if hasattr(model_parameters.neural_network_layers[i],
                       'bias'):
                model.neural_network_layers[i].bias = \
                    nn.Parameter(
                        model_parameters.neural_network_layers[i].bias.
                        float())
            if hasattr(model.neural_network_layers[i],
                       'alpha'):
                model.neural_network_layers[i].alpha = \
                    nn.Parameter(model_parameters.
                                 neural_network_layers[i].alpha)
            if hasattr(model.neural_network_layers[i],
                       'shortcut'):
                len_shortcut = len(model.neural_network_layers[i].shortcut)
                for k in range(len_shortcut):
                    if hasattr(model.neural_network_layers[i].shortcut[k],
                               'weight'):
                        model.neural_network_layers[i].shortcut[k].weight = \
                            nn.Parameter(model_parameters.
                                         neural_network_layers[i].shortcut[k].
                                         weight.float())
                        model.neural_network_layers[i].shortcut[k].bias = \
                            nn.Parameter(model_parameters.
                                         neural_network_layers[i].shortcut[k].
                                         bias.float())
                    if hasattr(model.neural_network_layers[i].shortcut[k],
                               'alpha'):
                        model.neural_network_layers[i].shortcut[k].alpha = \
                            nn.Parameter(model_parameters.
                                         neural_network_layers[i].
                                         shortcut[k].alpha)
            if hasattr(model.neural_network_layers[i],
                       'layers'):
                len_layers = len(model.neural_network_layers[i].layers)
                for k in range(len_layers):
                    if hasattr(model.neural_network_layers[i].layers[k],
                               'weight'):
                        model.neural_network_layers[i].layers[k].weight = \
                            nn.Parameter(model_parameters.
                                         neural_network_layers[i].
                                         layers[k].weight.float())
                    if hasattr(model.neural_network_layers[i].layers[k],
                               'bias'):
                        model.neural_network_layers[i].layers[k].bias = \
                            nn.Parameter(model_parameters.
                                         neural_network_layers[i].
                                         layers[k].bias.float())
                    if hasattr(model.neural_network_layers[i].layers[k],
                               'alpha'):
                        model.neural_network_layers[i].layers[k].alpha = \
                            nn.Parameter(model_parameters.
                                         neural_network_layers[i].
                                         layers[k].alpha)
    if hasattr(model, 'dense_neural_network_layers'):
        len_nn = len(model_parameters.dense_neural_network_layers)
        for i in range(len_nn):
            if hasattr(
                    model_parameters.dense_neural_network_layers[i],
                    'weight'):
                model.dense_neural_network_layers[i].weight = \
                    nn.Parameter(
                        model_parameters.dense_neural_network_layers[i].
                        weight.float())
            if hasattr(model_parameters.dense_neural_network_layers[i],
                       'bias'):
                model.dense_neural_network_layers[i].bias = \
                    nn.Parameter(
                        model_parameters.dense_neural_network_layers[i].
                        bias.float())
            if hasattr(model.neural_network_layers[i],
                       'alpha'):
                model.neural_network_layers[i].alpha = \
                    nn.Parameter(model_parameters.
                                 neural_network_layers[i].alpha)
    return model


class PytorchDistributed(object):
    """
    PytorchDistributed object. It is in charge of executing in parallel the
    small trainings inside each epoch of the main training.
    """
    def __init__(self):
        self.model = None
        self.loss = None
        self.optimizer = None

    def build(self, net, loss, optimizer, optimizer_parameters):
        """
        Sets all the needed parameters and objects for this object.

        Parameters
        ----------
        net:
            Network that will be used during the training
        loss:
            Loss used during the network training
        optimizer:
            Optimizer object used to updated the weights of the network
            during the training.
        optimizer_parameters:
            Parameters to set in the Optimizer
        """
        local_net = net
        self.model = local_net
        self.loss = loss()
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters

    def train_cnn_batch_GPU(self, model_parameters, x_train,
                            y_train, num_batches, shuffle_block_data):
        """
        Performs a training of one of the workers network
        on the corresponding part of the data.

        Parameters
        ----------
        model_parameters: tensor
            Weights and biases of the different layers of the network
            that will be used in this small training.
        x_train: torch.tensor
            Samples of the training data
        y_train: torch.tensor
            Labels, regression values or etc. of training data.
        num_batches: int
            Number of batches that will be done
        shuffle_block_data: boolean
            Whether to shuffle or not the training data used.

        Returns
        -------
        model_parameters: tensor
            Updated weights and biases of the different layers of the network
            after the training.
        """
        return self._train_cnn_batch_GPU(model_parameters, x_train,
                                         y_train, num_batches,
                                         shuffle_block_data)

    @constraint(processors=[
        {'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
        {'processorType': 'GPU', 'computingUnits': '${ComputingUnitsGPUs}'}])
    @task()
    def _train_cnn_batch_GPU_losses(self, model_parameters, x_train,
                                    y_train, num_batches, shuffle_block_data,
                                    transformations=None,
                                    gradient_clipping=None,
                                    score_to_compute=accuracy_score):
        if shuffle_block_data:
            idx = torch.randperm(x_train.shape[0])
            if not isinstance(x_train.size, int):
                x_train = x_train[idx].view(x_train.size())
            else:
                if not torch.is_tensor(x_train):
                    x_train = x_train[idx]
                else:
                    x_train = torch.from_numpy(x_train)
                    x_train = x_train[idx].view(x_train.size())
            if not isinstance(y_train.size, int):
                if len(y_train.size()) > 1:
                    y_train = y_train[idx].view(y_train.size())
                else:
                    y_train = y_train[idx]
            else:
                if not torch.is_tensor(y_train):
                    y_train = y_train[idx]
                else:
                    y_train = torch.from_numpy(y_train)
                    y_train = y_train[idx].view(y_train.size())
        self.model = assign_parameters(self.model, model_parameters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        optimizer = self.optimizer(self.model.parameters(),
                                   **self.optimizer_parameters)
        x_train = x_train.float().to(device)
        indexes = int(x_train.shape[0] / num_batches)
        if len(y_train.shape) == 1:
            y_train = y_train.float()
        elif y_train.shape[-1] == 1:
            y_train = y_train.float()
        if isinstance(self.loss, nn.CrossEntropyLoss):
            y_train = y_train.long()
        true_labels = y_train.to(device)
        output_list = []
        losses_epoch = []
        for idx in range(num_batches):
            optimizer.zero_grad()
            inputs = x_train[idx*indexes:(idx + 1)*indexes]
            if transformations is not None:
                inputs = transformations(inputs)
            outputs = self.model(inputs)
            loss = self.loss(outputs,
                             true_labels[idx*indexes:(idx+1)*indexes])
            loss.backward()
            if gradient_clipping is not None:
                gradient_clipping(self.model)
            optimizer.step()
            losses_epoch.append(loss.item())
            output_list.append(outputs.to("cpu"))
        output_list = torch.cat(output_list)
        _, output_list = torch.max(output_list, dim=1)
        true_labels = true_labels.to("cpu")
        self.model = self.model.to("cpu")
        return self.model, np.array(losses_epoch).mean(), \
            score_to_compute(true_labels.detach().numpy(),
                             output_list.detach().numpy())

    @constraint(processors=[
        {'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
        {'processorType': 'GPU', 'computingUnits': '${ComputingUnitsGPUs}'}])
    @task()
    def _train_cnn_batch_GPU(self, model_parameters, x_train,
                             y_train, num_batches, shuffle_block_data,
                             transformations=None,
                             gradient_clipping=None):
        if shuffle_block_data:
            idx = torch.randperm(x_train.shape[0])
            if not isinstance(x_train.size, int):
                x_train = x_train[idx].view(x_train.size())
            else:
                if not torch.is_tensor(x_train):
                    x_train = x_train[idx]
                else:
                    x_train = torch.from_numpy(x_train)
                    x_train = x_train[idx].view(x_train.size())
            if not isinstance(y_train.size, int):
                if len(y_train.size()) > 1:
                    y_train = y_train[idx].view(y_train.size())
                else:
                    y_train = y_train[idx]
            else:
                if not torch.is_tensor(y_train):
                    y_train = y_train[idx]
                else:
                    y_train = torch.from_numpy(y_train)
                    y_train = y_train[idx].view(y_train.size())
        self.model = assign_parameters(self.model, model_parameters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        optimizer = self.optimizer(self.model.parameters(),
                                   **self.optimizer_parameters)
        x_train = x_train.float().to(device)
        indexes = int(x_train.shape[0] / num_batches)
        if isinstance(self.loss, nn.CrossEntropyLoss):
            y_train = y_train.long()
        true_labels = y_train.to(device)
        for idx in range(num_batches):
            optimizer.zero_grad()
            inputs = x_train[idx*indexes:(idx + 1)*indexes]
            if transformations is not None:
                inputs = transformations(inputs)
            outputs = self.model(inputs)
            loss = self.loss(outputs,
                             true_labels[idx*indexes:(idx+1)*indexes])
            loss.backward()
            if gradient_clipping is not None:
                gradient_clipping(self.model)
            optimizer.step()
        self.model = self.model.to("cpu")
        return self.model

    def aggregate_parameters_async(self,
                                   model_params,
                                   parameters_to_aggregate):
        """
        Function that aggregates in commutative and without requiring a
        synchronization the weights of the network
        generated by the different trainings.

        Parameters
        ----------
        model_params: tensor
            Weights and biases of the different layers of the main
            network.
        parameters_to_aggregate:
            Weights and biases generated through the training

        Returns
        -------
        model_params: tensor
            Updated weights and biases of the different layers of the main
            network.
        """
        return self._aggregate_parameters_async(model_params,
                                                parameters_to_aggregate)

    @constraint(processors=[
        {'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
        {'processorType': 'GPU', 'computingUnits': '${ComputingUnitsGPUs}'}])
    @task(model_params=COMMUTATIVE, parameters_to_aggregate=IN,
          target_direction=IN)
    def _aggregate_parameters_async(self, model_params,
                                    parameters_to_aggregate):
        final_weights = []
        worker_weights = []
        for param in model_params.parameters():
            worker_weights.append(param)
        final_weights.append(worker_weights)
        worker_weights = []
        for param in parameters_to_aggregate.parameters():
            worker_weights.append(param)
        final_weights.append(worker_weights)
        final_added_parameters = final_weights[0]
        for i in range(len(final_weights[0])):
            for j in range(1, len(final_weights)):
                final_added_parameters[i] = final_added_parameters[i] + \
                                            final_weights[j][i]
            final_added_parameters[i] = final_added_parameters[i]
        for i in range(len(final_weights[0])):
            final_added_parameters[i] = final_added_parameters[i] / 2
        j = 0
        if hasattr(model_params, 'neural_network_layers'):
            len_nn = len(model_params.neural_network_layers)
            for i in range(len_nn):
                if hasattr(model_params.neural_network_layers[i], 'weight'):
                    model_params.neural_network_layers[i].weight = \
                        nn.Parameter(final_added_parameters[j].float())
                    j += 1
                if hasattr(model_params.neural_network_layers[i],
                           'bias'):
                    model_params.neural_network_layers[i].bias = \
                        nn.Parameter(final_added_parameters[j].float())
                    j += 1
                if hasattr(self.model.neural_network_layers[i],
                           'alpha'):
                    self.model.neural_network_layers[i].alpha = \
                        nn.Parameter(model_params.
                                     neural_network_layers[i].alpha)
        if hasattr(model_params, 'dense_neural_network_layers'):
            len_nn = len(model_params.dense_neural_network_layers)
            aux_j = 0
            for i in range(len_nn):
                if hasattr(model_params.dense_neural_network_layers[i],
                           'weight'):
                    model_params.dense_neural_network_layers[i].weight = \
                        nn.Parameter(final_added_parameters[aux_j + j].float())
                    aux_j += 1
                if hasattr(model_params.dense_neural_network_layers[i],
                           'bias'):
                    model_params.dense_neural_network_layers[i].bias = \
                        nn.Parameter(final_added_parameters[aux_j + j].float())
                    aux_j += 1
                if hasattr(self.model.dense_neural_network_layers[i],
                           'alpha'):
                    self.model.dense_neural_network_layers[i].alpha = \
                        nn.Parameter(model_params.
                                     neural_network_layers[i].alpha)
        return model_params
