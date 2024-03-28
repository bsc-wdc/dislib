import os
from pycompss.api.task import task
from pycompss.api.parameter import IN, COMMUTATIVE
from pycompss.api.constraint import constraint
import torch.nn as nn
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class PytorchDistributed(object):
    def __init__(self):
        self.model = None
        self.loss = None
        self.optimizer = None

    def build(self, net, loss, optimizer, optimizer_parameters):
        local_net = net
        self.model = local_net
        self.loss = loss()
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters

    @constraint(processors=[
        {'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
        {'processorType': 'GPU', 'computingUnits': '${ComputingUnitsGPUs}'}])
    @task()
    def train_cnn_batch_GPU(self, model_parameters, x_train,
                            y_train, num_batches, shuffle_block_data):
        if shuffle_block_data:
            idx = torch.randperm(x_train.shape[0])
            x_train = x_train[idx].view(x_train.size())
            y_train = y_train[idx].view(y_train.size())
        if hasattr(self.model, 'neural_network_layers'):
            len_nn = len(self.model.neural_network_layers)
            for i in range(len_nn):
                if hasattr(model_parameters.neural_network_layers[i],
                           'weight'):
                    self.model.neural_network_layers[i].weight = \
                        nn.Parameter(
                            model_parameters.neural_network_layers[i].
                            weight.float())
                    self.model.neural_network_layers[i].bias = \
                        nn.Parameter(
                            model_parameters.neural_network_layers[i].bias.
                            float())
        if hasattr(self.model, 'dense_neural_network_layers'):
            len_nn = len(model_parameters.dense_neural_network_layers)
            for i in range(len_nn):
                if hasattr(
                        model_parameters.dense_neural_network_layers[i],
                        'weight'):
                    self.model.dense_neural_network_layers[i].weight = \
                        nn.Parameter(
                            model_parameters.dense_neural_network_layers[i].
                            weight.float())
                    self.model.dense_neural_network_layers[i].bias = \
                        nn.Parameter(
                            model_parameters.dense_neural_network_layers[i].
                            bias.float())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        optimizer = self.optimizer(self.model.parameters(),
                                   **self.optimizer_parameters)
        x_train = x_train.float().to(device)
        true_labels = y_train.to(device)
        indexes = int(x_train.shape[0] / num_batches)
        for idx in range(num_batches):
            optimizer.zero_grad()
            outputs = self.model(x_train[idx*indexes:(idx+1)*indexes])
            loss = self.loss(outputs,
                             true_labels[idx*indexes:(idx+1)*indexes])
            loss.backward()
            optimizer.step()
        self.model = self.model.to("cpu")
        return self.model

    @constraint(processors=[
        {'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
        {'processorType': 'GPU', 'computingUnits': '${ComputingUnitsGPUs}'}])
    @task(model_params=COMMUTATIVE, parameters_to_aggregate=IN,
          mult_factor=IN, target_direction=IN)
    def aggregate_parameters_async(self, model_params,
                                   parameters_to_aggregate,
                                   mult_factor):
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
                    model_params.neural_network_layers[i].bias = \
                        nn.Parameter(final_added_parameters[j].float())
                    j += 1
        if hasattr(model_params, 'dense_neural_network_layers'):
            len_nn = len(model_params.dense_neural_network_layers)
            aux_j = 0
            for i in range(len_nn):
                if hasattr(model_params.dense_neural_network_layers[i],
                           'weight'):
                    model_params.dense_neural_network_layers[i].weight = \
                        nn.Parameter(final_added_parameters[aux_j + j].float())
                    aux_j += 1
                    model_params.dense_neural_network_layers[i].bias = \
                        nn.Parameter(final_added_parameters[aux_j + j].float())
                    aux_j += 1
        return model_params
