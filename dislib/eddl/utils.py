import copy
import numpy as np
from pyeddl.tensor import Tensor


def getParameters(model):
    parameters = []
    gradients = []
    for i, layer in enumerate(model.layers):
        params = []
        if len(layer.params) > 0:
            weights = np.array(layer.params[0], copy=True).astype(np.float32)
            params.append(weights)
            if len(layer.params) > 1:
                bias = np.array(layer.params[1], copy=True).astype(np.float32)
                params.append(bias)
        gradients.append(layer.gradients)
        parameters.append(params)
    return parameters, gradients


def aggregateParameters(workers_parameters):
    final_weights = copy.deepcopy(workers_parameters[0])
    for i, layer in enumerate(final_weights):
        if len(layer) > 0:
            for j in range(1, len(workers_parameters)):
                layer_recv = workers_parameters[j][i]
                layer[0] = np.add(layer[0], layer_recv[0])
                if len(layer) > 1:
                    layer[1] = np.add(layer[1], layer_recv[1])
            layer[0] = np.divide(layer[0], len(workers_parameters))
            if len(layer) > 1:
                layer[1] = np.divide(layer[1], len(workers_parameters))
    return final_weights


def parametersToNumpy(parameters):
    np_params = []
    for layer in parameters:
        params = []
        for param in layer:
            v = np.array(param, copy=True).astype(np.float32)
            params.append(v)
        np_params.append(params)
    return np_params


def parametersToEDDLTensor(parameters):
    tensor_params = []
    for layer in parameters:
        params = []
        for param in layer:
            v = Tensor.fromarray(np.array(param, copy=True).astype(np.float32))
            params.append(v)
        tensor_params.append(params)
    return tensor_params
