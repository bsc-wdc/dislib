import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

def net_getParameters(model):

	all_params = []
	gradients = []

	i = 0
	for layer in model.layers:

		params = list()

		if len(layer.params) > 0:

			weights = np.array(layer.params[0], copy=True).astype(np.float32)
			params.append(weights)



			if len(layer.params) > 1:

				bias = np.array(layer.params[1], copy=True).astype(np.float32)
				params.append(bias)
		gradients.append(layer.gradients)

		all_params.append(params)
		i += 1

	return all_params, gradients

def net_aggregateParameters(workers_parameters):

	NUM_WORKERS = len(workers_parameters)
	recv_weights = workers_parameters

	final_weights = recv_weights[0]

	for i in range(0, len(final_weights)):

		layer_final = final_weights[i]

		if len(layer_final) > 0:
		
			for j in range(1, NUM_WORKERS):

				layer_recv = recv_weights[j][i]
				layer_final[0] = np.add(layer_final[0], layer_recv[0])

				if len(layer_final) > 1:
					layer_final[1] = np.add(layer_final[1], layer_recv[1])

			layer_final[0] = np.divide(layer_final[0], NUM_WORKERS)

			if len(layer_final) > 1:
				layer_final[1] = np.divide(layer_final[1], NUM_WORKERS)

	return final_weights


def net_parametersToNumpy(parameters):

	np_params = list()

	for i in range(0, len(parameters)):

		params = list()

		for j in range(0, len(parameters[i])):

			v = np.array(parameters[i][j], copy=True).astype(np.float32)
			params.append(v)

		np_params.append(params)

	return np_params


def net_parametersToTensor(parameters):
	tensor_params = list()

	for i in range(0, len(parameters)):

		params = list()
		for j in range(0, len(parameters[i])):
			v = Tensor.fromarray(np.array(parameters[i][j], copy=True).astype(np.float32))
			params.append(v)

		tensor_params.append(params)
	return tensor_params

