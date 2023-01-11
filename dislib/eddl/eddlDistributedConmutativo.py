from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, COLLECTION_IN, Depth, \
    IN, COMMUTATIVE

from sklearn.utils import shuffle
from pycompss.api.task import task
import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from net_utils import net_parametersToTensor, net_parametersToNumpy
from net_utils import net_getParameters
import gc
import sys
import os

class EddlDistributedConmutativo(object):
    def __init__(self):
        self.num_gpu = None
        self.metric = None
        self.loss = None
        self.initialized = False
        self.optimizer = None
        self.model = None

    @task(is_replicated=True)
    def build(self, net, optimizer, loss, metric, num_gpu=0, gpu=0):
        if num_gpu > 0:
            num_gpu = np.zeros((num_gpu), dtype=np.int8)
            num_gpu[gpu] = 1
            comps_service = eddl.CS_GPU(num_gpu.tolist())
        #else:
            #comps_service = eddl.CS_CPU()
        eddl.build(
            net,
            self._build_optimizer(optimizer),
            [loss],
            [metric],
            comps_service
        )
        self.optimizer = optimizer
        self.model = net
        self.initialized = True
        self.loss = loss
        self.metric = metric
        self.num_gpu = num_gpu

    @constraint(processors=[{'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
                            {'processorType': 'GPU', 'computingUnits': '${ComputingGPUs}'}])
    @task(returns=1, target_direction=IN)
    def train_cnn_batch_GPU(self, initial_parameters, x_train, y_train, dtype,
                            num_batches, epoch=1,
                            shuffle_block_data=True):
        if os.environ["COMPSS_BINDED_GPUS"] != "":
            num_gpus = os.environ["COMPSS_BINDED_GPUS"].split(',')
            num_gpu = np.zeros((len(num_gpus)), dtype=np.int8)
            for i in num_gpus:
                num_gpu[int(i)] = 1
            sys.stdout.write("NUM GPU" + str(num_gpu.tolist()) + "\n")
            comps_service = eddl.CS_GPU(num_gpu.tolist())
        else:
            comps_service = eddl.CS_CPU()

        optimizer = self.optimizer.copy()
        weight_decay = self.optimizer.get('weight_decay', 0.0)
        optimizer['lr'] = optimizer['lr']/(1 + weight_decay*epoch)

        eddl.build(
            self.model,
            self._build_optimizer(optimizer),
            [self.loss],
            [self.metric],
            comps_service,
            False
        )
        eddl.set_parameters(self.model,
                            net_parametersToTensor(initial_parameters))
        eddl.reset_loss(self.model)
        if shuffle_block_data:
            x_train, y_train = shuffle(x_train, y_train)
            x_train = Tensor.fromarray(x_train)
            y_train = Tensor.fromarray(y_train.astype(np.float32))
        else:
            x_train = Tensor.fromarray(x_train)
            y_train = Tensor.fromarray(y_train.astype(np.float32))
        x_train.div_(255.0)
        num_images = int(x_train.shape[0] / (num_batches))
        eddl.reset_loss(self.model)
        for j in range(num_batches):
            index_batch = list(range(j * num_images, (j + 1) *
                                     num_images - 1))
            eddl.train_batch(self.model, [x_train], [y_train], index_batch)
        final_parameters = eddl.get_parameters(self.model)
        final_parameters = net_parametersToNumpy(final_parameters)
        x_train = None
        y_train = None
        comps_service = None
        del x_train
        del y_train
        del comps_service
        gc.collect()
        return final_parameters

    @task(returns=1, target_direction=IN)
    def train_cnn_batch(self, initial_parameters, x_train, y_train,
                        num_batches, epoch=1, gpu_to_use=0,
                        shuffle_block_data=True):
        if len(self.num_gpu) > 0:
            gpus = self.num_gpu
            gpus[gpu_to_use] = 1
            comps_service = eddl.CS_GPU(gpu_to_use.tolist())
        else:
            comps_service = eddl.CS_CPU()

        optimizer = self.optimizer.copy()
        weight_decay = self.optimizer.get('weight_decay', 0.0)
        optimizer['lr'] = optimizer['lr'] / (1 + weight_decay * epoch)

        eddl.build(
            self.model,
            self._build_optimizer(optimizer),
            [self.loss],
            [self.metric],
            comps_service,
            False
        )
        eddl.set_parameters(self.model,
                            net_parametersToTensor(initial_parameters))
        if isinstance(x_train, type(np)):
            if shuffle_block_data:
                x_train, y_train = shuffle(x_train, y_train)
                x_train = Tensor.fromarray(x_train)
                y_train = Tensor.fromarray(y_train.astype(np.float32))
            else:
                x_train = Tensor.fromarray(x_train)
                y_train = Tensor.fromarray(y_train.astype(np.float32))
        x_train.div_(255.0)
        num_images = int(x_train.shape[0] / (num_batches))
        eddl.reset_loss(self.model)
        for j in range(num_batches):
            index_batch = list(range(j * num_images, (j + 1) *
                                     num_images - 1))
            eddl.train_batch(self.model, [x_train], [y_train], index_batch)
        final_parameters, gradients = net_getParameters(self.model)
        final_parameters = net_parametersToNumpy(final_parameters)

        return final_parameters

    @constraint(processors=[{'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
                            {'processorType': 'GPU', 'computingUnits': '${ComputingGPUs}'}])
    @task(returns=1, target_direction=IN,
          x_train={Type: COLLECTION_IN, Depth: 2},
          y_train={Type: COLLECTION_IN, Depth: 2})
    def train_batch_GPU(self, initial_parameters, x_train, y_train,
                        num_batches, epoch=1, gpu_to_use=0,
                        shuffle_block_data=True):
        if len(self.num_gpu) > 0:
            gpus = self.num_gpu
            gpus[gpu_to_use] = 1
            comps_service = eddl.CS_GPU(gpu_to_use.tolist())
        else:
            comps_service = eddl.CS_CPU()

        optimizer = self.optimizer.copy()
        weight_decay = self.optimizer.get('weight_decay', 0.0)
        optimizer['lr'] = optimizer['lr'] / (1 + weight_decay * epoch)

        eddl.build(
            self.model,
            self._build_optimizer(optimizer),
            [self.loss],
            [self.metric],
            comps_service,
            False
        )
        eddl.set_parameters(self.model,
                            net_parametersToTensor(initial_parameters))
        x_train = np.block(x_train)
        y_train = np.block(y_train)
        if shuffle_block_data:
            x_train, y_train = shuffle(x_train, y_train)
        x = Tensor.fromarray(x_train)
        y = Tensor.fromarray(y_train.astype(np.float32))
        x.div_(255.0)
        num_images = int(x_train.shape[0] / (num_batches))
        eddl.reset_loss(self.model)
        for j in range(num_batches):
            index_batch = list(range(j * num_images,
                                     (j + 1) * num_images - 1))
            eddl.train_batch(self.model, [x], [y], index_batch)
        final_parameters, gradients = eddl.get_parameters(self.model)
        final_parameters = net_parametersToNumpy(final_parameters)
        return final_parameters

    @task(returns=1, target_direction=IN,
          x_train={Type: COLLECTION_IN, Depth: 2},
          y_train={Type: COLLECTION_IN, Depth: 2})
    def train_batch(self, initial_parameters, x_train, y_train,
                    num_batches, epoch=1, gpu_to_use=0,
                    shuffle_block_data=True):
        if len(self.num_gpu) > 0:
            gpus = self.num_gpu
            gpus[gpu_to_use] = 1
            comps_service = eddl.CS_GPU(gpu_to_use.tolist())
        else:
            comps_service = eddl.CS_CPU()

        optimizer = self.optimizer.copy()
        weight_decay = self.optimizer.get('weight_decay', 0.0)
        optimizer['lr'] = optimizer['lr'] / (1 + weight_decay * epoch)

        eddl.build(
            self.model,
            self._build_optimizer(optimizer),
            [self.loss],
            [self.metric],
            comps_service,
            False
        )
        eddl.set_parameters(self.model,
                            net_parametersToTensor(initial_parameters))
        x_train = np.block(x_train)
        y_train = np.block(y_train)
        if shuffle_block_data:
            x_train, y_train = shuffle(x_train, y_train)
        x = Tensor.fromarray(x_train)
        y = Tensor.fromarray(y_train.astype(np.float32))
        x.div_(255.0)
        num_images = int(x_train.shape[0] / (num_batches))
        eddl.reset_loss(self.model)
        for j in range(num_batches):
            index_batch = list(range(j * num_images,
                                     (j + 1) * num_images - 1))
            eddl.train_batch(self.model, [x], [y], index_batch)
        final_parameters, gradients = net_getParameters(self.model)
        final_parameters = net_parametersToNumpy(final_parameters)

        return final_parameters

    @task()
    def update_lr_optimizer(self, iterations):
        if self.optimizer['optimizer'] == 'sgd':
            weight_decay = self.optimizer.get('weight_decay', 0.0)
            if weight_decay != 0.0:
                self.optimizer['lr'] = self.optimizer['lr'] / \
                                       (1 + weight_decay * iterations)

    @constraint(processors=[{'processorType': 'CPU', 'computingUnits': '40'}])
    @task()
    def evaluate_batch_in_GPU(self):
        pass  # TODO

    @task()
    def evaluate_batch(self):
        pass  # TODO
    
    @constraint(processors=[{'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
                            {'processorType': 'GPU', 'computingUnits': '${ComputingGPUs}'}])
    @task(model_params=COMMUTATIVE, parameters_to_aggregate=IN,
          mult_factor=IN, target_direction=IN)
    def aggregate_parameters_async(self, model_params,
                                   parameters_to_aggregate,
                                   mult_factor):
        for i in range(0, len(model_params)):
            for j in range(0, len(model_params[i])):
                model_params[i][j] = (
                        (model_params[i][j] +
                         parameters_to_aggregate[i][j]) / 2).astype(
                    np.float32)

        return model_params

    def _build_optimizer(self, optimizer):
        if optimizer['optimizer'] == 'adam':
            return eddl.adam(optimizer.get('lr', 0.01),
                             optimizer.get('beta_1', 0.9),
                             optimizer.get('beta_2', 0.999),
                             optimizer.get('epsilon', 0.000001),
                             optimizer.get('weight_decay', 0),
                             optimizer.get('amsgrad', False))
        elif optimizer['optimizer'] == 'sgd':
            return eddl.sgd(optimizer.get('lr', 0.01),
                            optimizer.get('momentum', 0.0),
                            optimizer.get('weight_decay', 0.0),
                            optimizer.get('nesterov', False))
        elif optimizer['optimizer'] == 'rmsprop':
            return eddl.rmsprop(optimizer.get('lr', 0.01),
                                optimizer.get('rho', 0.9),
                                optimizer.get('epsilon', 0.00001),
                                optimizer.get('weight_decay', 0.0))
        elif optimizer['optimizer'] == 'adagrad':
            return eddl.adagrad(optimizer.get('lr', 0.01),
                                optimizer.get('epsilon', 0.00001),
                                optimizer.get('weight_decay', 0.0))
        elif optimizer['optimizer'] == 'adamax':
            return eddl.adamax(optimizer.get('lr', 0.01),
                               optimizer.get('beta_1', 0.9),
                               optimizer.get('beta_2', 0.999),
                               optimizer.get('epsilon', 0.00001),
                               optimizer.get('weight_decay', 0.0))
        elif optimizer['optimizer'] == 'adadelta':
            return eddl.adadelta(optimizer.get('lr', 0.01),
                                 optimizer.get('rho', 0.9),
                                 optimizer.get('epsilon', 0.00001),
                                 optimizer.get('weight_decay', 0.0))
        elif optimizer['optimizer'] == 'nadam':
            return eddl.nadam(optimizer.get('lr', 0.01),
                              optimizer.get('beta_1', 0.9),
                              optimizer.get('beta_2', 0.999),
                              optimizer.get('epsilon', 0.00001),
                              optimizer.get('schedule_decay', 0.0))

    def __setstate__(self, state):
        model = state['model']
        initialized = state['initialized']
        loss = state['loss']
        metric = state['metric']
        num_gpu = state['num_gpu']
        optimizer = state['optimizer']
        if model:
            if initialized:
                self.model = eddl.import_net_from_onnx_string(model)
                self.initialized = initialized
                self.loss = loss
                self.metric = metric
                self.num_gpu = num_gpu
                self.optimizer = optimizer
            else:
                self.model = model
                self.initialized = initialized
                self.loss = loss
                self.metric = metric
                self.num_gpu = num_gpu
                self.optimizer = optimizer
        else:
            self.model = None

    def __getstate__(self):
        if self.model:
            if self.initialized:
                return {
                    'model': eddl.serialize_net_to_onnx_string(self.model,
                                                               False),
                    'initialized': self.initialized,
                    'loss': self.loss,
                    'metric': self.metric,
                    'num_gpu': self.num_gpu,
                    'optimizer': self.optimizer
                }
            else:
                return {
                    'model': self.model,
                    'initialized': self.initialized,
                    'loss': self.loss,
                    'metric': self.metric,
                    'num_gpu': self.num_gpu,
                    'optimizer': self.optimizer
                }
        else:
            return {
                'model': None,
                'initialized': None,
                'loss': None,
                'metric': None,
                'num_gpu': None,
                'optimizer': None
            }
