import unittest
import pyeddl.eddl as eddl
from dislib.eddl import EncapsulatedFunctionsDistributedEddl
from dislib.eddl.eddl_distributed_conmutativo import EddlDistributedConmutativo
from dislib.eddl.models import TestsNetworkCNNEDDL, VGG19, VGG16
import numpy as np
import dislib as ds
from dislib.eddl.utils import parametersToEDDLTensor
from pyeddl.tensor import Tensor
import pyeddl
from sklearn.metrics import accuracy_score


class TensorEDDLDistributed(unittest.TestCase):
    def test_synchronous_shuffle_every_n_epochs_with_GPU_training(self):
        x = np.ones([1000, 1, 1, 1])
        y = np.ones([1000, 1])
        x[:500] = np.zeros([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = np.ones([500, 1, 1, 1]) * 510
        y[500:] = 1
        indices = np.random.permutation(x.size)
        x = x[indices]
        y = y[indices].astype(int)
        x_tensor_original = ds.from_array(np_array=x,
                                          shape=(2, 2))
        x_tensor = ds.from_array(np_array=x,
                                 shape=(2, 2))
        b = np.zeros((y.size, y.max() + 1))
        b[np.arange(y.size), y.flatten()] = 1
        y = b
        del b
        y_tensor_original = ds.from_array(np_array=y,
                                          shape=(2, 2))
        y_tensor = ds.from_array(np_array=y,
                                 shape=(2, 2))
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        encaps_function.build(net, optimizer, "binary_cross_entropy",
                              "categorical_accuracy", num_nodes=1, num_gpu=0)
        parameters = encaps_function.\
            fit_synchronous_shuffle_every_n_epochs_with_GPU(
              x_tensor, y_tensor, 25, 5, n_epocs_sync=2)
        net = eddl.Model([in_], [out])
        eddl.build(
            net,
            eddl.sgd(0.002, 0.9, 0.8),
            ["binary_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(),
        )
        eddl.set_parameters(net, parametersToEDDLTensor(parameters))
        x_test = x_tensor_original.collect()
        y_test = y_tensor_original.collect()
        x_test = Tensor.fromarray(x_test[0][0])
        y_test = Tensor.fromarray(y_test[0][0].astype(np.float32))
        x_test.div_(255.0)
        prediction = eddl.predict(net, [x_test])[0].getdata()
        total_score = accuracy_score(np.argmax(y_test.getdata(), axis=1),
                                     np.argmax(prediction, axis=1))
        self.assertTrue(total_score == 1)

    def test_synchronous_every_n_epochs_with_GPU_training(self):
        x = np.ones([1000, 1, 1, 1])
        y = np.ones([1000, 1])
        x[:500] = np.zeros([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = np.ones([500, 1, 1, 1]) * 510
        y[500:] = 1
        indices = np.random.permutation(x.size)
        x = x[indices]
        y = y[indices].astype(int)
        x_tensor = ds.from_array(np_array=x,
                                 shape=(2, 2))
        x_tensor_original = ds.from_array(np_array=x,
                                          shape=(2, 2))
        b = np.zeros((y.size, y.max() + 1))
        b[np.arange(y.size), y.flatten()] = 1
        y = b
        del b
        y_tensor = ds.from_array(np_array=y,
                                 shape=(2, 2))
        y_tensor_original = ds.from_array(np_array=y,
                                          shape=(2, 2))
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        encaps_function.build(net, optimizer, "binary_cross_entropy",
                              "categorical_accuracy", num_nodes=1, num_gpu=0)
        parameters = encaps_function.fit_synchronous_every_n_epochs_with_GPU(
            x_tensor, y_tensor, 25, 5, n_epocs_sync=2)
        net = eddl.Model([in_], [out])
        eddl.build(
            net,
            eddl.sgd(0.002, 0.9, 0.8),
            ["binary_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(),
        )
        eddl.set_parameters(net, parametersToEDDLTensor(parameters))
        x_test = x_tensor_original.collect()
        y_test = y_tensor_original.collect()
        x_test = Tensor.fromarray(x_test[0][0])
        y_test = Tensor.fromarray(y_test[0][0].astype(np.float32))
        x_test.div_(255.0)
        prediction = eddl.predict(net, [x_test])[0].getdata()
        total_score = accuracy_score(np.argmax(y_test.getdata(), axis=1),
                                     np.argmax(prediction, axis=1))
        self.assertTrue(total_score == 1)

    def test_fit_synchronous_with_GPU_training(self):
        x = np.ones([1000, 1, 1, 1])
        y = np.ones([1000, 1])
        x[:500] = np.zeros([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = np.ones([500, 1, 1, 1]) * 510
        y[500:] = 1
        indices = np.random.permutation(x.size)
        x = x[indices]
        y = y[indices].astype(int)
        x_tensor_original = ds.from_array(np_array=x,
                                          shape=(2, 2))
        x_tensor = ds.from_array(np_array=x,
                                 shape=(2, 2))
        b = np.zeros((y.size, y.max() + 1))
        b[np.arange(y.size), y.flatten()] = 1
        y = b
        del b
        y_tensor_original = ds.from_array(np_array=y,
                                          shape=(2, 2))
        y_tensor = ds.from_array(np_array=y,
                                 shape=(2, 2))
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        encaps_function.build(net, optimizer, "binary_cross_entropy",
                              "categorical_accuracy", num_nodes=1, num_gpu=0)
        parameters = encaps_function.fit_synchronous_with_GPU(
            x_tensor, y_tensor, 25, 5)
        net = eddl.Model([in_], [out])
        eddl.build(
            net,
            eddl.sgd(0.002, 0.9, 0.8),
            ["binary_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(),
        )
        eddl.set_parameters(net, parametersToEDDLTensor(parameters))
        x_test = x_tensor_original.collect()
        y_test = y_tensor_original.collect()
        x_test = Tensor.fromarray(x_test[0][0])
        y_test = Tensor.fromarray(y_test[0][0].astype(np.float32))
        x_test.div_(255.0)
        prediction = eddl.predict(net, [x_test])[0].getdata()
        total_score = accuracy_score(np.argmax(y_test.getdata(), axis=1),
                                     np.argmax(prediction, axis=1))
        self.assertTrue(total_score == 1)

    def test_synchronous_shuffle_with_GPU_training(self):
        x = np.ones([1000, 1, 1, 1])
        y = np.ones([1000, 1])
        x[:500] = np.zeros([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = np.ones([500, 1, 1, 1]) * 510
        y[500:] = 1
        indices = np.random.permutation(x.size)
        x = x[indices]
        y = y[indices].astype(int)
        x_tensor_original = ds.from_array(np_array=x,
                                          shape=(2, 2))
        x_tensor = ds.from_array(np_array=x,
                                 shape=(2, 2))
        b = np.zeros((y.size, y.max() + 1))
        b[np.arange(y.size), y.flatten()] = 1
        y = b
        del b
        y_tensor_original = ds.from_array(np_array=y,
                                          shape=(2, 2))
        y_tensor = ds.from_array(np_array=y,
                                 shape=(2, 2))
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        encaps_function.build(net, optimizer, "binary_cross_entropy",
                              "categorical_accuracy", num_nodes=1, num_gpu=0)
        parameters = encaps_function.fit_synchronous_shuffle_with_GPU(
            x_tensor, y_tensor, 25, 7)
        net = eddl.Model([in_], [out])
        eddl.build(
            net,
            eddl.sgd(0.002, 0.9, 0.8),
            ["binary_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(),
        )
        eddl.set_parameters(net, parametersToEDDLTensor(parameters))
        x_test = x_tensor_original.collect()
        y_test = y_tensor_original.collect()
        x_test = Tensor.fromarray(x_test[0][0])
        y_test = Tensor.fromarray(y_test[0][0].astype(np.float32))
        x_test.div_(255.0)
        prediction = eddl.predict(net, [x_test])[0].getdata()
        total_score = accuracy_score(np.argmax(y_test.getdata(), axis=1),
                                     np.argmax(prediction, axis=1))
        self.assertTrue(total_score == 1)

    def test_asynchronous_with_GPU_training(self):
        x = np.ones([1000, 1, 1, 1])
        y = np.ones([1000, 1])
        x[:500] = np.zeros([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = np.ones([500, 1, 1, 1]) * 510
        y[500:] = 1
        indices = np.random.permutation(x.size)
        x = x[indices]
        y = y[indices].astype(int)
        x_tensor = ds.from_array(np_array=x,
                                 shape=(2, 2))
        x_tensor_original = ds.from_array(np_array=x,
                                          shape=(2, 2))
        b = np.zeros((y.size, y.max() + 1))
        b[np.arange(y.size), y.flatten()] = 1
        y = b
        del b
        y_tensor = ds.from_array(np_array=y,
                                 shape=(2, 2))
        y_tensor_original = ds.from_array(np_array=y,
                                          shape=(2, 2))
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        encaps_function.build(net, optimizer, "binary_cross_entropy",
                              "categorical_accuracy", num_nodes=1, num_gpu=0)
        parameters = encaps_function.fit_asynchronous_with_GPU(
            x_tensor, y_tensor, 25, 7)
        net = eddl.Model([in_], [out])
        eddl.build(
            net,
            eddl.sgd(0.002, 0.9, 0.8),
            ["binary_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(),
        )
        eddl.set_parameters(net, parametersToEDDLTensor(parameters))
        x_test = x_tensor_original.collect()
        y_test = y_tensor_original.collect()
        x_test = Tensor.fromarray(x_test[0][0])
        y_test = Tensor.fromarray(y_test[0][0].astype(np.float32))
        x_test.div_(255.0)
        prediction = eddl.predict(net, [x_test])[0].getdata()
        total_score = accuracy_score(np.argmax(y_test.getdata(), axis=1),
                                     np.argmax(prediction, axis=1))
        self.assertTrue(total_score == 1)

    def test_asynchronous_shuffle_with_GPU_training(self):
        x = np.ones([1000, 1, 1, 1])
        y = np.ones([1000, 1])
        x[:500] = np.zeros([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = np.ones([500, 1, 1, 1]) * 510
        y[500:] = 1
        indices = np.random.permutation(x.size)
        x = x[indices]
        y = y[indices].astype(int)
        x_tensor_original = ds.from_array(np_array=x,
                                          shape=(2, 2))
        x_tensor = ds.from_array(np_array=x,
                                 shape=(2, 2))
        b = np.zeros((y.size, y.max() + 1))
        b[np.arange(y.size), y.flatten()] = 1
        y = b
        del b
        y_tensor_original = ds.from_array(np_array=y,
                                          shape=(2, 2))
        y_tensor = ds.from_array(np_array=y,
                                 shape=(2, 2))
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        encaps_function.build(net, optimizer, "binary_cross_entropy",
                              "categorical_accuracy", num_nodes=1, num_gpu=0)
        parameters = encaps_function.fit_asynchronous_shuffle_with_GPU(
            x_tensor, y_tensor, 25, 5)
        net = eddl.Model([in_], [out])
        eddl.build(
            net,
            eddl.sgd(0.002, 0.9, 0.8),
            ["binary_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(),
        )
        eddl.set_parameters(net, parametersToEDDLTensor(parameters))
        x_test = x_tensor_original.collect()
        y_test = y_tensor_original.collect()
        x_test = Tensor.fromarray(x_test[0][0])
        y_test = Tensor.fromarray(y_test[0][0].astype(np.float32))
        x_test.div_(255.0)
        prediction = eddl.predict(net, [x_test])[0].getdata()
        total_score = accuracy_score(np.argmax(y_test.getdata(), axis=1),
                                     np.argmax(prediction, axis=1))
        self.assertTrue(total_score == 1)

    def test_asynchronous_n_epochs_with_GPU_training(self):
        x = np.ones([1000, 1, 1, 1])
        y = np.ones([1000, 1])
        x[:500] = np.zeros([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = np.ones([500, 1, 1, 1]) * 510
        y[500:] = 1
        indices = np.random.permutation(x.size)
        x = x[indices]
        y = y[indices].astype(int)
        x_tensor = ds.from_array(np_array=x,
                                 shape=(2, 2))
        x_tensor_original = ds.from_array(np_array=x,
                                          shape=(2, 2))
        b = np.zeros((y.size, y.max() + 1))
        b[np.arange(y.size), y.flatten()] = 1
        y = b
        del b
        y_tensor = ds.from_array(np_array=y,
                                 shape=(2, 2))
        y_tensor_original = ds.from_array(np_array=y,
                                          shape=(2, 2))
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        encaps_function.build(net, optimizer, "binary_cross_entropy",
                              "categorical_accuracy", num_nodes=1, num_gpu=0)
        parameters = encaps_function.fit_asynchronous_n_epochs_with_GPU(
            x_tensor, y_tensor, 25, 5, n_epocs_sync=2)
        net = eddl.Model([in_], [out])
        eddl.build(
            net,
            eddl.sgd(0.002, 0.9, 0.8),
            ["binary_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(),
        )
        eddl.set_parameters(net, parametersToEDDLTensor(parameters))
        x_test = x_tensor_original.collect()
        y_test = y_tensor_original.collect()
        x_test = Tensor.fromarray(x_test[0][0])
        y_test = Tensor.fromarray(y_test[0][0].astype(np.float32))
        x_test.div_(255.0)
        prediction = eddl.predict(net, [x_test])[0].getdata()
        total_score = accuracy_score(np.argmax(y_test.getdata(), axis=1),
                                     np.argmax(prediction, axis=1))
        self.assertTrue(total_score == 1)

    def test_asynchronous_shuffle_n_epochs_with_GPU_training(self):
        x = np.ones([1000, 1, 1, 1])
        y = np.ones([1000, 1])
        x[:500] = np.zeros([500, 1, 1, 1])
        y[:500] = 0
        x[500:] = np.ones([500, 1, 1, 1]) * 510
        y[500:] = 1
        indices = np.random.permutation(x.size)
        x = x[indices]
        y = y[indices].astype(int)
        x_tensor_original = ds.from_array(np_array=x,
                                          shape=(2, 2))
        x_tensor = ds.from_array(np_array=x,
                                 shape=(2, 2))
        b = np.zeros((y.size, y.max() + 1))
        b[np.arange(y.size), y.flatten()] = 1
        y = b
        del b
        y_tensor_original = ds.from_array(np_array=y,
                                          shape=(2, 2))
        y_tensor = ds.from_array(np_array=y,
                                 shape=(2, 2))
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        encaps_function.build(net, optimizer, "binary_cross_entropy",
                              "categorical_accuracy", num_nodes=1, num_gpu=0)
        parameters = encaps_function.\
            fit_asynchronous_shuffle_n_epochs_with_GPU(
              x_tensor, y_tensor, 25, 8, n_epocs_sync=2)
        net = eddl.Model([in_], [out])
        eddl.build(
            net,
            eddl.sgd(0.002, 0.9, 0.8),
            ["binary_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(),
        )
        eddl.set_parameters(net, parametersToEDDLTensor(parameters))
        x_test = x_tensor_original.collect()
        y_test = y_tensor_original.collect()
        x_test = Tensor.fromarray(x_test[0][0])
        y_test = Tensor.fromarray(y_test[0][0].astype(np.float32))
        x_test.div_(255.0)
        prediction = eddl.predict(net, [x_test])[0].getdata()
        total_score = accuracy_score(np.argmax(y_test.getdata(), axis=1),
                                     np.argmax(prediction, axis=1))
        self.assertTrue(total_score == 1)

    def test_build_optimizer(self):
        encaps_function = EncapsulatedFunctionsDistributedEddl(num_workers=1)
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        self.assertTrue(isinstance(encaps_function.
                                   _build_optimizer(optimizer),
                                   type(eddl.sgd())))
        optimizer = {'optimizer': 'adam', 'lr': 0.002}
        self.assertTrue(isinstance(encaps_function.
                                   _build_optimizer(optimizer),
                                   type(eddl.adam())))
        optimizer = {'optimizer': 'rmsprop', 'lr': 0.002}
        self.assertTrue(isinstance(encaps_function.
                                   _build_optimizer(optimizer),
                                   type(eddl.rmsprop())))
        optimizer = {'optimizer': 'adagrad', 'lr': 0.002}
        self.assertTrue(isinstance(encaps_function.
                                   _build_optimizer(optimizer),
                                   type(eddl.adagrad(lr=0.002,
                                                     epsilon=0.000001,
                                                     weight_decay=0))))
        optimizer = {'optimizer': 'adamax', 'lr': 0.002}
        self.assertTrue(isinstance(encaps_function.
                                   _build_optimizer(optimizer),
                                   type(eddl.adamax(lr=0.002,
                                                    epsilon=0.000001,
                                                    beta_1=0.9, beta_2=0.99,
                                                    weight_decay=0))))
        optimizer = {'optimizer': 'adadelta', 'lr': 0.001}
        self.assertTrue(isinstance(encaps_function.
                                   _build_optimizer(optimizer),
                                   type(eddl.adadelta(lr=0.001,
                                                      epsilon=0.000001,
                                                      rho=0.9,
                                                      weight_decay=0))))
        optimizer = {'optimizer': 'nadam', 'lr': 0.002}
        self.assertTrue(isinstance(encaps_function.
                                   _build_optimizer(optimizer),
                                   type(eddl.nadam(lr=0.002,
                                                   epsilon=0.000001,
                                                   beta_1=0.9, beta_2=0.99,
                                                   schedule_decay=0.04))))
        eddldistrcon = EddlDistributedConmutativo()
        optimizer = {'optimizer': 'sgd', 'lr': 0.002}
        self.assertTrue(isinstance(eddldistrcon.
                                   _build_optimizer(optimizer),
                                   type(eddl.sgd())))
        optimizer = {'optimizer': 'adam', 'lr': 0.002}
        self.assertTrue(isinstance(eddldistrcon.
                                   _build_optimizer(optimizer),
                                   type(eddl.adam())))
        optimizer = {'optimizer': 'rmsprop', 'lr': 0.002}
        self.assertTrue(isinstance(eddldistrcon.
                                   _build_optimizer(optimizer),
                                   type(eddl.rmsprop())))
        optimizer = {'optimizer': 'adagrad', 'lr': 0.002}
        self.assertTrue(isinstance(eddldistrcon.
                                   _build_optimizer(optimizer),
                                   type(eddl.adagrad(lr=0.002,
                                                     epsilon=0.000001,
                                                     weight_decay=0))))
        optimizer = {'optimizer': 'adamax', 'lr': 0.002}
        self.assertTrue(isinstance(eddldistrcon.
                                   _build_optimizer(optimizer),
                                   type(eddl.adamax(lr=0.002,
                                                    epsilon=0.000001,
                                                    beta_1=0.9, beta_2=0.99,
                                                    weight_decay=0))))
        optimizer = {'optimizer': 'adadelta', 'lr': 0.001}
        self.assertTrue(isinstance(eddldistrcon.
                                   _build_optimizer(optimizer),
                                   type(eddl.adadelta(lr=0.001,
                                                      epsilon=0.000001,
                                                      rho=0.9,
                                                      weight_decay=0))))
        optimizer = {'optimizer': 'nadam', 'lr': 0.002}
        self.assertTrue(isinstance(eddldistrcon.
                                   _build_optimizer(optimizer),
                                   type(eddl.nadam(lr=0.002,
                                                   epsilon=0.000001,
                                                   beta_1=0.9, beta_2=0.99,
                                                   schedule_decay=0.04))))

    def test_setter_getter_state(self):
        eddldistrcon = EddlDistributedConmutativo()
        in_ = eddl.Input([1, 1, 1])
        model = TestsNetworkCNNEDDL(in_)
        out = eddl.Softmax(eddl.Dense(model, 2))
        net = eddl.Model([in_], [out])
        state = {"model": net, "initialized": False,
                 'loss': 'binary_cross_entropy',
                 'metric': 'categorical_accuracy',
                 'num_gpu': 0,
                 'optimizer': eddl.sgd}
        self.assertTrue(eddldistrcon.__getstate__()["model"] is None)
        eddldistrcon.__setstate__(state)
        self.assertTrue(eddldistrcon.__getstate__()["model"] is not None)

    def test_initialize_model(self):
        in_ = eddl.Input([3, 224, 224])
        out = VGG19(in_)
        out = eddl.Softmax(eddl.Dense(out, 2))
        net = eddl.Model([in_], [out])
        self.assertTrue(isinstance(net, pyeddl._core.Net))
        out = VGG16(in_)
        out = eddl.Softmax(eddl.Dense(out, 2))
        net = eddl.Model([in_], [out])
        self.assertTrue(isinstance(net, pyeddl._core.Net))
