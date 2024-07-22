import pyeddl.eddl as eddl
import numpy as np

from dislib.eddl.utils import getParameters, \
    aggregateParameters, parametersToNumpy
from dislib.eddl.eddl_distributed_conmutativo import EddlDistributedConmutativo
from pycompss.api.api import compss_wait_on, compss_delete_object
from dislib.data.tensor import shuffle


class EncapsulatedFunctionsDistributedEddl(object):
    """
    Object that encapsulates the different distributed trainings that can be
    done using PyCOMPSs. Each function implements a different version, the
    number of epochs and batches is specified in each of the functions.

    There are mainly three different types of training.

    - Synchronous training: At the end of each epoch, the weights are
    synchronized and the update is computed.
    - Partially asynchronous: The weights of each worker are updated
    commutatively with the general weights and viceversa.
    - Asynchronous training: A synchronization and update of the weigths is
    done after executing all the epochs or each n specified epochs.

    Attributes
    ----------
    model_parameters : tensor
        weights and biases of the different layers of the network that
        is being trained.
    compss_object: list
        List that contains objects of type PytorchDistributed, each of the
        objects in this list makes a small part of the epoch training in
        parallel to the rest.
    num_workers: int
        Number of parallel trainings existing.

    """
    def __init__(self, num_workers=10):
        self.num_workers = num_workers

    def build(self, net, optimizer, loss, metric, num_gpu=0, num_nodes=0):
        """
        Builds the model to obtain the initial parameters of the training
        and it also builds the model in each worker
        in order to be ready to start the training.

        Parameters
        ----------
        net : eddl.Model
            Neural network model to be used during the parallel training.
        optimizer: dict
            Dictionary containing the optimizer to be used and its parameters.
        loss: str
            String specifying the loss to be used during the training.
        metric: str
            Metric to be used during the training.
        num_gpu: int
            Number of GPUs to use during the training.
        num_nodes: int
            Number of nodes available during the training.
        Returns
        -------
        (void)
        """
        if loss not in ("mean_squared_error", "mse",
                        "soft_cross_entropy", "sce",
                        "binary_cross_entropy", "bce",
                        "categorical_cross_entropy", "cce",
                        "cross_entropy", "ce",
                        "softmax_cross_entropy", "dice"):
            raise ValueError("This loss is not supported.")
        if metric not in ("mse", "categorical_accuracy",
                          "accuracy", "mae", "mean_absolute_error",
                          "mre", "mean_relative_error"):
            raise ValueError("This metric is not supported.")
        if num_gpu > 0:
            num_gpus = np.ones((num_gpu), dtype=np.int8)
            comps_service = eddl.CS_GPU(num_gpus.tolist())
            eddl.build(
                net,
                self._build_optimizer(optimizer),
                [loss],
                [metric],
                comps_service
            )
        else:
            num_gpus = None
            comps_service = eddl.CS_CPU()
            eddl.build(
                net,
                self._build_optimizer(optimizer),
                [loss],
                [metric],
                comps_service
            )
        self.num_gpus = num_gpus
        self.num_gpu = num_gpu
        self.num_gpus_per_worker = int(num_nodes * num_gpu / self.num_workers)
        self.compss_object = EddlDistributedConmutativo()
        self.new_lr = optimizer['lr']
        self.compss_object.build(net, optimizer,
                                 loss, metric, num_gpu,
                                 gpu=np.arange(num_gpu))
        model_parameters, gradients = getParameters(net)
        self.model_parameters = parametersToNumpy(model_parameters)

    def get_parameters(self):
        """
        Returns the parameters (weights) of the neural network
        Returns
        -------
        model_parameters: np.array
        """
        return self.model_parameters

    def fit_synchronous_shuffle_every_n_epochs_with_GPU(self, x_train,
                                                        y_train,
                                                        num_batches_per_worker,
                                                        num_epochs,
                                                        n_epocs_sync=1):
        """
        Training of the neural network performing a syncrhonization every n
        specified epochs, it performs a total shuffle of the dataset used.

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every
            piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing a syncrhonization
            and between synchronizations
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [self.model_parameters for _ in
                                  range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            for row in range(x_train.shape[0]):
                for col in range(x_train.shape[1]):
                    parameters_for_workers[j] = \
                        self.compss_object.train_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            x_train.dtype,
                            num_batches_per_worker,
                            epoch=i,
                            shuffle_block_data=False)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                parameters_for_workers = compss_wait_on(
                    parameters_for_workers)
                self.model_parameters = \
                    aggregateParameters(parameters_for_workers)
                parameters_for_workers = [self.model_parameters for _
                                          in
                                          range(len(parameters_for_workers))]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = aggregateParameters(
            parameters_for_workers)
        return self.model_parameters

    def fit_synchronous_every_n_epochs_with_GPU(self, x_train, y_train,
                                                num_batches_per_worker,
                                                num_epochs,
                                                n_epocs_sync=1,
                                                shuffle_blocks=True,
                                                shuffle_block_data=True):
        """
        Training of the neural network performing a syncrhonization every
        n specified epochs, it performs a total shuffle of the tensors on
        the ds_tensor and the elements inside each tensor

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with
            every piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing a syncrhonization
            and between synchronizations
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside each
            tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [self.model_parameters for _
                                  in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        for i in range(num_epochs):
            j = 0
            if shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            for row in rows:
                for col in cols:
                    parameters_for_workers[j] = \
                        self.compss_object.train_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            x_train.dtype,
                            num_batches_per_worker,
                            epoch=i,
                            shuffle_block_data=shuffle_block_data)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                parameters_for_workers = compss_wait_on(
                    parameters_for_workers)
                self.model_parameters = aggregateParameters(
                    parameters_for_workers)
                parameters_for_workers = [self.model_parameters for _
                                          in range(self.num_workers)]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = aggregateParameters(
            parameters_for_workers)
        return self.model_parameters

    def fit_synchronous_with_GPU(self, x_train, y_train,
                                 num_batches_per_worker,
                                 num_epochs, shuffle_blocks=True,
                                 shuffle_block_data=True):
        """
        Training of the neural network performing a syncrhonization of
        the weights at the end of each epoch, it performs a total shuffle
         of the tensors on the ds_tensor and the elements inside each tensor

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with
            every piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside
            each tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [self.model_parameters for _
                                  in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        for i in range(num_epochs):
            j = 0
            if shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            for row in rows:
                for col in cols:
                    parameters_for_workers[j] = \
                        self.compss_object.train_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            x_train.dtype,
                            num_batches_per_worker,
                            epoch=i,
                            shuffle_block_data=shuffle_block_data)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            parameters_for_workers = compss_wait_on(parameters_for_workers)
            self.model_parameters = aggregateParameters(
                parameters_for_workers)
            parameters_for_workers = [self.model_parameters for _
                                      in range(self.num_workers)]
        compss_delete_object(parameters_for_workers)
        return self.model_parameters

    def fit_synchronous_shuffle_with_GPU(self, x_train, y_train,
                                         num_batches_per_worker,
                                         num_epochs):
        """
        Training of the neural network performing a syncrhonization of
        the weights every epoch, it performs a total shuffle of the dataset

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with
            every piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [self.model_parameters for _
                                  in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object.train_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            x_train.dtype,
                            num_batches_per_worker, epoch=i,
                            shuffle_block_data=False)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            parameters_for_workers = compss_wait_on(parameters_for_workers)
            self.model_parameters = \
                aggregateParameters(parameters_for_workers)
            parameters_for_workers = [self.model_parameters for _
                                      in range(self.num_workers)]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = aggregateParameters(
            parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_with_GPU(self, x_train, y_train,
                                  num_batches_per_worker, num_epochs,
                                  shuffle_blocks=True,
                                  shuffle_block_data=True):
        """
        Training of the neural network performing an asyncrhonous update of
        the weights every epoch,
        it performs a shuffle of the tensors on the ds_tensor and a local
        shuffle of the elements inside each tensor

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every
            piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside each
            tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [self.model_parameters for _ in
                                  range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        for i in range(num_epochs):
            j = 0
            if shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            for row in rows:
                for col in cols:
                    parameters_for_workers[j] = \
                        self.compss_object.train_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            x_train.dtype,
                            num_batches_per_worker,
                            epoch=i,
                            shuffle_block_data=shuffle_block_data)
                    parameters_for_workers[j] = \
                        self.compss_object.aggregate_parameters_async(
                            self.model_parameters, parameters_for_workers[j])
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
        self.model_parameters = compss_wait_on(self.model_parameters)
        return self.model_parameters

    def fit_asynchronous_shuffle_with_GPU(self, x_train, y_train,
                                          num_batches_per_worker,
                                          num_epochs):
        """
        Training of the neural network performing an asyncrhonous update of
        the weights every epoch,
        it performs a total shuffle of the dataset

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with
            every piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [self.model_parameters
                                  for _ in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            for row in range(x_train.shape[0]):
                for col in range(y_train.shape[1]):
                    parameters_for_workers[j] = \
                        self.compss_object.train_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            x_train.dtype,
                            num_batches_per_worker,
                            shuffle_block_data=False,
                            epoch=i)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            for j in range(self.num_workers):
                parameters_for_workers[j] = \
                    self.compss_object.aggregate_parameters_async(
                        self.model_parameters,
                        parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = aggregateParameters(
            parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_n_epochs_with_GPU(self, x_train, y_train,
                                           num_batches_per_worker, num_epochs,
                                           n_epocs_sync=0,
                                           shuffle_blocks=True,
                                           shuffle_block_data=True):
        """
        Training of the neural network performing an asyncrhonous update of
        the weights every n epochs,
        it performs a shuffle of the tensors and locally a shuffle of the
        elements inside each tensor

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every
            piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing an asyncrhonous
            update of the weights and between the following updates
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside each
            tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [self.model_parameters for _
                                  in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        if shuffle_blocks:
            for i in range(num_epochs):
                j = 0
                if shuffle_blocks:
                    rows = np.random.permutation(x_train.shape[0])
                    cols = np.random.permutation(x_train.shape[1])
                for row in rows:
                    for col in cols:
                        parameters_for_workers[j] = \
                            self.compss_object.train_batch_GPU(
                                parameters_for_workers[j],
                                x_train.tensors[int(row)][int(col)],
                                y_train.tensors[int(row)][int(col)],
                                x_train.dtype,
                                num_batches_per_worker,
                                shuffle_block_data=shuffle_block_data)
                        if n_epocs_sync != 0 and (i + 1) % n_epocs_sync == 0:
                            parameters_for_workers[j] = \
                                self.compss_object.aggregate_parameters_async(
                                    self.model_parameters,
                                    parameters_for_workers[j])
                        j = j + 1
                        if j == self.num_workers:
                            j = 0
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = aggregateParameters(
            parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_shuffle_n_epochs_with_GPU(self, x_train, y_train,
                                                   num_batches_per_worker,
                                                   num_epochs,
                                                   n_epocs_sync=0):
        """
        Training of the neural network performing an asyncrhonous update of
        the weights every n epochs,
        it performs a total shuffle of the dataset

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every
            piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing an asyncrhonous
            update of the weights and between the following updates
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [self.model_parameters for _
                                  in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            for row in range(x_train.shape[0]):
                for col in range(x_train.shape[1]):
                    parameters_for_workers[j] = \
                        self.compss_object.train_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            x_train.dtype,
                            num_batches_per_worker,
                            epoch=i,
                            shuffle_block_data=False)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            for j in range(self.num_workers):
                if n_epocs_sync != 0 and (i + 1) % n_epocs_sync == 0:
                    parameters_for_workers[j] = \
                        self.compss_object.aggregate_parameters_async(
                            self.model_parameters,
                            parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = aggregateParameters(
            parameters_for_workers)
        return self.model_parameters

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
