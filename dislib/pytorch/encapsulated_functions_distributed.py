import numpy as np
import copy
from pycompss.api.api import compss_wait_on, compss_delete_object
from dislib.data.tensor import shuffle
from dislib.pytorch.pytorch_distributed import PytorchDistributed
from sklearn.metrics import accuracy_score
from auxiliar_functions import pt_aggregate_parameters, \
        compute_validation_losses, assign_parameters, \
        evaluate_model_parameters


class EncapsulatedFunctionsDistributedPytorch(object):
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
        self.model_parameters = None
        self.num_workers = num_workers

    def build(self, net, optimizer, loss, optimizer_parameters,
              gradient_clipping=None, scheduler=None, T_max=1,
              eta_min=0.0, num_gpu=0, num_nodes=0):
        """
        Builds the model to obtain the initial parameters of the training
        and it also builds the model in each worker in order to be ready
        to start the training.

        Parameters
        ----------
        net : pytorch Model
            Neural network model to be used during the parallel training.
        optimizer: dict
            Dictionary containing the optimizer to be used and its parameters.
        loss: str
            String specifying the loss to be used during the training.
        num_gpu: int
            Number of GPUs to use during the training.
        num_nodes: int
            Number of nodes available during the training.
        Returns
        -------
        (void)
        """
        if num_gpu > 0:
            self.compss_object = [PytorchDistributed() for _ in
                                  range(num_gpu*num_nodes)]
            for i in range(num_gpu*num_nodes):
                self.compss_object[i].build(net, loss,
                                            copy.deepcopy(optimizer),
                                            optimizer_parameters)

        self.optimizer_parameters = optimizer_parameters
        self.num_gpu = num_gpu
        self.num_gpus_per_worker = int(num_nodes*num_gpu/self.num_workers)
        self.model_parameters = net
        self.optimizer = optimizer(self.model_parameters.parameters(),
                                   **optimizer_parameters)
        if gradient_clipping is not None:
            if callable(gradient_clipping):
                self.gradient_clipping = gradient_clipping
            else:
                self.gradient_clipping = None
                raise Warning("Gradient clipping specified should"
                              "be a callable function. Set to None")
        else:
            self.gradient_clipping = None
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer,
                                       T_max=T_max, eta_min=eta_min)
        else:
            self.scheduler = None

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
            Number of batches that each worker will be trained with every piece
             of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing a syncrhonization and
            between synchronizations
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=False,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                parameters_for_workers = compss_wait_on(
                    parameters_for_workers)
                self.model_parameters = \
                    pt_aggregate_parameters(parameters_for_workers)
                parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                          for _ in
                                          range(len(parameters_for_workers))]
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregate_parameters(
            parameters_for_workers)
        return self.model_parameters

    def fit_synchronous_every_n_epochs_with_GPU(self, x_train, y_train,
                                                num_batches_per_worker,
                                                num_epochs,
                                                n_epocs_sync=1,
                                                shuffle_blocks=True,
                                                shuffle_block_data=True):
        """
        Training of the neural network performing a syncrhonization every n
        specified epochs,  it performs a total shuffle of the tensors on the
        ds_tensor and the elements inside each tensor

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
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside each
            tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
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
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                parameters_for_workers = compss_wait_on(
                    parameters_for_workers)
                self.model_parameters = \
                    pt_aggregate_parameters(parameters_for_workers)
                parameters_for_workers = [
                    copy.deepcopy(self.model_parameters) for _
                    in range(len(parameters_for_workers))]
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregate_parameters(
            parameters_for_workers)
        return self.model_parameters

    def fit_synchronous_with_GPU_train_curves(self, x_train,
                                              y_train,
                                              num_batches_per_worker,
                                              num_epochs,
                                              shuffle_blocks=True,
                                              shuffle_block_data=True,
                                              complete_shuffle=False,
                                              x_test=None, y_test=None,
                                              y_test_to_loss=None,
                                              function_score=accuracy_score):
        """
        Training of the neural network performing a syncrhonization of the
        weights at the end of each epoch, it performs a total shuffle of
        the tensors on the ds_tensor and the elements inside each tensor

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
        x_test: list of tensors
            samples and features of the validation dataset
        y_test: list of tensors
            classes or values of the samples of the validation dataset
        y_test_to_loss: list of tensors
            classes or values of the samples of the validation dataset
            as the loss function expects
        function_score: function
            function to compute the score of the predictions during
            training and validation
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters) for
                                  _ in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        training_loss = []
        training_acc = []
        validation_acc = []
        validation_loss = []
        for i in range(num_epochs):
            j = 0
            if complete_shuffle:
                x_train, y_train = shuffle(x_train, y_train)
            elif shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            epoch_accuracy = []
            epoch_loss = []
            for row in rows:
                for col in cols:
                    parameters_for_workers[j], train_loss, train_accuracy = \
                        self.compss_object[j].train_cnn_batch_GPU_losses(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data,
                            score_to_compute=function_score,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
                    epoch_loss.append(train_loss)
                    epoch_accuracy.append(train_accuracy)
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
            parameters_for_workers = compss_wait_on(parameters_for_workers)
            epoch_loss = compss_wait_on(epoch_loss)
            epoch_accuracy = compss_wait_on(epoch_accuracy)
            training_acc.append(np.array(epoch_accuracy).mean())
            training_loss.append(np.array(epoch_loss).mean())
            self.model_parameters = pt_aggregate_parameters(
                parameters_for_workers)
            assign_parameters(self.compss_object[0].model,
                              self.model_parameters)
            torch_model = self.compss_object[0].model
            if y_test_to_loss is not None:
                val_loss, val_acc = compute_validation_losses(
                        torch_model, x_test, y_test_to_loss, y_test,
                        validation_score=function_score)
            else:
                val_loss, val_acc = compute_validation_losses(
                        torch_model, x_test, y_test, y_test,
                        validation_score=function_score)
            validation_loss.extend(val_loss)
            validation_acc.extend(val_acc)
            [compss_delete_object(params) for params in parameters_for_workers]
            del parameters_for_workers
            parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                      for _ in range(self.num_workers)]
        self.model_parameters = parameters_for_workers[0]
        return self.model_parameters, training_loss, training_acc, \
            validation_loss, validation_acc

    def fit_synchronous_with_GPU(self, x_train, y_train,
                                 num_batches_per_worker,
                                 num_epochs,
                                 shuffle_blocks=True,
                                 shuffle_block_data=True):
        """
        Training of the neural network performing a syncrhonization of the
        weights at the end of each epoch, it performs a total shuffle of
        the tensors on the ds_tensor and the elements inside each tensor

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
        parameters_for_workers = [copy.deepcopy(self.model_parameters) for
                                  _ in range(self.num_workers)]
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
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
            parameters_for_workers = compss_wait_on(parameters_for_workers)
            self.model_parameters = pt_aggregate_parameters(
                parameters_for_workers)
            [compss_delete_object(params) for params in parameters_for_workers]
            del parameters_for_workers
            parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                      for _ in range(self.num_workers)]
        self.model_parameters = parameters_for_workers[0]
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
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=False,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
            parameters_for_workers = compss_wait_on(parameters_for_workers)
            self.model_parameters = \
                pt_aggregate_parameters(parameters_for_workers)
            parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                      for _ in range(self.num_workers)]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregate_parameters(parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_with_GPU_train_curves(
            self, x_train, y_train,
            num_batches_per_worker,
            num_epochs,
            shuffle_blocks=True,
            shuffle_block_data=True,
            complete_shuffle=False,
            x_test=None,
            y_test=None,
            y_test_to_loss=None,
            function_score=accuracy_score):
        """
        Training of the neural network performing an asyncrhonous update
        of the weights every epoch, it performs a shuffle of the tensors
        on the ds_tensor and a local shuffle of the elements inside each
        tensor

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
        model_parameters: pytorch tensor
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        training_loss = []
        training_acc = []
        validation_loss = []
        validation_acc = []
        for i in range(num_epochs):
            j = 0
            if complete_shuffle:
                pass
            elif shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            epoch_accuracy = []
            epoch_loss = []
            for row in rows:
                for col in cols:
                    parameters_for_workers[j], train_loss, \
                            train_accuracy = \
                            self.compss_object[j].\
                            train_cnn_batch_GPU(
                                    parameters_for_workers[j],
                                    x_train.tensors[int(row)]
                                                   [int(col)],
                                    y_train.tensors[int(row)]
                                                   [int(col)],
                                    num_batches_per_worker,
                                    shuffle_block_data=shuffle_block_data,
                                    gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
                    epoch_loss.append(train_loss)
                    epoch_accuracy.append(train_accuracy)
            training_loss.append(epoch_loss)
            training_acc.append(epoch_accuracy)
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
            for j in range(self.num_workers):
                parameters_for_workers[j] = \
                    self.compss_object[j].aggregate_parameters_async(
                        self.model_parameters,
                        parameters_for_workers[j])
            val_loss, val_acc = evaluate_model_parameters(
                    x_test, y_test,
                    self.compss_object[0].model,
                    parameters_for_workers[-1],
                    self.loss,
                    score_function=function_score,
                    y_val_loss_function=y_test_to_loss)
            validation_acc.append(val_acc)
            validation_loss.append(val_loss)
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregate_parameters(parameters_for_workers)
        training_loss = compss_wait_on(training_loss)
        training_acc = compss_wait_on(training_acc)
        validation_loss = compss_wait_on(validation_loss)
        validation_acc = compss_wait_on(validation_acc)
        training_loss = [np.array(epoch_loss).mean()
                         for epoch_loss in training_loss]
        training_acc = [np.array(epoch_accuracy).mean()
                        for epoch_accuracy in training_acc]
        return self.model_parameters, training_loss, training_acc, \
            validation_loss, validation_acc

    def fit_asynchronous_with_GPU(self, x_train, y_train,
                                  num_batches_per_worker,
                                  num_epochs,
                                  shuffle_blocks=True,
                                  shuffle_block_data=True):
        """
        Training of the neural network performing an asyncrhonous update
        of the weights every epoch, it performs a shuffle of the tensors
        on the ds_tensor and a local shuffle of the elements inside each
        tensor

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
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
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
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
            for j in range(self.num_workers):
                parameters_for_workers[j] = \
                    self.compss_object[j].aggregate_parameters_async(
                        self.model_parameters,
                        parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregate_parameters(parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_shuffle_with_GPU(self, x_train, y_train,
                                          num_batches_per_worker,
                                          num_epochs):
        """
        Training of the neural network performing an asyncrhonous
        update of the weights every epoch, it performs a total shuffle
         of the dataset

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
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=False,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
            for j in range(self.num_workers):
                parameters_for_workers[j] = \
                    self.compss_object[j].aggregate_parameters_async(
                        self.model_parameters,
                        parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregate_parameters(parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_n_epochs_with_GPU(self, x_train, y_train,
                                           num_batches_per_worker,
                                           num_epochs,
                                           n_epocs_sync=0,
                                           shuffle_blocks=True,
                                           shuffle_block_data=True):
        """
        Training of the neural network performing an asyncrhonous update
        of the weights every n epochs, it performs a shuffle of the tensors
        and locally a shuffle of the elements inside each tensor

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
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
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
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = \
                    self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = \
                            self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = \
                            self.optimizer.param_groups[0]["weight_decay"]
            if (i + 1) % n_epocs_sync == 0:
                for j in range(self.num_workers):
                    parameters_for_workers[j] = \
                        self.compss_object[j].aggregate_parameters_async(
                            self.model_parameters,
                            parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregate_parameters(parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_shuffle_n_epochs_with_GPU(self, x_train,
                                                   y_train,
                                                   num_batches_per_worker,
                                                   num_epochs,
                                                   n_epocs_sync=0):
        """
        Training of the neural network performing an asyncrhonous update
        of the weights every n epochs, it performs a total shuffle of the
        dataset

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every piece
            of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing an asyncrhonous update
            of the weights and between the following updates
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        pt_aggregate_parameters(parameters_for_workers)
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=False,
                            gradient_clipping=self.gradient_clipping)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                for j in range(self.num_workers):
                    parameters_for_workers[j] = \
                        self.compss_object[j].aggregate_parameters_async(
                            self.model_parameters,
                            parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregate_parameters(parameters_for_workers)
        return self.model_parameters
