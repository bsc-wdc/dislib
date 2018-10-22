import mmap
import os
from collections import deque
from itertools import islice
from time import time

import numpy as np
from pycompss.api.api import compss_barrier as barrier
from pycompss.api.api import compss_delete_object
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import *
from pycompss.api.task import task
from scipy.sparse import vstack, issparse
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from uuid import uuid4


class CascadeSVM(object):
    name_to_kernel = {"linear": "_linear_kernel", "rbf": "_rbf_kernel"}

    def __init__(self, split_times=False):
        """

        :param split_times: boolean, optional (default=False)
            Whether to compute read and fit times separately.
        """

        self.iterations = []
        self.converged = []

        self.read_time = 0
        self.fit_time = 0
        self.total_time = 0

        self._split_times = split_times
        self._cascade_arity = []
        self._max_iterations = []
        self._nchunks = []
        self._tol = []
        self._last_W = []
        self._clf = []
        self._data = []
        self._clf_params = []
        self._kernel_f = []
        

    def fit(self, check_convergence=True):
        """
        Fits one or more models using training data. The fit function uses data previously loaded with the load_data
        function. The training process of each dataset is performed in parallel. The resulting models are stored in
        self._clf in the order in which the datasets were loaded.
        
        :param check_convergence: boolean, optional (default=True)
            Whether to test for convergence. If False, the algorithm will run for the number of iterations specified in
            load_data.
            
            Checking for convergence adds a synchronization point, and can negatively affect performance if multiple
            datasets are fit in parallel.
        """

        if self._split_times:
            barrier()
            self.read_time = time() - self.read_time
            self.fit_time = time()

        self._do_fit(check_convergence)

        barrier()

        if self._split_times:
            self.fit_time = time() - self.fit_time

        self.total_time = time() - self.total_time

    def load_data(self, X=None, y=None, path=None, n_features=None, data_format="csv", force_dense=False, cascade_arity=2, n_chunks=4,
                  cascade_iterations=5, tol=10 ** -3, C=1.0, kernel="rbf", gamma="auto"):
        """
        Loads a set of vectors to be used in the training process through the fit function. Multiple calls to load_data
        will result in the fit function computing multiple decision functions in parallel.
        
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features), optional (default=None)
            Training vectors, where n_samples is the number of samples and n_features is the number of features. 
            
        :param y: array-like, shape (n_samples,), optional (default=None)
            Class labels.
        
        :param path: string, optional (default=None)
            Path to a file or a directory containing files with train data. If path is defined, X and y are ignored and
            train data is read from path.
            
            All input files are assumed to be randomly shuffled, and to contain samples from both classes.
            
        :param n_features: int
            Number of features. This parameter is mandatory only if path is a directory and data_format is 'libsvm'.
            Otherwise, the parameter is optional although setting it saves counting the number of features when train
            data is read from files.
        
        :param data_format: string, optional (default='csv')
            The format of the data in path. It can be 'csv' for CSV with the label in the last column or 'libsvm'.

        :param force_dense: bool, optional (default=True)
            If set to True, parse data in libsvm format as a NumPy array instead of a sparse matrix.

        :param cascade_arity: int, optional (default=2)
            Arity of the reduction stage.
        
        :param n_chunks: int, optional (default=4)
            Number of chunks in which to split the training data (ignored if path is a directory).
        
        :param cascade_iterations: int, optional (default=5)
            Maximum number of iterations to perform.
        
        :param tol: float, optional (default=1e-3) 
            Tolerance for the stopping criterion.
            
        :param C: float, optional (default=1.0)
            Penalty parameter C of the error term.
            
        :param kernel: string, optional (default='rbf')
            Specifies the kernel type to be used in the algorithm. It must be one of 'linear' or 'rbf'.
            
        :param gamma: float, optional (default='auto')
            Kernel coefficient for 'rbf'. If gamma is 'auto' then 1/n_features will be used instead.
        """

        try:
            self._kernel_f.append(getattr(self, CascadeSVM.name_to_kernel[kernel]))
        except AttributeError:
            self._kernel_f.append(getattr(self, CascadeSVM.name_to_kernel["rbf"]))

        assert (gamma is "auto" or type(gamma) == float or type(float(gamma)) == float), "Gamma is not a valid float"
        assert (kernel is None or kernel in self.name_to_kernel.keys()), \
            "Incorrect kernel value [%s], available kernels are %s" % (kernel, self.name_to_kernel.keys())
        assert (C is None or type(C) == float or type(float(C)) == float), \
            "Incorrect C type [%s], type : %s" % (C, type(C))
        assert cascade_arity > 1, "Cascade arity must be greater than 1"
        assert cascade_iterations > 0, "Max iterations must be greater than 0"

        self._cascade_arity.append(cascade_arity)
        self._nchunks.append(n_chunks)
        self._max_iterations.append(cascade_iterations)
        self._force_dense = force_dense
        self._tol.append(tol)
        self._clf_params.append({"gamma": gamma, "C": C, "kernel": kernel})
        self.iterations.append(0)
        self._last_W.append(None)
        self._clf.append(None)

        if self._split_times and not self._data:
            self.read_time = time()

        if not self._data:
            self.total_time = time()

        # WARNING: when partitioning the data it is not guaranteed that all chunks contain vectors from both classes               
        if path and os.path.isdir(path):
            chunks = self._read_dir(path, data_format, n_features)
        elif path:
            chunks = self._read_file(path, data_format, n_features)
        else:
            chunks = self._read_data(X, y)

        self._data.append(chunks)
        self.converged.append(False)

    def predict(self, X, i=0):
        """
        Perform classification on samples in X using model i.
        
        :param X: array-like, shape (n_samples, n_features)
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel. 
        :return y_pred: array, shape (n_samples,)
            Class labels for samples in X.
        """

        if len(self._clf) > i and self._clf[i]:
            return self._clf[i].predict(X)
        else:
            raise Exception("Model %s has not been initialized. Try calling fit first." % i)

    def decision_function(self, X, i=0):
        """
        Distance of the samples X to the ith separating hyperplane.
        
        :param X: array-like, shape (n_samples, n_features)
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel. 
        :return: array-like, shape (n_samples, n_classes * (n_classes-1) / 2)        
            Returns the decision function of the sample for each class in the ith model.
        """

        if len(self._clf) > i and self._clf[i]:
            return self._clf[i].decision_function(X)
        else:
            raise Exception("Model %s has not been initialized. Try calling fit first." % i)

    def score(self, X, y, i=0):
        """
        Returns the mean accuracy on the given test data and labels using model
        i.
        
        :param X: array-like, shape = (n_samples, n_features)
            Test samples.
        :param y: array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.  
        :param i: int, optional (default=0)
            Model index in case multiple models have been built in parallel.
        :return score: Mean accuracy of self.predict(X, i) wrt. y.
        """

        if len(self._clf) > i and self._clf[i]:
            return self._clf[i].score(X, y)
        else:
            raise Exception("Model %s has not been initialized. Try calling fit first." % i)

    def _read_dir(self, path, data_format, n_features):
        files = os.listdir(path)

        if data_format == "libsvm":
            assert n_features > 0, "Number of features is required to read from multiple files using libsvm format"
        elif not n_features:
            n_features = self._count_features(os.path.join(path, files[0]), data_format)

        if self._clf_params[-1]["gamma"] == "auto":
            self._clf_params[-1]["gamma"] = 1. / n_features

        self._nchunks[-1] = len(files)

        chunks = []

        for f in files:
            chunks.append(read_chunk(os.path.join(path, f), data_format=data_format, n_features=n_features,
                                     force_dense=self._force_dense))

        return chunks

    def _read_file(self, path, data_format, n_features):
        n_lines = self._count_lines(path)

        assert n_lines > self._nchunks[-1], "Not enough vectors to divide into %s chunks\n" \
                                            " - Minimum required elements: %s\n" \
                                            " - Vectors available: %s\n" % \
                                            (self._nchunks[-1], self._nchunks[-1], n_lines)

        if not n_features:
            n_features = self._count_features(path, data_format)

        if self._clf_params[-1]["gamma"] == "auto":
            self._clf_params[-1]["gamma"] = 1. / n_features

        steps = np.linspace(0, n_lines + 1, self._nchunks[-1] + 1, dtype=int)
        chunks = []

        for s in range(len(steps) - 1):
            chunks.append(read_chunk(path, steps[s], steps[s + 1], data_format=data_format, n_features=n_features,
                                     force_dense=self._force_dense))

        return chunks

    def _read_data(self, X, y):
        chunks = self._get_chunks(X, y)

        if self._clf_params[-1]["gamma"] == "auto":
            self._clf_params[-1]["gamma"] = 1. / X.shape[1]

        return chunks

    def _do_fit(self, check_convergence):
        q = deque()
        feedback = []
        finished = []

        for _ in self._data:
            feedback.append(None)
            finished.append(None)

        while not np.array(finished).all():
            for idx, chunks in enumerate(self._data):
                if not finished[idx]:

                    # first level
                    for chunk in chunks:
                        data = filter(None, [chunk, feedback[idx]])
                        q.append(train(False, *data, **self._clf_params[idx]))

                    # reduction
                    while len(q) > 1:
                        data = []

                        while q and len(data) < self._cascade_arity[idx]:
                            data.append(q.popleft())

                        if q or not check_convergence:
                            q.append(train(False, *data, **self._clf_params[idx]))
                        elif not q:
                            sv, sl, si, self._clf[idx] = compss_wait_on(train(True, *data, **self._clf_params[idx]))
                            q.append((sv, sl, si))

                        # delete partial results
                        for d in data:
                            compss_delete_object(d)

                    feedback[idx] = q.popleft()
                    self.iterations[idx] += 1

                    if check_convergence:
                        self._check_convergence_and_update_w(feedback[idx][0], feedback[idx][1], idx)
                        print("Dataset %s iteration %s of %s. \n" % (
                            idx, self.iterations[idx], self._max_iterations[idx]))

                    if self.iterations[idx] >= self._max_iterations[idx] or self.converged[idx]:
                        finished[idx] = True

    def _lagrangian_fast(self, SVs, sl, coef, idx):
        set_sl = set(sl)
        assert len(set_sl) == 2, "Only binary problem can be handled"
        new_sl = sl.copy()
        new_sl[sl == 0] = -1

        if issparse(coef):
            coef = coef.todense()

        C1, C2 = np.meshgrid(coef, coef)
        L1, L2 = np.meshgrid(new_sl, new_sl)
        double_sum = C1 * C2 * L1 * L2 * self._kernel_f[idx](SVs, idx)
        double_sum = double_sum.sum()
        W = -0.5 * double_sum + coef.sum()

        return W

    def _rbf_kernel(self, x, idx):
        # Trick: || x - y || ausmultipliziert
        sigmaq = -1 / (2 * self._clf_params[idx]["gamma"])
        n = x.shape[0]
        K = x.dot(x.T) / sigmaq

        if issparse(K):
            K = K.todense()

        d = np.diag(K).reshape((n, 1))
        K = K - np.ones((n, 1)) * d.T / 2
        K = K - d * np.ones((1, n)) / 2
        K = np.exp(K)
        return K

    def _check_convergence_and_update_w(self, sv, sl, idx):
        self.converged[idx] = False
        clf = self._clf[idx]
        print("Checking convergence for model %s:" % idx)

        if clf:
            W = self._lagrangian_fast(sv, sl, clf.dual_coef_, idx)
            print("     Computed W %s" % W)

            if self._last_W[idx]:
                delta = np.abs((W - self._last_W[idx]) / self._last_W[idx])
                if delta < self._tol[idx]:
                    print("     Converged with delta: %s " % delta)
                    self.converged[idx] = True
                else:
                    print("     No convergence with delta: %s " % delta)
            else:
                print("     First iteration, not testing convergence.")
            self._last_W[idx] = W
            print()

    def _get_chunks(self, X, y):
        chunks = []

        steps = np.linspace(0, X.shape[0], self._nchunks[-1] + 1, dtype=int)

        for s in range(len(steps) - 1):
            chunkx = X[steps[s]:steps[s + 1]]
            chunky = y[steps[s]:steps[s + 1]]

            chunks.append((chunkx, chunky))

        return chunks

    @staticmethod
    def _count_lines(filename):
        f = open(filename, "r+")
        buf = mmap.mmap(f.fileno(), 0)
        buf.readline()
        lines = 1
        readline = buf.readline

        while readline():
            lines += 1

        f.close()

        return lines

    @staticmethod
    def _count_features(filename, data_format=None):
        if data_format == "libsvm":
            X, y = load_svmlight_file(filename)
            features = X.shape[1]
        else:
            f = open(filename, "r+")
            buf = mmap.mmap(f.fileno(), 0)
            line = buf.readline()
            features = len(line.split(",")) - 1
            f.close()

        return features

    @staticmethod
    def _linear_kernel(x1):
        return np.dot(x1, x1.T)


@task(returns=tuple)
def train(return_classifier, *args, **kwargs):
    if len(args) > 1:
        X, y, idx = merge(*args)
    else:
        X, y, idx = args[0]

    clf = SVC(random_state=1, **kwargs)
    clf.fit(X, y)

    sv = X[clf.support_]
    sl = y[clf.support_]
    idx = idx[clf.support_]

    if return_classifier:
        return sv, sl, idx, clf
    else:
        return sv, sl, idx


@task(filename=FILE, returns=tuple)
def read_chunk(filename, start=None, stop=None, data_format=None, n_features=None, force_dense=None):
    if data_format == "libsvm":
        X, y = load_svmlight_file(filename, n_features)

        if force_dense:
            X = X.toarray()

        if start and stop:
            X = X[start:stop]
            y = y[start:stop]
    else:
        with open(filename) as f:
            vecs = np.genfromtxt(islice(f, start, stop), delimiter=",")

        X, y = vecs[:, :-1], vecs[:, -1]

    # create array of unique identifiers for each vector
    idx = np.array([uuid4().int for _ in range(X.shape[0])])

    return X, y, idx


def merge(*args):
    if issparse(args[0][0]):
        sv = vstack([t[0] for t in args])
    else:
        sv = np.concatenate([t[0] for t in args])

    sl = np.concatenate([t[1] for t in args])
    si = np.concatenate([t[2] for t in args])

    si, uniques = np.unique(si, return_index=True)
    sv = sv[uniques]
    sl = sl[uniques]

    return sv, sl, si
