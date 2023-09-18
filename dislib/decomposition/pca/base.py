import json
import os
import pickle

import numpy as np
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN, Depth, Type, COLLECTION_OUT
from pycompss.api.task import task
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator

from dislib.data.array import Array
from dislib.math.base import svd
from math import ceil
import dislib
from dislib.data.util import encoder_helper, decoder_helper
import dislib.data.util.model as utilmodel


class PCA(BaseEstimator):
    """ Principal component analysis (PCA).

    Parameters
    ----------
    n_components : int or None, optional (default=None)
        Number of components to keep. If None, all components are kept.
    arity : int, optional (default=50)
        Arity of the reductions. Only if method='eig'.
    method : str, optional (default='eig')
        Method to use in the decomposition. Can be 'svd' for singular value
        decomposition and 'eig' for eigendecomposition of the covariance
        matrix. 'svd' is recommended when having a large number of
        features. Falls back to 'eig' if the method is not recognized.
    eps : float, optional (default=1e-9)
        Tolerance for the convergence criterion when method='svd'.

    Attributes
    ----------
    components_ : ds-array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum
        variance in the data. The components are sorted by explained_variance_.

        Equal to the n_components eigenvectors of the covariance matrix with
        greater eigenvalues.
    explained_variance_ : ds-array, shape (1, n_components)
        The amount of variance explained by each of the selected components.

        Equal to the first n_components largest eigenvalues of the covariance
        matrix.
    mean_ : ds-array, shape (1, n_features)
        Per-feature empirical mean, estimated from the training set.

    Examples
    --------
    >>> import dislib as ds
    >>> from dislib.decomposition import PCA
    >>> import numpy as np
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>>     bn, bm = 2, 2
    >>>     data = ds.array(x=x, block_size=(bn, bm))
    >>>     pca = PCA()
    >>>     transformed_data = pca.fit_transform(data)
    >>>     print(transformed_data)
    >>>     print(pca.components_.collect())
    >>>     print(pca.explained_variance_.collect())
    """

    def __init__(self, n_components=None, arity=50, method="eig", eps=1e-9):
        self.n_components = n_components
        self.arity = arity
        self.method = method
        self.eps = eps

    def fit(self, x, y=None):
        """ Fit the model with the dataset.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : PCA
        """
        if self.method == 'svd' and x._sparse:
            raise NotImplementedError(
                "SVD method not supported for sparse arrays.")

        self.mean_ = x.mean(axis=0)
        norm_x = x - self.mean_

        if self.method == "svd":
            return self._fit_svd(norm_x)
        else:
            return self._fit_eig(norm_x)

    def fit_transform(self, x):
        """ Fit the model with the dataset and apply the dimensionality
        reduction to it.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        transformed_darray : ds-array, shape (n_samples, n_components)
        """
        self.fit(x)

        if self.method == "svd":
            return self._u * self._s
        else:
            return self._transform_eig(x)

    def transform(self, x):
        """
        Apply dimensionality reduction to ds-array.

        The given dataset is projected on the first principal components
        previously extracted from a training ds-array.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            New ds-array, with the same n_features as the training dataset.

        Returns
        -------
        transformed_darray : ds-array, shape (n_samples, n_components)
        """
        return self._transform_eig(x)

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
        The model is synchronized before saving and can be reinstantiated in
        the exact same state, without any of the code used for model
        definition or fitting.
        Parameters
        ----------
        filepath : str
            Path where to save the model
        overwrite : bool, optional (default=True)
            Whether any existing model at the target
            location should be overwritten.
        save_format : str, optional (default='json')
            Format used to save the models.
        Examples
        --------
        >>> from dislib.decomposition import PCA
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = ds.random_array((1000, 100),
        >>> block_size=(100, 50), random_state=0)
        >>> pca = PCA()
        >>> x_transformed = pca.fit_transform(x)
        >>> pca.save_model('/tmp/model')
        >>> load_pca = PCA()
        >>> load_pca.load_model('/tmp/model')
        >>> x_load_transform = load_pca.transform(x)
        >>> assert np.allclose(x_transformed.collect(),
        >>> x_load_transform.collect())
        """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        utilmodel.sync_obj(self.__dict__)
        model_metadata = self.__dict__
        model_metadata["model_name"] = "pca"

        # Save model
        if save_format == "json":
            with open(filepath, "w") as f:
                json.dump(model_metadata, f, default=_encode_helper)
        elif save_format == "cbor":
            if utilmodel.cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "wb") as f:
                utilmodel.cbor2.dump(model_metadata, f,
                                     default=_encode_helper_cbor)
        elif save_format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(model_metadata, f)
        else:
            raise ValueError("Wrong save format.")

    def load_model(self, filepath, load_format="json"):
        """Loads a model from a file.
        The model is reinstantiated in the exact same state in which it was
        saved, without any of the code used for model definition or fitting.
        Parameters
        ----------
        filepath : str
            Path of the saved the model
        load_format : str, optional (default='json')
            Format used to load the model.
        Examples
        --------
        >>> from dislib.decomposition import PCA
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = ds.random_array((1000, 100),
        >>> block_size=(100, 50), random_state=0)
        >>> pca = PCA()
        >>> x_transformed = pca.fit_transform(x)
        >>> pca.save_model('/tmp/model')
        >>> load_pca = PCA()
        >>> load_pca.load_model('/tmp/model')
        >>> x_load_transform = load_pca.transform(x)
        >>> assert np.allclose(x_transformed.collect(),
        >>> x_load_transform.collect())
        """
        # Load model
        if load_format == "json":
            with open(filepath, "r") as f:
                model_metadata = json.load(
                    f, object_hook=_decode_helper)
        elif load_format == "cbor":
            if utilmodel.cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "rb") as f:
                model_metadata = utilmodel.cbor2.\
                    load(f,
                         object_hook=_decode_helper_cbor)
        elif load_format == "pickle":
            with open(filepath, "rb") as f:
                model_metadata = pickle.load(f)
        else:
            raise ValueError("Wrong load format.")

        for key, val in model_metadata.items():
            setattr(self, key, val)

    def _fit_eig(self, x):
        scatter_matrix = _scatter_matrix(x, self.arity)
        cov_matrix = _estimate_covariance(scatter_matrix, x.shape[0])

        if self.n_components:
            shape1 = self.n_components
        else:
            shape1 = x.shape[1]

        n_blocks = int(ceil(shape1 / x._reg_shape[1]))

        val_blocks = Array._get_out_blocks((1, n_blocks))
        vec_blocks = Array._get_out_blocks((n_blocks, x._n_blocks[1]))

        if dislib.__gpu_available__:
            decompose_func = _decompose_gpu
        else:
            decompose_func = _decompose

        decompose_func(cov_matrix, self.n_components, x._reg_shape[1],
                       val_blocks,
                       vec_blocks)

        bshape = (x._reg_shape[1], x._reg_shape[1])

        self.components_ = Array(vec_blocks, bshape, bshape,
                                 (shape1, x.shape[1]), False)

        ex_var_bshape = (1, bshape)
        self.explained_variance_ = Array(val_blocks, ex_var_bshape,
                                         ex_var_bshape, (1, shape1), False)

        return self

    def _fit_svd(self, x):
        self._u, self._s, v = svd(x, copy=False, eps=self.eps)

        if self.n_components:
            self._u = self._u[:, :self.n_components]
            self._s = self._s[:, :self.n_components]
            v = v[:, :self.n_components]

        self.components_ = v.T
        self.explained_variance_ = (self._s ** 2) / (x.shape[0] - 1)

        return self

    def _transform_eig(self, x):
        new_blocks = []
        n_components = self.components_.shape[0]
        reg_shape = x._reg_shape[1]
        div, mod = divmod(n_components, reg_shape)
        n_col_blocks = div + (1 if mod else 0)

        if dislib.__gpu_available__:
            subset_trans_func = _subset_transform_gpu
        else:
            subset_trans_func = _subset_transform

        for rows in x._iterator('rows'):
            out_blocks = [object() for _ in range(n_col_blocks)]
            subset_trans_func(rows._blocks, self.mean_._blocks,
                              self.components_._blocks, reg_shape, out_blocks)
            new_blocks.append(out_blocks)

        return Array(blocks=new_blocks, top_left_shape=x._top_left_shape,
                     reg_shape=x._reg_shape, shape=(x.shape[0], n_components),
                     sparse=x._sparse)


def _scatter_matrix(x, arity):
    partials = []

    if dislib.__gpu_available__:
        scatter_func = _subset_scatter_matrix_gpu
    else:
        scatter_func = _subset_scatter_matrix

    for rows in x._iterator('rows'):
        partials.append(scatter_func(rows._blocks))
    return _reduce_scatter_matrix(partials, arity)


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _subset_scatter_matrix(blocks):
    data = Array._merge_blocks(blocks)

    if issparse(data):
        data = data.toarray()

    return np.dot(data.T, data)


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _subset_scatter_matrix_gpu(blocks):
    import cupy as cp

    data = Array._merge_blocks(blocks)

    if issparse(data):
        data = data.toarray()

    data_gpu = cp.asarray(data)

    return cp.asnumpy(cp.dot(data_gpu.T, data_gpu))


def _reduce_scatter_matrix(partials, arity):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(_merge_partial_scatter_matrix(*partials_chunk))
    return partials[0]


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _merge_partial_scatter_matrix(*partials):
    return sum(partials)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _estimate_covariance(scatter_matrix, n_samples):
    return scatter_matrix / (n_samples - 1)


@constraint(computing_units="${ComputingUnits}")
@task(val_blocks={Type: COLLECTION_OUT, Depth: 2},
      vec_blocks={Type: COLLECTION_OUT, Depth: 2})
def _decompose(covariance_matrix, n_components, bsize, val_blocks, vec_blocks):
    eig_val, eig_vec = np.linalg.eigh(covariance_matrix)

    if n_components is None:
        n_components = len(eig_val)

    # first n_components eigenvalues in descending order:
    eig_val = eig_val[::-1][:n_components]

    # first n_components eigenvectors in rows, with the corresponding order:
    eig_vec = eig_vec.T[::-1][:n_components]

    # normalize eigenvectors sign to ensure deterministic output
    max_abs_cols = np.argmax(np.abs(eig_vec), axis=1)
    signs = np.sign(eig_vec[range(len(eig_vec)), max_abs_cols])
    eig_vec *= signs[:, np.newaxis]

    if len(eig_val.shape) == 1:
        eig_val = np.expand_dims(eig_val, axis=0)

    for i in range(len(vec_blocks)):
        val_blocks[0][i] = eig_val[:, i * bsize:(i + 1) * bsize]

        for j in range(len(vec_blocks[i])):
            vec_blocks[i][j] = \
                eig_vec[i * bsize:(i + 1) * bsize, j * bsize:(j + 1) * bsize]


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(val_blocks={Type: COLLECTION_OUT, Depth: 2},
      vec_blocks={Type: COLLECTION_OUT, Depth: 2})
def _decompose_gpu(covariance_matrix, n_components, bsize,
                   val_blocks, vec_blocks):
    import cupy as cp

    eig_val_gpu, eig_vec_gpu = cp.linalg.eigh(cp.asarray(covariance_matrix))

    if n_components is None:
        n_components = len(eig_val_gpu)

    # first n_components eigenvalues in descending order:
    eig_val_gpu = eig_val_gpu[::-1][:n_components]

    # first n_components eigenvectors in rows, with the corresponding order:
    eig_vec_gpu = eig_vec_gpu.T[::-1][:n_components]

    # normalize eigenvectors sign to ensure deterministic output
    max_abs_cols = cp.argmax(cp.abs(eig_vec_gpu), axis=1)
    s = eig_vec_gpu[list(range(len(eig_vec_gpu))), max_abs_cols]
    signs_gpu = cp.sign(s)
    eig_vec, signs = cp.asnumpy(eig_vec_gpu), cp.asnumpy(signs_gpu)
    eig_val = cp.asnumpy(eig_val_gpu)
    eig_vec *= signs[:, np.newaxis]

    for i in range(len(vec_blocks)):
        val_blocks[0][i] = eig_val[i * bsize:(i + 1) * bsize]

        for j in range(len(vec_blocks[i])):
            vec_blocks[i][j] = \
                eig_vec[i * bsize:(i + 1) * bsize, j * bsize:(j + 1) * bsize]


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      u_blocks={Type: COLLECTION_IN, Depth: 2},
      c_blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_OUT, Depth: 1})
def _subset_transform(blocks, u_blocks, c_blocks, reg_shape, out_blocks):
    data = Array._merge_blocks(blocks)
    mean = Array._merge_blocks(u_blocks)
    components = Array._merge_blocks(c_blocks)

    if issparse(data):
        data = data.toarray()
        mean = mean.toarray()

    res = (np.matmul(data - mean, components.T))

    if issparse(data):
        res = csr_matrix(res)

    for j in range(0, len(blocks[0])):
        out_blocks[j] = res[:, j * reg_shape:(j + 1) * reg_shape]


@constraint(processors=[
                {"processorType": "CPU", "computingUnits": "1"},
                {"processorType": "GPU", "computingUnits": "1"},
            ])
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      u_blocks={Type: COLLECTION_IN, Depth: 2},
      c_blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_OUT, Depth: 1})
def _subset_transform_gpu(blocks, u_blocks, c_blocks, reg_shape, out_blocks):
    import cupy as cp

    data = Array._merge_blocks(blocks)
    mean = Array._merge_blocks(u_blocks)
    components = Array._merge_blocks(c_blocks)

    if issparse(data):
        data = data.toarray()
        mean = mean.toarray()

    data_sub_mean = cp.subtract(cp.asarray(data), cp.asarray(mean))

    matmul_gpu_res = cp.matmul(data_sub_mean, cp.asarray(components).T)
    res = cp.asnumpy(matmul_gpu_res)

    if issparse(data):
        res = csr_matrix(res)

    for j in range(0, len(blocks[0])):
        out_blocks[j] = res[:, j * reg_shape:(j + 1) * reg_shape]


def _encode_helper_cbor(encoder, obj):
    encoder.encode(_encode_helper(obj))


def _encode_helper(obj):
    encoded = encoder_helper(obj)
    if encoded is not None:
        return encoded
    else:
        return {
            "class_name": "PCA",
            "module_name": "decomposition",
            "items": obj.__dict__,
        }


def _decode_helper_cbor(decoder, obj):
    """Special decoder wrapper for dislib using cbor2."""
    return _decode_helper(obj)


def _decode_helper(obj):
    if isinstance(obj, dict) and "class_name" in obj:
        class_name = obj["class_name"]
        decoded = decoder_helper(class_name, obj)
        if decoded is not None:
            return decoded
    return obj
