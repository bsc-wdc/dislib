import numpy as np
from pycompss.api.parameter import COLLECTION_IN, Depth, Type, COLLECTION_OUT
from pycompss.api.task import task
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator

from dislib.data.array import Array
from dislib.math.base import svd
from math import ceil


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
    >>> from dislib.decomposition import PCA
    >>> import numpy as np
    >>> import dislib as ds
    >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> bn, bm = 2, 2
    >>> data = ds.array(x=x, block_size=(bn, bm))
    >>> pca = PCA()
    >>> transformed_data = pca.fit_transform(data)
    >>> print(transformed_data)
    >>> print(pca.components_.collect())
    >>> print(pca.explained_variance_.collect())
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

        _decompose(cov_matrix, self.n_components, x._reg_shape[1],
                   val_blocks,
                   vec_blocks)

        bshape = (x._reg_shape[1], x._reg_shape[1])

        self.components_ = Array(vec_blocks, bshape, bshape,
                                 (shape1, x.shape[1]), False)
        self.explained_variance_ = Array(val_blocks, bshape, bshape,
                                         (1, shape1), False)

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

        for rows in x._iterator('rows'):
            out_blocks = [object() for _ in range(n_col_blocks)]
            _subset_transform(rows._blocks, self.mean_._blocks,
                              self.components_._blocks, reg_shape, out_blocks)
            new_blocks.append(out_blocks)

        return Array(blocks=new_blocks, top_left_shape=x._top_left_shape,
                     reg_shape=x._reg_shape, shape=(x.shape[0], n_components),
                     sparse=x._sparse)


def _scatter_matrix(x, arity):
    partials = []
    for rows in x._iterator('rows'):
        partials.append(_subset_scatter_matrix(rows._blocks))
    return _reduce_scatter_matrix(partials, arity)


@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _subset_scatter_matrix(blocks):
    data = Array._merge_blocks(blocks)

    if issparse(data):
        data = data.toarray()

    return np.dot(data.T, data)


def _reduce_scatter_matrix(partials, arity):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(_merge_partial_scatter_matrix(*partials_chunk))
    return partials[0]


@task(returns=1)
def _merge_partial_scatter_matrix(*partials):
    return sum(partials)


@task(returns=1)
def _estimate_covariance(scatter_matrix, n_samples):
    return scatter_matrix / (n_samples - 1)


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

    for i in range(len(vec_blocks)):
        val_blocks[0][i] = eig_val[i * bsize:(i + 1) * bsize]

        for j in range(len(vec_blocks[i])):
            vec_blocks[i][j] = \
                eig_vec[i * bsize:(i + 1) * bsize, j * bsize:(j + 1) * bsize]


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
