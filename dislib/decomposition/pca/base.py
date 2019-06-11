from pycompss.api.task import task
import numpy as np

from dislib.data import Dataset, Subset


class PCA:
    """ Principal component analysis (PCA) using the covariance method.

    Performs a full eigendecomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int or None, optional (default=None)
        Number of components to keep. If None, all components are kept.
    arity : int, optional (default=50)
        Arity of the reductions.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum
        variance in the data. The components are sorted by explained_variance_.

        Equal to the n_components eigenvectors of the covariance matrix with
        greater eigenvalues.
    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to the first n_components largest eigenvalues of the covariance
        matrix.
    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

    Examples
    --------
    >>> from dislib.decomposition import PCA
    >>> import numpy as np
    >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> from dislib.data import load_data
    >>> data = load_data(x=x, subset_size=2)
    >>> pca = PCA()
    >>> transformed_data = pca.fit_transform(data)
    >>> print(transformed_data)
    >>> print(pca.components_)
    >>> print(pca.explained_variance_)
    """
    def __init__(self, n_components=None, arity=50):
        self.n_components = n_components
        self.arity = arity

    def fit(self, dataset):
        """ Fit the model with the dataset.

        Parameters
        ----------
        dataset : Dataset, shape (n_samples, n_features)
            Training dataset.

        Returns
        -------
        self: PCA
            Returns the instance itself.
        """
        n_samples = sum(dataset.subsets_sizes())
        self.mean_ = _features_mean(dataset, self.arity, n_samples)
        normalized_dataset = Dataset(n_features=None)
        for subset in dataset:
            normalized_dataset.append(_normalize(subset, self.mean_))
        scatter_matrix = _scatter_matrix(normalized_dataset, self.arity)
        covariance_matrix = _estimate_covariance(scatter_matrix, n_samples)
        eig_val, eig_vec = _decompose(covariance_matrix, self.n_components)
        self.components_ = eig_vec
        self.explained_variance_ = eig_val

        return self

    def fit_transform(self, dataset):
        """ Fit the model with the dataset and apply the dimensionality
        reduction to it.

        Parameters
        ----------
        dataset : Dataset, shape (n_samples, n_features)
            Training dataset.

        Returns
        -------
        transformed_dataset : Dataset, shape (n_samples, n_components)
        """
        return self.fit(dataset).transform(dataset)

    def transform(self, dataset):
        """
        Apply dimensionality reduction to dataset.

        The given dataset is projected on the first principal components
        previously extracted from a training dataset.

        Parameters
        ----------
        dataset : Dataset, shape (n_samples, n_features)
            New dataset, with the same n_features as the training dataset.

        Returns
        -------
        transformed_dataset : Dataset, shape (n_samples, n_components)
        """
        return _transform(dataset, self.mean_, self.components_)


def _features_mean(dataset, arity, n_samples):
    partials = []
    for subset in dataset:
        partials.append(_subset_feature_sum(subset))
    return _reduce_features_mean(partials, arity, n_samples)


@task(returns=1)
def _subset_feature_sum(subset):
    return subset.samples.sum(axis=0)


def _reduce_features_mean(partials, arity, n_samples):
    while len(partials) > 1:
        partials_chunk = partials[:arity]
        partials = partials[arity:]
        partials.append(_merge_features_sum(*partials_chunk))
    return _finalize_features_mean(partials[0], n_samples)


@task(returns=1)
def _merge_features_sum(*partials):
    return sum(partials)


@task(returns=1)
def _finalize_features_mean(feature_sums, n_samples):
    return feature_sums / n_samples


@task(returns=1)
def _normalize(subset, means):
    return Subset(subset.samples - means)


def _scatter_matrix(dataset, arity):
    partials = []
    for subset in dataset:
        partials.append(_subset_scatter_matrix(subset))
    return _reduce_scatter_matrix(partials, arity)


@task(returns=1)
def _subset_scatter_matrix(subset):
    return np.dot(subset.samples.T, subset.samples)


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
    print(n_samples)
    return scatter_matrix / (n_samples - 1)


@task(returns=2)
def _decompose(covariance_matrix, n_components):
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

    return eig_val, eig_vec


def _transform(dataset, mean, components):
    transformed = Dataset(n_features=None)
    for subset in dataset:
        transformed.append(_subset_transform(subset, mean, components))
    return transformed


@task(returns=1)
def _subset_transform(subset, mean, components):
    return Subset(np.matmul(subset.samples - mean, components.T))
