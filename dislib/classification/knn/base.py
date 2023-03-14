import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN, COLLECTION_OUT, Depth, Type
from pycompss.api.task import task
from sklearn.base import BaseEstimator
from dislib.data.array import Array
from dislib.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors as SKNeighbors
from sklearn.neighbors import KDTree

from collections import defaultdict

import os
import json
import dislib.data.util.model as utilmodel
import pickle
from dislib.data.util import sync_obj, decoder_helper, encoder_helper


class KNeighborsClassifier(BaseEstimator):
    """Classifier implementing the k-nearest neighbors vote.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator used when shuffling the
        data for probability estimates. If int, random_state is the seed used
        by the random number generator; If RandomState instance, random_state
        is the random number generator; If None, the random number generator is
        the RandomState instance used by np.random.

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.
    .. warning::
       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances
       but different labels, the results will depend on the ordering of the
       training data.
    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    Examples
    --------
    >>> import dislib as ds
    >>> from dislib.classification import KNeighborsClassifier
    >>> import numpy as np
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>>     y = np.array([1, 1, 2, 2])
    >>>     train_data = ds.array(x, block_size=(4, 2))
    >>>     train_labels = ds.array(y, block_size=(1, 2))
    >>>     knn = KNeighborsClassifier(n_neighbors=3)
    >>>     knn.fit(train_data, train_labels)
    >>>     test_data = ds.array(np.array([[-0.8, -1]]), block_size=(1, 2))
    >>>     y_pred = knn.predict(test_data)
    >>>     print(y_pred)
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform',
                 random_state=None):

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.random_state = random_state
        self.nn = NearestNeighbors(n_neighbors)

    def fit(self, x: Array, y: Array):
        """ Fit the model using training data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Training data.
        y : ds-array, shape=(n_samples, 1)
            Class labels of x.
        Returns
        -------
        self : KNeighborsClassifier
        """
        self.y = y
        self.nn.fit(x)

        return self

    def predict(self, q: Array):
        """ Perform classification on samples.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Input samples.

        Returns
        -------
        y : ds-array, shape(n_samples, 1)
            Class labels of x.
        """
        dist, ind = self.nn.kneighbors(q)

        out_blocks = Array._get_out_blocks(self.y._n_blocks)

        _indices_to_classes(ind._blocks, self.y._blocks, dist._blocks,
                            out_blocks, self.weights)

        return Array(blocks=out_blocks, top_left_shape=self.y._top_left_shape,
                     reg_shape=self.y._reg_shape,
                     shape=self.y.shape, sparse=False)

    def score(self, q: Array, y: Array, collect=False):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Test samples.
        y : ds-array, shape=(n_samples, 1)
            True labels for x.
        collect : bool, optional (default=False)
            When True, a synchronized result is returned.

        Returns
        -------
        score : float (as future object)
            Mean accuracy of self.predict(x) wrt. y.
        """

        y_pred = self.predict(q)
        score = _get_score(y._blocks, y_pred._blocks)

        return compss_wait_on(score) if collect else score

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
        The model is synchronized before saving and can be reinstantiated
        in the exact same state, without any of the code used for model
        definition or fitting.
        Parameters
        ----------
        filepath : str
            Path where to save the model
        overwrite : bool, optional (default=True)
            Whether any existing model at the target
            location should be overwritten.
        save_format : str, optional (default='json)
            Format used to save the models.
        Examples
        --------
        >>> from dislib.classification import KNeighborsClassifier
        >>> import numpy as np
        >>> import dislib as ds
        >>>  data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
        >>> y_data = np.array([2, 1, 1, 2, 0])
        >>> train = ds.array(x=ratings, block_size=(1, 1))
        >>> knn = KNeighborsClassifier()
        >>> knn.fit(train)
        >>> knn.save_model("./model_KNN")
        """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        sync_obj(self.__dict__)
        model_metadata = self.__dict__
        model_metadata["model_name"] = "knn"

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
        >>> from dislib.clasiffication import KNeighborsClassifier
        >>> import numpy as np
        >>> import dislib as ds
        >>> x_data = np.array([[1, 2], [2, 0], [3, 1], [4, 4], [5, 3]])
        >>> y_data = np.array([2, 1, 1, 2, 0])
        >>> x_test_m = np.array([[3, 2], [4, 4], [1, 3]])
        >>> bn, bm = 2, 2
        >>> x = ds.array(x=x_data, block_size=(bn, bm))
        >>> y = ds.array(x=y_data, block_size=(bn, 1))
        >>> test_data_m = ds.array(x=x_test_m, block_size=(bn, bm))
        >>> knn = KNeighborsClassifier()
        >>> knn.fit(x, y)
        >>> knn.save_model("./model_KNN")
        >>> knn_loaded = KNeighborsClassifier()
        >>> knn_loaded.load_model("./model_KNN")
        >>> pred = knn_loaded.predict(test_data).collect()
        """
        # Load model
        if load_format == "json":
            with open(filepath, "r") as f:
                model_metadata = json.load(f, object_hook=_decode_helper)
        elif load_format == "cbor":
            if utilmodel.cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "rb") as f:
                model_metadata = utilmodel.cbor2. \
                    load(f, object_hook=_decode_helper_cbor)
        elif load_format == "pickle":
            with open(filepath, "rb") as f:
                model_metadata = pickle.load(f)
        else:
            raise ValueError("Wrong load format.")
        for key, val in model_metadata.items():
            setattr(self, key, val)


@constraint(computing_units="${ComputingUnits}")
@task(ind_blocks={Type: COLLECTION_IN, Depth: 2},
      y_blocks={Type: COLLECTION_IN, Depth: 2},
      dist_blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_OUT, Depth: 2})
def _indices_to_classes(ind_blocks, y_blocks, dist_blocks,
                        out_blocks, weights):
    ind = Array._merge_blocks(ind_blocks)
    y = Array._merge_blocks(y_blocks).flatten()
    dist = Array._merge_blocks(dist_blocks)

    classes = y[ind]

    final_class = []
    for crow, drow in zip(classes, dist):
        d = defaultdict(int)
        crow = crow.flatten()
        for j in range(ind.shape[1]):

            if weights == 'uniform':
                w = 1
            else:
                w = (drow[j] + np.finfo(drow.dtype).eps)

            d[crow[j]] += 1/w

        final_class.append(max(d, key=d.get))

    blocks = np.array_split(final_class, len(y_blocks))

    for i in range(len(y_blocks)):
        out_blocks[i][0] = np.expand_dims(blocks[i][:], axis=1)


@constraint(computing_units="${ComputingUnits}")
@task(y_blocks={Type: COLLECTION_IN, Depth: 2},
      ypred_blocks={Type: COLLECTION_IN, Depth: 2},
      returns=float)
def _get_score(y_blocks, ypred_blocks):
    y = Array._merge_blocks(y_blocks).flatten()
    y_pred = Array._merge_blocks(ypred_blocks).flatten()

    return accuracy_score(y, y_pred)


def _decode_helper_cbor(decoder, obj):
    """Special decoder wrapper for dislib using cbor2."""
    return _decode_helper(obj)


def _decode_helper(obj):
    if isinstance(obj, dict) and "class_name" in obj:
        class_name = obj["class_name"]
        if class_name == "NearestNeighbors":
            nn = NearestNeighbors(obj["n_neighbors"])
            nn.__setstate__(_decode_helper(obj["items"]))
            return nn
        elif class_name == "SKNeighbors":
            dict_ = _decode_helper(obj["items"])
            model = SKNeighbors()
            model.__setstate__(dict_)
            return model
        elif class_name == "KDTree":
            dict_ = _decode_helper(obj["items"])
            model = KDTree(dict_[0])
            return model
        else:
            decoded = decoder_helper(class_name, obj)
            if decoded is not None:
                return decoded
    return obj


def _encode_helper_cbor(encoder, obj):
    encoder.encode(_encode_helper(obj))


def _encode_helper(obj):
    encoded = encoder_helper(obj)
    if encoded is not None:
        return encoded
    elif isinstance(obj, SKNeighbors):
        return {
            "class_name": "SKNeighbors",
            "n_neighbors": obj.n_neighbors,
            "radius": obj.radius,
            "items": obj.__getstate__(),
        }
    elif isinstance(obj, KDTree):
        return {
            "class_name": "KDTree",
            "items": obj.__getstate__(),
        }
    elif isinstance(obj, NearestNeighbors):
        return {
            "class_name": obj.__class__.__name__,
            "n_neighbors": obj.n_neighbors,
            "items": obj.__getstate__(),
        }
