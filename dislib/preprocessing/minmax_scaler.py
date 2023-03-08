import json
import os
import pickle

import numpy as np
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Depth, Type, COLLECTION_IN, COLLECTION_OUT
from pycompss.api.task import task
from scipy.sparse import csr_matrix, issparse

from dislib.data.array import Array
import dislib as ds

import dislib.data.util.model as utilmodel
from dislib.data.util import encoder_helper, decoder_helper, sync_obj


class MinMaxScaler(object):
    """ Standardize features by rescaling them to the provided range

    Scaling happen independently on each feature by computing the relevant
    statistics on the samples in the training set. Minimum and Maximum
    values are then stored to be used on later data using the transform method.

    Attributes
    ----------
    feature_range : tuple
        The desired range of values in the ds-array.
    """

    def __init__(self, feature_range=(0, 1)):
        self._feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, x):
        """ Compute the min and max values for later scaling.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        self : MinMaxScaler
        """

        self.data_min_ = ds.apply_along_axis(np.min, 0, x)
        self.data_max_ = ds.apply_along_axis(np.max, 0, x)

        return self

    def fit_transform(self, x):
        """ Fit to data, then transform it.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        x_new : ds-array, shape=(n_samples, n_features)
            Scaled data.
        """
        return self.fit(x).transform(x)

    def transform(self, x):
        """
        Scale data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        x_new : ds-array, shape=(n_samples, n_features)
            Scaled data.
        """
        if self.data_min_ is None or self.data_max_ is None:
            raise Exception("Model has not been initialized.")

        n_blocks = x._n_blocks[1]
        blocks = []
        min_blocks = self.data_min_._blocks
        max_blocks = self.data_max_._blocks

        for row in x._iterator(axis=0):
            out_blocks = [object() for _ in range(n_blocks)]
            _transform(row._blocks, min_blocks, max_blocks, out_blocks,
                       self._feature_range[0], self._feature_range[1])
            blocks.append(out_blocks)

        return Array(blocks, top_left_shape=x._top_left_shape,
                     reg_shape=x._reg_shape, shape=x.shape,
                     sparse=x._sparse)

    def inverse_transform(self, x):
        """
        Returns data to its original values. The Scaler should be fitted
        before using this function.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)

        Returns
        -------
        x_new : ds-array, shape=(n_samples, n_features)
            Original valued data.
        """
        if self.data_min_ is None or self.data_max_ is None:
            raise Exception("Model has not been initialized.")

        n_blocks = x._n_blocks[1]
        blocks = []
        min_blocks = self.data_min_._blocks
        max_blocks = self.data_max_._blocks

        for row in x._iterator(axis=0):
            out_blocks = [object() for _ in range(n_blocks)]
            _inverse_transform(row._blocks, min_blocks, max_blocks, out_blocks,
                               self._feature_range[0], self._feature_range[1])
            blocks.append(out_blocks)

        return Array(blocks, top_left_shape=x._top_left_shape,
                     reg_shape=x._reg_shape, shape=x.shape,
                     sparse=x._sparse)

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
        save_format : str, optional (default='json)
            Format used to save the models.
        Examples
        --------
        >>> from dislib.preprocessing import MinMaxScaler
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = ds.array(np.array([[1, 2], [2, 1], [-1, -2],
        >>> [-2, -1]]), (2, 2))
        >>> y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))
        >>> model = MinMaxScaler()
        >>> model.fit(x)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = MinMaxScaler()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[1, 2], [2, 1], [-1, -2], [-2, -1],
        >>> [1, 1], [-1, -1]]), (2, 2))
        >>> x_transformed = model.transform(x_test)
        >>> x_loaded_pred = loaded_model.transform(x_test)
        >>> assert np.allclose(x_transformed.collect(),
        >>> x_loaded_pred.collect())
        """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        sync_obj(self.__dict__)
        model_metadata = self.__dict__
        model_metadata["model_name"] = "minmaxscaler"

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
        >>> from dislib.classification import CascadeSVM
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = ds.array(np.array([[1, 2], [2, 1], [-1, -2],
        >>> [-2, -1]]), (2, 2))
        >>> y = ds.array(np.array([0, 1, 1, 0]).reshape(-1, 1), (2, 1))
        >>> model = MinMaxScaler()
        >>> model.fit(x)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = MinMaxScaler()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[1, 2], [2, 1], [-1, -2], [-2, -1],
        >>> [1, 1], [-1, -1]]), (2, 2))
        >>> x_transformed = model.transform(x_test)
        >>> x_loaded_pred = loaded_model.transform(x_test)
        >>> assert np.allclose(x_transformed.collect(),
        >>>                    x_loaded_pred.collect())
        """
        # Load model
        if load_format == "json":
            with open(filepath, "r") as f:
                model_metadata = json.load(f, object_hook=_decode_helper)
        elif load_format == "cbor":
            if utilmodel.cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "rb") as f:
                model_metadata = utilmodel.cbor2.\
                    load(f, object_hook=_decode_helper_cbor)
        elif load_format == "pickle":
            with open(filepath, "rb") as f:
                model_metadata = pickle.load(f)
        else:
            raise ValueError("Wrong load format.")

        for key, val in model_metadata.items():
            setattr(self, key, val)


def _encode_helper_cbor(encoder, obj):
    encoder.encode(_encode_helper(obj))


def _encode_helper(obj):
    encoded = encoder_helper(obj)
    if encoded is not None:
        return encoded


def _decode_helper_cbor(decoder, obj):
    """Special decoder wrapper for dislib using cbor2."""
    return _decode_helper(obj)


def _decode_helper(obj):
    if isinstance(obj, dict) and "class_name" in obj:
        class_name = obj["class_name"]
        decoded = decoder_helper(class_name, obj)
        if decoded is not None:
            return decoded
        elif class_name == "RandomState":
            random_state = np.random.RandomState()
            random_state.set_state(_decode_helper(obj["items"]))
            return random_state
    return obj


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      min_blocks={Type: COLLECTION_IN, Depth: 2},
      max_blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks=COLLECTION_OUT)
def _transform(blocks, min_blocks, max_blocks, out_blocks,
               range_min, range_max):
    x = Array._merge_blocks(blocks)
    min_val = Array._merge_blocks(min_blocks)
    max_val = Array._merge_blocks(max_blocks)
    sparse = issparse(x)

    if sparse:
        x = x.toarray()
        min_val = min_val.toarray()
        max_val = max_val.toarray()

    std_x = (x - min_val) / (max_val - min_val)
    std_x = np.nan_to_num(std_x)
    scaled_x = std_x * (range_max - range_min) + range_min

    constructor_func = np.array if not sparse else csr_matrix
    start, end = 0, 0

    for i, block in enumerate(blocks[0]):
        end += block.shape[1]
        out_blocks[i] = constructor_func(scaled_x[:, start:end])
        start += block.shape[1]


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      min_blocks={Type: COLLECTION_IN, Depth: 2},
      max_blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks=COLLECTION_OUT)
def _inverse_transform(blocks, min_blocks, max_blocks, out_blocks,
                       range_min, range_max):
    x = Array._merge_blocks(blocks)
    min_val = Array._merge_blocks(min_blocks)
    max_val = Array._merge_blocks(max_blocks)
    sparse = issparse(x)

    if sparse:
        x = x.toarray()
        min_val = min_val.toarray()
        max_val = max_val.toarray()

    x = (x - range_min) / (range_max - range_min)
    x = np.nan_to_num(x, nan=1.0)
    x = x * (max_val - min_val) + min_val

    constructor_func = np.array if not sparse else csr_matrix
    start, end = 0, 0

    for i, block in enumerate(blocks[0]):
        end += block.shape[1]
        out_blocks[i] = constructor_func(x[:, start:end])
        start += block.shape[1]
