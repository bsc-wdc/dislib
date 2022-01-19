"""
ADMM Lasso

@Authors: Aleksandar Armacki and Lidija Fodor
@Affiliation: Faculty of Sciences, University of Novi Sad, Serbia

This work is supported by the I-BiDaaS project, funded by the European
Commission under Grant Agreement No. 780787.
"""
import json
import os
import pickle

from dislib.data.util.model import sync_obj, decoder_helper, encoder_helper

try:
    import cvxpy as cp
except ImportError:
    import warnings
    warnings.warn('Cannot import cvxpy module. Lasso estimator will not work.')
from sklearn.base import BaseEstimator
from dislib.optimization import ADMM
try:
    import cbor2
except ImportError:
    cbor2 = None


class Lasso(BaseEstimator):
    """ Lasso represents the Least Absolute Shrinkage and Selection Operator
    (Lasso) for regression analysis, solved in a distributed manner with ADMM.

    Parameters
    ----------
    lmbd : float, optional (default=1e-3)
        The regularization parameter for Lasso regression.
    rho : float, optional (default=1)
        The penalty parameter for constraint violation.
    max_iter : int, optional (default=100)
        The maximum number of iterations of ADMM.
    atol : float, optional (default=1e-4)
        The absolute tolerance used to calculate the early stop criterion
        for ADMM.
    rtol : float, optional (default=1e-2)
        The relative tolerance used to calculate the early stop criterion
        for ADMM.
    verbose : boolean, optional (default=False)
        Whether to print information about the optimization process.

    Attributes
    ----------
    coef_ : ds-array, shape=(1, n_features)
        Parameter vector.
    n_iter_ : int
        Number of iterations run by ADMM.
    converged_ : boolean
        Whether ADMM converged.

    See also
    --------
    ADMM
    """

    def __init__(self, lmbd=1e-3, rho=1, max_iter=100, atol=1e-4, rtol=1e-2,
                 verbose=False):
        self.max_iter = max_iter
        self.lmbd = lmbd
        self.rho = rho
        self.atol = atol
        self.rtol = rtol
        self.verbose = verbose

    @staticmethod
    def _loss_fn(x, y, w):
        return 1 / 2 * cp.norm(cp.matmul(x, w) - y, p=2) ** 2

    def fit(self, x, y):
        """ Fits the model with training data. Optimization is carried out
        using ADMM.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Training samples.
        y : ds-array, shape=(n_samples, 1)
            Class labels of x.

        Returns
        -------
        self :  Lasso
        """
        k = self.lmbd / self.rho

        admm = ADMM(Lasso._loss_fn, k, self.rho, max_iter=self.max_iter,
                    rtol=self.rtol, atol=self.atol, verbose=self.verbose)
        admm.fit(x, y)

        self.n_iter_ = admm.n_iter_
        self.converged_ = admm.converged_
        self.coef_ = admm.z_

        return self

    def predict(self, x):
        """ Predict using the linear model.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Samples.

        Returns
        -------
        y : ds-array, shape=(n_samples, 1)
            Predicted values.
        """
        coef = self.coef_.T

        # this rechunk can be removed as soon as matmul supports multiplying
        # ds-arrays with different block shapes
        if coef._reg_shape[0] != x._reg_shape[1]:
            coef = coef.rechunk(x._reg_shape)

        return x @ coef

    def fit_predict(self, x):
        """ Fits the model and predicts using the same data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Training samples.

        Returns
        -------
        y : ds-array, shape=(n_samples, 1)
            Predicted values.
        """
        return self.fit(x).predict(x)

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
        The model is synchronized before saving and can be reinstantiated in the
        exact same state, without any of the code used for model definition or
        fitting.
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
        >>> from dislib.cluster import KMeans
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> x_train = ds.array(x, (2, 2))
        >>> model = KMeans(n_clusters=2, random_state=0)
        >>> model.fit(x_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = KMeans()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(), loaded_model_pred.collect())
        """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        sync_obj(self.__dict__)
        model_metadata = self.__dict__
        model_metadata["model_name"] = "kmeans"

        # Save model
        if save_format == "json":
            with open(filepath, "w") as f:
                json.dump(model_metadata, f, default=_encode_helper)
        elif save_format == "cbor":
            if cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "wb") as f:
                cbor2.dump(model_metadata, f, default=_encode_helper_cbor)
        elif save_format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(model_metadata, f)
        else:
            raise ValueError("Wrong save format.")

    def load_model(self, filepath, load_format="json"):
        """Loads a model from a file.
        The model is reinstantiated in the exact same state in which it was saved,
        without any of the code used for model definition or fitting.
        Parameters
        ----------
        filepath : str
            Path of the saved the model
        load_format : str, optional (default='json')
            Format used to load the model.
        Examples
        --------
        >>> from dislib.cluster import KMeans
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> x_train = ds.array(x, (2, 2))
        >>> model = KMeans(n_clusters=2, random_state=0)
        >>> model.fit(x_train)
        >>> model.save_model('/tmp/model')
        >>> loaded_model = KMeans()
        >>> loaded_model.load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(), loaded_model_pred.collect())
        """
        # Load model
        if load_format == "json":
            with open(filepath, "r") as f:
                model_metadata = json.load(f, object_hook=_decode_helper)
        elif load_format == "cbor":
            if cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "rb") as f:
                model_metadata = cbor2.load(f, object_hook=_decode_helper_cbor)
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
    return obj