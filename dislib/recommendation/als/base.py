import json
import os
import pickle
from math import sqrt

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.task import task
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from dislib.data.array import Array
from dislib.data.util import sync_obj, decoder_helper, encoder_helper

import dislib.data.util.model as utilmodel


class ALS(BaseEstimator):
    """ Alternating Least Squares recommendation.

    Implements distributed alternating least squares recommendation based on
    Zhou et al. [1]_.

    Parameters
    ----------
    max_iter : int, optional (default=100)
        Maximum number of iterations to perform.
    tol : float, optional (default=1e-4)
        Tolerance for the stopping criterion.
    n_f : int, optional (default=100)
        Number of latent factors (i.e. dimensions) for the matrices U and I.
    lambda_ : float, optional (default=0.065)
        Regularization parameters value.
    check_convergence : boolean, optional (default=True)
        Whether to test for convergence at the end of each iteration.
    random_state : int, orNone, optional (default=None)
        The seed of the pseudo random number generator used to initialize the
        items matrix I.
    arity : int, optional (default=5)
        The arity of the tasks during the merge of each matrix chunk.
    verbose : boolean, optional (default=False)
        Whether to print progress information.

    Attributes
    ----------
    users : np.array
        User matrix.
    items : np.array
        Items matrix.
    converged : boolean
        Whether the model has converged.

    References
    ----------

    .. [1] Zhou Y., Wilkinson D., Schreiber R., Pan R. (2008) Large-Scale
        Parallel Collaborative Filtering for the Netflix Prize. In: Fleischer
        R., Xu J. (eds) Algorithmic Aspects in Information and Management.
        AAIM 2008. Lecture Notes in Computer Science, vol 5034. Springer,
        Berlin, Heidelberg

    Examples
    --------
    >>> import dislib as ds
    >>> from dislib.recommendation import ALS
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
    >>>     ratings = csr_matrix(data).transpose().tocsr()
    >>>     train = ds.array(ratings, block_size=(1, 3))
    >>>     als = ALS()
    >>>     als.fit(train)
    >>>     print('Ratings for user 0: %s' % als.predict_user(user_id=0))
    """

    def __init__(self, random_state=None, n_f=100, lambda_=0.065,
                 tol=1e-4, max_iter=100, arity=5,
                 check_convergence=True, verbose=False):
        # params
        self.random_state = random_state
        self.n_f = n_f
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.arity = arity
        self.check_convergence = check_convergence

    def _update(self, r, x, axis):
        """ Returns updated matrix M given U (if x=U), or matrix U given M
        otherwise

        Parameters
        ----------
        r : Dataset
            copy of R with items as samples (if x=U), users as samples
            otherwise
        x : Dataset
            User or Item feature matrix
        """
        res = []
        for darray in r._iterator(axis=axis):
            params = (self.n_f, self.lambda_, axis)
            chunk_res = _update_chunk(darray._blocks, x, params)
            res.append(chunk_res)

        while len(res) > 1:
            q = []

            while len(res) > 0:
                # we pop the future objects to merge
                to_merge = res[:self.arity]
                del res[:self.arity]
                # if it's a single object, just add it to next step
                aux = _merge(*to_merge) if len(to_merge) > 1 else to_merge[0]
                q.append(aux)
            res = q

        return res.pop()

    def _has_finished(self, i):
        return i >= self.max_iter or self.converged

    def _has_converged(self, last_rmse, rmse):
        return abs(last_rmse - rmse) < self.tol

    def _compute_rmse(self, dataset, u, i):
        rmses = [_get_rmse(sb._blocks, u, i) for sb in
                 dataset._iterator(axis=0)]
        rmses = np.array(compss_wait_on(rmses))
        # remove NaN errors that come from empty chunks
        return np.mean(rmses[~np.isnan(rmses)])

    def fit(self, x, test=None):
        """ Fits a model using training data. Training data is also used to
        check for convergence unless test data is provided.

        Parameters
        ----------
        x : ds-array, shape=(n_ratings, n_users)
            ds-array where each row is the collection of ratings given by a
            user
        test : csr_matrix
            Sparse matrix used to check convergence with users as rows and
            items as columns. If not passed, uses training data to check
            convergence.
        """
        self.converged = False
        self.users = None
        self.items = None

        n_u = x.shape[0]
        n_i = x.shape[1]

        if self.verbose:
            print("Item blocks: %s" % n_i)
            print("User blocks: %s" % n_u)

        if self.random_state:
            np.random.seed(self.random_state)

        self.converged = False
        users = None
        items = np.random.rand(n_i, self.n_f)

        # Assign average rating as first feature
        # average_ratings = dataset.mean(axis='columns').collect()
        average_ratings = _mean(x)

        items[:, 0] = average_ratings

        rmse, last_rmse = np.inf, np.NaN
        i = 0
        while not self._has_finished(i):
            last_rmse = rmse

            users = self._update(r=x, x=items, axis=0)
            items = self._update(r=x, x=users, axis=1)

            if self.check_convergence:

                _test = x if test is None else test
                rmse = compss_wait_on(self._compute_rmse(_test, users, items))
                self.converged = self._has_converged(last_rmse, rmse)
                if self.verbose:
                    test_set = "Train" if test is None else "Test"
                    print("%s RMSE: %.3f  [%s]" % (test_set, rmse,
                                                   abs(last_rmse - rmse)))
            i += 1

        self.users = compss_wait_on(users)
        self.items = compss_wait_on(items)

        return users, items

    def predict_user(self, user_id):
        """ Returns the expected ratings for user_id. Each index represents
        the rating for i-th item. If the user was not present in the training
        set, a np.NaN vector is returned.

        Parameters
        ----------
        user_id : int

        Returns
        -------
        ratings : np.array containing all estimated items ratings for user_id.
        """
        if self.users is None or self.items is None:
            raise Exception("Model not trained, call first model.fit()")
        if user_id > self.users.shape[1]:
            return np.full([self.items.shape[1]], np.nan)

        return self.users[user_id].dot(self.items.T)

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
        >>> from dislib.recommendation import ALS
        >>> import numpy as np
        >>> import dislib as ds
        >>>  data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
        >>> ratings = csr_matrix(data)
        >>> train = ds.array(x=ratings, block_size=(1, 1))
        >>> als = ALS(tol=0.01, random_state=666, n_f=5, verbose=False)
        >>> als.fit(train)
        >>> als.save_model("model_als")
        """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        sync_obj(self.__dict__)
        model_metadata = self.__dict__
        model_metadata["model_name"] = "als"

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
        The model is reinstantiated in the exact same state in which it
        was saved, without any of the code used for model definition or
        fitting.
        Parameters
        ----------
        filepath : str
            Path of the saved the model
        load_format : str, optional (default='json')
            Format used to load the model.
        Examples
        --------
        >>> from dislib.recommendation import ALS
        >>> import numpy as np
        >>> import dislib as ds
        >>> als2 = ALS()
        >>> als2.load_model("model_als")
        >>> predictions2 = als2.predict_user(user_id=0)
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
    return obj


def _mean(dataset):
    averages = []
    for col in dataset._iterator('columns'):
        averages.append(_col_mean(col._blocks))

    averages = compss_wait_on(averages)

    return np.bmat(averages)


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _col_mean(blocks):
    cols = Array._merge_blocks(blocks)
    averages = cols.sum(axis=0) / (cols != 0).toarray().sum(axis=0)

    return averages


@constraint(computing_units="${ComputingUnits}")
@task(returns=np.array)
def _merge(*chunks):
    res = np.vstack(chunks)
    return res


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=np.array)
def _update_chunk(blocks, x, params):
    n_f, lambda_, axis = params
    r_chunk = Array._merge_blocks(blocks)
    if axis == 1:
        r_chunk = r_chunk.transpose()

    n = r_chunk.shape[0]
    y = np.zeros((n, n_f), dtype=np.float32)
    n_c = np.array(
        [len(sparse.find(r_chunk[i])[0]) for i in
         range(0, r_chunk.shape[0])])
    for element in range(0, n):
        indices = sparse.find(r_chunk[element])[1]

        x_xt = x[indices].T.dot(x[indices])

        a_i = x_xt + lambda_ * n_c[element] * np.eye(n_f)
        v_i = x[indices].T.dot(r_chunk[element, indices].toarray().T)

        # TODO: decide if atol should be changed when default is changed
        y[element] = sparse.linalg.cg(a_i, v_i, atol='legacy')[0].reshape(-1)

    return y


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _get_rmse(blocks, users, items):
    test = Array._merge_blocks(blocks)
    x_idxs, y_idxs, recs = sparse.find(test)
    indices = zip(x_idxs, y_idxs)

    rmse = np.NaN
    if len(recs) > 0:
        preds = [users[x].dot(items[y].T) for x, y in indices]
        rmse = sqrt(mean_squared_error(recs, preds))

    return rmse
