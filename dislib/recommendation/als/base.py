import sys
from collections import deque
from math import sqrt

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy import sparse
from sklearn.metrics import mean_squared_error


class ALS(object):
    """ Alternating Least Squares recommendation.

    Implements distributed alternating least squares recommendation based on
    Zhou et al. [1]_.

    Parameters
    ----------
    max_iter : int, optional (default=5)
        Maximum number of iterations to perform.
    tol : float, optional (default=1e-3)
        Tolerance for the stopping criterion.
    n_f : int, optional (default=1.0)
        Number of latent factors (i.e. dimensions) for the matrices U and I.
    lambda_ : float, optional (default=0.065)
        Regularization parameters value.
    check_convergence : boolean, optional (default=True)
        Whether to test for convergence. If False, the algorithm will run
        for cascade_iterations. Checking for convergence adds a
        synchronization point after each iteration.

        If ``check_convergence=False'' synchronization does not happen until
        a call to ``predict'', ``decision_function'' or ``score''. This can
        be useful to fit multiple models in parallel.
    seed : int, optional (default=None)
        The seed of the pseudo random number generator used to initialize the
        items matrix I.
    merge_arity : int, optional (default=5)
        The arity of the tasks during the merge of each matrix chunk.
    verbose : boolean, optional (default=False)
        Whether to print progress information.

    Attributes
    ----------
    U : np.array
        User matrix.
    I : np.array
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

    """

    def __init__(self, seed=None, n_f=100, lambda_=0.065,
                 tol=0.0001 ** 2, max_iter=np.inf, merge_arity=5,
                 check_convergence=True, verbose=False):
        # params
        self._seed = seed
        self._n_f = n_f
        self._lambda = lambda_
        self._tol = tol
        self._max_iter = max_iter
        self._verbose = verbose
        self._merge_arity = merge_arity
        self._check_convergence = check_convergence
        self.converged = False
        self.U = None
        self.I = None

    def _update(self, r, x):
        """ Returns updated matrix M given U (if x=U), or matrix U given M
        otherwise

        Parameters
        ----------
        r : Dataset
            copy of R with items as rows (if x=U), users as rows otherwise
        x : Dataset
            User or Movie feature matrix
        """
        # res = []
        res = deque()
        for subset in r:
            chunk_res = _update_chunk(subset, x, self._n_f, self._lambda)
            res.append(chunk_res)

        while len(res) > 1:
            q = deque()
            while len(res) > 0:
                # we pop the future objects to merge
                to_merge = [res.popleft() for _ in range(self._merge_arity)
                            if len(res) > 0]
                # if it's a single object, just add it to next step
                aux = _merge(*to_merge) if len(to_merge) > 1 else to_merge[0]
                q.append(aux)
            res = q

        return res.pop()

    def _has_finished(self, i):
        if i >= self._max_iter or self.converged:
            return True
        return False

    def _has_converged(self, last_rmse, rmse):
        if abs(last_rmse - rmse) < self._tol:
            return True
        return False

    def _compute_rmse(self, d_u, U, I):
        rmses = [_get_rmse(sb, U, I) for sb in d_u._subsets]
        rmses = compss_wait_on(rmses)
        return np.mean(rmses)

    def fit(self, dataset, test=None):
        """ Fits a model using training data. Training data is also used to
        check for convergence unless test data is provided.

        Parameters
        ----------
        dataset : Dataset
            Ratings matrix with items as rows and users as columns.
        test : DataFrame
            Dataframe used to check convergence (used training data otherwise).
        """

        d_i = dataset
        d_u = d_i.transpose()

        n_m = d_u.n_features

        if self._verbose:
            print("Item chunks: %s" % len(d_i))
            print("User chunks: %s" % len(d_u))

        if self._seed:
            np.random.seed(self._seed)
        U = None
        I = np.random.rand(n_m, self._n_f)

        # Assign average rating as first feature
        average_ratings = d_i._apply(lambda row: np.mean(row.data),
                                     sparse=False, return_dataset=True)
        average_ratings = compss_wait_on(average_ratings)

        I[:, 0] = average_ratings.samples.reshape(-1)

        rmse, last_rmse = np.inf, np.NaN
        i = 0
        while not self._has_finished(i):
            last_rmse = rmse

            U = self._update(r=d_u, x=I)
            I = self._update(r=d_i, x=U)

            if self._check_convergence:
                if test is not None:
                    x_idxs, y_idxs, recs = sparse.find(test)
                    indices = zip(x_idxs, y_idxs)
                    preds = [U[x].dot(I[y].T) for x, y in indices]
                    rmse = sqrt(mean_squared_error(recs, preds))
                    if self._verbose:
                        print("Test RMSE: %.3f  [%s]" % (
                            rmse, abs(last_rmse - rmse)))

                else:
                    rmse = self._compute_rmse(d_u, U, I)
                    self.converged = self._has_converged(last_rmse, rmse)
                    if self._verbose:
                        print("Train RMSE: %.3f  [%s]" % (
                            rmse, abs(last_rmse - rmse)))
            i += 1

        self.U = compss_wait_on(U)
        self.I = compss_wait_on(I)

        return U, I

    def predict_user(self, user_id):
        if self.U is None or self.I is None:
            raise Exception("Model not trained, call first model.fit()")
        if user_id > self.U.shape[1]:
            return np.full([self.I.shape[1]], np.nan)

        return self.U[user_id].dot(self.I.T)


@task(returns=np.array)
def _merge(*chunks):
    pro_f = sys.getprofile()
    sys.setprofile(None)

    res = np.vstack(chunks)

    sys.setprofile(pro_f)
    return res


@task(returns=np.array)
def _update_chunk(subset, x, n_f, lambda_):
    pro_f = sys.getprofile()
    sys.setprofile(None)

    r_chunk = subset.samples
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

        y[element] = sparse.linalg.cg(a_i, v_i)[0].reshape(-1)

    sys.setprofile(pro_f)

    return y


@task(returns=float)
def _get_rmse(test, U, I):
    pro_f = sys.getprofile()
    sys.setprofile(None)

    x_idxs, y_idxs, recs = sparse.find(test.samples)
    indices = zip(x_idxs, y_idxs)

    preds = [U[x].dot(I[y].T) for x, y in indices]
    rmse = sqrt(mean_squared_error(recs, preds))

    sys.setprofile(pro_f)
    return rmse
