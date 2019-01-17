from math import sqrt

import numpy as np
from numpy.linalg import inv
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy import sparse
from sklearn.metrics import mean_squared_error

from dislib.data import load_data


class ALS(object):
    def __init__(self, seed=None, n_f=100, lambda_=0.065,
                 convergence_threshold=0.0001 ** 2,
                 max_iter=np.inf, verbose=False):
        # params
        self.seed = seed
        self.n_f = n_f
        self.lambda_ = lambda_
        self.conv = convergence_threshold
        self.max_iter = max_iter
        self.verbose = verbose
        self.u = None
        self.m = None

    def _update(self, r, X):
        """ Returns updated matrix M given U (if X=U), or matrix U given M
        otherwise

        Parameters
        ----------
        r : Dataset
            copy of R with movies as rows (if X=U), users as rows otherwise
        X : Dataset
            User or Movie feature matrix
        """
        results = []
        for subset in r:
            chunk_res = self._update_chunk(subset, X, n_f=self.n_f,
                                           lambda_=self.lambda_)
            results.append(chunk_res)

        results = compss_wait_on(results)

        return np.vstack(results)

    @task(returns=np.array, isModifier=False)
    def _update_chunk(self, subset, X, n_f, lambda_):

        r_chunk = subset.samples
        n = r_chunk.shape[0]
        Y = np.zeros((n, n_f), dtype=np.float32)
        n_c = np.array(
            [len(sparse.find(r_chunk[i])[0]) for i in
             range(0, r_chunk.shape[0])])
        for element in range(0, n):
            indices = sparse.find(r_chunk[element])[1]

            X_Xt = X[indices].T.dot(X[indices])

            A_i = X_Xt + lambda_ * n_c[element] * np.eye(n_f)
            V_i = X[indices].T.dot(r_chunk[element, indices].toarray().T)

            Y[element] = inv(A_i).dot(V_i).reshape(-1)
        return Y

    def _has_converged(self, last_rmse, rmse, i):
        if i > self.max_iter:
            if self.verbose:
                print("Max iterations reached [%s]" % self.max_iter)
            return True
        if i > 0 and abs(last_rmse - rmse) < self.conv:
            if self.verbose:
                print("Converged in %s iterations to difference < %s" % (
                    i, abs(last_rmse - rmse)))
            return True
        return False

    def fit(self, r, test=None):

        r_u = r
        r_m = r.transpose(copy=True).tocsr()

        d_u = load_data(r_u, r_u.shape[0] // 4)
        d_m = load_data(r_m, r_m.shape[0] // 4)

        n_u = r.shape[0]
        n_m = r.shape[1]

        if self.seed:
            np.random.seed(self.seed)
        u = None
        m = np.random.rand(n_m, self.n_f)

        # Assign average rating as first feature
        average_ratings = [np.mean(r[:, i].data) for i in range(0, r.shape[1])]
        m[:, 0] = average_ratings

        rmse, last_rmse = np.inf, np.NaN
        i = 0
        while not self._has_converged(last_rmse, rmse, i):
            last_rmse = rmse

            u = self._update(r=d_u, X=m)
            m = self._update(r=d_m, X=u)

            if test is not None:
                x_idxs, y_idxs, recs = sparse.find(test)
                indices = zip(x_idxs, y_idxs)
                preds = [u[x].dot(m[y].T) for x, y in indices]
                rmse = sqrt(mean_squared_error(recs, preds))
                if self.verbose:
                    print("RMSE: %.3f  [%s]" % (rmse, abs(last_rmse - rmse)))

            i += 1

        self.u, self.m = u, m

        return u, m

    def predict_user(self, user_id):
        if user_id > self.u.shape[1]:
            return np.full([self.m.shape[1]], np.nan)

        return self.u[user_id].dot(self.m.T)
