from math import sqrt

import numpy as np

from numpy.linalg import inv
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from scipy import sparse
from sklearn.metrics import mean_squared_error


from dislib.data import load_data


class ALS(object):
    def __init__(self, seed=666, n_f=100, lambda_=0.065,
                 convergence_threshold=0.0001 ** 2,
                 max_iter=np.inf):
        # params
        self.seed = seed
        self.n_f = n_f
        self.lambda_ = lambda_
        self.conv = convergence_threshold
        self.max_iter = max_iter
        self.U = None
        self.M = None

    def _update(self, r, X):
        """ Returns updated matrix M given U (if X=U), or matrix U given M
        otherwise

        Parameters
        ----------
        r : Dataset
            copy of R with movies as rows (if X=U), users as rows otherwise
        X : Dataset
            User or Movie feature matrix
        n_c : np.array
            Number of ratings of a given movie (if X=U), user ratings otherwise
        """
        results = []
        for subset in r:

            chunk_res = self._update_chunk(subset, X, n_f=self.n_f, lambda_=self.lambda_)
            results.append(chunk_res)

        results = compss_wait_on(results)

        return np.vstack(results)

    @task(returns=np.array, is_modifier=False)
    def _update_chunk(self, subset, X, n_f, lambda_):

        r_chunk = subset.samples
        n = r_chunk.shape[0]
        Y = np.zeros((n, n_f), dtype=np.float32)
        n_c = np.array(
            [len(sparse.find(r_chunk[i])[0]) for i in range(0, r_chunk.shape[0])])
        for element in range(0, n):
            indices = sparse.find(r_chunk[element])[1]

            X_Xt = X[indices].T.dot(X[indices])

            A_i = X_Xt + lambda_ * n_c[element] * np.eye(n_f)
            V_i = X[indices].T.dot(r_chunk[element, indices].toarray().T)

            Y[element] = inv(A_i).dot(V_i).reshape(-1)
        return Y

    def _update_u(self, r_u, M, n_ui):
        """ Update matrix U given M

        Parameters
        ----------
        r_u : Dataset
            copy of R with users as rows
        M : Dataset
            movie feature matrix
        """

        U = np.zeros((n_u, self.n_f), dtype=np.float32)

        for u in range(0, n_u):
            movies = sparse.find(r_u[u])[1]

            M_u = M[movies]
            M_Mt = M_u.T.dot(M_u)

            A_i = M_Mt + self.lambda_ * n_ui[u] * np.eye(self.n_f)
            V_i = M_u.T.dot(r_u[u, movies].toarray().T)

            U[u] = inv(A_i).dot(V_i).reshape(-1)

        return U

    def _update_m(self, r_m, U, n_mj):
        """ Update matrix M given U

        Parameters
        ----------
        r_m : Dataset
            copy of R with movies as rows
        U : Dataset
            user feature matrix
        """

        M = np.zeros((n_m, self.n_f), dtype=np.float32)

        for m in range(0, n_m):
            users = sparse.find(r_m[m])[1]

            U_m = U[users]
            U_Ut = U_m.T.dot(U_m)

            A_i = U_Ut + self.lambda_ * n_mj[m] * np.eye(self.n_f)
            V_i = U_m.T.dot(r_m[m, users].toarray().T)

            M[m] = inv(A_i).dot(V_i).reshape(-1)

        return M

    def _has_converged(self, last_rmse, rmse, i):
        if i > self.max_iter:
            print("Max iterations reached [%s]" % self.max_iter)
            return True
        if i > 0 and abs(last_rmse - rmse) < self.conv:
            print("Converged in %s iterations to difference < %s" % (
                i, abs(last_rmse - rmse)))
            return True
        return False

    def fit(self, r, test=None):

        r_u = r
        r_m = r.transpose(copy=True).tocsr()

        d_u = r_u
        d_m = r_m
        d_u = load_data(r_u, r_u.shape[0] // 4)
        d_m = load_data(r_m, r_m.shape[0] // 4)

        n_u = r.shape[0]
        n_m = r.shape[1]

        np.random.seed(self.seed)
        U = None
        M = np.random.rand(n_m, self.n_f)
        # Assign average rating as first feature
        average_ratings = [np.mean(r[:, i].data) for i in range(0, r.shape[1])]
        M[:, 0] = average_ratings

        rmse, last_rmse = np.inf, np.NaN
        i = 0
        while not self._has_converged(last_rmse, rmse, i):
            last_rmse = rmse

            # U = self._update_u(r_u, M, n_ui)
            U = self._update(r=d_u, X=M)
            # M = self._update_m(r_m, U, n_mj)
            M = self._update(r=d_m, X=U)

            if test is not None:
                x_idxs, y_idxs, recs = sparse.find(test)
                indices = zip(x_idxs, y_idxs)
                preds = [U[x].dot(M[y].T) for x, y in indices]
                rmse = sqrt(mean_squared_error(recs, preds))
                print("Test RMSE: %.3f  [%s]" % (rmse, abs(last_rmse - rmse)))

            i += 1

        self.U, self.M = U, M

        return U, M

    def predict_user(self, user_id):
        if user_id > self.U.shape[1]:
            return np.full([self.M.shape[1]], np.nan)

        return self.U[user_id].dot(self.M.T)


