import argparse
from math import sqrt

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import sparse
from sklearn.metrics import mean_squared_error


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

    def _update_m(self, r, U, n_mj):
        """ Update matrix M given U

        Parameters
        ----------
        r : Dataset
            copy of R distributed by columns
        U : Dataset
            user feature matrix
        """

        M = np.zeros((n_m, self.n_f), dtype=np.float32)

        for m in range(0, n_m):
            users = sparse.find(r[:, m])[0]

            U_m = U[users]
            U_Ut = U_m.T.dot(U_m)

            A_i = U_Ut + self.lambda_ * n_mj[m] * np.eye(self.n_f)
            V_i = U_m.T.dot(r[users, m].toarray())

            M[m] = inv(A_i).dot(V_i).reshape(-1)

        return M

    def _update_u(self, r, M, n_ui):
        """ Update matrix U given M

        Parameters
        ----------
        r : Dataset
            copy of R distributed by columns
        U : Dataset
            movie feature matrix
        """

        U = np.zeros((n_u, self.n_f), dtype=np.float32)

        for u in range(0, n_u):
            movies = sparse.find(r[u])[1]

            M_u = M[movies]
            M_Mt = M_u.T.dot(M_u)

            A_i = M_Mt + self.lambda_ * n_ui[u] * np.eye(self.n_f)
            V_i = M_u.T.dot(r[u, movies].toarray().T)

            U[u] = inv(A_i).dot(V_i).reshape(-1)

        return U

    def _has_converged(self, last_rmse, rmse, i):
        if i > self.max_iter or (i > 0 and abs(last_rmse - rmse) < self.conv):
            return True
        return False

    def fit(self, r, test=None):

        n_u = r.shape[0]
        n_m = r.shape[1]

        n_ui = np.array(
            [len(sparse.find(r[i])[0]) for i in range(0, r.shape[0])])
        n_mj = np.array(
            [len(sparse.find(r[:, i])[0]) for i in range(0, r.shape[1])])

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

            U = self._update_u(r, M, n_ui=n_ui)
            M = self._update_m(r, U, n_mj=n_mj)

            if test is not None:
                x_idxs, y_idxs, recs = sparse.find(test)
                indices = zip(x_idxs, y_idxs)
                preds = [U[i].dot(M[j].T) for i, j in indices]
                rmse = sqrt(mean_squared_error(recs, preds))
                print("Test RMSE: %.3f  [%s]" % (rmse, abs(last_rmse - rmse)))

            i += 1

        self.U, self.M = U, M

        return U, M

    def predict(self, user_id):
        if user_id > self.U.shape[1]:
            return np.full([self.M.shape[1]], np.nan)

        return self.U[user_id].dot(self.M.T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # data
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv('./sample_movielens_ratings.txt',
                     delimiter='::',
                     names=cols,
                     usecols=cols[0:3]).sample(frac=1)

    # just in case there are movies/user without rating
    n_m = max(df.movie_id.nunique(), max(df.movie_id) + 1)
    n_u = max(df.user_id.nunique(), max(df.user_id) + 1)

    idx = int(df.shape[0] * 0.8)

    train_df = df.iloc[:idx]
    test_df = df.iloc[idx:]

    train = sparse.csr_matrix(
        (train_df.rating, (train_df.user_id, train_df.movie_id)),
        shape=(n_u, n_m))
    test = sparse.csr_matrix(
        (test_df.rating, (test_df.user_id, test_df.movie_id)))

    als = ALS()

    als.fit(train, test)
