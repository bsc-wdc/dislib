import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification, make_regression

import dislib as ds
from dislib.cluster import KMeans
from dislib.cluster import GaussianMixture
from dislib.classification import CascadeSVM
from dislib.classification import RandomForestClassifier
from dislib.regression import RandomForestRegressor
from dislib.regression import Lasso
from dislib.regression import LinearRegression
from dislib.recommendation import ALS
from dislib.utils import save_model, load_model

from pycompss.api.api import compss_wait_on


class JSONSavingTest(unittest.TestCase):

    def test_saving_kmeans(self):
        file_ = "tests/files/libsvm/2"
        filepath = "tests/files/saving/kmeans.json"

        x_sp, _ = ds.load_svmlight_file(file_, (10, 300), 780, True)
        x_ds, _ = ds.load_svmlight_file(file_, (10, 300), 780, False)

        kmeans = KMeans(random_state=170)
        kmeans.fit(x_sp)

        save_model(kmeans, filepath, save_format="json")
        kmeans2 = load_model(filepath, load_format="json")

        y_sparse = kmeans.predict(x_sp).collect()
        y_sparse2 = kmeans2.predict(x_sp).collect()

        sparse_c = kmeans.centers.toarray()
        sparse_c2 = kmeans2.centers.toarray()

        kmeans = KMeans(random_state=170)

        y_dense = kmeans.fit_predict(x_ds).collect()
        dense_c = kmeans.centers

        self.assertTrue(np.allclose(sparse_c, dense_c))
        self.assertTrue(np.allclose(sparse_c2, dense_c))
        self.assertTrue(np.array_equal(y_sparse, y_dense))
        self.assertTrue(np.array_equal(y_sparse2, y_dense))

    def test_saving_gm(self):
        file_ = "tests/files/libsvm/2"
        filepath = "tests/files/saving/gm.json"

        x_sparse, _ = ds.load_svmlight_file(file_, (10, 780), 780, True)
        x_dense, _ = ds.load_svmlight_file(file_, (10, 780), 780, False)

        covariance_types = "full", "tied", "diag", "spherical"

        for cov_type in covariance_types:
            gm = GaussianMixture(
                n_components=4, random_state=0, covariance_type=cov_type
            )
            gm.fit(x_sparse)
            save_model(gm, filepath, save_format="json")
            gm2 = load_model(filepath, load_format="json")
            labels_sparse = gm.predict(x_sparse).collect()
            labels_sparse2 = gm2.predict(x_sparse).collect()

            gm = GaussianMixture(
                n_components=4, random_state=0, covariance_type=cov_type
            )
            gm.fit(x_dense)
            save_model(gm, filepath, save_format="json")
            gm2 = load_model(filepath, load_format="json")
            labels_dense = gm.predict(x_dense).collect()
            labels_dense2 = gm2.predict(x_dense).collect()

            self.assertTrue(np.array_equal(labels_sparse, labels_sparse2))
            self.assertTrue(np.array_equal(labels_sparse, labels_dense))
            self.assertTrue(np.array_equal(labels_sparse2, labels_dense2))

    def test_saving_csvm(self):
        seed = 666
        train = "tests/files/libsvm/3"
        filepath = "tests/files/saving/csvm.json"

        x_sp, y_sp = ds.load_svmlight_file(train, (10, 300), 780, True)
        x_d, y_d = ds.load_svmlight_file(train, (10, 300), 780, False)

        csvm_sp = CascadeSVM(random_state=seed)
        csvm_sp.fit(x_sp, y_sp)
        save_model(csvm_sp, filepath, save_format="json")
        csvm_sp2 = load_model(filepath, load_format="json")

        csvm_d = CascadeSVM(random_state=seed)
        csvm_d.fit(x_d, y_d)
        save_model(csvm_d, filepath, save_format="json")
        csvm_d2 = load_model(filepath, load_format="json")

        sv_d = csvm_d._clf.support_vectors_
        sv_sp = csvm_sp._clf.support_vectors_.toarray()
        sv_d2 = csvm_d2._clf.support_vectors_
        sv_sp2 = csvm_sp2._clf.support_vectors_.toarray()

        self.assertTrue(np.array_equal(sv_d, sv_sp))
        self.assertTrue(np.array_equal(sv_d2, sv_sp2))
        self.assertTrue(np.array_equal(sv_d, sv_d2))

        coef_d = csvm_d._clf.dual_coef_
        coef_sp = csvm_sp._clf.dual_coef_.toarray()
        coef_d2 = csvm_d2._clf.dual_coef_
        coef_sp2 = csvm_sp2._clf.dual_coef_.toarray()

        self.assertTrue(np.array_equal(coef_d, coef_sp))
        self.assertTrue(np.array_equal(coef_d2, coef_sp2))
        self.assertTrue(np.array_equal(coef_d, coef_d2))

    def test_saving_rf_class(self):
        filepath = "tests/files/saving/rf_class.json"
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(random_state=0, sklearn_max=10)
        rf.fit(x_train, y_train)
        save_model(rf, filepath, save_format="json")
        rf2 = load_model(filepath, load_format="json")

        probabilities = rf.predict_proba(x_test).collect()
        probabilities2 = rf2.predict_proba(x_test).collect()
        rf.classes = compss_wait_on(rf.classes)
        rf2.classes = compss_wait_on(rf2.classes)
        y_pred = rf.classes[np.argmax(probabilities, axis=1)]
        y_pred2 = rf2.classes[np.argmax(probabilities2, axis=1)]
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        accuracy2 = np.count_nonzero(y_pred2 == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy2, 0.7)

    def test_saving_rf_regr(self):
        filepath = "tests/files/saving/rf_regr.json"

        def determination_coefficient(y_true, y_pred):
            u = np.sum(np.square(y_true - y_pred))
            v = np.sum(np.square(y_true - np.mean(y_true)))
            return 1 - u / v

        x, y = make_regression(
            n_samples=3000,
            n_features=10,
            n_informative=4,
            shuffle=True,
            random_state=0,
        )
        x_train = ds.array(x[: len(x) // 2], (300, 10))
        y_train = ds.array(y[: len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = ds.array(y[len(y) // 2:][:, np.newaxis], (300, 1))

        rf = RandomForestRegressor(random_state=0, sklearn_max=10)

        rf.fit(x_train, y_train)
        save_model(rf, filepath, save_format="json")
        rf2 = load_model(filepath, load_format="json")

        accuracy1 = compss_wait_on(rf.score(x_test, y_test))
        accuracy2 = compss_wait_on(rf2.score(x_test, y_test))
        y_pred = rf.predict(x_test).collect()
        y_true = y[len(y) // 2:]
        y_pred2 = rf2.predict(x_test).collect()
        y_true2 = y[len(y) // 2:]
        coef1 = determination_coefficient(y_true, y_pred)
        coef2 = determination_coefficient(y_true2, y_pred2)

        self.assertGreater(accuracy1, 0.85)
        self.assertGreater(accuracy2, 0.85)
        self.assertGreater(coef1, 0.85)
        self.assertGreater(coef2, 0.85)
        self.assertAlmostEqual(accuracy1, accuracy2)
        self.assertAlmostEqual(coef1, coef2)

    def test_saving_lasso(self):
        filepath = "tests/files/saving/lasso.json"
        np.random.seed(42)

        n_samples, n_features = 50, 100
        X = np.random.randn(n_samples, n_features)

        # Decreasing coef w. alternated signs for visualization
        idx = np.arange(n_features)
        coef = (-1) ** idx * np.exp(-idx / 10)
        coef[10:] = 0  # sparsify coef
        y = np.dot(X, coef)

        # Add noise
        y += 0.01 * np.random.normal(size=n_samples)

        n_samples = X.shape[0]
        X_train, y_train = X[: n_samples // 2], y[: n_samples // 2]
        X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

        lasso = Lasso(lmbd=0.1, max_iter=50)

        lasso.fit(ds.array(X_train, (5, 100)), ds.array(y_train, (5, 1)))
        save_model(lasso, filepath, save_format="json")
        lasso2 = load_model(filepath, load_format="json")

        y_pred_lasso = lasso.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso = r2_score(y_test, y_pred_lasso.collect())
        y_pred_lasso2 = lasso2.predict(ds.array(X_test, (25, 100)))
        r2_score_lasso2 = r2_score(y_test, y_pred_lasso2.collect())

        self.assertAlmostEqual(r2_score_lasso, 0.9481746925431124)
        self.assertAlmostEqual(r2_score_lasso2, 0.9481746925431124)

    def test_saving_linear(self):
        filepath = "tests/files/saving/linear_regression.json"

        x_data = np.array([[1, 2], [2, 0], [3, 1], [4, 4], [5, 3]])
        y_data = np.array([2, 1, 1, 2, 4.5])

        bn, bm = 2, 2

        x = ds.array(x=x_data, block_size=(bn, bm))
        y = ds.array(x=y_data, block_size=(bn, 1))

        reg = LinearRegression()
        reg.fit(x, y)
        save_model(reg, filepath, save_format="json")
        reg2 = load_model(filepath, load_format="json")

        self.assertTrue(np.allclose(reg.coef_.collect(), [0.421875, 0.296875]))
        self.assertTrue(
            np.allclose(reg2.coef_.collect(), [0.421875, 0.296875])
        )
        self.assertTrue(np.allclose(reg.intercept_.collect(), 0.240625))
        self.assertTrue(np.allclose(reg2.intercept_.collect(), 0.240625))

        # Predict one sample
        x_test = np.array([3, 2])
        test_data = ds.array(x=x_test, block_size=(1, bm))
        pred = reg.predict(test_data).collect()
        pred2 = reg2.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, 2.1))
        self.assertTrue(np.allclose(pred2, 2.1))

        # Predict multiple samples
        x_test = np.array([[3, 2], [4, 4], [1, 3]])
        test_data = ds.array(x=x_test, block_size=(bn, bm))
        pred = reg.predict(test_data).collect()
        self.assertTrue(np.allclose(pred, [2.1, 3.115625, 1.553125]))

    def test_saving_als(self):
        filepath = "tests/files/saving/als.json"

        data = np.array([[0, 0, 5], [3, 0, 5], [3, 1, 2]])
        ratings = csr_matrix(data)
        train = ds.array(x=ratings, block_size=(1, 1))
        als = ALS(tol=0.01, random_state=666, n_f=5, verbose=False)
        als.fit(train)
        save_model(als, filepath, save_format="json")
        als2 = load_model(filepath, load_format="json")

        predictions = als.predict_user(user_id=0)
        predictions2 = als2.predict_user(user_id=0)

        # Check that the ratings for user 0 are similar to user 1 because they
        # share preferences (third movie), thus it is expected that user 0
        # will rate movie 1 similarly to user 1.
        self.assertTrue(
            2.75 < predictions[0] < 3.25
            and predictions[1] < 1
            and predictions[2] > 4.5
        )
        self.assertTrue(
            2.75 < predictions2[0] < 3.25
            and predictions2[1] < 1
            and predictions2[2] > 4.5
        )
        self.assertTrue(
            np.array_equal(predictions, predictions2, equal_nan=True)
        )


def load_movielens(train_ratio=0.9):
    file = "tests/files/sample_movielens_ratings.csv"

    # 'user_id', 'movie_id', 'rating', 'timestamp'

    data = np.genfromtxt(file, dtype="int", delimiter=",", usecols=range(3))

    # just in case there are movies/user without rating
    # movie_id
    n_m = max(len(np.unique(data[:, 1])), max(data[:, 1]) + 1)
    # user_id
    n_u = max(len(np.unique(data[:, 0])), max(data[:, 0]) + 1)

    idx = int(data.shape[0] * train_ratio)

    train_data = data[:idx]
    test_data = data[idx:]

    train = csr_matrix(
        (train_data[:, 2], (train_data[:, 0], train_data[:, 1])),
        shape=(n_u, n_m),
    )

    test = csr_matrix((test_data[:, 2], (test_data[:, 0], test_data[:, 1])))

    x_size, y_size = train.shape[0] // 4, train.shape[1] // 4
    train_arr = ds.array(train, block_size=(x_size, y_size))

    x_size, y_size = test.shape[0] // 4, test.shape[1] // 4
    test_arr = ds.array(test, block_size=(x_size, y_size))

    return train_arr, test_arr


def main():
    unittest.main()


if __name__ == "__main__":
    main()
