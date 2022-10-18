import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier as skKNeighborsClassifier
from sklearn.datasets import make_classification

import dislib as ds
from dislib.classification import KNeighborsClassifier
import dislib.data.util.model as utilmodel


class KNearestNeighborsTest(unittest.TestCase):

    def test_kneighbors(self):
        """ Tests kneighbors against scikit-learn """

        X, Y = make_classification(n_samples=200, n_features=5)
        x, y = ds.array(X, (50, 5)), ds.array(Y, (50, 1))

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x, y)
        ds_y_hat = knn.predict(x)
        knn.score(x, y)

        sknn = skKNeighborsClassifier(n_neighbors=3)
        sknn.fit(X, Y)
        sk_y_hat = sknn.predict(X)

        self.assertTrue(np.all(ds_y_hat.collect() == sk_y_hat))

    def test_kneighbors_sparse(self):
        """ Tests kneighbors against scikit-learn with sparse data """
        X, Y = make_classification(n_samples=200, n_features=5)
        X, Y = csr_matrix(X), Y
        x, y = ds.array(X, (50, 5)), ds.array(Y, (50, 1))

        knn = KNeighborsClassifier(n_neighbors=3, weights='')
        knn.fit(x, y)
        ds_y_hat = knn.predict(x)

        sknn = skKNeighborsClassifier(n_neighbors=3, weights='distance')
        sknn.fit(X, Y)
        sk_y_hat = sknn.predict(X)

        self.assertTrue(np.all(ds_y_hat.collect() == sk_y_hat))

    def test_save_load(self):
        """
        Tests that the save and load methods of the KNN work properly with
        the implemented formats and that an exception is retuned when the
        requested format is not supported.
        """
        X, Y = make_classification(n_samples=200, n_features=5)
        x, y = ds.array(X, (50, 5)), ds.array(Y, (50, 1))

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x, y)
        knn.save_model("./model_KNN")
        knn2 = KNeighborsClassifier(n_neighbors=3)
        knn2.load_model("./model_KNN")
        self.assertTrue(knn2.score(x, y, collect=True) > 0.8)
        # negative points belong to class 1, positives to 0
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 = \
            [1, 2], [2, 1], [-1, -2], [-2, -1], [1, 2], [2, 1], \
            [-1, -2], [-2, -1], [1, 2], [2, 1], [-1, -2], [-2, -1]

        x = ds.array(np.array([p1, p4, p3, p2, p5, p8, p7, p6, p9,
                               p12, p11, p10]), (4, 2))
        y = ds.array(np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]).
                     reshape(-1, 1), (4, 1))

        knn = KNeighborsClassifier(n_neighbors=3, weights='')

        knn.fit(x, y)
        knn.save_model("./saved_knn")
        knn2 = KNeighborsClassifier()
        knn2.load_model("./saved_knn")
        p13, p14 = np.array([1, 1]), np.array([-1, -1])

        x_test = ds.array(np.array([p1, p2, p3, p4, p13, p14]), (2, 2))

        y_pred = knn2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        knn.save_model("./saved_knn", save_format="cbor")
        knn2 = KNeighborsClassifier()
        knn2.load_model("./saved_knn", load_format="cbor")

        y_pred = knn2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        knn.save_model("./saved_knn", save_format="pickle")
        knn2 = KNeighborsClassifier()
        knn2.load_model("./saved_knn", load_format="pickle")

        y_pred = knn2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        with self.assertRaises(ValueError):
            knn.save_model("./saved_knn", save_format="txt")

        with self.assertRaises(ValueError):
            knn2 = KNeighborsClassifier()
            knn2.load_model("./saved_knn", load_format="txt")

        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 = \
            [1, 2], [2, 1], [-1, -2], [-2, -1], [1, 2], [2, 1], \
            [-1, -2], [-2, -1], [1, 2], [2, 1], [-1, -2], [-2, -1]

        x = ds.array(np.array([p1, p4, p3, p2, p5, p8, p7, p6,
                               p9, p12, p11, p10]), (2, 2))
        y = ds.array(np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]).
                     reshape(-1, 1), (2, 1))

        knn = KNeighborsClassifier(n_neighbors=3, weights='')

        knn.fit(x, y)
        knn.save_model("./saved_knn", overwrite=False)

        knn2 = KNeighborsClassifier()
        knn2.load_model("./saved_knn", load_format="pickle")

        y_pred = knn2.predict(x_test)

        l1, l2, l3, l4, l5, l6 = y_pred.collect()
        self.assertTrue(l1 == l2 == l5 == 0)
        self.assertTrue(l3 == l4 == l6 == 1)

        cbor2_module = utilmodel.cbor2
        utilmodel.cbor2 = None
        with self.assertRaises(ModuleNotFoundError):
            knn.save_model("./saved_knn", save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            knn2.load_model("./saved_knn", load_format="cbor")
        utilmodel.cbor2 = cbor2_module


def main():
    unittest.main()


if __name__ == '__main__':
    main()
