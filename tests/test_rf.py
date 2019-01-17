import unittest

from sklearn.datasets import make_classification

from dislib.classification import RandomForestClassifier
from dislib.data import load_data
import numpy as np


class RFTest(unittest.TestCase):
    def test_make_classification_score(self):
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0)
        x_train = x[:len(x) // 2]
        y_train = y[:len(y) // 2]
        x_test = x[len(x) // 2:]
        y_test = y[len(y) // 2:]

        train_ds = load_data(x=x_train, y=y_train, subset_size=300)
        test_ds = load_data(x=x_test, y=y_test, subset_size=300)

        rf = RandomForestClassifier(random_state=0)

        rf.fit(train_ds)
        accuracy = rf.score(test_ds)
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_predict_and_distr_depth(self):
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0)
        x_train = x[:len(x) // 2]
        y_train = y[:len(y) // 2]
        x_test = x[len(x) // 2:]
        y_test = y[len(y) // 2:]

        train_ds = load_data(x=x_train, y=y_train, subset_size=300)
        test_ds = load_data(x=x_test, subset_size=300)

        rf = RandomForestClassifier(distr_depth=2, random_state=0)

        rf.fit(train_ds)
        rf.predict(test_ds)
        accuracy = np.count_nonzero(test_ds.labels == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_fit_predict(self):
        x, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_classes=3,
            n_informative=4,
            n_redundant=2,
            n_repeated=1,
            n_clusters_per_class=2,
            shuffle=True,
            random_state=0)
        x_train = x[:len(x) // 2]
        y_train = y[:len(y) // 2]

        train_ds = load_data(x=x_train, y=y_train, subset_size=300)

        rf = RandomForestClassifier(random_state=0)

        rf.fit_predict(train_ds)
        accuracy = np.count_nonzero(train_ds.labels == y_train) / len(y_train)
        self.assertGreater(accuracy, 0.7)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
