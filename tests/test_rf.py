import unittest

import numpy as np
from pycompss.api.api import compss_wait_on
from sklearn import datasets
from sklearn.datasets import make_classification

import dislib as ds
from dislib.classification import RandomForestClassifier


class RFTest(unittest.TestCase):
    def test_make_classification_score(self):
        """Tests RandomForestClassifier fit and score with default params."""
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
        x_train = ds.array(x[:len(x) // 2], (300, 10))
        y_train = ds.array(y[:len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = ds.array(y[len(y) // 2:][:, np.newaxis], (300, 1))

        rf = RandomForestClassifier(random_state=0)

        rf.fit(x_train, y_train)
        accuracy = compss_wait_on(rf.score(x_test, y_test))
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_predict_and_distr_depth(self):
        """Tests RandomForestClassifier fit and predict with a distr_depth."""
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
        x_train = ds.array(x[:len(x) // 2], (300, 10))
        y_train = ds.array(y[:len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(distr_depth=2, random_state=0)

        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test).collect()
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_fit_predict(self):
        """Tests RandomForestClassifier fit_predict with default params."""
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
        x_train = ds.array(x[:len(x) // 2], (300, 10))
        y_train = ds.array(y[:len(y) // 2][:, np.newaxis], (300, 1))

        rf = RandomForestClassifier(random_state=0)

        y_pred = rf.fit(x_train, y_train).predict(x_train).collect()
        y_train = y_train.collect()
        accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_sklearn_max_predict(self):
        """Tests RandomForestClassifier predict with sklearn_max."""
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
        x_train = ds.array(x[:len(x) // 2], (300, 10))
        y_train = ds.array(y[:len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(random_state=0, sklearn_max=10)

        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test).collect()
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_sklearn_max_predict_proba(self):
        """Tests RandomForestClassifier predict_proba with sklearn_max."""
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
        x_train = ds.array(x[:len(x) // 2], (300, 10))
        y_train = ds.array(y[:len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(random_state=0, sklearn_max=10)

        rf.fit(x_train, y_train)
        probabilities = rf.predict_proba(x_test).collect()
        rf.classes = compss_wait_on(rf.classes)
        y_pred = rf.classes[np.argmax(probabilities, axis=1)]
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_hard_vote_predict(self):
        """Tests RandomForestClassifier predict with hard_vote."""
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
        x_train = ds.array(x[:len(x) // 2], (300, 10))
        y_train = ds.array(y[:len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = y[len(y) // 2:]

        rf = RandomForestClassifier(random_state=0, sklearn_max=10,
                                    hard_vote=True)

        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test).collect()
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_hard_vote_score_mix(self):
        """Tests RandomForestClassifier score with hard_vote, sklearn_max,
        distr_depth and max_depth."""
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
        x_train = ds.array(x[:len(x) // 2], (300, 10))
        y_train = ds.array(y[:len(y) // 2][:, np.newaxis], (300, 1))
        x_test = ds.array(x[len(x) // 2:], (300, 10))
        y_test = ds.array(y[len(y) // 2:][:, np.newaxis], (300, 1))

        rf = RandomForestClassifier(random_state=0, sklearn_max=100,
                                    distr_depth=2, max_depth=12,
                                    hard_vote=True)

        rf.fit(x_train, y_train)
        accuracy = compss_wait_on(rf.score(x_test, y_test))
        self.assertGreater(accuracy, 0.7)

    def test_iris(self):
        """Tests RandomForestClassifier with a minimal example."""
        x, y = datasets.load_iris(return_X_y=True)
        ds_fit = ds.array(x[::2], block_size=(30, 2))
        fit_y = ds.array(y[::2].reshape(-1, 1), block_size=(30, 1))
        ds_validate = ds.array(x[1::2], block_size=(30, 2))
        validate_y = ds.array(y[1::2].reshape(-1, 1), block_size=(30, 1))

        rf = RandomForestClassifier(n_estimators=1, max_depth=1,
                                    random_state=0)
        rf.fit(ds_fit, fit_y)
        accuracy = compss_wait_on(rf.score(ds_validate, validate_y))

        # Accuracy should be <= 2/3 for any seed, often exactly equal.
        self.assertAlmostEqual(accuracy, 2 / 3)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
