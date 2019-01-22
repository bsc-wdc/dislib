import unittest

from pycompss.api.api import compss_wait_on
from sklearn.datasets import make_classification

from dislib.classification import RandomForestClassifier
from dislib.data import load_data
import numpy as np


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
        x_train = x[:len(x) // 2]
        y_train = y[:len(y) // 2]

        train_ds = load_data(x=x_train, y=y_train, subset_size=300)

        rf = RandomForestClassifier(random_state=0)

        rf.fit_predict(train_ds)
        accuracy = np.count_nonzero(train_ds.labels == y_train) / len(y_train)
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
        x_train = x[:len(x) // 2]
        y_train = y[:len(y) // 2]
        x_test = x[len(x) // 2:]
        y_test = y[len(y) // 2:]

        train_ds = load_data(x=x_train, y=y_train, subset_size=300)
        test_ds = load_data(x=x_test, subset_size=300)

        rf = RandomForestClassifier(random_state=0, sklearn_max=10)

        rf.fit(train_ds)
        rf.predict(test_ds)
        accuracy = np.count_nonzero(test_ds.labels == y_test) / len(y_test)
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
        x_train = x[:len(x) // 2]
        y_train = y[:len(y) // 2]
        x_test = x[len(x) // 2:]
        y_test = y[len(y) // 2:]

        train_ds = load_data(x=x_train, y=y_train, subset_size=300)
        test_ds = load_data(x=x_test, subset_size=300)

        rf = RandomForestClassifier(random_state=0, sklearn_max=10)

        rf.fit(train_ds)
        rf.predict_proba(test_ds)
        rf.classes = compss_wait_on(rf.classes)
        y_pred = rf.classes[np.argmax(test_ds.labels, axis=1)]
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
        x_train = x[:len(x) // 2]
        y_train = y[:len(y) // 2]
        x_test = x[len(x) // 2:]
        y_test = y[len(y) // 2:]

        train_ds = load_data(x=x_train, y=y_train, subset_size=300)
        test_ds = load_data(x=x_test, subset_size=300)

        rf = RandomForestClassifier(random_state=0, sklearn_max=10,
                                    hard_vote=True)

        rf.fit(train_ds)
        rf.predict(test_ds)
        accuracy = np.count_nonzero(test_ds.labels == y_test) / len(y_test)
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
        x_train = x[:len(x) // 2]
        y_train = y[:len(y) // 2]
        x_test = x[len(x) // 2:]
        y_test = y[len(y) // 2:]

        train_ds = load_data(x=x_train, y=y_train, subset_size=300)
        test_ds = load_data(x=x_test, y=y_test, subset_size=300)

        rf = RandomForestClassifier(random_state=0, sklearn_max=100,
                                    distr_depth=2, max_depth=12,
                                    hard_vote=True)

        rf.fit(train_ds)
        accuracy = rf.score(test_ds)
        self.assertGreater(accuracy, 0.7)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
