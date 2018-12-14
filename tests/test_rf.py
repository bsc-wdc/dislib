import unittest

from sklearn.datasets import make_classification

from dislib.classification import RandomForestClassifier
from dislib.data import load_data


class RFTest(unittest.TestCase):
    def test_make_classification(self):
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

        rf = RandomForestClassifier()

        rf.fit(train_ds)
        accuracy = rf.score(x_test, y_test)
        self.assertGreater(accuracy, 0.7)

    def test_make_classification_distr_depth(self):
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

        rf = RandomForestClassifier(distr_depth=2)

        rf.fit(train_ds)
        accuracy = rf.score(x_test, y_test)
        self.assertGreater(accuracy, 0.7)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
