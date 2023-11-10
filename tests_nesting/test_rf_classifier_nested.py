import numpy as np
from parameterized import parameterized
from pycompss.api.api import compss_wait_on
from sklearn import datasets
from sklearn.datasets import make_classification

import dislib as ds
from dislib.classification import RandomForestClassifier
import dislib.data.util.model as utilmodel
from dislib.trees.nested.forest import _resolve_try_features
from tests import BaseTimedTestCase
from pycompss.api.task import task


def test_make_classification_score():
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
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

    rf = RandomForestClassifier(n_classes=3,
                                random_state=0, mmap=False,
                                nested=True)

    rf.fit(x_train, y_train)
    accuracy = rf.score(x_test, y_test, collect=True)
    return accuracy > 0.7


def test_make_classification_predict_and_distr_depth():
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
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = y[1::2]

    rf = RandomForestClassifier(n_estimators=4, n_classes=3,
                                distr_depth=2,
                                n_split_points="auto",
                                random_state=0, mmap=False,
                                nested=True)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).collect()
    accuracy1 = np.count_nonzero(y_pred == y_test) / len(y_test)

    rf = RandomForestClassifier(n_estimators=4, n_classes=3,
                                distr_depth=2,
                                n_split_points="sqrt",
                                random_state=0, mmap=False,
                                nested=True)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).collect()
    accuracy2 = np.count_nonzero(y_pred == y_test) / len(y_test)

    rf = RandomForestClassifier(n_estimators=4, n_classes=3,
                                distr_depth=2,
                                n_split_points=0.2,
                                random_state=0, mmap=False,
                                nested=True)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).collect()
    accuracy3 = np.count_nonzero(y_pred == y_test) / len(y_test)

    rf = RandomForestClassifier(n_estimators=4, n_classes=3,
                                distr_depth=2,
                                n_split_points="sqrt",
                                random_state=0, mmap=False,
                                nested=True)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).collect()
    accuracy2 = np.count_nonzero(y_pred == y_test) / len(y_test)

    rf = RandomForestClassifier(n_estimators=4, n_classes=3,
                                distr_depth=2,
                                n_split_points=0.2,
                                random_state=0, mmap=False,
                                nested=True, hard_vote=True)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).collect()
    accuracy4 = np.count_nonzero(y_pred == y_test) / len(y_test)

    return accuracy1 > 0.7 and accuracy2 > 0.7 and accuracy3 > 0.7 \
        and accuracy4 > 0.7


def test_make_classification_fit_predict():
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
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))

    rf = RandomForestClassifier(n_classes=3, distr_depth=1,
                                random_state=0, mmap=False,
                                nested=True)

    y_pred = rf.fit(x_train, y_train).predict(x_train).collect()
    y_train = y_train.collect()
    accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
    return accuracy > 0.7


def test_make_classification_sklearn_max_predict():
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
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = y[1::2]

    rf = RandomForestClassifier(n_classes=3, distr_depth=1,
                                random_state=0, sklearn_max=10,
                                mmap=False,
                                nested=True)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).collect()
    accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
    return accuracy > 0.7


def test_make_classification_sklearn_max_predict_proba():
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
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = y[1::2]

    rf = RandomForestClassifier(n_classes=3, distr_depth=1, random_state=0,
                                sklearn_max=10, mmap=False,
                                nested=True)

    rf.fit(x_train, y_train)
    probabilities = rf.predict_proba(x_test).collect()
    rf.classes = np.arange(rf.n_classes)
    y_pred = rf.classes[np.argmax(probabilities, axis=1)]
    accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
    return accuracy > 0.7


def test_make_classification_hard_vote_predict():
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
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = y[1::2]

    rf = RandomForestClassifier(
        n_classes=3, distr_depth=1, random_state=0,
        sklearn_max=10, hard_vote=True, mmap=False,
        nested=True
    )

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).collect()
    accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
    return accuracy > 0.7


def test_make_classification_hard_vote_score_mix():
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
        random_state=0,
    )
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
    x_test = ds.array(x[1::2], (1000, 10))
    y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

    rf = RandomForestClassifier(
        n_classes=3,
        random_state=0,
        sklearn_max=100,
        distr_depth=1,
        max_depth=12,
        hard_vote=True,
        mmap=False,
        nested=True,
    )

    rf.fit(x_train, y_train)
    accuracy = compss_wait_on(rf.score(x_test, y_test))
    return accuracy > 0.7


def test_score_on_iris():
    """Tests RandomForestClassifier with a minimal example."""
    x, y = datasets.load_iris(return_X_y=True)
    ds_fit = ds.array(x[::2], block_size=(30, 2))
    fit_y = ds.array(y[::2].reshape(-1, 1), block_size=(30, 1))
    ds_validate = ds.array(x[1::2], block_size=(30, 2))
    validate_y = ds.array(y[1::2].reshape(-1, 1), block_size=(30, 1))

    rf = RandomForestClassifier(
        n_classes=3, distr_depth=1,
        n_estimators=1, max_depth=2, random_state=0,
        mmap=False, nested=True
    )
    rf.fit(ds_fit, fit_y)
    accuracy1 = rf.score(ds_validate, validate_y, True)
    accuracy2 = rf.score(ds_validate, validate_y, False)
    accuracy2 = compss_wait_on(accuracy2)

    # Accuracy should be <= 2/3 for any seed, often exactly equal.
    return accuracy1 > (2 / 3) and accuracy2 > (2 / 3)


def test_save_load():
    """
    Tests that the save and load methods work properly with the three
    expected formats and that an exception is raised when a non-supported
    format is provided.
    """
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
    x_train = ds.array(x[::2], (1000, 10))
    y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))

    rf = RandomForestClassifier(n_classes=3, distr_depth=1, random_state=0,
                                n_estimators=5, mmap=False, nested=True)
    rf.fit(x_train, y_train)
    rf.save_model("./saved_model")

    rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                 random_state=0,
                                 n_estimators=5, mmap=False, nested=True)
    rf2.load_model("./saved_model")
    y_pred = rf2.predict(x_train).collect()
    y_train = y_train.collect()
    accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
    condition = accuracy > 0.7

    rf.save_model("./saved_model", save_format="cbor")

    rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                 random_state=0,
                                 n_estimators=5, mmap=False, nested=True)
    rf2.load_model("./saved_model", load_format="cbor")

    y_pred = rf2.predict(x_train).collect()
    accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
    condition = condition and accuracy > 0.7

    rf.save_model("./saved_model", save_format="pickle")

    rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                 random_state=0,
                                 n_estimators=5, mmap=False, nested=True)
    rf2.load_model("./saved_model", load_format="pickle")
    y_pred = rf2.predict(x_train).collect()
    accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
    condition = condition and accuracy > 0.7
    condition_error = False
    try:
        rf.save_model("./saved_model", save_format="txt")
    except ValueError:
        condition_error = True
    condition = condition and condition_error

    condition_error = False
    try:
        rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                     random_state=0,
                                     n_estimators=5, mmap=False,
                                     nested=True)
        rf2.load_model("./saved_model", load_format="txt")
    except ValueError:
        condition_error = True
    condition = condition and condition_error

    rf = RandomForestClassifier(n_classes=3, distr_depth=1, random_state=0,
                                n_estimators=1, mmap=False, nested=True)
    x_train2 = ds.array(x[::2], (1000, 10))
    y_train2 = ds.array(y[::2][:, np.newaxis], (1000, 1))
    rf.fit(x_train2, y_train2)
    rf.save_model("./saved_model", overwrite=False)

    rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                 random_state=0,
                                 n_estimators=5, mmap=False, nested=True)
    rf2.load_model("./saved_model", load_format="pickle")
    y_pred = rf2.predict(x_train).collect()
    accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
    condition = condition and accuracy > 0.7

    cbor2_module = utilmodel.cbor2
    utilmodel.cbor2 = None
    condition_error = False
    try:
        rf.save_model("./saved_model_error", save_format="cbor")
    except ModuleNotFoundError:
        condition_error = True
    condition = condition and condition_error
    condition_error = False
    try:
        rf2.load_model("./saved_model_error", load_format="cbor")
    except ModuleNotFoundError:
        condition_error = True
    condition = condition and condition_error
    utilmodel.cbor2 = cbor2_module
    return condition


class RFTest(BaseTimedTestCase):
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
            random_state=0,
        )
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

        rf = RandomForestClassifier(n_classes=3,
                                    random_state=0, mmap=False,
                                    nested=True)

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
            random_state=0,
        )
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = y[1::2]

        rf = RandomForestClassifier(n_estimators=2, n_classes=3,
                                    distr_depth=2,
                                    n_split_points="auto",
                                    random_state=0, mmap=False,
                                    nested=True)

        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test).collect()
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)

        rf = RandomForestClassifier(n_estimators=2, n_classes=3,
                                    distr_depth=2,
                                    n_split_points="sqrt",
                                    random_state=0, mmap=False,
                                    nested=True)

        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test).collect()
        accuracy = np.count_nonzero(y_pred == y_test) / len(y_test)
        self.assertGreater(accuracy, 0.7)

        rf = RandomForestClassifier(n_estimators=2, n_classes=3,
                                    distr_depth=2,
                                    n_split_points=0.2,
                                    random_state=0, mmap=False,
                                    nested=True)

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
            random_state=0,
        )
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))

        rf = RandomForestClassifier(n_classes=3, distr_depth=1,
                                    random_state=0, mmap=False,
                                    nested=True)

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
            random_state=0,
        )
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = y[1::2]

        rf = RandomForestClassifier(n_classes=3, distr_depth=1,
                                    random_state=0, sklearn_max=10,
                                    mmap=False,
                                    nested=True)

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
            random_state=0,
        )
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = y[1::2]

        rf = RandomForestClassifier(n_classes=3, distr_depth=1, random_state=0,
                                    sklearn_max=10, mmap=False,
                                    nested=True)

        rf.fit(x_train, y_train)
        probabilities = rf.predict_proba(x_test).collect()
        rf.classes = np.arange(rf.n_classes)
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
            random_state=0,
        )
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = y[1::2]

        rf = RandomForestClassifier(
            n_classes=3, distr_depth=1, random_state=0,
            sklearn_max=10, hard_vote=True, mmap=False,
            nested=True
        )

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
            random_state=0,
        )
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))
        x_test = ds.array(x[1::2], (1000, 10))
        y_test = ds.array(y[1::2][:, np.newaxis], (1000, 1))

        rf = RandomForestClassifier(
            n_classes=3,
            random_state=0,
            sklearn_max=100,
            distr_depth=1,
            max_depth=12,
            hard_vote=True,
            mmap=False,
            nested=True,
        )

        rf.fit(x_train, y_train)
        accuracy = compss_wait_on(rf.score(x_test, y_test))
        self.assertGreater(accuracy, 0.7)

    @parameterized.expand([(True,), (False,)])
    def test_score_on_iris(self, collect):
        """Tests RandomForestClassifier with a minimal example."""
        x, y = datasets.load_iris(return_X_y=True)
        ds_fit = ds.array(x[::2], block_size=(30, 2))
        fit_y = ds.array(y[::2].reshape(-1, 1), block_size=(30, 1))
        ds_validate = ds.array(x[1::2], block_size=(30, 2))
        validate_y = ds.array(y[1::2].reshape(-1, 1), block_size=(30, 1))

        rf = RandomForestClassifier(
            n_classes=3, distr_depth=1,
            n_estimators=1, max_depth=2, random_state=0,
            mmap=False, nested=True
        )
        rf.fit(ds_fit, fit_y)
        accuracy = rf.score(ds_validate, validate_y, collect)
        if not collect:
            accuracy = compss_wait_on(accuracy)

        # Accuracy should be <= 2/3 for any seed, often exactly equal.
        self.assertGreater(accuracy, 2 / 3)

    def test_save_load(self):
        """
        Tests that the save and load methods work properly with the three
        expected formats and that an exception is raised when a non-supported
        format is provided.
        """
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
        x_train = ds.array(x[::2], (1000, 10))
        y_train = ds.array(y[::2][:, np.newaxis], (1000, 1))

        rf = RandomForestClassifier(n_classes=3, distr_depth=1, random_state=0,
                                    n_estimators=5, mmap=False, nested=True)
        rf.fit(x_train, y_train)
        rf.save_model("./saved_model")

        rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                     random_state=0,
                                     n_estimators=5, mmap=False, nested=True)
        rf2.load_model("./saved_model")
        y_pred = rf2.predict(x_train).collect()
        y_train = y_train.collect()
        accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
        self.assertGreater(accuracy, 0.7)

        rf.save_model("./saved_model", save_format="cbor")

        rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                     random_state=0,
                                     n_estimators=5, mmap=False, nested=True)
        rf2.load_model("./saved_model", load_format="cbor")

        y_pred = rf2.predict(x_train).collect()
        accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
        self.assertGreater(accuracy, 0.7)

        rf.save_model("./saved_model", save_format="pickle")

        rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                     random_state=0,
                                     n_estimators=5, mmap=False, nested=True)
        rf2.load_model("./saved_model", load_format="pickle")
        y_pred = rf2.predict(x_train).collect()
        accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
        self.assertGreater(accuracy, 0.7)

        with self.assertRaises(ValueError):
            rf.save_model("./saved_model", save_format="txt")

        with self.assertRaises(ValueError):
            rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                         random_state=0,
                                         n_estimators=5, mmap=False,
                                         nested=True)
            rf2.load_model("./saved_model", load_format="txt")

        rf = RandomForestClassifier(n_classes=3, distr_depth=1, random_state=0,
                                    n_estimators=1, mmap=False, nested=True)
        x_train2 = ds.array(x[::2], (1000, 10))
        y_train2 = ds.array(y[::2][:, np.newaxis], (1000, 1))
        rf.fit(x_train2, y_train2)
        rf.save_model("./saved_model", overwrite=False)

        rf2 = RandomForestClassifier(n_classes=3, distr_depth=1,
                                     random_state=0,
                                     n_estimators=5, mmap=False, nested=True)
        rf2.load_model("./saved_model", load_format="pickle")
        y_pred = rf2.predict(x_train).collect()
        accuracy = np.count_nonzero(y_pred == y_train) / len(y_train)
        self.assertGreater(accuracy, 0.7)

        cbor2_module = utilmodel.cbor2
        utilmodel.cbor2 = None
        with self.assertRaises(ModuleNotFoundError):
            rf.save_model("./saved_model_error", save_format="cbor")
        with self.assertRaises(ModuleNotFoundError):
            rf2.load_model("./saved_model_error", load_format="cbor")
        utilmodel.cbor2 = cbor2_module

    def test_other_functions(self):
        number_features = _resolve_try_features("sqrt", 9)
        self.assertTrue(number_features == 3)
        number_features = _resolve_try_features("third", 12)
        self.assertTrue(number_features == 4)
        number_features = _resolve_try_features(None, 12)
        self.assertTrue(number_features == 12)
        number_features = _resolve_try_features(2, 12)
        self.assertTrue(number_features == 2)
        number_features = _resolve_try_features(0.5, 12)
        self.assertTrue(number_features == 6)


@task()
def main():
    test = test_make_classification_score()
    test2 = test_make_classification_predict_and_distr_depth()
    test3 = test_make_classification_fit_predict()
    test4 = test_make_classification_sklearn_max_predict()
    test5 = test_make_classification_sklearn_max_predict_proba()
    test = test and test2 and test3 and test4 and test5
    test6 = test_make_classification_hard_vote_predict()
    test7 = test_make_classification_hard_vote_score_mix()
    test8 = test_score_on_iris()
    test9 = test_save_load()
    test = test and test6 and test7 and test8 and test9
    if test:
        print("Result tests: Passed", flush=True)
    else:
        print("Result tests: Failed", flush=True)


if __name__ == "__main__":
    main()
