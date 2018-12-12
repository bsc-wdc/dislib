import math

from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT
from pycompss.api.task import task

from dislib.classification.rf.decision_tree import DecisionTreeClassifier

import numpy as np

from .data import RfDataset, transform_to_rf_dataset
from dislib.data import Dataset, load_data


class RandomForestClassifier:
    """A distributed random forest classifier.

    Parameters
    ----------
    n_estimators : int, optional (default=10)
        Number of trees to fit.
    try_features : int, str or None, optional (default='sqrt')
        The number of features to consider when looking for the best split:

        - If "sqrt", then `try_features=sqrt(n_features)`.
        - If "third", then `try_features=n_features // 3`.
        - If None, then `try_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires
        to effectively inspect more than ``try_features`` features.
    max_depth : int or float, optional (default=np.inf)
        The maximum depth of the tree. If np.inf, then nodes are expanded
        until all leaves are pure.
    distr_depth : int or str, optional (default='auto')
        Number of levels of the tree in which the nodes are split in a
        distributed way.

    Attributes
    ----------
    classes : None or ndarray
        Array of distinct classes, set at fit().
    trees : list of DecisionTreeClassifier
        List of the tree classifiers of this forest, populated at fit().

    Methods
    -------
    fit(dataset)
        Fits the RandomForestClassifier.
    predict_proba(dataset)
        Predicts class probabilities using a fitted forest.
    predict(dataset, soft_voting=True)
        Predicts classes using a fitted forest.
    score(x_test, y_test)
        Accuracy classification score.

    """

    def __init__(self,
                 n_estimators=10,
                 try_features='sqrt',
                 max_depth=np.inf,
                 distr_depth='auto'):
        self.n_estimators = n_estimators
        self.try_features = try_features
        self.max_depth = max_depth
        self.distr_depth = distr_depth

        self.classes = None
        self.trees = []

    def fit(self, dataset):
        """Fits the RandomForestClassifier.

        Parameters
        ----------
        dataset : dislib.data.Dataset
            Note: In the implementation of this method, the dataset is
            transformed to a dislib.classification.rf.data.RfDataset. To avoid
            the cost of the transformation, RfDataset objects are additionally
            accepted as argument.

        """

        if not isinstance(dataset, (Dataset, RfDataset)):
            raise TypeError('Invalid type for param dataset.')
        if isinstance(dataset, Dataset):
            dataset = transform_to_rf_dataset(dataset)

        if isinstance(dataset.features_path, str):
            dataset.validate_features_file()

        n_features = dataset.get_n_features()
        self.try_features = _resolve_try_features(self.try_features,
                                                  n_features)
        self.classes = dataset.get_classes()

        if self.distr_depth is 'auto':
            dataset.n_samples = compss_wait_on(dataset.get_n_samples())
            self.distr_depth = max(0, int(math.log10(dataset.n_samples)) - 4)
            self.distr_depth = min(self.distr_depth, self.max_depth)

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(self.try_features, self.max_depth,
                                          self.distr_depth, bootstrap=True)
            self.trees.append(tree)

        for tree in self.trees:
            tree.fit(dataset)

    def predict_proba(self, dataset):
        """Predicts class probabilities using a fitted forest.

        The probabilities are obtained as an average of the probabilities of
        each decision tree.

        Parameters
        ----------
        dataset : dislib.data.Dataset
            Dataset with samples for predicting their probabilities.

        Returns
        -------
        dataset : dislib.data.Dataset
            The given dataset, where the labels attribute for each dataset has
            been set to a 2-dimensional array with the predicted probabilities.
            The order of the classes is given by self.classes.

        """
        assert self.trees is not None, 'The random forest is not fitted.'
        for subset in dataset:
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.predict_proba(subset))
            _join_predictions(subset, *tree_predictions)
        return dataset

    def predict(self, dataset, soft_voting=True):
        """Predicts classes using a fitted forest.

        Parameters
        ----------
        dataset : dislib.data.Dataset
            Dataset with samples to predict.

        soft_voting : bool, optional (default=True)
            If True, it takes the class with the higher probability given by
            predict_proba(), which is an average of the probabilities given by
            the decision trees. If False, it uses majority voting over the
            predict() result of the decision tree predictions.

        Returns
        -------
        dataset : dislib.data.Dataset
            The given dataset, with the labels set to their predicted values.

        """
        assert self.trees is not None, 'The random forest is not fitted.'
        if soft_voting:
            for subset in dataset:
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict_proba(subset))
                _soft_vote(subset, self.classes, *tree_predictions)
        else:
            for subset in dataset:
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict(subset))
                _hard_vote(subset, self.classes, *tree_predictions)
        return dataset

    def score(self, x_test, y_test):
        """Accuracy classification score.

        Parameters
        ----------
        x_test : ndarray
            Test samples.
        y_test : ndarray
            Correct test labels.

        Returns
        -------
        score : float
            Fraction of correctly classified samples.

        """
        ds = load_data(x=x_test, subset_size=x_test.shape[0])
        self.predict(ds)
        ds.collect()
        y_pred = ds[0].labels
        return np.count_nonzero(y_pred == y_test) / len(y_test)


@task(returns=1)
def _resolve_try_features(try_features, n_features):
    if try_features is None:
        return n_features
    elif try_features == 'sqrt':
        return int(math.sqrt(n_features))
    elif try_features == 'third':
        return max(1, n_features // 3)
    else:
        return int(try_features)


@task(subset=INOUT, returns=1)
def _join_predictions(subset, *predictions):
    aggregate = predictions[0]
    for p in predictions[1:]:
        aggregate += p
    subset.labels = aggregate/len(predictions)


@task(subset=INOUT, returns=1)
def _soft_vote(subset, classes, *predictions):
    aggregate = predictions[0]
    for p in predictions[1:]:
        aggregate += p
    subset.labels = classes[np.argmax(aggregate, axis=1)]


@task(subset=INOUT, returns=1)
def _hard_vote(subset, classes, *predictions):
    subset.labels = classes[np.argmax(*predictions, axis=1)]
