import math
from collections import Counter

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import INOUT, Type, COLLECTION_IN, Depth
from pycompss.api.task import task
from sklearn.utils import check_random_state

from dislib.classification.rf.decision_tree import DecisionTreeClassifier
from dislib.data import Dataset
from dislib.data.array import Array
from .data import RfDataset, transform_to_rf_dataset


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
    max_depth : int or np.inf, optional (default=np.inf)
        The maximum depth of the tree. If np.inf, then nodes are expanded
        until all leaves are pure.
    distr_depth : int or str, optional (default='auto')
        Number of levels of the tree in which the nodes are split in a
        distributed way.
    sklearn_max: int or float, optional (default=1e8)
        Maximum size (len(subsample)*n_features) of the arrays passed to
        sklearn's DecisionTreeClassifier.fit(), which is called to fit subtrees
        (subsamples) of our DecisionTreeClassifier. sklearn fit() is used
        because it's faster, but requires loading the data to memory, which can
        cause memory problems for large datasets. This parameter can be
        adjusted to fit the hardware capabilities.
    hard_vote : bool, optional (default=False)
        If True, it uses majority voting over the predict() result of the
        decision tree predictions. If False, it takes the class with the higher
        probability given by predict_proba(), which is an average of the
        probabilities given by the decision trees.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    classes : None or ndarray
        Array of distinct classes, set at fit().
    trees : list of DecisionTreeClassifier
        List of the tree classifiers of this forest, populated at fit().
    """

    def __init__(self,
                 n_estimators=10,
                 try_features='sqrt',
                 max_depth=np.inf,
                 distr_depth='auto',
                 sklearn_max=1e8,
                 hard_vote=False,
                 random_state=None):
        self.n_estimators = n_estimators
        self.try_features_init = try_features
        self.max_depth = max_depth
        self.distr_depth_init = distr_depth
        self.sklearn_max = sklearn_max
        self.hard_vote = hard_vote
        self.random_state = check_random_state(random_state)

    def fit(self, x, y, rf_dataset=False):
        """Fits the RandomForestClassifier.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``.
        y : ds-array, shape (n_samples, 1)
            The target values.
        rf_dataset : RfDataset (default=False)
            Dataset in the particular format used by RandomForestClassifier,
            which can reduce the execution time by avoiding the conversion.
            If this argument is provided, x and y are ignored.

        """
        self.classes = None
        self.trees = []

        if isinstance(rf_dataset, RfDataset):
            dataset = rf_dataset
            if isinstance(dataset.features_path, str):
                dataset.validate_features_file()
        else:
            dataset = transform_to_rf_dataset(x, y)

        n_features = dataset.get_n_features()
        self.try_features = _resolve_try_features(self.try_features_init,
                                                  n_features)
        self.classes = dataset.get_classes()

        if self.distr_depth_init == 'auto':
            dataset.n_samples = compss_wait_on(dataset.get_n_samples())
            self.distr_depth = max(0, int(math.log10(dataset.n_samples)) - 4)
            self.distr_depth = min(self.distr_depth, self.max_depth)
        else:
            self.distr_depth = self.distr_depth_init

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(self.try_features, self.max_depth,
                                          self.distr_depth, self.sklearn_max,
                                          bootstrap=True,
                                          random_state=self.random_state)
            self.trees.append(tree)

        for tree in self.trees:
            tree.fit(dataset)

    def fit_predict(self, x, y):
        """Fits the forest and predicts the classes for the same samples.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            The training input samples.
        y : ds-array, shape (n_samples, 1)
            The target values.

        Returns
        -------
        y_pred : ds-array, shape(n_samples, 1)
            Predicted class labels for x.

        """
        self.fit(x, y)
        return self.predict(x)

    def predict_proba(self, x):
        """Predicts class probabilities using a fitted forest.

        The probabilities are obtained as an average of the probabilities of
        each decision tree.


        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        probabilities : list of unsynchronized arrays
            List with an array of the predicted probabilities for each rows
            block. The shape of each array is (rows_block_size, n_classes).
            The class corresponding to each column of the arrays is given by
            self.classes.

        """
        assert self.trees is not None, 'The random forest is not fitted.'
        probabilities = []
        for x_row in x._iterator(axis=0):
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.predict_proba(x_row))
            probabilities.append(_join_predictions(*tree_predictions))
        return probabilities

    def predict(self, x):
        """Predicts classes using a fitted forest.

        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ds-array, shape(n_samples, 1)
            Predicted class labels for x.

        """
        assert self.trees is not None, 'The random forest is not fitted.'
        pred_blocks = []
        if self.hard_vote:
            for x_row in x._iterator(axis=0):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict(x_row))
                pred_blocks.append(_hard_vote(self.classes, *tree_predictions))
        else:
            for x_row in x._iterator(axis=0):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict_proba(x_row))
                pred_blocks.append(_soft_vote(self.classes, *tree_predictions))
        s = x._blocks_shape
        pred_blocks_shape = (s[0], 1) if len(s) == 2 \
            else ((s[0][0], 1), (s[1][0], 1), (s[2][0], 1))
        y_pred = Array(blocks=[pred_blocks], blocks_shape=pred_blocks_shape,
                       shape=(x.shape[0], 1), sparse=False)
        return y_pred

    def score(self, x, y):
        """Accuracy classification score.

        Returns the mean accuracy on the given test dataset.


        Parameters
        ----------
        x : ds-array, shape (n_samples, n_features)
            The training input samples.
        y : ds-array, shape (n_samples, 1)
            The true labels.

        Returns
        -------
        score : float
            Fraction of correctly classified samples.

        """
        assert self.trees is not None, 'The random forest is not fitted.'
        partial_scores = []

        def paired_blocks(x, y):
            regular_shapes = len(x._blocks_shape) == 2
            if regular_shapes:
                top_num_rows = x._blocks_shape[0]
                regular_num_rows = x._blocks_shape[0]
            else:
                top_num_rows = x._blocks_shape[0][0]
                regular_num_rows = x._blocks_shape[1][0]
            start_idx = 0
            end_idx = top_num_rows
            for x_row in x._iterator(axis=0):
                y_row = y[start_idx:end_idx]
                yield x_row, y_row
                start_idx = end_idx
                end_idx = min(end_idx + regular_num_rows, x.shape[0])

        if self.hard_vote:
            for x_row, y_row in paired_blocks(x, y):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict(x_row))
                subset_score = _hard_vote_score(y_row._blocks, self.classes,
                                                *tree_predictions)
                partial_scores.append(subset_score)
        else:
            for x_row, y_row in paired_blocks(x, y):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict_proba(x_row))
                subset_score = _soft_vote_score(y_row._blocks, self.classes,
                                                *tree_predictions)
                partial_scores.append(subset_score)
        score = compss_wait_on(_merge_scores(*partial_scores))

        return score


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


@task(returns=1)
def _join_predictions(*predictions):
    aggregate = predictions[0]
    for p in predictions[1:]:
        aggregate += p
    labels = aggregate / len(predictions)
    return labels


@task(returns=1)
def _soft_vote(classes, *predictions):
    aggregate = predictions[0]
    for p in predictions[1:]:
        aggregate += p
    labels = classes[np.argmax(aggregate, axis=1)]
    return labels


@task(returns=1)
def _hard_vote(classes, *predictions):
    mode = np.empty((len(predictions[0]),), dtype=int)
    for sample_i, votes in enumerate(zip(*predictions)):
        mode[sample_i] = Counter(votes).most_common(1)[0][0]
    labels = classes[mode]
    return labels


@task(y_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _soft_vote_score(y_blocks, classes, *predictions):
    real_labels = Array._merge_blocks(y_blocks).flatten()
    aggregate = predictions[0]
    for p in predictions[1:]:
        aggregate += p
    predicted_labels = classes[np.argmax(aggregate, axis=1)]
    correct = np.count_nonzero(predicted_labels == real_labels)
    return correct, len(real_labels)


@task(y_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _hard_vote_score(y_blocks, classes, *predictions):
    real_labels = Array._merge_blocks(y_blocks).flatten()
    mode = np.empty((len(predictions[0]),), dtype=int)
    for sample_i, votes in enumerate(zip(*predictions)):
        mode[sample_i] = Counter(votes).most_common(1)[0][0]
    predicted_labels = classes[mode]
    correct = np.count_nonzero(predicted_labels == real_labels)
    return correct, len(real_labels)


@task(returns=1)
def _merge_scores(*partial_scores):
    correct = sum(subset_score[0] for subset_score in partial_scores)
    total = sum(subset_score[1] for subset_score in partial_scores)
    return correct / total
