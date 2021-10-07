import math
from collections import Counter

import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, COLLECTION_IN, Depth
from pycompss.api.task import task
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from dislib.trees.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from dislib.data.array import Array
from dislib.utils.base import _paired_partition
from dislib.trees.data import (
    RfClassifierDataset,
    RfRegressorDataset,
    transform_to_rf_dataset
)


class BaseRandomForest(BaseEstimator):
    """Base class for distributed random forests.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
        self,
        n_estimators,
        try_features,
        max_depth,
        distr_depth,
        sklearn_max,
        hard_vote,
        random_state,
        base_tree,
        base_dataset,
    ):
        self.n_estimators = n_estimators
        self.try_features = try_features
        self.max_depth = max_depth
        self.distr_depth = distr_depth
        self.sklearn_max = sklearn_max
        self.hard_vote = hard_vote
        self.random_state = random_state
        self.base_tree = base_tree
        self.base_dataset = base_dataset

    def fit(self, x, y):
        """Fits a RandomForest.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``.
        y : ds-array, shape=(n_samples, 1)
            The target values.

        Returns
        -------
        self : RandomForest
        """

        dataset = transform_to_rf_dataset(x, y, self.base_dataset)

        n_features = dataset.get_n_features()
        try_features = _resolve_try_features(self.try_features, n_features)
        random_state = check_random_state(self.random_state)

        self.classes = dataset.get_classes()

        if self.distr_depth == "auto":
            dataset.n_samples = compss_wait_on(dataset.get_n_samples())
            distr_depth = max(0, int(math.log10(dataset.n_samples)) - 4)
            distr_depth = min(distr_depth, self.max_depth)
        else:
            distr_depth = self.distr_depth

        self.trees = []
        for _ in range(self.n_estimators):
            tree = self.base_tree(
                try_features,
                self.max_depth,
                distr_depth,
                self.sklearn_max,
                bootstrap=True,
                random_state=random_state,
            )
            self.trees.append(tree)

        for tree in self.trees:
            tree.fit(dataset)

        return self


class RandomForestClassifier(BaseRandomForest):
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

    def __init__(
        self,
        n_estimators=10,
        try_features="sqrt",
        max_depth=np.inf,
        distr_depth="auto",
        sklearn_max=1e8,
        hard_vote=False,
        random_state=None,
    ):
        super().__init__(
            n_estimators,
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            hard_vote,
            random_state,
            base_tree=DecisionTreeClassifier,
            base_dataset=RfClassifierDataset,
        )

    def predict(self, x):
        """Predicts target classes using a fitted forest.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ds-array, shape=(n_samples, 1)
            Predicted class labels for x.
        """
        assert self.trees is not None, "The random forest is not fitted."

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

        y_pred = Array(
            blocks=[pred_blocks],
            top_left_shape=(x._top_left_shape[0], 1),
            reg_shape=(x._reg_shape[0], 1),
            shape=(x.shape[0], 1),
            sparse=False,
        )

        return y_pred

    def predict_proba(self, x):
        """Predicts class probabilities using a fitted forest.

        The probabilities are obtained as an average of the probabilities of
        each decision tree.


        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        probabilities : ds-array, shape=(n_samples, n_classes)
            Predicted probabilities for the samples to belong to each class.
            The columns of the array correspond to the classes given at
            self.classes.
        """
        assert self.trees is not None, "The random forest is not fitted."

        prob_blocks = []
        for x_row in x._iterator(axis=0):
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.predict_proba(x_row))
            prob_blocks.append([_join_predictions(*tree_predictions)])
        self.classes = compss_wait_on(self.classes)
        n_classes = len(self.classes)

        probabilities = Array(
            blocks=prob_blocks,
            top_left_shape=(x._top_left_shape[0], n_classes),
            reg_shape=(x._reg_shape[0], n_classes),
            shape=(x.shape[0], n_classes),
            sparse=False,
        )
        return probabilities

    def score(self, x, y, collect=False):
        """Accuracy classification score.

        Returns the mean accuracy of the predictions on the given test data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The training input samples.
        y : ds-array, shape (n_samples, 1)
            The true labels.
        collect : bool, optional (default=False)
            When True, a synchronized result is returned.


        Returns
        -------
        score : float (as future object)
            Fraction of correctly classified samples.
        """
        assert self.trees is not None, "The random forest is not fitted."

        partial_scores = []
        if self.hard_vote:
            for x_row, y_row in _paired_partition(x, y):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict(x_row))
                subset_score = _hard_vote_score(
                    y_row._blocks, self.classes, *tree_predictions
                )
                partial_scores.append(subset_score)

        else:
            for x_row, y_row in _paired_partition(x, y):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict_proba(x_row))
                subset_score = _soft_vote_score(
                    y_row._blocks, self.classes, *tree_predictions
                )
                partial_scores.append(subset_score)

        score = _merge_classification_scores(*partial_scores)

        return compss_wait_on(score) if collect else score


class RandomForestRegressor(BaseRandomForest):
    """A distributed random forest regressor.

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
        sklearn's DecisionTreeRegressor.fit(), which is called to fit subtrees
        (subsamples) of our DecisionTreeRegressor. sklearn fit() is used
        because it's faster, but requires loading the data to memory, which can
        cause memory problems for large datasets. This parameter can be
        adjusted to fit the hardware capabilities.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    trees : list of DecisionTreeRegressor
        List of the tree regressors of this forest, populated at fit().
    """

    def __init__(
        self,
        n_estimators=10,
        try_features="sqrt",
        max_depth=np.inf,
        distr_depth="auto",
        sklearn_max=1e8,
        random_state=None,
    ):
        hard_vote = None
        super().__init__(
            n_estimators,
            try_features,
            max_depth,
            distr_depth,
            sklearn_max,
            hard_vote,
            random_state,
            base_tree=DecisionTreeRegressor,
            base_dataset=RfRegressorDataset,
        )

    def predict(self, x):
        """Predicts target values using a fitted forest.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ds-array, shape=(n_samples, 1)
            Predicted values for x.
        """
        assert self.trees is not None, "The random forest is not fitted."

        pred_blocks = []
        for x_row in x._iterator(axis=0):
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.predict(x_row))
            pred_blocks.append(_join_predictions(*tree_predictions))

        y_pred = Array(
            blocks=[pred_blocks],
            top_left_shape=(x._top_left_shape[0], 1),
            reg_shape=(x._reg_shape[0], 1),
            shape=(x.shape[0], 1),
            sparse=False,
        )

        return y_pred

    def score(self, x, y, collect=False):
        """R2 regression score.

        Returns the coefficient of determination $R^2$ of the prediction.
        The coefficient $R^2$ is defined as $(1-u/v)$, where $u$
        is the residual sum of squares `((y_true - y_pred) ** 2).sum()` and
        $v$ is the total sum of squares
        `((y_true - y_true.mean()) ** 2).sum()`.
        The best possible score is 1.0 and it can be negative
        if the model is arbitrarily worse.
        A constant model that always predicts the expected value of y,
        disregarding the input features, would get a $R^2$ score of 0.0.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            The training input samples.
        y : ds-array, shape (n_samples, 1)
            The true values.
        collect : bool, optional (default=False)
            When True, a synchronized result is returned.


        Returns
        -------
        score : float (as future object)
            Coefficient of determination $R^2$.
        """
        assert self.trees is not None, "The random forest is not fitted."

        partial_scores = []
        for x_row, y_row in _paired_partition(x, y):
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.predict(x_row))
            subset_score = _regression_score(y_row._blocks, *tree_predictions)
            partial_scores.append(subset_score)

        score = _merge_regression_scores(*partial_scores)

        return compss_wait_on(score) if collect else score


def _base_soft_vote(classes, *predictions):
    aggregate = predictions[0]
    for p in predictions[1:]:
        aggregate += p
    predicted_labels = classes[np.argmax(aggregate, axis=1)]
    return predicted_labels


def _base_hard_vote(classes, *predictions):
    mode = np.empty((len(predictions[0]),), dtype=int)
    for sample_i, votes in enumerate(zip(*predictions)):
        mode[sample_i] = Counter(votes).most_common(1)[0][0]
    labels = classes[mode]
    return labels


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _resolve_try_features(try_features, n_features):
    if try_features is None:
        return n_features
    elif try_features == "sqrt":
        return int(math.sqrt(n_features))
    elif try_features == "third":
        return max(1, n_features // 3)
    else:
        return int(try_features)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _join_predictions(*predictions):
    aggregate = predictions[0]
    for p in predictions[1:]:
        aggregate += p
    labels = aggregate / len(predictions)
    return labels


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _soft_vote(classes, *predictions):
    predicted_labels = _base_soft_vote(classes, *predictions)
    return predicted_labels


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _hard_vote(classes, *predictions):
    predicted_labels = _base_hard_vote(classes, *predictions)
    return predicted_labels


@constraint(computing_units="${ComputingUnits}")
@task(y_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _soft_vote_score(y_blocks, classes, *predictions):
    predicted_labels = _base_soft_vote(classes, *predictions)
    real_labels = Array._merge_blocks(y_blocks).flatten()
    correct = np.count_nonzero(predicted_labels == real_labels)
    return correct, len(real_labels)


@constraint(computing_units="${ComputingUnits}")
@task(y_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _hard_vote_score(y_blocks, classes, *predictions):
    predicted_labels = _base_hard_vote(classes, *predictions)
    real_labels = Array._merge_blocks(y_blocks).flatten()
    correct = np.count_nonzero(predicted_labels == real_labels)
    return correct, len(real_labels)


@constraint(computing_units="${ComputingUnits}")
@task(y_blocks={Type: COLLECTION_IN, Depth: 2}, returns=1)
def _regression_score(y_blocks, *predictions):
    y_true = Array._merge_blocks(y_blocks).flatten()
    y_pred = np.mean(predictions, axis=0)
    n_samples = y_true.shape[0]
    y_avg = np.mean(y_true)
    u_partial = np.sum(np.square(y_true - y_pred), axis=0)
    v_partial = np.sum(np.square(y_true - y_avg), axis=0)
    return u_partial, v_partial, y_avg, n_samples


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _merge_classification_scores(*partial_scores):
    correct = sum(subset_score[0] for subset_score in partial_scores)
    total = sum(subset_score[1] for subset_score in partial_scores)
    return correct / total


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _merge_regression_scores(*partial_scores):
    u = v = avg = n = 0
    for u_p, v_p, avg_p, n_p in partial_scores:
        u += u_p

        delta = avg_p - avg
        avg += delta * n_p / (n + n_p)
        v += v_p + delta ** 2 * n * n_p / (n + n_p)
        n += n_p

    return 1 - u / v
