from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
import math
import numpy as np
from pycompss.api.parameter import COLLECTION_IN, Type, Depth

from dislib.trees.nested.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor, encode_forest_helper, decode_forest_helper,
)
from pycompss.api.api import compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.task import task
from dislib.data.array import Array
from dislib.utils.base import _paired_partition
from dislib.data.util import decoder_helper, encoder_helper, sync_obj
import json
import numbers
import os
import pickle
import dislib.data.util.model as utilmodel
from sklearn.svm import SVC as SklearnSVC
from sklearn.tree import DecisionTreeClassifier as SklearnDTClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnDTRegressor
from sklearn.tree._tree import Tree as SklearnTree
SKLEARN_CLASSES = {
    "SVC": SklearnSVC,
    "DecisionTreeClassifier": SklearnDTClassifier,
    "DecisionTreeRegressor": SklearnDTRegressor,
}


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
        n_classes=None,
        range_max=None,
        range_min=None,
        bootstrap=True,
        n_split_points="auto",
        split_computation="raw",
        sync_after_fit=True,
    ):
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.try_features = try_features
        self.max_depth = max_depth
        self.distr_depth = distr_depth
        self.sklearn_max = sklearn_max
        self.hard_vote = hard_vote
        self.random_state = random_state
        self.base_tree = base_tree
        self.range_max = range_max
        self.range_min = range_min
        self.bootstrap = bootstrap
        self.n_split_points = n_split_points
        self.split_computation = split_computation
        self.sync_after_fit = sync_after_fit

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

        try_features = _resolve_try_features(self.try_features, x.shape[1])

        if self.range_max is None:
            self.range_max = x.max()
        if self.range_min is None:
            self.range_min = x.min()
        self.range_max._blocks = compss_wait_on(self.range_max._blocks)
        self.range_min._blocks = compss_wait_on(self.range_min._blocks)

        if self.distr_depth == "auto":
            distr_depth = max(0, int(math.log10(x.shape[0])) - 4)
            distr_depth = min(distr_depth, self.max_depth)
            if distr_depth < 1:
                self.distr_depth = 1
            else:
                self.distr_depth = distr_depth

        self.trees = []

        for _ in range(self.n_estimators):
            random_state = check_random_state(self.random_state)
            if isinstance(self.random_state, numbers.Integral):
                self.random_state = self.random_state+np.random.randint(100)
            if self.n_classes is not None:
                tree = self.base_tree(
                    try_features=try_features,
                    max_depth=self.max_depth,
                    distr_depth=self.distr_depth,
                    sklearn_max=self.sklearn_max,
                    bootstrap=self.bootstrap,
                    random_state=random_state,
                    n_classes=self.n_classes,
                    range_min=self.range_min,
                    range_max=self.range_max,
                    n_split_points=self.n_split_points,
                    split_computation=self.split_computation,
                    sync_after_fit=False,
                )
            else:
                tree = self.base_tree(
                    try_features=try_features,
                    max_depth=self.max_depth,
                    distr_depth=self.distr_depth,
                    sklearn_max=self.sklearn_max,
                    bootstrap=self.bootstrap,
                    random_state=random_state,
                    range_min=self.range_min,
                    range_max=self.range_max,
                    n_split_points=self.n_split_points,
                    split_computation=self.split_computation,
                    sync_after_fit=False,
                )
            self.trees.append(tree)

        for tree in self.trees:
            tree.fit(x, y)
        self.trees = compss_wait_on(self.trees)

        return self

    def save_model(self, filepath, overwrite=True, save_format="json"):
        """Saves a model to a file.
        The model is synchronized before saving and can be reinstantiated in
        the exact same state, without any of the code used for model
        definition or fitting.
        Parameters
        ----------
        filepath : str
            Path where to save the model
        overwrite : bool, optional (default=True)
            Whether any existing model at the target
            location should be overwritten.
        save_format : str, optional (default='json)
            Format used to save the models.
        Examples
        --------
        >>> from dislib.cluster import DecisionTreeClassifier
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> x_train = ds.array(x, (2, 2))
        >>> model = DecisionTreeClassifier(n_clusters=2, random_state=0)
        >>> model.fit(x_train)
        >>> save_model(model, '/tmp/model')
        >>> loaded_model = load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        loaded_model_pred.collect())
        """

        # Check overwrite
        if not overwrite and os.path.isfile(filepath):
            return

        _sync_rf(self)

        sync_obj(self.__dict__)

        model_metadata = self.__dict__
        model_metadata["model_name"] = self.__class__.__name__

        # Save model
        if save_format == "json":
            with open(filepath, "w") as f:
                json.dump(model_metadata, f, default=_encode_helper)
        elif save_format == "cbor":
            if utilmodel.cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "wb") as f:
                utilmodel.cbor2.dump(model_metadata, f,
                                     default=_encode_helper_cbor)
        elif save_format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(model_metadata, f)
        else:
            raise ValueError("Wrong save format.")

    def load_model(self, filepath, load_format="json"):
        """Loads a model from a file.
        The model is reinstantiated in the exact same state in which it
        was saved, without any of the code used for model definition or
        fitting.
        Parameters
        ----------
        filepath : str
            Path of the saved the model
        load_format : str, optional (default='json')
            Format used to load the model.
        Examples
        --------
        >>> from dislib.cluster import DecisionTreeClassifier
        >>> import numpy as np
        >>> import dislib as ds
        >>> x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> x_train = ds.array(x, (2, 2))
        >>> model = DecisionTreeClassifier(n_clusters=2, random_state=0)
        >>> model.fit(x_train)
        >>> save_model(model, '/tmp/model')
        >>> loaded_model = load_model('/tmp/model')
        >>> x_test = ds.array(np.array([[0, 0], [4, 4]]), (2, 2))
        >>> model_pred = model.predict(x_test)
        >>> loaded_model_pred = loaded_model.predict(x_test)
        >>> assert np.allclose(model_pred.collect(),
        """
        # Load model
        if load_format == "json":
            with open(filepath, "r") as f:
                model_metadata = json.load(f, object_hook=_decode_helper)
        elif load_format == "cbor":
            if utilmodel.cbor2 is None:
                raise ModuleNotFoundError("No module named 'cbor2'")
            with open(filepath, "rb") as f:
                model_metadata = utilmodel.cbor2.\
                    load(f, object_hook=_decode_helper_cbor)
        elif load_format == "pickle":
            with open(filepath, "rb") as f:
                model_metadata = pickle.load(f)
        else:
            raise ValueError("Wrong load format.")

        for key, val in model_metadata.items():
            setattr(self, key, val)


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
        n_classes,
        n_estimators=10,
        try_features="sqrt",
        max_depth=np.inf,
        distr_depth="auto",
        sklearn_max=1e8,
        hard_vote=False,
        random_state=None,
        range_max=None,
        range_min=None,
        bootstrap=True,
        n_split_points="auto",
        split_computation="raw",
        sync_after_fit=True,
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
            n_classes=n_classes,
            range_max=range_max,
            range_min=range_min,
            bootstrap=bootstrap,
            n_split_points=n_split_points,
            split_computation=split_computation,
            sync_after_fit=sync_after_fit,
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

        if self.hard_vote:
            for x_row in x._iterator(axis=0):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict(x_row))
                pred_blocks.append([_hard_vote(np.arange(self.n_classes),
                                               compss_wait_on(
                                                   tree_predictions))])
        else:
            for x_row in x._iterator(axis=0):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict_proba(x_row))
                pred_blocks.append([_soft_vote(np.arange(self.n_classes),
                                               compss_wait_on(
                                                   tree_predictions))])
        pred_blocks = compss_wait_on(pred_blocks)
        y_pred = Array(
            blocks=pred_blocks,
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
            prob_blocks.append([_join_predictions(tree_predictions)])

        probabilities = Array(
            blocks=prob_blocks,
            top_left_shape=(x._top_left_shape[0], self.n_classes),
            reg_shape=(x._reg_shape[0], self.n_classes),
            shape=(x.shape[0], self.n_classes),
            sparse=False,
        )
        return probabilities

    def score(self, x, y, collect=False):
        assert self.trees is not None, "The random forest is not fitted."
        partial_scores = []
        if self.hard_vote:
            for x_row, y_row in _paired_partition(x, y):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict(x_row))
                subset_score = _hard_vote_score(
                    y_row._blocks, np.arange(self.n_classes), tree_predictions
                )
                partial_scores.append(subset_score)
        else:
            for x_row, y_row in _paired_partition(x, y):
                tree_predictions = []
                for tree in self.trees:
                    tree_predictions.append(tree.predict_proba(x_row))
                subset_score = _soft_vote_score(
                    y_row._blocks, np.arange(self.n_classes), tree_predictions
                )
                partial_scores.append(subset_score)
        score = _merge_classification_scores(partial_scores)

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
            sklearn's DecisionTreeRegressor.fit(), which is
            called to fit subtrees (subsamples) of our DecisionTreeRegressor.
            sklearn fit() is used because it's faster, but requires loading
            the data to memory, which can cause memory problems
            for large datasets.
            This parameter can be adjusted to fit the hardware capabilities.
        random_state : int, RandomState instance or None, optional
        (default=None)
            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState
            instance used
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
        range_max=None,
        range_min=None,
        bootstrap=True,
        n_split_points="auto",
        split_computation="raw",
        sync_after_fit=True,
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
            n_classes=None,
            range_max=range_max,
            range_min=range_min,
            bootstrap=bootstrap,
            n_split_points=n_split_points,
            split_computation=split_computation,
            sync_after_fit=sync_after_fit,
        )

    def predict(self, x):
        pred_blocks = []
        for x_row in x._iterator(axis=0):
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.predict(x_row))
            pred_blocks.append(tree_predictions)
        final_blocks = []
        for tree_predictions in pred_blocks:
            final_blocks.append([_join_predictions(
                compss_wait_on(tree_predictions))])

        y_pred = Array(
            blocks=final_blocks,
            top_left_shape=(x._top_left_shape[0], 1),
            reg_shape=(x._reg_shape[0], 1),
            shape=(x.shape[0], 1),
            sparse=False,
        )

        return y_pred

    def score(self, x, y, collect=False):
        assert self.trees is not None, "The random forest is not fitted."

        partial_scores = []
        for x_row, y_row in _paired_partition(x, y):
            tree_predictions = []
            for tree in self.trees:
                tree_predictions.append(tree.predict(x_row))
            subset_score = _regression_score(y_row._blocks, tree_predictions)
            partial_scores.append(subset_score)

        score = _merge_regression_scores(partial_scores)

        return compss_wait_on(score) if collect else score


def _base_soft_vote(classes, predictions):
    aggregate = predictions[0][0]
    for p in predictions[1:]:
        aggregate += p[0]
    predicted_labels = classes[np.argmax(aggregate, axis=1)]
    return np.expand_dims(predicted_labels, axis=1)


def _base_hard_vote(classes, predictions):
    mode = predictions[0][0]
    for p in predictions[1:]:
        mode += p[0]
    predicted_labels = classes[np.argmax(mode, axis=1)]
    return np.expand_dims(predicted_labels, axis=1)


def _soft_vote(classes, predictions):
    predicted_labels = _base_soft_vote(classes, predictions)
    return predicted_labels


@constraint(computing_units="${ComputingUnits}")
@task(y_blocks={Type: COLLECTION_IN, Depth: 2},
      predictions=COLLECTION_IN, returns=1)
def _soft_vote_score(y_blocks, classes, predictions):
    predicted_labels = _base_soft_vote(classes, predictions)
    real_labels = Array._merge_blocks(y_blocks).flatten()
    correct = np.count_nonzero(predicted_labels.squeeze() == real_labels)
    return correct, len(real_labels)


def _hard_vote(classes, predictions):
    predicted_labels = _base_hard_vote(classes, predictions)
    return predicted_labels


@constraint(computing_units="${ComputingUnits}")
@task(y_blocks={Type: COLLECTION_IN, Depth: 2},
      predictions=COLLECTION_IN, returns=1)
def _hard_vote_score(y_blocks, classes, predictions):
    predicted_labels = _base_hard_vote(classes, predictions)
    real_labels = Array._merge_blocks(y_blocks).flatten()
    correct = np.count_nonzero(predicted_labels.squeeze() == real_labels)
    return correct, len(real_labels)


def _resolve_try_features(try_features, n_features):
    if try_features is None:
        return n_features
    elif try_features == "sqrt":
        return int(math.sqrt(n_features))
    elif try_features == "third":
        return max(1, n_features // 3)
    elif try_features >= 1:
        return int(try_features)
    else:
        return int(try_features*n_features)


@constraint(computing_units="${ComputingUnits}")
@task(predictions=COLLECTION_IN, returns=1)
def _join_predictions(predictions):
    aggregate = np.block(predictions[0])
    for p in predictions[1:]:
        aggregate += np.block(p)
    labels = aggregate / len(predictions)
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    return labels


@constraint(computing_units="${ComputingUnits}")
@task(y_blocks={Type: COLLECTION_IN, Depth: 2},
      predictions=COLLECTION_IN, returns=1)
def _regression_score(y_blocks, predictions):
    y_true = Array._merge_blocks(y_blocks).flatten()
    y_pred = np.mean(np.squeeze(predictions), axis=0)
    n_samples = y_true.shape[0]
    y_avg = np.mean(y_true)
    u_partial = np.sum(np.square(y_true - y_pred), axis=0)
    v_partial = np.sum(np.square(y_true - y_avg), axis=0)
    return u_partial, v_partial, y_avg, n_samples


@constraint(computing_units="${ComputingUnits}")
@task(partial_scores=COLLECTION_IN, returns=1)
def _merge_classification_scores(partial_scores):
    correct = sum(subset_score[0] for subset_score in partial_scores)
    total = sum(subset_score[1] for subset_score in partial_scores)
    return correct / total


@constraint(computing_units="${ComputingUnits}")
@task(partial_scores=COLLECTION_IN, returns=1)
def _merge_regression_scores(partial_scores):
    u = v = avg = n = 0
    for u_p, v_p, avg_p, n_p in partial_scores:
        u += u_p

        delta = avg_p - avg
        avg += delta * n_p / (n + n_p)
        v += v_p + delta ** 2 * n * n_p / (n + n_p)
        n += n_p

    return 1 - u / v


def _encode_helper_cbor(encoder, obj):
    encoder.encode(_encode_helper(obj))


def _encode_helper(obj):
    encoded = encoder_helper(obj)
    if encoded is not None:
        return encoded
    elif callable(obj):
        return {
            "class_name": "callable",
            "module": obj.__module__,
            "name": obj.__name__,
        }
    elif isinstance(obj, SklearnTree):
        return {
            "class_name": obj.__class__.__name__,
            "n_features": obj.n_features,
            "n_classes": obj.n_classes,
            "n_outputs": obj.n_outputs,
            "items": obj.__getstate__(),
        }
    elif isinstance(obj, (RandomForestClassifier, RandomForestRegressor,
                          DecisionTreeClassifier, DecisionTreeRegressor,
                          SklearnDTClassifier, SklearnDTRegressor)):
        return {
            "class_name": obj.__class__.__name__,
            "module_name": obj.__module__,
            "items": obj.__dict__,
        }
    else:
        return encode_forest_helper(obj)


def _decode_helper_cbor(decoder, obj):
    """Special decoder wrapper for dislib using cbor2."""
    return _decode_helper(obj)


def _decode_helper(obj):
    if isinstance(obj, dict) and "class_name" in obj:
        class_name = obj["class_name"]
        decoded = decoder_helper(class_name, obj)
        if decoded is not None:
            return decoded
        elif class_name == "RandomState":
            random_state = np.random.RandomState()
            random_state.set_state(_decode_helper(obj["items"]))
            return random_state
        elif class_name == "Tree":
            dict_ = _decode_helper(obj["items"])
            model = SklearnTree(
                obj["n_features"], obj["n_classes"], obj["n_outputs"]
            )
            model.__setstate__(dict_)
            return model
        elif class_name == "callable":
            if obj["module"] == "numpy":
                return getattr(np, obj["name"])
            return None
        elif (
                class_name in SKLEARN_CLASSES.keys()
                and "sklearn" in obj["module_name"]
        ):
            dict_ = _decode_helper(obj["items"])
            model = SKLEARN_CLASSES[obj["class_name"]]()
            model.__dict__.update(dict_)
            return model
        else:
            dict_ = _decode_helper(obj["items"])
            return decode_forest_helper(class_name, dict_)
    return obj


def _sync_rf(rf):
    """Sync the `try_features` and `n_classes` attribute of the different trees
    since they cannot be synced recursively.
    """
    try_features = compss_wait_on(rf.trees[0].try_features)
    n_classes = compss_wait_on(rf.trees[0].n_classes)
    for tree in rf.trees:
        tree.try_features = try_features
        tree.n_classes = n_classes
