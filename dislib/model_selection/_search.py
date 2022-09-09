from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from itertools import product

import numpy as np
from pycompss.api.api import compss_wait_on
from scipy.stats import rankdata
from sklearn import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler
from numpy.ma import MaskedArray

from dislib.model_selection._split import infer_cv
from dislib.model_selection._validation import check_scorer, \
    validate_score, aggregate_score_dicts, fit, score_func, \
    sklearn_fit, sklearn_score


class BaseSearchCV(ABC):
    """Abstract base class for hyper parameter search with cross-validation."""

    def __init__(self, estimator, scoring=None, cv=None, refit=True):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.refit = refit

    @abstractmethod
    def _run_search(self, evaluate_candidates):
        """Abstract method to perform the search. The parameter
        `evaluate_candidates` is a function that evaluates a ParameterGrid at a
        time """
        pass

    def fit(self, x, y=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        x : ds-array
            Training data samples.
        y : ds-array, optional (default = None)
            Training data labels or values.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        cv = infer_cv(self.cv)

        scorers, refit_metric = self._infer_scorers()

        base_estimator = clone(estimator)

        n_splits = None
        all_candidate_params = []
        all_out = []

        def evaluate_candidates_sklearn(candidate_params):
            """Evaluate some parameters"""
            candidate_params = list(candidate_params)

            validation_data = []
            fits = []
            for parameters, (train, validation) in product(candidate_params,
                                                           cv.split(x, y)):
                validation_data.append(validation)
                fits.append(sklearn_fit(clone(base_estimator), train,
                                        parameters=parameters,
                                        fit_params=fit_params))
            out = [sklearn_score(estimator, validation, scorer=scorers) for
                   estimator, validation in zip(fits, validation_data)]

            out = compss_wait_on(out)

            nonlocal n_splits
            n_splits = cv.get_n_splits()

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

        def evaluate_candidates(candidate_params):
            """Evaluate some parameters"""
            candidate_params = list(candidate_params)

            validation_data = []
            fits = []
            for parameters, (train, validation) in product(candidate_params,
                                                           cv.split(x, y)):
                validation_data.append(validation)
                fits.append(fit(clone(base_estimator), train,
                                parameters=parameters,
                                fit_params=fit_params))
            out = [score_func(estimator, validation, scorer=scorers) for
                   estimator, validation in zip(fits, validation_data)]

            nonlocal n_splits
            n_splits = cv.get_n_splits()

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)
        if 'sklearn' in str(type(estimator)):
            self._run_search(evaluate_candidates_sklearn)
        else:
            self._run_search(evaluate_candidates)

        for params_result in all_out:
            scores = params_result[0]
            for scorer_name, score in scores.items():
                score = compss_wait_on(score)
                scores[scorer_name] = validate_score(score, scorer_name)

        results = self._format_results(all_candidate_params, scorers,
                                       n_splits, all_out)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, (int, np.integer)):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                        self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                    self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            if 'sklearn' in str(type(estimator)):
                x = x.collect()
                y = y.collect()
            self.best_estimator_.fit(x, y, **fit_params)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    @staticmethod
    def _format_results(candidate_params, scorers, n_splits, out):
        n_candidates = len(candidate_params)

        (test_score_dicts,) = zip(*out)

        test_scores = aggregate_score_dicts(test_score_dicts)

        results = {}

        def _store(key_name, array, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.mean(array, axis=1)
            results['mean_%s' % key_name] = array_means
            array_stds = np.std(array, axis=1)
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates, ),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        for scorer_name in scorers.keys():
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True)

        return results

    def _infer_scorers(self):
        estimator = self.estimator
        scoring = self.scoring
        refit = self.refit
        if scoring is None or callable(scoring):
            scorers = {"score": check_scorer(estimator, scoring)}
            refit_metric = 'score'
            self.multimetric_ = False
        elif isinstance(scoring, dict):
            scorers = {key: check_scorer(estimator, scorer)
                       for key, scorer in scoring.items()}
            if refit is not False and (
                    not isinstance(refit, str) or
                    refit not in scorers) and not callable(refit):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key or a "
                                 "callable to refit an estimator with the "
                                 "best parameter setting on the whole "
                                 "data and make the best_* attributes "
                                 "available for that metric. If this is "
                                 "not needed, refit should be set to "
                                 "False explicitly. %r was passed."
                                 % refit)
            refit_metric = refit
            self.multimetric_ = True
        else:
            raise ValueError('scoring is not valid')

        return scorers, refit_metric


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    GridSearchCV implements a "fit" and a "score" method.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    scoring : callable, dict or None, optional (default=None)
        A callable to evaluate the predictions on the test set. It should take
        3 parameters, estimator, x and y, and return a score (higher meaning
        better). For evaluating multiple metrics, give a dict with names as
        keys and callables as values. If None, the estimator's score method is
        used.
    cv : int or cv generator, optional (default=None)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `KFold`,
        - custom cv generator.
    refit : boolean, string, or callable, optional (default=True)
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer. ``best_score_`` is not returned if refit is callable.
        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    Examples
    --------
    >>> import dislib as ds
    >>> from dislib.model_selection import GridSearchCV
    >>> from dislib.classification import RandomForestClassifier
    >>> import numpy as np
    >>> from sklearn import datasets
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     x_np, y_np = datasets.load_iris(return_X_y=True)
    >>>     x = ds.array(x_np, (30, 4))
    >>>     y = ds.array(y_np[:, np.newaxis], (30, 1))
    >>>     param_grid = {'n_estimators': (2, 4), 'max_depth': range(3, 5)}
    >>>     rf = RandomForestClassifier()
    >>>     searcher = GridSearchCV(rf, param_grid)
    >>>     searcher.fit(x, y)
    >>>     searcher.cv_results_

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table:

        +------------+------------+-----------------+---+---------+
        |param_kernel|param_degree|split0_test_score|...|rank_t...|
        +============+============+=================+===+=========+
        |  'poly'    |      2     |       0.80      |...|    2    |
        +------------+------------+-----------------+---+---------+
        |  'poly'    |      3     |       0.70      |...|    4    |
        +------------+------------+-----------------+---+---------+
        |  'rbf'     |     --     |       0.80      |...|    3    |
        +------------+------------+-----------------+---+---------+
        |  'rbf'     |     --     |       0.93      |...|    1    |
        +------------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.68, 0.78],
            'split2_test_score'  : [0.79, 0.55, 0.71, 0.93],
            ...
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTES:

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above ('split0_test_precision', 'mean_train_precision' etc.).

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.
        See ``refit`` parameter for more information on allowed values.
    best_score_ : float
        Mean cross-validated score of the best_estimator
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    """

    def __init__(self, estimator, param_grid, scoring=None, cv=None,
                 refit=True):
        super().__init__(estimator=estimator, scoring=scoring, cv=cv,
                         refit=refit)
        self.param_grid = param_grid
        self._check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(ParameterGrid(self.param_grid))

    @staticmethod
    def _check_param_grid(param_grid):
        if hasattr(param_grid, 'items'):
            param_grid = [param_grid]

        for p in param_grid:
            for name, v in p.items():
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    raise ValueError("Parameter array should be "
                                     "one-dimensional.")

                if (isinstance(v, str) or
                        not isinstance(v, (np.ndarray, Sequence))):
                    raise ValueError(
                        "Parameter values for parameter ({0}) need "
                        "to be a sequence (but not a string) or"
                        " np.ndarray.".format(name))

                if len(v) == 0:
                    raise ValueError(
                        "Parameter values for parameter ({0}) need "
                        "to be a non-empty sequence.".format(name))


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, optional (default=10)
        Number of parameter settings that are sampled.

    scoring : callable, dict or None, optional (default=None)
        A callable to evaluate the predictions on the test set. It should take
        3 parameters, estimator, x and y, and return a score (higher meaning
        better). For evaluating multiple metrics, give a dict with names as
        keys and callables as values. If None, the estimator's score method is
        used.

    cv : int or cv generator, optional (default=None)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `KFold`,
        - custom cv generator.

    refit : boolean, string, or callable, optional (default=True)
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer. ``best_score_`` is not returned if refit is callable.
        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    random_state : int, RandomState instance or None, optional, default=None
        Pseudo random number generator state used for random sampling of params
        in param_distributions. This is not passed to each estimator.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> import dislib as ds
    >>> from dislib.model_selection import RandomizedSearchCV
    >>> from dislib.classification import CascadeSVM
    >>> import numpy as np
    >>> import scipy.stats as stats
    >>> from sklearn import datasets
    >>>
    >>>
    >>> if __name__ == '__main__':
    >>>     x_np, y_np = datasets.load_iris(return_X_y=True)
    >>>     # Pre-shuffling required for CSVM
    >>>     p = np.random.permutation(len(x_np))
    >>>     x = ds.array(x_np[p], (30, 4))
    >>>     y = ds.array((y_np[p] == 0)[:, np.newaxis], (30, 1))
    >>>     param_distributions = {'c': stats.expon(scale=0.5),
    >>>                            'gamma': stats.expon(scale=10)}
    >>>     csvm = CascadeSVM()
    >>>     searcher = RandomizedSearchCV(csvm, param_distributions, n_iter=10)
    >>>     searcher.fit(x, y)
    >>>     searcher.cv_results_

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +---------+-------------+-------------------+---+---------------+
        | param_c | param_gamma | split0_test_score |...|rank_test_score|
        +=========+=============+===================+===+===============+
        |  0.193  |    1.883    |       0.82        |...|       3       |
        +---------+-------------+-------------------+---+---------------+
        |  1.452  |    0.327    |       0.81        |...|       2       |
        +---------+-------------+-------------------+---+---------------+
        |  0.926  |    3.452    |       0.94        |...|       1       |
        +---------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.82, 0.81, 0.94],
            'split1_test_score'  : [0.66, 0.75, 0.79],
            'split2_test_score'  : [0.82, 0.87, 0.84],
            ...
            'mean_test_score'    : [0.76, 0.84, 0.86],
            'std_test_score'     : [0.01, 0.20, 0.04],
            'rank_test_score'    : [3, 2, 1],
            'params'             : [{'c' : 0.193, 'gamma' : 1.883}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    """
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 cv=None, refit=True, random_state=None):
        super().__init__(estimator=estimator, scoring=scoring, cv=cv,
                         refit=refit)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        ps = ParameterSampler(self.param_distributions, self.n_iter,
                              random_state=self.random_state)
        evaluate_candidates(ps)
