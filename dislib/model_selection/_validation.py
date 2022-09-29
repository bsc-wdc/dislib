import numbers

from dislib.data.array import Array
from pycompss.api.task import task
from pycompss.api.parameter import INOUT, Depth, Type, COLLECTION_IN
import sys
import numpy as np


def fit(estimator, train_ds, parameters, fit_params):
    if parameters is not None:
        estimator.set_params(**parameters)
    x_train, y_train = train_ds
    estimator.fit(x_train, y_train, **fit_params)
    return estimator


def score_func(estimator, validation_ds, scorer):
    x_test, y_test = validation_ds
    test_scores = _score(estimator, x_test, y_test, scorer)

    return [test_scores]


@task(est=INOUT, blocks_x={Type: COLLECTION_IN, Depth: 2},
      blocks_y={Type: COLLECTION_IN, Depth: 2})
def fit_sklearn_estimator(est, blocks_x, blocks_y, **fit_params):
    x = Array._merge_blocks(blocks_x)
    y = Array._merge_blocks(blocks_y)
    return est.fit(x, y, **fit_params)


@task(blocks_x={Type: COLLECTION_IN, Depth: 2},
      blocks_y={Type: COLLECTION_IN, Depth: 2},
      returns=1)
def score_sklearn_estimator(est, scorer,  blocks_x, blocks_y):
    x = Array._merge_blocks(blocks_x)
    y = Array._merge_blocks(blocks_y)
    return _score(est, x, y, scorer)


def execute_simulation(simulation, **parameters):
    sys.stdout.write("PARAMETERS")
    for param in parameters:
        sys.stdout.write(str(param))
    return simulation(**parameters)


def simulation_execution(simulation, parameters,
                         simulation_params, number_simulations):
    simulations_result = []
    if parameters is not None:
        for _ in range(number_simulations):
            simulations_result.append(execute_simulation(simulation,
                                                         **parameters,
                                                         **simulation_params))
    return simulations_result


def sklearn_fit(estimator, train_ds,
                parameters, fit_params):
    if parameters is not None:
        estimator.set_params(**parameters)
    x_train, y_train = train_ds

    return fit_sklearn_estimator(estimator, x_train._blocks,
                                 y_train._blocks, **fit_params)


def sklearn_score(estimator, validation_ds, scorer):
    x_test, y_test = validation_ds
    test_scores = score_sklearn_estimator(estimator, scorer,
                                          x_test._blocks, y_test._blocks)

    return [test_scores]


def _score(estimator, x, y, scorers):
    """Return a dict of scores"""
    scores = {}

    for name, scorer in scorers.items():
        score = scorer(estimator, x, y)
        scores[name] = score
    return scores


def validate_score(score, name):
    if not isinstance(score, numbers.Number) and \
            not (isinstance(score, np.ndarray) and len(score.shape) == 0):
        raise ValueError("scoring must return a number, got %s (%s) "
                         "instead. (scorer=%s)"
                         % (str(score), type(score), name))
    return score


def aggregate_score_dicts(scores):
    """Aggregate the results of each scorer
    Example
    -------
    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]
    >>> aggregate_score_dicts(scores)
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {key: np.asarray([score[key] for score in scores])
            for key in scores[0]}


def check_scorer(estimator, scorer):
    if scorer is None:
        if hasattr(estimator, 'score'):
            def _passthrough_scorer(estimator, *args, **kwargs):
                """Function that wraps estimator.score"""
                return estimator.score(*args, **kwargs)
            return _passthrough_scorer
        else:
            raise TypeError(
                "If a scorer is None, the estimator passed should have a "
                "'score' method. The estimator %r does not." % estimator)
    elif callable(scorer):
        return scorer
    raise ValueError("Invalid scorer %r" % scorer)
