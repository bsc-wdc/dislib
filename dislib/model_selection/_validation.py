import numbers

import numpy as np
from pycompss.api.task import task
from itertools import product
from sklearn import clone
from pycompss.api.parameter import COMMUTATIVE, COLLECTION_IN, Type, Depth
from pycompss.api.exceptions import COMPSsException

@task()
def eval_candidates(base_estimator, candidate_params, cv, x, y, scorers, fit_params):
    return [fit_and_score(clone(base_estimator), train, validation,
                   scorer=scorers, parameters=parameters,
                   fit_params=fit_params)
     for parameters, (train, validation)
     in product(candidate_params, cv.split(x, y))]

def fit_and_score(estimator, train_ds, validation_ds, scorer, parameters,
                  fit_params):
    if parameters is not None:
        estimator.set_params(**parameters)

    x_train, y_train = train_ds
    estimator.fit(x_train, y_train, **fit_params)
    x_test, y_test = validation_ds
    test_scores = _score(estimator, x_test, y_test, scorer)
    return [test_scores]

def early_stopping(value, known_values, threshold):
    return threshold_stopping(threshold, value) or median_stopping(known_values, value)

def threshold_stopping(thresh, value):
    return value >= thresh

def median_stopping(known_values, value):
    if len(known_values) > 3:
        return value <= median(known_values)
    else:
        return False

@task(out_files=COMMUTATIVE, priority=True)
def check_convergence(out_files, out):
    out_files.append(out)
    print("OUT VAL " + str(out) + " OUT FILES  " + str(out_files))
    if threshold_stopping(threshold, out) or median_stopping(out_files, out):
      raise COMPSsException("XXX Exception " + str(out_files))

@task()
def fit_and_score_task(estimator, train_x, train_y, test_x, test_y, scorer, parameters, size_x, size_y,
                  fit_params):
    import dislib as ds
    if parameters is not None:
        estimator.set_params(**parameters)
    x_train = ds.array(train_x, size_x)
    y_train = ds.array(train_y, size_y)
    estimator.fit(x_train, y_train, **fit_params)

    x_test = ds.array(test_x, size_x)
    y_test = ds.array(test_y, size_y)
    scores = estimator.score(x_test, y_test, True)
    return [{"score": scores}]


@task(is_distributed=True, train_x={Type: COLLECTION_IN, Depth: 4}, train_y={Type: COLLECTION_IN, Depth: 4}, test_x={Type: COLLECTION_IN, Depth: 4}, test_y={Type: COLLECTION_IN, Depth: 4})
def fit_and_score_task_reassemble(estimator, train_x, train_y, test_x, test_y, shape_t_x, shape_t_y, shape_v_x, shape_v_y, parameters, size_x, size_y,
                  fit_params):
    from dislib.data.array import reassemblearray
    import sys
    if parameters is not None:
        estimator.set_params(**parameters)

    x_train = reassemblearray(train_x, shape_t_x, size_x)
    y_train = reassemblearray(train_y, shape_t_y, size_y)
    #print(XTRAIN " + str(x_train))

    estimator.fit(x_train, y_train, **fit_params)

    x_test = reassemblearray(test_x, shape_v_x, size_x)
    y_test = reassemblearray(test_y, shape_v_y, size_y)
    scores = estimator.score(x_test, y_test, True)
    return [{"score": scores}]

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
