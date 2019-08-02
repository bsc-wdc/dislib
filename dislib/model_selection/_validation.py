import numbers
import numpy as np


def fit_and_score(estimator, train_ds, validation_ds, scorer, parameters,
                  fit_params):

    if parameters is not None:
        estimator.set_params(**parameters)

    estimator.fit(train_ds, **fit_params)
    test_scores = _score(estimator, validation_ds, scorer)
    return [test_scores]


def _score(estimator, dataset, scorers):
    """Return a dict of scores"""
    scores = {}

    for name, scorer in scorers.items():
        score = scorer(estimator, dataset)
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
    """Aggregate the list of dict to dict of np ndarray
    The aggregated output of _fit_and_score will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}
    Parameters
    ----------
    scores : list of dict of string keys
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.
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
