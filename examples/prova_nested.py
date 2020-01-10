from uuid import uuid4

from pycompss.api.api import compss_wait_on, compss_wait_on_file
from pycompss.util.serialization.serializer import deserialize_from_file
import numpy as np

from dislib import random_array, array
from dislib.classification import CascadeSVM
from dislib.model_selection import GridSearchCV
from dislib.model_selection._nested_search import evaluate_candidate_nested
from dislib.model_selection._validation import check_scorer


def main():
    train_x = random_array((10, 10), (5, 5))
    train_y = array(np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])[:, np.newaxis],
                    (5, 1))
    test_x = random_array((10, 10), (5, 5))
    test_y = array(np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])[:, np.newaxis],
                   (5, 1))
    data = (train_x, train_y), (test_x, test_y)
    scorers = {"score": check_scorer(CascadeSVM(), None)}
    id1 = str(uuid4())
    id2 = str(uuid4())
    out = evaluate_candidate_nested(id1, CascadeSVM(), scorers,
                                    {'cascade_arity': 4}, {}, data)
    out2 = evaluate_candidate_nested(id2, CascadeSVM(), scorers,
                                     {'cascade_arity': 4}, {}, data)
    print(compss_wait_on(out))
    print(compss_wait_on(out2))
    compss_wait_on_file(id1)
    compss_wait_on_file(id2)
    print(deserialize_from_file(id1))
    print(deserialize_from_file(id2))


if __name__ == "__main__":
    main()
