'''
from dislib.cluster import DBSCAN
import dislib as ds
import numpy as np

def main():
    arr = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    x = ds.array(arr, block_size=(2, 2))
    dbscan = DBSCAN(eps=3, min_samples=2)
    y = dbscan.fit_predict(x)
    print(y.collect())

if __name__ == '__main__':
    main()
'''
import dislib as ds
import numpy as np
from dislib.cluster import DBSCAN
from dislib.model_selection import RandomizedSearchCV
from dislib.model_selection import GridSearchCV
from dislib.classification import CascadeSVM
from dislib.classification import RandomForestClassifier
from sklearn import datasets
import scipy.stats as stats
from uuid import uuid4

from pycompss.api.api import compss_wait_on, compss_wait_on_file
from pycompss.util.serialization.serializer import deserialize_from_file
import numpy as np

from dislib import random_array, array
from dislib.classification import CascadeSVM
from dislib.model_selection import GridSearchCV
from dislib.model_selection._validation import check_scorer


def CSVM():
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

def dbscan():
    param_grid = {'eps': (2, 4), 'min_samples': range(3, 8)}
    arr = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    dbscan = DBSCAN()
    x = ds.array(arr, block_size=(2, 2))
    y = [0,0,0,1,1,-1]
    searcher = GridSearchCV(dbscan, param_grid, scoring=score)
    searcher.fit(x, y)
    print(searcher.cv_results_)


def score(x, y, collect=False):
    count = 0
    sum = 0
    for (i,j) in (x,y):
        count += 1;
        if i == j: sum += 1
    if count > 0: return sum/count
    else: return 1

def RandomForest():
    x_np, y_np = datasets.load_iris(return_X_y=True)
    x = ds.array(x_np, (30, 4))
    y = ds.array(y_np[:, np.newaxis], (30, 1))
    param_grid = {'n_estimators': (2, 4), 'max_depth': range(3, 5)}
    rf = RandomForestClassifier()
    searcher = GridSearchCV(rf, param_grid)
    searcher.fit(x, y)
    print(searcher.cv_results_)
    print(searcher.scorer_)

def cascadeCSVM():
    x_np, y_np = datasets.load_iris(return_X_y=True)
    p = np.random.permutation(len(x_np))
    x = ds.array(x_np, (30, 4))
    y = ds.array((y_np[p] == 0)[:, np.newaxis], (30, 1))
    param_grid = {'c': (0.25, 0.5), 'gamma': (0.1, 0.2)}
    rf = CascadeSVM()
    searcher = GridSearchCV(rf, param_grid)
    searcher.fit(x, y)
    print(searcher.cv_results_)
    print(searcher.scorer_)


if __name__ == '__main__':
    cascadeCSVM()
