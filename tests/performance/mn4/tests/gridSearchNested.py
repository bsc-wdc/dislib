import dislib as ds
import numpy as np
from dislib.model_selection import GridSearchCV, RandomizedSearchCV
from dislib.classification import CascadeSVM
from sklearn import datasets
from pycompss.api.task import task
from dislib import array
import time

import sys

@task()
def main():

    print("SYYYS " + str(sys.argv))

    a = int(sys.argv[1])
    b = int(sys.argv[2])
    init = time.time()
    x_np, y_np = datasets.load_iris(return_X_y=True)
    #x_np, y_np = datasets.load_breast_cancer(return_X_y=True)

    p = np.random.permutation(len(x_np))
    x = ds.array(x_np[p], (30, 4))
    y = ds.array((y_np[p] == 0)[:, np.newaxis], (30, 1))
    # csvm = CascadeSVM(c=10000, gamma=0.01, max_iter=1)
    param_grid = {'c': range(a), 'gamma': range(b)}
    param_grid = {'c':(10, 20)}
    csvm = CascadeSVM(max_iter=1, check_convergence=True)

    searcher = RandomizedSearchCV(csvm, param_grid, n_iter=3)
    searcher.fit(x, y, type=2, threshold=0.3)
    print("CV RESUTLS " + str(searcher.cv_results_))
    print("CV RESUTLS KEYS " + str(searcher.cv_results_.keys()))
    searcher.scorer_
    print("Grid search time " + str(time.time()-init))

if __name__ == '__main__':
    main()
