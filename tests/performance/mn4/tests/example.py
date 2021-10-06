import dislib as ds
import numpy as np
from dislib.model_selection import RandomizedSearchCV
from dislib.classification import CascadeSVM
from sklearn import datasets
import scipy.stats as stats

def main():
    x_np, y_np = datasets.load_iris(return_X_y=True)
    p = np.random.permutation(len(x_np))  # Pre-shuffling required for CSVM
    x = ds.array(x_np[p], (30, 4))
    y = ds.array((y_np[p] == 0)[:, np.newaxis], (30, 1))
    param_distributions = {'c': stats.expon(scale=0.5),
                           'gamma': stats.expon(scale=10)}
    csvm = CascadeSVM()
    searcher = RandomizedSearchCV(csvm, param_distributions, n_iter=10)
    searcher.fit(x, y)
    print(searcher.cv_results_)

if __name__ == '__main__':
    main()