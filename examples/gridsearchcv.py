from sklearn import datasets
import pandas as pd
import dislib as ds
import numpy as np
from dislib.classification import RandomForestClassifier
from dislib.model_selection import GridSearchCV


def main():
    x_np, y_np = datasets.load_iris(return_X_y=True)
    x = ds.array(x_np, (30, 4))
    y = ds.array(y_np[:, np.newaxis], (30, 1))
    parameters = {'n_estimators': (1, 2, 4, 8, 16, 32),
                  'max_depth': range(3, 5)}
    rf = RandomForestClassifier()
    searcher = GridSearchCV(rf, parameters, cv=5)
    np.random.seed(0)
    searcher.fit(x, y)
    print(searcher.cv_results_['params'])
    print(searcher.cv_results_['mean_test_score'])
    pd_df = pd.DataFrame.from_dict(searcher.cv_results_)
    print(pd_df[['params', 'mean_test_score']])
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(pd_df)
    print(searcher.best_estimator_)
    print(searcher.best_score_)
    print(searcher.best_params_)
    print(searcher.best_index_)
    print(searcher.scorer_)
    print(searcher.n_splits_)


if __name__ == "__main__":
    main()
