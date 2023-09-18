from dislib.regression.linear.base import LinearRegression
from dislib.regression.lasso.base import Lasso
from dislib.trees.mmap.forest import (RandomForestRegressor as
                                      RandomForestRegressorMMap)
from dislib.trees.distributed.forest import RandomForestRegressor

__all__ = ["LinearRegression", "Lasso",
           "RandomForestRegressorMMap", "RandomForestRegressor"]
