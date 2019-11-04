import numpy as np
from pycompss.api.api import compss_wait_on
from pylab import scatter, plot, show

import dislib as ds
from dislib.regression import LinearRegression


def main():
    """
    Linear regression example with plot
    """

    # Example data
    x = np.array([1000, 4000, 5000, 4500, 3000, 4000, 9000, 11000, 15000,
                  12000, 7000, 3000])
    y = np.array([9914, 40487, 54324, 50044, 34719, 42551, 94871, 118914,
                  158484, 131348, 78504, 36284])
    x_ds = ds.array(x[:, np.newaxis], (4, 1))
    y_ds = ds.array(y[:, np.newaxis], (4, 1))
    reg = LinearRegression()
    reg.fit(x_ds, y_ds)
    reg.coef_ = compss_wait_on(reg.coef_)
    reg.intercept_ = compss_wait_on(reg.intercept_)
    print(reg.coef_, reg.intercept_)

    # plot_result:
    scatter(x, y, marker='x')
    x_mesh = np.linspace(min(x), max(x), 1000)
    plot(x_mesh, [reg.coef_*x + reg.intercept_ for x in x_mesh])
    show()


if __name__ == "__main__":
    main()
