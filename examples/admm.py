import sys
import time

from dislib.optimization import ADMM
from dislib.regression import Lasso


def main():
    start = time.time()
    n = int(sys.argv[1])

    optimizer = ADMM(rho=1, abstol=1e-4, reltol=1e-2)
    lasso = Lasso(n=n, max_iter=500, lmbd=1e-3, optimizer=optimizer)
    optimizer.obj_func = lasso.objective

    z = lasso.fit()

    print("\nTotal elapsed time: %s" % str((time.time() - start) / 100))
    np.savetxt("Solution.COMPSs.txt", z)

    print(lasso.predict(np.random.rand(50, 50)))


if __name__ == '__main__':
    main()
