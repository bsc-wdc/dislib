"""
ADMM Lasso

@Authors: Aleksandar Armacki and Lidija Fodor
@Affiliation: Faculty of Sciences, University of Novi Sad, Serbia

This work is supported by the I-BiDaaS project, funded by the European
Commission under Grant Agreement No. 780787.
"""
from pycompss.api.constraint import constraint

try:
    import cvxpy as cp
except ImportError:
    import warnings
    warnings.warn('Cannot import cvxpy module. ADMM estimator will not work.')
import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import Type, Depth, COLLECTION_IN, COLLECTION_OUT
from pycompss.api.task import task
from sklearn.base import BaseEstimator

import dislib as ds
from dislib.data.array import Array
from dislib.utils.base import _paired_partition


class ADMM(BaseEstimator):
    """ Alternating Direction Method of Multipliers (ADMM) solver. ADMM is
    renowned for being well suited to the distributed settings [1]_, for its
    guaranteed convergence and general robustness with respect
    to the parameters. Additionally, the algorithm has a generic form that
    can be easily adapted to a wide range of machine learning problems with
    only minor tweaks in the code.

    Parameters
    ----------
    loss_fn : func
        Loss function.
    k : float
        Soft thresholding value.
    rho : float, optional (default=1)
        The penalty parameter for constraint violation.
    max_iter : int, optional (default=100)
        Maximum number of iterations to perform.
    atol : float, optional (default=1e-4)
        The absolute tolerance used to calculate the early stop criterion.
    rtol : float, optional (default=1e-2)
        The relative tolerance used to calculate the early stop criterion.
    verbose : boolean, optional (default=False)
        Whether to print information about the optimization process.

    Attributes
    ----------
    z_ : ds-array shape=(1, n_features)
        Computed z.
    n_iter_ : int
        Number of iterations performed.
    converged_ : boolean
        Whether the optimization converged.

    References
    ----------
    .. [1] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein (2011).
        Distributed Optimization and Statistical Learning via the Alternating
        Direction Method of Multipliers. In Foundations and Trends in Machine
        Learning, 3(1):1–122.
    """

    def __init__(self, loss_fn, k, rho=1, max_iter=100, rtol=1e-2, atol=1e-4,
                 verbose=False):
        self.rho = rho
        self.atol = atol
        self.rtol = rtol
        self.loss_fn = loss_fn
        self.k = k
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, x, y):
        """
        Fits the model with training data.

        Parameters
        ----------
        x : ds-array, shape=(n_samples, n_features)
            Training samples.
        y : ds-array, shape=(n_samples, 1)
            Class labels of x.

        Returns
        -------
        self : ADMM
        """
        if not x._is_regular():
            x_reg = x.rechunk(x._reg_shape)
        else:
            x_reg = x

        self._init_model(x_reg)

        while not self.converged_ and self.n_iter_ < self.max_iter:
            self._step(x_reg, y)
            self.n_iter_ += 1

            if self.verbose:
                print("Iteration ", self.n_iter_)

        z_blocks = [object() for _ in range(x_reg._n_blocks[1])]
        _split_z(self._z, x._reg_shape[1], z_blocks)
        self.z_ = Array([z_blocks], (1, x._reg_shape[1]), (1, x._reg_shape[1]),
                        (1, x.shape[1]), False)

        return self

    def _init_model(self, x):
        n_features = x.shape[1]

        self.converged_ = False
        self.n_iter_ = 0
        self._z = np.zeros(n_features)
        # u has one row per each row-block in x
        self._u = ds.zeros((x._n_blocks[0], n_features), (1, x._reg_shape[1]))

    def _step(self, x, y):
        # update w
        self._w_step(x, y)

        z_old = self._z

        # update z
        self._z_step()

        # update u
        self._u_step()

        # after norm in axis=1 and sum in axis=0, these should be ds-arrays
        # of a single element, so we keep the only block
        nxstack = (self._w.norm(axis=1) ** 2).sum().sqrt()
        nystack = (self._u.norm(axis=1) ** 2).sum().sqrt()

        # termination check
        n_samples, n_features = self._u.shape
        dualres = _compute_dual_res(n_samples, self.rho, self._z, z_old)
        prires = self._compute_primal_res(z_old)
        n_total = n_samples * n_features

        self.converged_ = _check_convergence(prires._blocks[0][0], dualres,
                                             n_samples, n_total,
                                             nxstack._blocks[0][0],
                                             nystack._blocks[0][0],
                                             self.atol, self.rtol, self._z)
        self.converged_ = compss_wait_on(self.converged_)

    def _compute_primal_res(self, z_old):
        blocks = []

        for w_hblock in self._w._iterator():
            out_blocks = [object() for _ in range(self._w._n_blocks[1])]
            _substract(w_hblock._blocks, z_old, out_blocks)
            blocks.append(out_blocks)

        prires = Array(blocks, self._w._reg_shape, self._w._reg_shape,
                       self._w.shape, self._w._sparse)

        # this should be a ds-array of a single element. We return only the
        # block
        return (prires.norm(axis=1) ** 2).sum().sqrt()

    def _u_step(self):
        u_blocks = []

        for u_hblock, w_hblock in zip(self._u._iterator(),
                                      self._w._iterator()):
            out_blocks = [object() for _ in range(self._u._n_blocks[1])]
            _update_u(self._z, u_hblock._blocks, w_hblock._blocks, out_blocks)
            u_blocks.append(out_blocks)

        r_shape = self._u._reg_shape
        shape = self._u.shape
        self._u = Array(u_blocks, r_shape, r_shape, shape, self._u._sparse)

    def _z_step(self):
        w_mean = self._w.mean(axis=0)
        u_mean = self._u.mean(axis=0)
        self._z = _soft_thresholding(w_mean._blocks, u_mean._blocks, self.k)

    def _w_step(self, x, y):
        w_blocks = []

        for xy_hblock, u_hblock in zip(_paired_partition(x, y),
                                       self._u._iterator()):
            x_hblock, y_hblock = xy_hblock
            w_hblock = [object() for _ in range(x._n_blocks[1])]
            x_blocks = x_hblock._blocks
            y_blocks = y_hblock._blocks
            u_blocks = u_hblock._blocks

            _update_w(x_blocks, y_blocks, self._z, u_blocks, self.rho,
                      self.loss_fn, w_hblock)
            w_blocks.append(w_hblock)

        r_shape = self._u._reg_shape
        self._w = Array(w_blocks, r_shape, r_shape, self._u.shape, x._sparse)


@constraint(computing_units="${ComputingUnits}")
@task(z_blocks={Type: COLLECTION_OUT, Depth: 1})
def _split_z(z, block_size, z_blocks):
    for i in range(len(z_blocks)):
        z_blocks[i] = z[i * block_size: (i + 1) * block_size]


@constraint(computing_units="${ComputingUnits}")
@task(x_blocks={Type: COLLECTION_IN, Depth: 2},
      y_blocks={Type: COLLECTION_IN, Depth: 2},
      u_blocks={Type: COLLECTION_IN, Depth: 2},
      w_blocks={Type: COLLECTION_OUT, Depth: 1})
def _update_w(x_blocks, y_blocks, z, u_blocks, rho, loss, w_blocks):
    x_np = Array._merge_blocks(x_blocks)
    y_np = np.squeeze(Array._merge_blocks(y_blocks))
    u_np = np.squeeze(Array._merge_blocks(u_blocks))

    w_new = cp.Variable(x_np.shape[1])

    problem = cp.Problem(cp.Minimize(_objective(loss, x_np, y_np, w_new, z,
                                                u_np, rho)))
    problem.solve()
    status = problem.status

    if 'infeasible' in status or 'unbounded' in status:
        raise Exception("Cannot solve the problem. CVXPY status: %s" % status)

    w_np = w_new.value
    n_cols = x_blocks[0][0].shape[1]

    for i in range(len(w_blocks)):
        w_blocks[i] = w_np[i * n_cols:(i + 1) * n_cols].reshape(1, -1)


def _objective(loss, x, y, w, z, u, rho):
    reg = cp.norm(w - z + u, p=2) ** 2
    return loss(x, y, w) + (rho / 2) * reg


@constraint(computing_units="${ComputingUnits}")
@task(w_blocks={Type: COLLECTION_IN, Depth: 2},
      u_blocks={Type: COLLECTION_IN, Depth: 2},
      returns=np.array)
def _soft_thresholding(w_blocks, u_blocks, k):
    w_mean = np.squeeze(Array._merge_blocks(w_blocks))
    u_mean = np.squeeze(Array._merge_blocks(u_blocks))
    v = w_mean + u_mean

    z = np.zeros(v.shape)
    for i in range(z.shape[0]):
        if np.abs(v[i]) <= k:
            z[i] = 0
        else:
            if v[i] > k:
                z[i] = v[i] - k
            else:
                z[i] = v[i] + k
    return z


@constraint(computing_units="${ComputingUnits}")
@task(u_blocks={Type: COLLECTION_IN, Depth: 2},
      w_blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_OUT, Depth: 1})
def _update_u(z, u_blocks, w_blocks, out_blocks):
    u_np = np.squeeze(Array._merge_blocks(u_blocks))
    w_np = np.squeeze(Array._merge_blocks(w_blocks))
    u_new = u_np + w_np - z
    n_cols = u_blocks[0][0].shape[1]

    for i in range(len(out_blocks)):
        out_blocks[i] = u_new[i * n_cols: (i + 1) * n_cols].reshape(1, -1)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _compute_dual_res(n_samples, rho, z, z_old):
    return np.sqrt(n_samples) * rho * np.linalg.norm(z - z_old)


@constraint(computing_units="${ComputingUnits}")
@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_OUT, Depth: 1})
def _substract(blocks, z, out_blocks):
    w_np = Array._merge_blocks(blocks) - z
    n_cols = blocks[0][0].shape[1]

    for i in range(len(out_blocks)):
        out_blocks[i] = w_np[i * n_cols: (i + 1) * n_cols].reshape(1, -1)


@constraint(computing_units="${ComputingUnits}")
@task(returns=bool)
def _check_convergence(prires, dualres, n_samples, n_total, nxstack,
                       nystack, abstol, reltol, z):
    eps_pri = (np.sqrt(n_total)) * abstol + reltol * (
        max(nxstack, np.sqrt(n_samples) * np.linalg.norm(z)))
    eps_dual = np.sqrt(n_total) * abstol + reltol * nystack

    if prires <= eps_pri and dualres <= eps_dual:
        return True

    return False
