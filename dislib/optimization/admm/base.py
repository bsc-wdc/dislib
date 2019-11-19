"""
ADMM Lasso

@Authors: Aleksandar Armacki and Lidija Fodor
@Affiliation: Faculty of Sciences, University of Novi Sad, Serbia

This work is supported by the I-BiDaaS project, funded by the European
Commission under Grant Agreement No. 780787.
"""

import functools

import cvxpy as cp
import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task


class ADMM(object):
    """
    Alternating Direction Method of Multipliers (ADMM) solver. ADMM is
    renowned for being well suited
    to the distributed settings, for its guaranteed convergence and general
    robustness with respect
    to the parameters. Additionally, the algorithm has a generic form that
    can be easily adapted to a
    wide range of machine learning problems with only minor tweaks in the code.
    :param rho: The penalty parameter for constraint violation in ADMM
    :param abstol: The absolute tolerance used to calculate the early
    stoppage criterion for ADMM
    :param reltol: The relative tolerance used to calculate the early
    stoppage criterion for ADMM
    :param warm_start: cvxpy warm start option
    :param objective_fn: objective function
    """

    def __init__(self, z, u, n, rho=1, soft_thres=None, abstol=1e-4,
                 reltol=1e-2, warm_start=False, objective_fn=None):
        self.rho = rho
        self.abstol = abstol
        self.reltol = reltol
        self.warm_start = warm_start
        self.objective_fn = objective_fn
        self.N = n
        self.soft_thres = soft_thres
        self.converged = False
        self.z = z
        self.u = u
        self.x = None

    @staticmethod
    def _soft_thresholding(v, k):
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

    def step(self, data_chunk, target_chunk, n, N):
        # update x
        self.x = list(
            map(functools.partial(x_update, self.z, self.objective_fn,
                                  self.warm_start), data_chunk, target_chunk,
                self.u))
        self.x = compss_wait_on(self.x)

        z_old = self.z

        # update z
        self.z = self._soft_thresholding(
            np.mean(self.x, axis=0) + np.mean(self.u, axis=0),
            self.soft_thres)

        # update u
        self.u = list(map(functools.partial(u_update, self.z), self.u, self.x))
        self.u = compss_wait_on(self.u)

        nxstack = np.sqrt(np.sum(np.linalg.norm(self.x, axis=1) ** 2))
        nystack = np.sqrt(np.sum(np.linalg.norm(self.u, axis=1) ** 2))

        # termination check
        dualres = np.sqrt(N) * self.rho * np.linalg.norm(self.z - z_old)
        prires = np.sqrt(
            np.sum(np.linalg.norm(np.array(self.x) - z_old, axis=1) ** 2))

        eps_pri = (np.sqrt(N * n)) * self.abstol + self.reltol * \
                  (max(nxstack, np.sqrt(N) * np.linalg.norm(self.z)))
        eps_dual = np.sqrt(N * n) * self.abstol + self.reltol * nystack

        if prires <= eps_pri and dualres <= eps_dual:
            self.converged = True


@task(returns=np.array)
def x_update(z, objective_fn, warm_start, a, b, u):
    n = a.shape[1]
    sol = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(objective_fn(a, b, sol, z, u)))
    problem.solve(warm_start=warm_start)
    return sol.value


@task(returns=np.array)
def u_update(z, u, x):
    return u + x - z
