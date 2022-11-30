# -*- coding: utf-8 -*-
"""
Comparisons of various norms for GP preconditioners.
"""

import startup, torch, gpytorch, linear_operator

from time import time
from matplotlib import pyplot as plt

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import base_model
from backend.functions.Functions import GP_nll
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner

gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(10e-4)


#set up model
df = get_regression_data('parkinsons')
df.shuffle()
train_x, test_x, train_y, test_y = df.X_train, df.X_test, df.Y_train, df.Y_test

likelihood, mean, covar = gpytorch.likelihoods.GaussianLikelihood(), gpytorch.means.ConstantMean(), gpytorch.kernels.MaternKernel()
model = base_model(train_x, train_y, likelihood, mean, covar)

lo = likelihood(model(train_x)).lazy_covariance_matrix
log_det_exact = 2*sum(torch.log(lo.cholesky().diag())).item()

def precon_params(lo_, preconditioner=None, rank=15):
    """
    

    Parameters
    ----------
    linear_operator : linear_operator
        regularised system.
    preconditioner : gp preconditioner, optional
        override strategy, if none PivChol is used. The default is None.
    rank : int, optional
        preconditioner quality. The default is 15.

    Returns
    -------
    pik : linear_operator
        P^{-1}K.
    P_mat : linear_operator
        P.
    log_det : fl64
        log|P|.

    """
    lo_._q_cache = None
    lo_.preconditioner_override=preconditioner
    with gpytorch.settings.max_preconditioner_size(rank):
        P, P_mat, log_det = lo_._preconditioner()
    
    return P, P_mat, log_det

preconditioners = [None,
                   rSVD_Preconditioner,
                   recursiveNystrom_Preconditioner]

precon_sizes = [5, 6, 7, 8, 9,
                10, 20, 30, 40,
                50, 75,
                100, 200, 300, 400,
                500, 750,1000]

results_chol =[]
results_rsvd = []
results_rnys = []

# for rank in precon_sizes:
#     params = PiK(lo, preconditioner=None, rank=rank)
#     results_chol.append(params)
#     params = PiK(lo, preconditioner=rSVD_Preconditioner, rank=rank)
#     results_rsvd.append(params)
#     params = PiK(lo, preconditioner=recursiveNystrom_Preconditioner, rank=rank)
#     results_rnys.append(params)

    
