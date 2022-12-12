import startup
import gpytorch
import torch
import math

from time import time

from data.data import *
from backend.models.regression_model import GPRegressionModel, base_model
from backend.functions.Functions import GP_nll, exact_log_det
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner
from backend.sampling.recursive_nystrom_gpytorch import recursiveNystrom

gpytorch.settings.max_preconditioner_size._set_value(50)
gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(10e-4)

df = get_regression_data('winered')

train_x, test_x, train_y, test_y = df.X_train, df.X_test, df.Y_train, df.Y_test
n = int(0.9*df.N)

lhood = gpytorch.likelihoods.GaussianLikelihood()
base = base_model(train_x, train_y, lhood, gpytorch.means.ConstantMean(), gpytorch.kernels.MaternKernel())
lo = lhood(base(train_x)).lazy_covariance_matrix
lo_ = lhood(base(train_x)).lazy_covariance_matrix
lo_.preconditioner_override=rSVD_Preconditioner

K = lo.linear_ops[0]

k = gpytorch.settings.max_preconditioner_size.value()
index,_ = recursiveNystrom(K, 500, return_leverage_score=True)

