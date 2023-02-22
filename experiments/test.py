import startup
import gpytorch
import torch
import math

from time import time

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import GPRegressionModel, base_model
from backend.functions.Functions import *
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner, Pivoted_Cholesky
from backend.sampling.recursive_nystrom_gpytorch import recursiveNystrom

gpytorch.settings.max_preconditioner_size._set_value(100)
gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(0.1)
gpytorch.settings.preconditioner_tolerance._set_value(1e-6)

df = get_regression_data('energy')

train_x, train_y = df.X_train, df.Y_train

lhood = gpytorch.likelihoods.GaussianLikelihood()
loss_fn = GP_nll
optim = torch.optim.Adam

base = base_model(train_x, train_y, lhood, gpytorch.means.ConstantMean(), gpytorch.kernels.MaternKernel())
model = GPRegressionModel(base, lhood, loss_fn, optim)

base_op = base(train_x).lazy_covariance_matrix 
lin_op = lhood(base(train_x)).lazy_covariance_matrix
ldet = exact_log_det(linop_cholesky(lin_op))
ldets = []

max_rank = min(int(0.9*df.N - 1), 50)

train_x, train_y, likelihood, model = train_x.cuda(), train_y.cuda(), lhood.cuda(), base.cuda()
cov = likelihood(model(train_x)).lazy_covariance_matrix