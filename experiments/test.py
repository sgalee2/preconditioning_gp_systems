import startup
import gpytorch
import torch
import math

from time import time

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import GPRegressionModel, base_model
from backend.functions.Functions import *
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner
from backend.sampling.recursive_nystrom_gpytorch import recursiveNystrom

gpytorch.settings.max_preconditioner_size._set_value(50)
gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(10e-2)

df = get_regression_data('winewhite')

train_x, test_x, train_y, test_y = df.X_train, df.X_test, df.Y_train[0], df.Y_test[0]
n = int(0.9*df.N)

print(n)

lhood = gpytorch.likelihoods.GaussianLikelihood()
loss_fn = GP_nll
optim = torch.optim.Adam

base = base_model(train_x, train_y, lhood, gpytorch.means.ConstantMean(), gpytorch.kernels.MaternKernel())
model = GPRegressionModel(base, lhood, loss_fn, optim, cuda=True)

model.Fit(train_x.cuda(), train_y.cuda(), lr=0.1, iters=50)
model_rsvd.Fit(train_x.cuda(), train_y.cuda(), lr=0.1, iters=50)