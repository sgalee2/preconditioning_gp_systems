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

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x_ = torch.linspace(0, 1, 1500)
train_x = train_x_[0:100]
test_x = train_x_[100:]

# True function is sin(2*pi*x) with Gaussian noise
train_y_ = torch.sin(train_x_ * (2 * math.pi)) + torch.randn(train_x_.size()) * math.sqrt(0.04)
train_y = train_y_[0:100]
test_y = train_y_[100:]

lhood = gpytorch.likelihoods.GaussianLikelihood()
loss_fn = GP_nll
optim = torch.optim.Adam

base = base_model(train_x, train_y, lhood, gpytorch.means.ConstantMean(), gpytorch.kernels.MaternKernel())
model = GPRegressionModel(base, lhood, loss_fn, optim)

model.Fit(train_x, train_y, 0.1, 5)

mll = gpytorch.ExactMarginalLogLikelihood(lhood, model.model)
print(-mll(model.model(train_x), train_y))

model.eval()
print(-mll(model.model(test_x), test_y))