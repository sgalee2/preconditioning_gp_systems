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

base = base_model(train_x, train_y, lhood, gpytorch.means.ConstantMean(), gpytorch.kernels.RBFKernel() + gpytorch.kernels.MaternKernel())
model = GPRegressionModel(base, lhood, loss_fn, optim)

base_op = base(train_x).lazy_covariance_matrix 
lin_op = lhood(base(train_x)).lazy_covariance_matrix
ldet = exact_log_det(linop_cholesky(lin_op))
ldets = []

max_rank = min(int(0.9*df.N - 1), 500)

for i in range(1,max_rank):
    with gpytorch.settings.max_preconditioner_size(i):
        lin_op.preconditioner_override = Pivoted_Cholesky
        ldets.append(math.sqrt( (lin_op._preconditioner()[2].item() - ldet.item())**2 ))
        lin_op._q_cache = None
from matplotlib import pyplot as plt
plt.plot(ldets, label='Piv_chol')







ldets = []
for i in range(1,max_rank):
    with gpytorch.settings.max_preconditioner_size(i):
        lin_op.preconditioner_override = rSVD_Preconditioner
        ldets.append(math.sqrt( (lin_op._preconditioner()[2].item() - ldet.item())**2 ))
        lin_op._q_cache = None
        
plt.plot(ldets, label = 'rSVD')

ldets = []
for i in range(2,max_rank):
    with gpytorch.settings.max_preconditioner_size(i):
        lin_op.preconditioner_override = recursiveNystrom_Preconditioner
        ldets.append(math.sqrt( (lin_op._preconditioner()[2].item() - ldet.item())**2 ))
        lin_op._q_cache = None
        
plt.plot(ldets, label = 'rNys')        
plt.ylabel('$|\log|\hat{K}| - \log|\hat{P}||$')
plt.xlabel('Preconditioner quality')
plt.legend()