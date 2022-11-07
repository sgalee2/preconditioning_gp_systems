import startup
import gpytorch
import torch

from time import time

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import GPRegressionModel
from backend.functions.Functions import GP_nll
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner

gpytorch.settings.max_preconditioner_size._set_value(100)
gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(10e-2)

df = get_regression_data('parkinsons')

tens = torch.Tensor
train_x, test_x, train_y, test_y = df.X_train, df.X_test, df.Y_train[0], df.Y_test[0]

class ExactGPR(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, mean, covar):
        super(ExactGPR, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = mean
        self.covar_module = covar
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

mean, covar = gpytorch.means.ConstantMean() , gpytorch.kernels.RBFKernel()
likelihood = gpytorch.likelihoods.GaussianLikelihood()

base_model = ExactGPR(train_x, train_y, likelihood, mean, covar)

mvn = likelihood(base_model(train_x))
lo = mvn.lazy_covariance_matrix

from recursive_nystrom.recursive_nystrom_gpytorch import recursiveNystrom

indices = recursiveNystrom(lo._linear_op, 10)
ks = lo[:,indices]
sks = ks[indices,:]
P = ks @ torch.linalg.inv(sks.evaluate()) @ ks.T
vals, vecs = torch.linalg.eig(P)
