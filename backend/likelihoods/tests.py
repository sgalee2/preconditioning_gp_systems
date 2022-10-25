import torch, startup, gpytorch, math, sys

sys.path.append(r'C:\Users\adayr\OneDrive\Documents\preconditioning_gp_systems\PyTorch-LBFGS')
from functions.LBFGS import FullBatchLBFGS

from gpytorch.distributions import MultivariateNormal

from linear_operator.operators import AddedDiagLinearOperator , DiagLinearOperator , RootLinearOperator

from backend.conjugate_gradients.preconditioners.Preconditioners import Eig_Preconditioner, rSVD_Preconditioner, rSVD_Preconditioner_cuda, nystrom_SVD
from backend.functions.Functions import *
from matplotlib import pyplot as plt

from gpytorch import settings

from time import time

import urllib.request
import os.path
from scipy.io import loadmat
from math import floor

gpytorch.settings.cg_tolerance._set_value(1e-3)
gpytorch.settings.max_preconditioner_size._set_value(15)

# n = 5000
# train_x = torch.linspace(0.,25.,n)

# kernel = gpytorch.kernels.rbf_kernel.RBFKernel()
# kernel.lengthscale = 0.2

# K = kernel(train_x)
# train_y = MultivariateNormal(torch.zeros(n), K).sample()

import urllib.request
import os.path
from scipy.io import loadmat
from math import floor

if not os.path.isfile('../3droad.mat'):
    print('Downloading \'3droad\' UCI dataset...')
    urllib.request.urlretrieve('https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1', '../3droad.mat')

data = torch.Tensor(loadmat('../3droad.mat')['data'])

import numpy as np

N = data.shape[0]
# make train/val/test
n_train = int(5000)
train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
test_x, test_y = data[n_train:, :-1], data[n_train:, -1]

# normalize features
mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# normalize labels
mean, std = train_y.mean(),train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def model_params(self, x):
        """
        Returns mean and covariance for model at input point x.

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return mean_x, covar_x

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

train_x = train_x.cuda()
train_y = train_y.cuda()
model = model.cuda()
likelihood = likelihood.cuda()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1.)  # Includes GaussianLikelihood parameters

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def optim_step(train_x, train_y, precon_override = None):

    optimizer.zero_grad()
    
    diff, output = model.model_params(train_x)
    diff = train_y - diff
    diag = DiagLinearOperator(likelihood(train_y).variance)
    
    overrode_lt = AddedDiagLinearOperator(output, diag, preconditioner_override=precon_override)
    
    inv_quad, log_det = overrode_lt.inv_quad_log_det(diff, logdet=True)
    
    ll = -0.5 * sum([inv_quad, log_det, diff.size(-1) * math.log(2 * math.pi)])
    loss = -ll/diff.size(-1)
    loss.backward(retain_graph=True)
    return loss

loss_chol = []
loss_rsvd = []
timer_chol = []
timer_rsvd = []

print("Training...")
t1 = time()
for i in range(0):
    t_i = time()
    #loss_choli = optim_step(train_x, train_y)
    #t = time() - t1
    #timer_chol.append(t)
    output = model(train_x)
    loss_rsvdi = optim_step(train_x, train_y)
    optimizer.step()
    time_train_i = time() - t_i
    print("Training time for iteration:",time_train_i)
    print("Loss:",loss_rsvdi.item(),"\n")
    
    #loss_chol.append(loss_choli.cpu().item())
    loss_rsvd.append(loss_rsvdi.cpu().item())
time_train = time() - t1