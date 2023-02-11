# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 12:38:31 2023

@author: adayr
"""

import startup
import gpytorch
import torch
import math

from time import time
from all_params import *

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import GPRegressionModel, base_model
from backend.functions.Functions import *

gpytorch.settings.max_preconditioner_size._set_value(0)
gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(10e-2)


df = get_regression_data('protein')

train_x, test_x, train_y, test_y = df.X_train, df.X_test, df.Y_train[0], df.Y_test[0]
n = int(0.9*df.N)

lhood = gpytorch.likelihoods.GaussianLikelihood()
loss_fn = exact_GP_nll
optim = torch.optim.Adam

base = base_model(train_x, train_y, lhood, gpytorch.means.ConstantMean(), gpytorch.kernels.MaternKernel(1.5))
model = GPRegressionModel(base, lhood, loss_fn, optim)

model.Fit(train_x, train_y, 0.1, 50)

mll = gpytorch.ExactMarginalLogLikelihood(lhood, model.model)
with gpytorch.settings.max_cholesky_size(100000):
    print(-mll(model.model(train_x), train_y))

model.eval()
with gpytorch.settings.max_cholesky_size(100000):
    print(-mll(model.model(test_x), test_y))
    
    gpytorch.settings