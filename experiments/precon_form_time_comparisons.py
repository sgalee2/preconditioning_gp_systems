# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:56:37 2022

@author: adayr
"""

import startup

from all_params import *

import gpytorch
import torch

from time import time

from torch.utils.data import Dataset

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import GPRegressionModel, base_model
from backend.functions.Functions import *
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner, Pivoted_Cholesky
from backend.sampling.recursive_nystrom_gpytorch import recursiveNystrom

def train_data_loader(data_title):
    
    df = get_regression_data(data_title)

    train_x, train_y, N, D = df.X_train, df.Y_train, df.N, df.D
    
    return train_x, train_y, N, D

def data_to_cuda(train_x, train_y, model, likelihood):
    model = model.cuda()
    likelihood = likelihood.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda
    return train_x, train_y, model, likelihood

def define_model(mean_module, cov_module):
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = base_model(train_x, train_y, likelihood, mean_module, cov_module)
    
    return model, likelihood

def model_likelihood_covariance(location, preconditioner = None):
    
    loc_likelihood = likelihood(model(location))
    lin_op = loc_likelihood.lazy_covariance_matrix
    
    if preconditioner is not None:
        lin_op.preconditioner_override = preconditioner
        
    return lin_op

def precon_matmul(lin_op, vec):
    
    precon_func = lin_op._preconditioner()[0]
    sol = precon_fun(vec)
    return sol
    
if __name__ == '__main__':
    
    print("Testing preconditioner formation and computation times \n")
    
    preconditioners = [Pivoted_Cholesky, rSVD_Preconditioner, recursiveNystrom_Preconditioner, rSVD_Preconditioner_cuda]