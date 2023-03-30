import startup

from all_params import *

import gpytorch
import torch

from torch.utils.data import Dataset

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import GPRegressionModel, base_model
from backend.functions.Functions import *
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner, Pivoted_Cholesky
from backend.sampling.recursive_nystrom_gpytorch import recursiveNystrom

gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(0.1)

def train_data_loader(data_title):
    
    df = get_regression_data(data_title)

    train_x, train_y, N, D = df.X_train, df.Y_train, df.N, df.D
    
    return train_x, train_y, N, D

def data_to_cuda(train_x, train_y):
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    return train_x, train_y

def data_to_cpu(train_x, train_y):
    train_x = train_x.cpu()
    train_y = train_y.cpu()
    return train_x, train_y

def define_model(mean_module, cov_module):
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = base_model(train_x, train_y, likelihood, mean_module, cov_module)
    
    return model, likelihood

def model_to_cuda(model, likelihood):
    model = model.cuda()
    likelihood = likelihood.cuda()
    return model, likelihood

def model_to_cpu(model, likelihood):
    model = model.cpu()
    likelihood = likelihood.cpu()
    return model, likelihood

def model_likelihood_covariance(location, preconditioner = None):
    
    loc_likelihood = likelihood(model(location))
    lin_op = loc_likelihood.lazy_covariance_matrix
    
    if preconditioner is not None:
        lin_op.preconditioner_override = preconditioner
        
    return lin_op

def precon_matmul(lin_op, vec):
    
    lin_op._q_cache = None
    precon_func = lin_op._preconditioner()[0]
    sol = precon_func(vec)
    return sol

def time_solve(lin_op, vec):
    
    from time import time
    
    t1 = time()
    sol = precon_matmul(lin_op, vec)
    t_sol = time() - t1
    
    return sol, t_sol
    
if __name__ == '__main__':
    
    
    print("Testing preconditioner formation and computation times \n")
    
    preconditioners = [Pivoted_Cholesky, rSVD_Preconditioner, rSVD_Preconditioner_cuda, Pivoted_Cholesky]
    precon_names = ['Pivoted Cholesky', 'Randomised SVD', 'Randomised SVD CUDA', 'Pivoted Cholesky CUDA']
    
    train_x, train_y, N, D = train_data_loader('naval')
    model, likelihood = define_model(gpytorch.means.ConstantMean(), gpytorch.kernels.MaternKernel(0.5))
    
    trials = 2
    times = torch.zeros(trials, 4)
    j = 0
    
    for i in range(trials):
        
        for precons in enumerate(preconditioners):
        
            train_x, train_y = data_to_cuda(train_x, train_y)
            model, likelihood = model_to_cuda(model, likelihood)
            
            with gpytorch.settings.max_preconditioner_size(3000):
                cov = model_likelihood_covariance(train_x, preconditioner=precons[1])
                sol, t_sol = time_solve(cov, train_y[0])
                times[i,j] = t_sol
                torch.cuda.empty_cache()
            j += 1
    
    
    
    
    
    
    
    