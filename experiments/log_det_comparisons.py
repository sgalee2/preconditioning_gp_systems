import startup

from all_params import *

import gpytorch
import torch
import argparse
import os

from time import time
from pathlib import Path

from torch.utils.data import Dataset

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import GPRegressionModel, base_model
from backend.functions.Functions import *
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner, Pivoted_Cholesky
from backend.sampling.recursive_nystrom_gpytorch import recursiveNystrom

import matplotlib.pyplot as plt

gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(0.1)

def train_data_loader(data_title):
    
    df = get_regression_data(data_title)

    train_x, train_y, N, D = df.X_train, df.Y_train, df.N, df.D
    
    return train_x, train_y, N, D

def define_model(train_x, train_y, mean_module, cov_module):
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = base_model(train_x, train_y, likelihood, mean_module, cov_module)
    
    return model, likelihood

def model_likelihood_covariance(model, likelihood, location, preconditioner = None):
    
    loc_likelihood = likelihood(model(location))
    lin_op = loc_likelihood.lazy_covariance_matrix
    
    if preconditioner is not None:
        lin_op.preconditioner_override = preconditioner
        
    return lin_op

def precon_log_det(lin_op, precon = Pivoted_Cholesky):
    
    lin_op.preconditioner_override = precon
    lin_op._q_cache = None
    P_log_det = lin_op._preconditioner()[2].item()
    
    return P_log_det

def exact_log_det_K(lin_op):
    
    L = linop_cholesky(lin_op)
    
    return exact_log_det(L)

def compute_preconditioner_logdet(dataset, mean_module, cov_module, precons, precon_rank, hypers):
    
    
    train_x, train_y, N, _ = train_data_loader(dataset)
    
    mean_module, cov_module = mean_module(), cov_module(ard_num_dims=train_x.shape[1])
    model, likelihood = define_model(train_x, train_y, mean_module, cov_module)
    model.initialize(**hypers)
    
    if N <= 10000:
        l_det_K = exact_log_det_K(model_likelihood_covariance(model, likelihood, train_x))
    else:
        l_det_K = model_likelihood_covariance(model, likelihood, train_x).logdet()
        
    precon_ldets = [l_det_K.item()]
    
    for precon in precons:
        with gpytorch.settings.max_preconditioner_size(precon_rank):
            cov = model_likelihood_covariance(model, likelihood, train_x)
            precon_ldets.append(precon_log_det(cov, precon=precon))
    return precon_ldets

class demo:
    
    def __init__(self, args):
        
        self.args = args
        
        if args.mean == 'zero':
            self.mean_module = gpytorch.means.ZeroMean
            
        if args.mean == 'constant':
            self.mean_module = gpytorch.means.ConstantMean
            
        if args.kernel == 'rbf':
            self.covar_module = gpytorch.kernels.RBFKernel
            
        if args.kernel == 'matern':
            self.covar_module = gpytorch.kernels.MaternKernel
        
        self.hypers = {
             'likelihood.noise': torch.tensor(1.),
             'covar_module.lengthscale': torch.tensor(0.5),}
        
        if args.save:
            save_path = '../results' + '/%s' % args.dataset
            os.makedirs(save_path, exist_ok=True)
            self.save_path = os.path.join(save_path, Path(args.kernel).stem + '_' + Path(str(args.max_rank)).stem + '.t')
            
            
    def run(self, precons):
        
        print("==> Running log_det tests for",args.dataset)
        results = torch.zeros([self.args.max_rank-1, len(precons)+1])
        for i in range(2, self.args.max_rank+1):
            if i%10 == 0:
                print("==> Testing for precon rank " + "%s/{}".format(self.args.max_rank) % i)
            results[i-2] = torch.tensor( compute_preconditioner_logdet(self.args.dataset, self.mean_module, self.covar_module, precons, i, self.hypers) )
            
        if self.args.save:
            torch.save([self.args, [precons[i].__name__ for i in range(len(precons))], results], self.save_path)
            
        return results

    
        
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=datasets, default='yacht')
    parser.add_argument("--mean", type=str, choices=['zero', 'constant'], default='zero')
    parser.add_argument("--kernel", type=str, choices=['rbf', 'matern'], default='rbf')
    parser.add_argument("--max_rank", type=int, default=20)
    parser.add_argument("--save", type=bool, default=False)
        
    args = parser.parse_args()
    precons = [Pivoted_Cholesky, rSVD_Preconditioner, recursiveNystrom_Preconditioner]
    
    Demo = demo(args)
    res = Demo.run(precons)


