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

import matplotlib.pyplot as plt

gpytorch.settings.min_preconditioning_size._set_value(0)
gpytorch.settings.cg_tolerance._set_value(0.1)

def train_data_loader(data_title):
    
    df = get_regression_data(data_title)

    train_x, train_y, N, D = df.X_train, df.Y_train, df.N, df.D
    
    return train_x, train_y, N, D

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

def precon_log_det(lin_op, precon = Pivoted_Cholesky):
    
    lin_op.preconditioner_override = precon
    lin_op._q_cache = None
    P_log_det = lin_op._preconditioner()[2].item()
    
    return P_log_det

def exact_log_det_K(lin_op):
    
    L = linop_cholesky(lin_op)
    
    return exact_log_det(L)

if __name__ == '__main__':
    print("Running log_det comparisons. . .\n")
    datasets = ['winered', 'winewhite', 'parkinsons', 'power', 'naval']
    covariances = [gpytorch.kernels.RBFKernel(), gpytorch.kernels.MaternKernel(nu=0.5)]
    covariances_named = ['RBF kernel', 'Matern 0.5 kernel']
    precon_named = ['Pivoted Cholesky', 'Randomised SVD', 'Recursive Nystrom']
    
    for i in datasets:
        print("Testing on data-set:",i)
        
        train_x, train_y, N, D = train_data_loader(i)
        
        num_figs = len(covariances)
        
        fig, ((ax1, ax2)) = plt.subplots(num_figs, figsize=[12,10], sharey=True)
        fig.suptitle('Comparing error in preconditioner log-det for ' + i)
        fig.supylabel('$|\log|\hat{K}| - \log|\hat{P}||$')
        fig.supxlabel('Preconditioner Quality')
        
        for kernels in enumerate(covariances):
            
            print("Tests using kernel:", kernels[1], "\n")
            model, likelihood = define_model(gpytorch.means.ZeroMean(), kernels[1])
            likelihood.noise = 1.
            
            if N <= 10000:
                l_det_K = exact_log_det_K(model_likelihood_covariance(train_x))
            else:
                l_det_K = model_likelihood_covariance(train_x).logdet()
            
            max_rank = min(train_x.shape[0], 200)
            
            precon_ldets = torch.zeros([max_rank, 3])
            for j in range(2, max_rank):
                with gpytorch.settings.max_preconditioner_size(j):
                    cov = model_likelihood_covariance(train_x)
                    precon_ldets[j,0] = precon_log_det(cov)
                    precon_ldets[j,1] = precon_log_det(cov, rSVD_Preconditioner)
                    for k in range(10):
                        precon_ldets[j,2] += precon_log_det(cov, recursiveNystrom_Preconditioner)
                    precon_ldets[j,2] = precon_ldets[j,2]/10
            error = abs(precon_ldets[2:] - l_det_K)
            for plot_ in range(3):
                fig.get_axes()[kernels[0]].plot(error[:,plot_].detach(), label=precon_named[plot_])
                fig.get_axes()[kernels[0]].legend()
                
            fig.get_axes()[kernels[0]].set_title(covariances_named[kernels[0]])
        
        
        
        
        