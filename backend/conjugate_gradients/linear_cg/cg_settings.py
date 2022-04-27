import startup

from gpytorch.utils.linear_cg import linear_cg
from gpytorch.settings import cg_tolerance, max_cg_iterations, max_cholesky_size, max_preconditioner_size

import torch
import gpytorch

def set_cg_tol(tol):
    cg_tolerance._set_value(tol)
    
def set_cg_iters(max_its):
    max_cg_iterations._set_value(max_its)
    
def set_max_cholesky_size(max_n):
    max_cholesky_size(max_n)
    
def set_max_precon_size(max_p):
    max_preconditioner_size._set_value(max_p)