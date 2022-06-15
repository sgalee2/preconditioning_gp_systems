# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:39:37 2022

@author: adayr
"""
import torch, startup, gpytorch
from backend.conjugate_gradients.utils.common_terms import precon_terms
from gpytorch.lazy.non_lazy_tensor import lazify
from gpytorch.lazy.lazy_tensor import LazyTensor
from gpytorch.utils.linear_cg import linear_cg

from gpytorch.lazy import AddedDiagLazyTensor, DiagLazyTensor, RootLazyTensor
from gpytorch.lazy.non_lazy_tensor import lazify
    
def Eig_Preconditioner(self):
    

    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(-1) < gpytorch.settings.min_preconditioning_size.value():
        return None, None, None
    
    if self._q_cache is None:
        
        #need to evaluate the full unregularised system
        system = self._lazy_tensor.evaluate()
        
        #set rank of approximation
        max_iter = gpytorch.settings.max_preconditioner_size.value()
        
        #compute truncated eigendecomposition
        vals_, vecs_ = torch.linalg.eigh(system)
        vecs, vals =  vecs_[:, -max_iter:], vals_[-max_iter:]
        
        #set the L factor & add it into GPyTorch language
        L = vecs * (vals ** 0.5)
        self._piv_chol_self = L
        if torch.any(torch.isnan(self._piv_chol_self)).item():
            warnings.warn(
                "NaNs encountered in preconditioner computation. Attempting to continue without preconditioning.",
                NumericalWarning,
            )
            return None, None, None
        self._init_cache()
        
    def precondition_closure(tensor):
        # This makes it fast to compute solves with it
        qqt = self._q_cache.matmul(self._q_cache.transpose(-2, -1).matmul(tensor))
        if self._constant_diag:
            return (1 / self._noise) * (tensor - qqt)
        return (tensor / self._noise) - qqt

    return (precondition_closure, self._precond_lt, self._precond_logdet_cache)

def rSVD_Preconditioner(self):
    
    
    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(-1) < gpytorch.settings.min_preconditioning_size.value():
        return None, None, None
    
    if self._q_cache is None:
        
        from gpytorch.lazy.matmul_lazy_tensor import MatmulLazyTensor
        
        n, k = self.shape[0], gpytorch.settings.max_preconditioner_size.value()
        omega = torch.distributions.normal.Normal(0.,1.).sample([n, k])
        
        Z = MatmulLazyTensor(self._lazy_tensor, omega)
        
        Q, R = Z.evaluate().qr()
        
        Y = MatmulLazyTensor(Q.T, self._lazy_tensor)
        
        U, S, V = Y.evaluate().svd()
        
        L = V * (S ** 0.5)
        
        self._piv_chol_self = L
        
        if torch.any(torch.isnan(self._piv_chol_self)).item():
            warnings.warn(
                "NaNs encountered in preconditioner computation. Attempting to continue without preconditioning.",
                NumericalWarning,
            )
            return None, None, None
        self._init_cache()
        
    def precondition_closure(tensor):
        # This makes it fast to compute solves with it
        qqt = self._q_cache.matmul(self._q_cache.transpose(-2, -1).matmul(tensor))
        if self._constant_diag:
            return (1 / self._noise) * (tensor - qqt)
        return (tensor / self._noise) - qqt

    return (precondition_closure, self._precond_lt, self._precond_logdet_cache)
        
class AddedDiagLazyTensor_rng(AddedDiagLazyTensor):
    
    def __init__(self, *lazy_tensors, preconditioner_override=None, distribution=None):
        
        super(AddedDiagLazyTensor_rng, self).__init__(*lazy_tensors, preconditioner_override=preconditioner_override)
        
        self.distribution = distribution

n = 3000
m = 50
A = torch.rand(n,m)
A_ = gpytorch.lazy.RootLazyTensor(A)
diag = DiagLazyTensor(2. * torch.ones(n))
rhs = torch.rand(n,1)

import math
k = int( math.sqrt(n) )
gpytorch.settings.min_preconditioning_size._set_value(1)
gpytorch.settings.max_preconditioner_size._set_value(1)

distribution = torch.distributions.Normal(0.,1.)

standard_lt = AddedDiagLazyTensor(A_, diag)
overrode_lt = AddedDiagLazyTensor(A_, diag, preconditioner_override=rSVD_Preconditioner)

sol = torch.linalg.inv( standard_lt.evaluate() ) @ rhs

