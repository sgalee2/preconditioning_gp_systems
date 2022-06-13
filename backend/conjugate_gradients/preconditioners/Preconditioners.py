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
        max_iter = gpytorch.settings.max_preconditioner_size.value()
        vals_, vecs_ = self._lazy_tensor.symeig(eigenvectors = True)
        vecs, vals =  lazify(vecs_[:, -max_iter:]), vals_[-max_iter:]
        L = vecs
        print(L)
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

n = 1000
A = torch.rand(n,n)
A = A.T @ A
diag = DiagLazyTensor(2. * torch.ones(n))
rhs = torch.rand(n,1)

gpytorch.settings.min_preconditioning_size._set_value(1)

standard_lt = AddedDiagLazyTensor(A, diag)
overrode_lt = AddedDiagLazyTensor(A, diag, preconditioner_override=Eig_Preconditioner)
overrode_lt.inv_quad_logdet(rhs)