# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:39:37 2022

@author: adayr
"""
import torch, startup, gpytorch
from backend.conjugate_gradients.utils.common_terms import precon_terms
from gpytorch.lazy.non_lazy_tensor import lazify
from gpytorch.lazy.lazy_tensor import LazyTensor

def svd_preconditioner(self):
    """
    Calling this needs to return a tuple:
        (preconditioner_closure, preconditioner_matrix, preconditioner_logdet)
    using Torch tensor arithmetic.
    """
    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(-1) < gpytorch.settings.min_preconditioning_size.value():
            return None, None, None
    A = self._lazy_tensor
    k = gpytorch.settings.max_preconditioner_size.value()
    U, S, V = A.svd()
    U_, S_ = U[:,0:k], S[0:k]
    log_det = S_[0:k].log().sum()
    
    precon_lt = gpytorch.lazy.root_lazy_tensor.RootLazyTensor(U_.evaluate() * S_**(0.5))
    
    def precon_closure(rhs):
        return torch.linalg.inv(U_ @ torch.diag(S_) @ U_.evaluate().T + self._diag_tensor.evaluate()) @ rhs
    
    return precon_closure, precon_lt, log_det

from gpytorch.lazy import AddedDiagLazyTensor, DiagLazyTensor, RootLazyTensor

seed = 4
torch.random.manual_seed(seed)

tensor = torch.randn(1000, 800)
diag = torch.abs(torch.randn(1000))

standard_lt = AddedDiagLazyTensor(RootLazyTensor(tensor), DiagLazyTensor(diag))
evals, evecs = standard_lt.symeig(eigenvectors=True)
