# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:39:37 2022

@author: adayr
"""
import torch, startup, gpytorch
from gpytorch.lazy import AddedDiagLazyTensor, DiagLazyTensor, RootLazyTensor
    
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
    print("using rsvd")    
    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(-1) < gpytorch.settings.min_preconditioning_size.value():
        return None, None, None
    
    if self._q_cache is None:
        
        from gpytorch.lazy.matmul_lazy_tensor import MatmulLazyTensor
        
        #get quantities & form sample matrix
        n, k = self.shape[0], gpytorch.settings.max_preconditioner_size.value()
        omega = torch.distributions.normal.Normal(0.,1.).sample([n, k + 4])
        
        #Z = A @ Omega = Q @ R
        Z = MatmulLazyTensor(self._lazy_tensor, omega)
        
        Q, R = Z.evaluate().qr()
        
        #Y = Q^T @ A = U @ S @ V
        Y = MatmulLazyTensor(Q.T, self._lazy_tensor)
        
        U, S, V = Y.evaluate().svd()
        
        #L = V @ S^0.5
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
        