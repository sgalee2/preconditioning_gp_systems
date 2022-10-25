# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:39:37 2022

@author: adayr
"""
import torch, startup, gpytorch, linear_operator
from linear_operator.operators import AddedDiagLinearOperator, DiagLinearOperator, RootLinearOperator
    
def Eig_Preconditioner(self):
    

    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(-1) < gpytorch.settings.min_preconditioning_size.value():
        return None, None, None
    
    if self._q_cache is None:
        
        #need to evaluate the full unregularised system
        try:
            system = self._linear_op.evaluate()
        except:
            system = self._linear_op
        
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

def nystrom_SVD(self):
    print("Sampling columns for preconditioner...")
    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(-1) < gpytorch.settings.min_preconditioning_size.value():
        return None, None, None
    
    if self._q_cache is None:
        
        import math
        
        n, k = self.shape[0], gpytorch.settings.max_preconditioner_size.value()
        index = torch.randint(0,n, size=[k])
        S_ = self._linear_op[index].evaluate()/math.sqrt(k/n)
        mat = S_ @ S_.T
        
        U,S,V = torch.linalg.svd(mat)
        
        L = S_.T @ U
        L = L / torch.norm(L, dim=0)
        L *= S ** 0.25
        
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
    print("Using rSVD Preconditioner")
    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(-1) < gpytorch.settings.min_preconditioning_size.value():
        return None, None, None
    
    if self._q_cache is None:
        
        from linear_operator.operators import MatmulLinearOperator
        
        #get quantities & form sample matrix
        n, k = self.shape[0], gpytorch.settings.max_preconditioner_size.value()
        omega = torch.distributions.normal.Normal(0.,1.).sample([n, k])
        
        #Z = A @ Omega = Q @ R
        Z = MatmulLinearOperator(self._linear_op, omega)
        
        Q, R = Z.evaluate().qr()
        
        #Y = Q^T @ A = U @ S @ V
        Y = MatmulLinearOperator(Q.T, self._linear_op)
        
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
        
def rSVD_Preconditioner_cuda(self):
    print("using rsvd")    
    if gpytorch.settings.max_preconditioner_size.value() == 0 or self.size(-1) < gpytorch.settings.min_preconditioning_size.value():
        return None, None, None
    
    if self._q_cache is None:
        
        from linear_operator.operators import MatmulLinearOperator
        
        #get quantities & form sample matrix
        n, k = self.shape[0], gpytorch.settings.max_preconditioner_size.value()
        omega = torch.distributions.normal.Normal(0.,1.).sample([n, k]).cuda()
        
        #Z = A @ Omega = Q @ R
        Z = MatmulLinearOperator(self._linear_op, omega).cuda()
        
        Q, R = Z.evaluate().qr()
        
        #Y = Q^T @ A = U @ S @ V
        Y = MatmulLinearOperator(Q.T, self._linear_op).cuda()
        
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