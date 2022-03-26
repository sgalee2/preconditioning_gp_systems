# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:35:24 2022

@author: adayr
"""
import torch

def precon(self, V, C, sigma):
    """
    Terms that commonly appear in GP approximations for preconditioning:
        
        K_xx + sigma ** 2 * I \approx Q_xx = UCV + sigma ** 2 * I
        
    then to compute Q_xx^{-1} v:
        
        A = sigma ** {-1} * L^{-1}V, where C = LL^T
        B = I + AA^T
        L_B = cholesky(B)
        
        then Q_xx^{-1} v = 1/sigma**2 [ v - A.T @ L_B^{-T} @ L_B^{-1} @ A @ v].
        
    This routine will compute and return all of A, B, L_B and L.
    """
    jitter = lambda x: 0.0001 * torch.eye(x)
    m, n = V.shape
    L = torch.linalg.cholesky(C + jitter(m))
    
    tri_solve = torch.triangular_solve
    A = tri_solve(V, L, upper=False).solution
    A = A/sigma
    
    B = A @ A.T + torch.eye(m)
    L_B = torch.linalg.cholesky(B)
    
    return A, B, L_B, L

n, d = 100, 1
X = torch.rand(n, d)
jitter = lambda x: 0.00001*torch.eye(x)
sigma = 1.

from gpytorch.kernels.rbf_kernel import RBFKernel
kernel = RBFKernel()
K_xx = kernel(X).evaluate().detach()
L = torch.linalg.cholesky(K_xx + jitter(n))

y_ = torch.normal(0, 1, size=[n,d])
y_obs = L @ y_

sol = torch.linalg.inv(K_xx + sigma ** 2 * torch.eye(n)) @ y_obs
print(sol)