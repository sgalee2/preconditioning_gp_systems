# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:39:37 2022

@author: adayr
"""
import torch, startup
from backend.conjugate_gradients.utils.common_terms import *

class Preconditioner(object):
    
    def mvp(self, v):
        raise NotImplementedError
        
    def imvp(self, v):
        raise NotImplementedError()

class Nystrom(Preconditioner):
    def __init__(self, K_ux = None, K_uu = None, sigma = None):
        if K_ux is not None:
            self.K_ux = K_ux
            self.m, self.n = K_ux.shape

        if K_uu is not None:
            self.K_uu = K_uu
        if sigma is not None:
            self.sigma = sigma
            if abs(self.sigma) < 1e-1:
                self.safe_woodbury = False
            else:
                self.safe_woodbury = True
        
    def set_U(self, U):
        self.K_ux = U
        
    def set_V(self, V):
        self.K_xu = V
        
    def set_C(self, C):
        self.K_uu = C
        
    def set_sigma(self, sigma):
        self.sigma = sigma
        
    def mvp(self, v):
        """
        

        Parameters
        ----------
        v : rhs vector

        Returns
        -------
        sol: returns P^{-1}v
        
        Given K_xu and K_uu, we compute the quantities:
            A = sigma ** -1 L^{-1} K_ux, B = AA^T + I
        and then the mvp
            Pv = (K_xu K_uu K_xu + sigma ** 2 I)^{-1} v = 1/sigma ** 2 [v - A^B^{-1}A^Tv]
        If safe_woodbury is False, use the computed solution at the risk of large errors.
        We will instead provide the scaled solution vector and sigma for timesing through.

        """
        if self.safe_woodbury == True:
            tri_solve = torch.triangular_solve
            A, B, L_B, L = precon_terms(self.K_ux, self.K_uu, self.sigma)
            
            a_til = tri_solve(A, L_B, upper=False).solution
            a_til_v = a_til @ v
            inner = a_til.T @ a_til_v 
            
            prod = v - inner
            prod = prod / self.sigma ** 2
            
        else:
            tri_solve = torch.triangular_solve
            A, B, L_B, L = precon_terms(self.K_ux, self.K_uu, self.sigma)
            
            a_til = tri_solve(A, L_B, upper=False).solution
            a_til_v = a_til @ v
            inner = a_til.T @ a_til_v 
            
            prod = v - inner
            prod = [prod, self.sigma ** 2]
            
        return prod
        
