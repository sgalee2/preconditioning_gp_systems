# -*- coding: utf-8 -*-
"""
Make sure preconditioners can perform inverse solves & log determinant estimations.
"""

import torch, startup, gpytorch, math
from backend.conjugate_gradients.preconditioners.Preconditioners import Eig_Preconditioner, rSVD_Preconditioner

from gpytorch.models import ExactGP
from matplotlib import pyplot as plt

from gpytorch.lazy import AddedDiagLazyTensor, DiagLazyTensor

import unittest



class TestAddedDiagLazyTensorPrecondOverride(unittest.TestCase):
    
    def test_rSVD_precond_solve(self):
        
        n = 1000
        
        A = torch.rand(n,n)
        A = A.T @ A
        diag = DiagLazyTensor(0.5 * torch.ones(n))
        
        standard_lt = AddedDiagLazyTensor(A, diag)
        overrode_lt = AddedDiagLazyTensor(A, diag, preconditioner_override=rSVD_Preconditioner)
        
        rhs = torch.rand([n,1])
        
        with gpytorch.settings.min_preconditioning_size(1), gpytorch.settings.cg_tolerance(0.01):
            standard_solve = standard_lt.inv_matmul(rhs)
            overrode_solve = overrode_lt.inv_matmul(rhs)
            exact_solve = torch.linalg.inv(A + diag.evaluate()) @ rhs
        
        self.assertLess(torch.norm(standard_solve - overrode_solve) / standard_solve.norm(), 1e-1)
        self.assertLess(torch.norm(exact_solve - overrode_solve)/exact_solve.norm(), 1e-1)
    
    def test_eig_precond_solve(self):
        
        n = 1000
        
        A = torch.rand(n,n)
        A = A.T @ A
        diag = DiagLazyTensor(0.5 * torch.ones(n))
        
        standard_lt = AddedDiagLazyTensor(A, diag)
        overrode_lt = AddedDiagLazyTensor(A, diag, preconditioner_override=Eig_Preconditioner)
        
        rhs = torch.rand([n,1])
        
        with gpytorch.settings.min_preconditioning_size(1), gpytorch.settings.cg_tolerance(0.01), gpytorch.settings.max_cholesky_size(1) :
            standard_solve = standard_lt.inv_matmul(rhs)
            overrode_solve = overrode_lt.inv_matmul(rhs)
            exact_solve = torch.linalg.inv(A + diag.evaluate()) @ rhs
        
        self.assertLess(torch.norm(standard_solve - overrode_solve) / standard_solve.norm(), 1e-1)
        self.assertLess(torch.norm(exact_solve - overrode_solve)/exact_solve.norm(), 1e-1)

    def test_rSVD_precon_logdet(self):
        
        n = 1000
        
        A = torch.rand(n,n)
        A = A.T @ A
        diag = DiagLazyTensor(0.5 * torch.ones(n))
        
        standard_lt = AddedDiagLazyTensor(A, diag)
        overrode_lt = AddedDiagLazyTensor(A, diag, preconditioner_override=rSVD_Preconditioner)
        
        with gpytorch.settings.min_preconditioning_size(1), gpytorch.settings.cg_tolerance(0.01), gpytorch.settings.max_cholesky_size(1) :
            standard_ldet = standard_lt.log_det()
            overrode_ldet = overrode_lt.log_det()
        
        self.assertLess(torch.norm(standard_ldet - overrode_ldet) / standard_ldet.norm(), 1e-1)
        
    def test_eig_precon_logdet(self):
        
        n = 1000
        
        A = torch.rand(n,n)
        A = A.T @ A
        diag = DiagLazyTensor(0.5 * torch.ones(n))
        
        standard_lt = AddedDiagLazyTensor(A, diag)
        overrode_lt = AddedDiagLazyTensor(A, diag, preconditioner_override=Eig_Preconditioner)
        
        rhs = torch.rand([n,1])
        
        with gpytorch.settings.min_preconditioning_size(1), gpytorch.settings.cg_tolerance(0.01), gpytorch.settings.max_cholesky_size(1) :
            standard_ldet = standard_lt.log_det()
            overrode_ldet = overrode_lt.log_det()
        
        self.assertLess(torch.norm(standard_ldet - overrode_ldet) / standard_ldet.norm(), 1e-1)

if __name__ == "__main__":
    unittest.main()