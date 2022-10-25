# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:32:19 2022

@author: adayr
"""
import torch, gpytorch, startup

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean, cov):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = cov

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def model_params(self, x):
        """
        Returns mean and covariance for model at input point x.

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return mean_x, covar_x
