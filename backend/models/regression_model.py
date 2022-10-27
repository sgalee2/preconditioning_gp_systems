import torch
import gpytorch
import time
import math

from model import Model
from backend.functions.Functions import *
from backend.conjugate_gradients.preconditioners.Preconditioners import *

class GPRegressionModel(Model):
    
    def __init__(self, reg_model, likelihood, loss_fn, optimizer, cuda=False, precon_override=None):
        
        self.model = reg_model
        self.likelihood = likelihood
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cuda = cuda
        self.precon_override = precon_override
        
        try:
            self.model.covar_module.base_kernel
            self.bker = True
        except:
            self.bker = False
        
    def Fit(self, X, y, lr, iters, *params):
        
        if self.cuda:
            X.cuda(), y.cuda(), self.model.cuda(), self.likelihood.cuda()
            
        
        #set model & likelihood to train mode
        self.model.train()
        self.likelihood.train()
        
        #define optimizer environment
        optimizer = self.optimizer(self.model.parameters(), lr = lr)
        
        print("Starting training...","\n")
        start_time = time.time()
        
        for i in range(iters):
            optimizer.zero_grad()
            with gpytorch.settings.max_cholesky_size(0):
                loss = self.loss_fn(self.model, self.likelihood, X, y, self.precon_override)
            loss.backward()
            
            if self.bker:
                
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, iters, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
                print("\n")
            else:
                
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, iters, loss.item(),
                    self.model.covar_module.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
                print("\n")
            optimizer.step()
        print("Finished training. Elapsed time: {}".format(time.time() - start_time))
                
