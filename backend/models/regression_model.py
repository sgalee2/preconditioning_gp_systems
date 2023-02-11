import torch
import gpytorch
import time
import math

from backend.models.model import Model
from backend.functions.Functions import *
from backend.conjugate_gradients.preconditioners.Preconditioners import *

class base_model(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, mean, covar):
        
        super(base_model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = covar
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRegressionModel(Model):
    
    def __init__(self, reg_model, likelihood, loss_fn, optimizer, cuda=False, precon_override=None, ard=False):
        
        self.model = reg_model
        self.likelihood = likelihood
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cuda = cuda
        self.precon_override = precon_override
        self.devices = [torch.device('cuda', i)
                        for i in range(torch.cuda.device_count())]
        self.output_device = self.devices[0]
        self.ard = ard
        self.iteration_times = None
        
        try:
            self.model.covar_module.base_kernel
            self.bker = True
        except:
            self.bker = False
        
    def Fit(self, X, y, lr, iters, *params):
        
        self.outputscale, self.lengthscale, self.noise = self.get_model_params()
        
        #need to fix CUDA
        if self.cuda is True:
            X.cuda(), y.cuda(), self.model.cuda(), self.likelihood.cuda()
        
        #set model & likelihood to train mode
        self.model.train()
        self.likelihood.train()
        
        #define optimizer environment
        optimizer = self.optimizer(self.model.parameters(), lr = lr)
        
        #start training
        print("Starting training...","\n")
        start_time = time.time()
        iteration_times = torch.tensor([[0]])
        self.loss = torch.tensor([[0]])
        
        for i in range(iters):
            iter_time = time.time()
            optimizer.zero_grad()
            with gpytorch.settings.max_cholesky_size(0):
                loss = self.loss_fn(self.model, self.likelihood, X, y, self.precon_override)
            loss.backward()
            self.loss = torch.vstack([self.loss, loss.cpu()])
            
            if self.bker:
                
                print('Iter %d/%d - Loss: %.3f' % (
                    i + 1, iters, loss.item(),
                ))
                print("\n")
            else:
                
                print('Iter %d/%d - Loss: %.3f' % (
                    i + 1, iters, loss.item(),
                ))
                print("\n")
            optimizer.step()
            outputscale, lengthscale, noise = self.get_model_params()
            self.outputscale, self.lengthscale, self.noise = torch.vstack([self.outputscale, outputscale]), torch.vstack([self.lengthscale, lengthscale]), torch.vstack([self.noise, noise])
            iteration_times = torch.vstack([iteration_times, torch.tensor([[time.time() - iter_time]])])
            
        finish_time = time.time()
        self.training_time = finish_time - start_time
        self.iteration_times = iteration_times[1:, :]
        self.loss = self.loss[1:, :].detach()

        print("Finished training. Elapsed time: {}".format(self.training_time))
        
    def Predict(self, Xs):
        
        if self.cuda:
            Xs.cuda()
            
        self.model.eval()
        self.likelihood.eval()
        
        print("Beginning predictions...")
        start_time = time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.latent_pred = self.model(Xs)
            self.post_pred = self.likelihood(self.latent_pred)
            mean, var = self.post_pred.mean, self.post_pred.variance
        print("Finished predicting. Elapsed time: {}".format(time.time() - start_time))
        return mean, var
        
    def get_model_params(self):
        
        if self.bker:
            outputscale = self.model.covar_module.outputscale.detach()
            lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        else:
            outputscale = torch.ones([1])
            lengthscale = self.model.covar_module.lengthscale.detach()
        
        noise = self.model.likelihood.noise.detach()
        
        return outputscale.cpu(), lengthscale.cpu(), noise.cpu()
    
    def get_exact_nll(self, inputs, targets):
        
        self.exact_nll = exact_nll(self.model, self.likelihood, inputs, targets)
        
        return self.exact_nll
    
    def eval(self):
        self.model.eval()
        self.likelihood.eval()
        print("Evaluation mode")
        
    def train(self):
        self.model.train()
        self.likelihood.train()
        print("Training mode")
        
        
        
        
        
        
        
        
        
        
        
        
        
                
