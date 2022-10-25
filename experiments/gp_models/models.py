import torch, startup, gpytorch, math, sys

class RBF_GPmodel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(RBF_GPmodel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

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
    
class RBF_ARD_GPmodel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ARD_dims):
        super(RBF_ARD_GPmodel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=ARD_dims)

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
    
class MATERN_GPmodel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, smoothness):
        super(MATERN_GPmodel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel(nu = float(smoothness) + 0.5)

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
