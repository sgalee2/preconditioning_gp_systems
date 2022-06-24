import torch, startup, gpytorch, math
from backend.conjugate_gradients.preconditioners.Preconditioners import Eig_Preconditioner, rSVD_Preconditioner

from gpytorch.distributions import MultivariateNormal

"""
A class of GPyTorch distributions which must be defined within the GP model e.g.

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormalrsvd(mean_x, covar_x)
    
Then:
    
    output = model(train_x)
    
returns a preconditioned_likelihood, and

    loss = -mll(output, train_y)
    
will use the desired preconditioner, if applicable.
"""

torch.manual_seed(0)

class MultivariateNormal_rsvd(MultivariateNormal):
    
    def __init__(self, mean, covariance_matrix, precon_override=rSVD_Preconditioner):
        super().__init__(mean, covariance_matrix)
        if type(self.lazy_covariance_matrix) is gpytorch.lazy.added_diag_lazy_tensor.AddedDiagLazyTensor:
            self.lazy_covariance_matrix.preconditioner_override = precon_override
            
class MultivariateNormal_eig(MultivariateNormal):
    
    def __init__(self, mean, covariance_matrix, precon_override=Eig_Preconditioner):
        super().__init__(mean, covariance_matrix)
        if type(self.lazy_covariance_matrix) is gpytorch.lazy.added_diag_lazy_tensor.AddedDiagLazyTensor:
            self.lazy_covariance_matrix.preconditioner_override = precon_override
        

