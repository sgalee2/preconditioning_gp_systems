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
                
    def log_prob(self, value):
        if gpytorch.settings.fast_computations.log_prob.off():
            return super().log_prob(value)

        if self._validate_args:
            self._validate_sample(value)

        mean, covar = self.loc, self.lazy_covariance_matrix
        diff = value - mean

        # Repeat the covar to match the batch shape of diff
        if diff.shape[:-1] != covar.batch_shape:
            if len(diff.shape[:-1]) < len(covar.batch_shape):
                diff = diff.expand(covar.shape[:-1])
            else:
                padded_batch_shape = (*(1 for _ in range(diff.dim() + 1 - covar.dim())), *covar.batch_shape)
                covar = covar.repeat(
                    *(diff_size // covar_size for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)),
                    1,
                    1,
                )

        # Get log determininant and first part of quadratic form
        inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)

        res = -0.5 * sum([inv_quad, logdet, diff.size(-1) * math.log(2 * math.pi)])
        return res
            
            
class MultivariateNormal_eig(MultivariateNormal):
    
    def __init__(self, mean, covariance_matrix, precon_override=Eig_Preconditioner):
        super().__init__(mean, covariance_matrix)
        if type(self.lazy_covariance_matrix) is gpytorch.lazy.added_diag_lazy_tensor.AddedDiagLazyTensor:
            self.lazy_covariance_matrix.preconditioner_override = precon_override
        

