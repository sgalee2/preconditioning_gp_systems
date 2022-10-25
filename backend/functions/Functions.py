import torch, startup, gpytorch, math

from gpytorch.distributions import MultivariateNormal

from linear_operator.operators import AddedDiagLinearOperator, DiagLinearOperator, RootLinearOperator
from backend.conjugate_gradients.preconditioners.Preconditioners import Eig_Preconditioner, rSVD_Preconditioner, rSVD_Preconditioner_cuda, nystrom_SVD

def gaussian_nll(rhs, mean, covariance, noise, precon = None):
    
    with gpytorch.settings.max_cholesky_size(0):
    
        diff = rhs - mean
        
        diag = DiagLinearOperator(noise)
        overrode_lt = AddedDiagLinearOperator(covariance, diag, preconditioner_override=precon)
        
        inv_quad, log_det = overrode_lt.inv_quad_log_det(diff, logdet=True)
        
        ll = -0.5 * sum([inv_quad, log_det, diff.size(-1) * math.log(2 * math.pi)])
        
        loss = -ll/diff.size(-1)
        
        return loss


def gp_nll(model, likelihood, train_x, train_y, precon = None):
    
    diff, output = model.model_params(train_x)
    noise = likelihood(train_y).variance
    loss = gaussian_nll(train_y, diff, output, noise, precon)
    
    return loss

##################################################################################

def GP_lazy_tensor(model, likelihood, train_x, precon_override=None):
    
    output = model(train_x)
    MVN = likelihood(output)
    tensor, loc = MVN.lazy_covariance_matrix, MVN.loc
    
    if precon_override is not None:
        tensor.preconditioner_override = precon_override
    
    return loc, tensor

def GP_nll(loc, tensor, target):
    
    diff = target - loc
    
    inv_quad, log_det = tensor.inv_quad_log_det(diff, logdet=True)
    ll = -0.5 * sum([inv_quad, log_det, diff.size(-1) * math.log(2 * math.pi)])
    loss = -ll/diff.size(-1)
    
    return loss

###################################################################################
