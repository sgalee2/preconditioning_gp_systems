import torch, startup, gpytorch, math

from gpytorch.distributions import MultivariateNormal

from linear_operator.operators import AddedDiagLinearOperator, DiagLinearOperator, RootLinearOperator
from backend.conjugate_gradients.preconditioners.Preconditioners import Eig_Preconditioner, rSVD_Preconditioner, rSVD_Preconditioner_cuda

def GP_nll(model, likelihood, train_x, target, precon_override=None):
    
    output = model(train_x)
    MVN = likelihood(output)
    tensor, loc = MVN.lazy_covariance_matrix, MVN.loc
    
    if precon_override is not None:
        tensor.preconditioner_override = precon_override
    
    diff = target - loc
    
    inv_quad, log_det = tensor.inv_quad_log_det(diff, logdet=True)
    ll = -0.5 * sum([inv_quad, log_det, diff.size(-1) * math.log(2 * math.pi)])
    loss = -ll/diff.size(-1)
    
    return loss

    jfuqireqf
    