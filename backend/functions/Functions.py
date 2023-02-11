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



def linop_cholesky(linear_operator):
    
    try:
        L = linear_operator.cholesky()
    except:
        L = torch.linalg.cholesky(linear_operator)
        
    return L


def exact_log_det(L):
        
    log_det = 2 * sum( torch.log(L.diag()) )
    return log_det

def exact_inv_quad(L, target):
    
    "target.T @ K^{-1} @ target = target.T (LL.T)^{-1} @ target"
    solve = torch.linalg.solve_triangular
    v = solve(L, target, upper=False)
    v = solve(L.T, v, upper = True)
    inv_quad = torch.dot(target.reshape(-1), v.reshape(-1))
    
    return inv_quad

def exact_GP_nll(model, likelihood, train_x, target, *params):
    
    output = model(train_x)
    MVN = likelihood(output)
    tensor, loc = MVN.lazy_covariance_matrix, MVN.loc
    diff = target - loc 
    
    L = linop_cholesky(tensor)
    
    inv_quad, log_det = exact_inv_quad(L, diff), exact_log_det(L)
    ll = -0.5 * sum([inv_quad, log_det, diff.size(-1) * math.log(2 * math.pi)])
    loss = -ll/diff.size(-1)
    
    return loss