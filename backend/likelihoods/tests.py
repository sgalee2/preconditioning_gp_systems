import torch, startup, gpytorch, math

from gpytorch.distributions import MultivariateNormal

from gpytorch.lazy import AddedDiagLazyTensor, DiagLazyTensor, RootLazyTensor

from backend.conjugate_gradients.preconditioners.Preconditioners import Eig_Preconditioner, rSVD_Preconditioner
from preconditioned_likelihoods import MultivariateNormal_rsvd, MultivariateNormal_eig
from matplotlib import pyplot as plt

from gpytorch import settings

from time import time

gpytorch.settings.cg_tolerance._set_value(1e-5)

torch.manual_seed(123)

# Training data is points in [0,4] inclusive regularly spaced
train_x = torch.linspace(0, 4, 3000)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.14)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def train_(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return mean_x, covar_x

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

print(model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item())

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

def optim_step(train_x, train_y, precon_override):

    optimizer.zero_grad()
    
    diff, output = model.train_(train_x)
    diff = train_y - diff
    diag = DiagLazyTensor(likelihood(train_y).variance)
    
    overrode_lt = AddedDiagLazyTensor(output, diag, preconditioner_override=precon_override)
    
    inv_quad, log_det = overrode_lt.inv_quad_log_det(diff, logdet=True)
    
    ll = -0.5 * sum([inv_quad, log_det, diff.size(-1) * math.log(2 * math.pi)])
    
    loss = -ll/diff.size(-1)
    
    loss.backward()
    optimizer.step()

print("Training...")
t1 = time()
for i in range(20):
    optim_step(train_x, train_y, rSVD_Preconditioner)
    print("\n .")
timer = time() - t1
    
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 4, 51)
    observed_pred = likelihood(model(test_x))
    

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), torch.sin(train_x * (2 * math.pi)).numpy(),color='red',alpha=0.4)
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])