import startup

from all_params import *

import gpytorch
import torch

from time import time

from bayesian_benchmarks.bayesian_benchmarks.data import *
from backend.models.regression_model import GPRegressionModel, base_model
from backend.conjugate_gradients.preconditioners.Preconditioners import rSVD_Preconditioner, rSVD_Preconditioner_cuda, recursiveNystrom_Preconditioner, Nystrom_Preconditioner

from torch.utils.data import Dataset


model = base_model()