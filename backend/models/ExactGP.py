# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:14:17 2022

@author: adayr
"""

import gpytorch, torch, startup

from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch, fit_gpytorch_scipy
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import (
    deterministic_probes, 
    cg_tolerance,
    max_preconditioner_size,
    max_cholesky_size,
    cholesky_jitter
)