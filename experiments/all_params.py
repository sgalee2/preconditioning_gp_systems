import startup
import gpytorch

from bayesian_benchmarks.bayesian_benchmarks.data import *

datasets = ['boston',
            'energy',
            'concrete',
            'naval',
            'power',
            'protein',
            'winered',
            'winewhite',
            'yacht',
            'parkinsons']

cov1 = gpytorch.kernels.RBFKernel()
cov2 = gpytorch.kernels.MaternKernel()
cov3 = cov1 + cov2
cov4 = gpytorch.kernels.ScaleKernel(cov1)
cov5 = gpytorch.kernels.ScaleKernel(cov2)
cov6 = gpytorch.kernels.ScaleKernel(cov3)

cov_modules = [cov1,
               cov2,
               cov3,
               cov4,
               cov5,
               cov6]

precon_sizes = [10,
                11,
                12,
                13,
                14,
                15,
                20,
                30,
                40,
                50,
                75,
                100,
                150,
                200,
                300,
                400,
                500,
                1000,
                2000,
                3000,
                4000,
                5000]