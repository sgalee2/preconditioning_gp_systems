# axelv 16/05/2019
#rewritten for pytorch sgalee2 02/11/2022

import numpy as np
import scipy.linalg as spl

import torch, gpytorch, math

import time
from tqdm import tqdm
import gc


def gauss(X: np.ndarray, Y: np.ndarray=None, gamma=0.01):
    # todo make this implementation more python like!

    if Y is None:
        Ksub = np.ones((X.shape[0], 1))
    else:
        nsq_rows = np.sum(X ** 2, axis=1, keepdims=True)
        nsq_cols = np.sum(Y ** 2, axis=1, keepdims=True)
        Ksub = nsq_rows - np.matmul(X, Y.T * 2)
        Ksub = nsq_cols.T + Ksub
        Ksub = np.exp(-gamma * Ksub)

    return Ksub


def uniformNystrom(X, n_components: int, kernel_func=gauss):
    indices = np.random.choice(X.shape[0], n_components)
    C = kernel_func(X, X[indices,:])
    SKS = C[indices, :]
    W = np.linalg.inv(SKS + 10e-6 * np.eye(n_components))

    return C, W


def recursiveNystrom(K, n_components: int, accelerated_flag=False, random_state=None, lmbda_0=0, return_leverage_score=False, **kwargs):
    '''

    :param X:
    :param n_components:
    :param kernel_func:
    :param accelerated_flag:
    :param random_state:
    :return:
    '''
    if random_state is not None:
        torch.random.manual_seed(random_state)
    
    n = K.shape[0]
    n_oversample = math.log(n_components)
    k = math.ceil(n_components / (4 * n_oversample))
    n_levels = math.ceil(math.log(n / n_components) / math.log(2))
    perm = torch.randperm(n)

    # set up sizes for recursive levels
    size_list = [n]
    for l in range(1, n_levels+1):
        size_list += [math.ceil(size_list[l - 1] / 2)]
    
    # indices of poitns selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    sample = torch.tensor([i for i in range(size_list[-1])])
    indices = perm[sample]
    weights = torch.ones((indices.shape[0],))

    # we need the diagonal of the whole kernel matrix, so compute upfront
    try:
        k_diag = K.diag().reshape(-1,1)
    except:
        k_diag = torch.diag(K).reshape(-1,1)
    rec = 0
    # Main recursion, unrolled for efficiency
    for l in reversed(range(n_levels)):
        # indices of current uniform sample
        current_indices = perm[:size_list[l]]
        # build sampled kernel

        # all rows and sampled columns
        #try:
            #KS = K[current_indices, :][:, indices].evaluate()
        #except:
        KS = K[current_indices, :][:, indices]
            
            
        SKS = KS[sample, :].evaluate().detach() # sampled rows and sampled columns
        
        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[0]:
            # for the rare chance we take less than k samples in a round
            lmbda = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:
            # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
            # can be interpret as the zoom level
            lmbda = (torch.sum(torch.diag(SKS) * (weights ** 2))
                    - torch.sum(torch.linalg.eigvalsh(SKS * weights[:,None] * weights[None,:])[SKS.shape[0]-k:]))/k
        #lmbda = np.maximum(lmbda_0*SKS.shape[0], lmbda)
        #if lmbda == lmbda_0*SKS.shape[0]:
            #print("Set lambda to %d." % lmbda)
        #lmbda = np.minimum(lmbda, 5)
            # lmbda = spl.eigvalsh(SKS * weights * weights.T, eigvals=(0, SKS.shape[0]-k-1)).sum()/k
            # calculate the n-k smallest eigenvalues

        # compute and sample by lambda ridge leverage scores
        R = torch.linalg.inv(SKS + torch.diag(lmbda * weights ** (-2)))
        R = KS.matmul(R)
        #R = np.linalg.lstsq((SKS + np.diag(lmbda * weights ** (-2))).T,KS.T)[0].T
        if l != 0:
            # max(0, . ) helps avoid numerical issues, unnecessary in theory
            leverage_score = torch.minimum(torch.tensor([1.0]), n_oversample * (1 / lmbda) * torch.maximum(torch.tensor([+0.0]), (
                    k_diag[current_indices, 0] - torch.sum((R * KS).evaluate().detach(), axis=1))))
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is n_components in expectation
            sample = torch.where(torch.rand(size=[size_list[l]]) < leverage_score)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            ##if sample.size == 0:
                ##leverage_score[:] = n_components / size_list[l]
                ##sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = np.sqrt(1. / leverage_score[sample])

        else:
            leverage_score = torch.minimum(torch.tensor(1.0), (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - torch.sum((R * KS).evaluate().detach(), axis=1))))
            p = leverage_score/leverage_score.sum()
            sample = p.multinomial(n_components)
        indices = perm[sample]

    if return_leverage_score:
        return indices, leverage_score[np.argsort(perm)]
    else:
        return indices


# Below the copyright info that came with the original MATLAB implementation
# -------------------------------------------------------------------------------------
# Copyright (c) 2017 Christopher Musco, Cameron Musco
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# Small check to test if the algorithms output makes sense
if __name__ == "__main__":

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import scipy.io as sio

    global y
    n1 = 4000
    n2 = 20000
    n3 = 19000
    n = np.asarray([n1, n2, n3])
    np.random.seed(10)
    X = np.concatenate([np.random.multivariate_normal(mean=[50, 10], cov=np.eye(2), size=(n1,)),
                        np.random.multivariate_normal(mean=[-70, -70], cov=np.eye(2), size=(n2,)),
                        np.random.multivariate_normal(mean=[90, -40], cov=np.eye(2), size=(n3,))], axis=0)
    X_tensor = torch.tensor(X)
    y = np.concatenate([np.ones((n1,)) * 1,
                        np.ones((n2,)) * 2,
                        np.ones((n3,)) * 3])
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    covar = gpytorch.kernels.RBFKernel()
    K = covar(torch.tensor(X))
