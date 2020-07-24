"""
Conjugate gradient method
==========================

Different implementations of the conjugate gradient (CG) exist. Here, we compare the CG implemented in scipy which uses Fortran
against it's pythonized version and the older version of the algorithm available in pykeops.

We want to solve the positive definite linear system :math:`(K_{x,x} + \\alpha Id)a = b` for :math:`a, b, x \in \mathbb R^N`.

Let the Gaussian RBF kernel be defined as

.. math::

    K_{x,x}=\left[\exp(-\gamma \|x_i - x_j\|^2)\\right]_{i,j=1}^N. 


Choosing :math:`x` such that :math:`x_i = i/N,\ i=1,\dots, N` makes :math:`K_{x,x}` be a highly unwell-conditioned matrix for :math:`N\geq 10`.

"""

#############################
# Setup
# ----------
# Imports needed

import importlib
import os
import time
import inspect

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, cg

from pykeops.numpy import KernelSolve as KernelSolve_np, LazyTensor
from pykeops.torch import KernelSolve
from pykeops.torch.utils import squared_distances
from pykeops.torch import Genred as Genred_tch
from pykeops.numpy import Vi, Vj, Pm
from pykeops.numpy import Genred as Genred_np

use_cuda = torch.cuda.is_available()

device = torch.device("cuda") if use_cuda else torch.device("cpu")
print("The device used is {}.".format(device))

########################################
# Gaussian radial basis function kernel
########################################

n = 100000
dv = 1
formula = 'Exp(- g * SqDist(x,y)) * a' # linear w.r.t a


############################
# Functions to benchmark
###########################
#
# All systems are regularized with a ridge parameter ``alpha``. 
#
# The originals :
# 


def keops_tch(x, b, gamma, alpha, aliases, callback=None):
    Kinv = KernelSolve(formula, aliases, "a", axis=1, dtype='float32')
    res = Kinv(x, x, b, gamma, alpha=alpha)
    return res


def keops_np(x, b, gamma, alpha, aliases, callback=None):
    Kinv = KernelSolve_np(formula, aliases, "a", axis=1, dtype='float32')
    res = Kinv(x, x, b, gamma, alpha=alpha, callback=callback)
    return res


####################################
# Scipy :
# 
#


def scipy_cg(x, b, gamma, alpha, aliases, callback=None):
    K_ij = (-Pm(gamma) * Vi(x).sqdist(Vj(x))).exp()
    A = aslinearoperator(
        diags(alpha * np.ones(x.shape[0]))) + aslinearoperator(K_ij)
    A.dtype = np.dtype('float32')
    res = cg(A, b, callback=callback)
    return res


####################################
# Pythonized scipy :
# 


def dic_cg_np(x, b, gamma, alpha, aliases, callback=None, check_cond=False):
    Kinv = KernelSolve_np(formula, aliases, "a", axis=1, dtype='float32')
    ans = Kinv.cg(x, x, b, gamma, alpha=alpha,
                      callback=callback, check_cond=check_cond)
    return ans


def dic_cg_tch(x, b, gamma, alpha, aliases, check_cond=False):
    Kinv = KernelSolve(formula, aliases, "a", axis=1, dtype='float32')
    ans = Kinv.cg(x, x, b, gamma, alpha=alpha, check_cond=check_cond)
    return ans


#########################
# Benchmarking
#########################

functions = [(scipy_cg, "numpy"),
             (keops_np, "numpy"), (keops_tch, "torch"),
             (dic_cg_np, "numpy"), (dic_cg_tch, "torch")]

sizes_d = [10, 50, 75, 100, 200]
reps =    [10, 5,  5,  5,   5]


def compute_error(func, pack, result, errors, x, b, alpha, gamma, aliases):
    if str(func)[10:15] == "keops":
        code = "a = func(x, b, gamma, alpha, aliases).reshape(b.shape);\
                err = ( (alpha * a + K(x, x, a, gamma) - b) ** 2).sum();\
                errors.append(err);"
    else:
        code = "a = func(x, b, gamma, alpha, aliases)[0].reshape(b.shape);\
                err = ( (alpha * a + K(x, x, a, gamma) - b) ** 2).sum();\
                errors.append(err);"

    if pack == 'numpy':
        K = Genred_np(formula, aliases, axis=1, dtype='float32')
    else:
        K = Genred_tch(formula, aliases, axis=1, dtype='float32')

    exec(code, locals())
    return errors
 

def to_bench(funcpack, d, rep):
    importlib.reload(torch)
    if device == 'cuda':
        torch.cuda.manual_seed_all(112358)
    else:
        torch.manual_seed(112358)
    code = "func(x, b, gamma, alpha, aliases)"
    func, pack = funcpack

    times = []
    errors = []

    if use_cuda:
        torch.cuda.synchronize()
    
    aliases = ['x = Vi(' + str(d) + ')',   # First arg:  i-variable of size d
                'y = Vj(' + str(d) + ')',   # Second arg: j-variable of size d
                'a = Vj(' + str(dv) + ')',  # Third arg:  j-variable of size dv
                'g = Pm(1)']

    for i in range(rep):

        x = torch.rand(n, d, device=device, dtype=torch.float32)
        b = torch.randn(n, dv, device=device, dtype=torch.float32)
        # kernel bandwidth
        gamma = torch.ones(
            1, device=device, dtype=torch.float32) * .5 / .01 ** 2
        # regularization
        alpha = torch.ones(1, device=device, dtype=torch.float32) * 2

        if pack == 'numpy':
            x, b = x.cpu().numpy().astype("float32"), b.cpu().numpy().astype("float32")
            gamma, alpha = gamma.cpu().numpy().astype(
                "float32"), alpha.cpu().numpy().astype("float32")

        if i == 0:
            exec(code, locals())  # Warmup run, to compile and load everything

        start = time.perf_counter()
        result = func(x, b, gamma, alpha, aliases)
        if use_cuda:
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)
        errors = compute_error(func, pack, result, errors, x, b, alpha, gamma, aliases)

    return sum(times)/rep, sum(errors)/rep


def global_bench(functions, sizes_d, reps):
    list_times = [[] for _ in range(len(functions))]
    list_errors = [[] for _ in range(len(functions))]

    for j, one_to_bench in enumerate(functions):
        print("~~~~~~~~~~~~~Benchmarking {}~~~~~~~~~~~~~~.".format(one_to_bench))
        for i in range(len(sizes_d)):
            try:
                time, err = to_bench(one_to_bench, sizes_d[i], reps[i])
                list_times[j].append(time)
                list_errors[j].append(err)
            except:
                 while len(list_times[j]) != len(reps):
                     list_times[j].append(np.nan)
                     list_errors[j].append(np.nan)
                 break
            print("Finished size {}.".format(sizes_d[i]))

        print("Finished", one_to_bench[0], "in a cumulated time of {:3.9f}s.".format(
            sum(list_times[j])))

    return list_times, list_errors


#########################################
# Plot the results of the benchmarking
#########################################

list_times, list_errors = global_bench(functions, sizes_d, reps)
labels = ["scipy + keops", "keops_np", "keops_tch",
          "dico + keops_np", "dico + keops_tch"]

plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.subplot(121)
for i in range(len(functions)):
    plt.plot(sizes_d, list_times[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel made from points of size {} $\times$ {}.".format(n, 'd'))
plt.ylabel("Computational time (s)")
plt.legend()
plt.subplot(122)
for i in range(len(functions)):
    plt.plot(sizes_d, list_errors[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel made from points of size {} $\times$ {}.".format(n, 'd'))
plt.ylabel(r"Error $||Ax_{k_{end}} -b||^2$")
plt.legend()
plt.tight_layout()
plt.show()
