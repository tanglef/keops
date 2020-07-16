"""
Conjugate gradient method
==========================

Different implementations of the conjugate gradient (CG) exist. Here, we compare the CG implemented in scipy which uses Fortran
against it's pythonized version and the older version of the algorithm available in pykeops.

We want to solve the positive definite linear system :math:`(K_{x,x} + \alpha Id)a = b` for :math:`a, b, x \in \mathbb R^N`.

Let the Gaussian RBF kernel be defined as
.. math::

    K_{x,x}=\left[\exp(-\gamma \|x_i - x_j\|^2)\right]_{i,j=1}^N. 

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
from matplotlib import pyplot as plt

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
print("The device used is", device, '.')

########################################
# gaussian radial basis function kernel
########################################

formula = 'Exp(- g * SqDist(x,y)) * a' # linear w.r.t a
aliases = ['x = Vi(1)',   # First arg:  i-variable of size 1
           'y = Vj(1)',   # Second arg: j-variable of size 1
           'a = Vj(1)',  # Third arg:  j-variable of size 1
           'g = Pm(1)']


############################
# Functions to benchmark
###########################

# All systems are regularized with a ridge parameter ``alpha``. 

# The originals :
# -------------------

def keops_tch(x, b, gamma, alpha, callback=None):
    Kinv = KernelSolve(formula, aliases, "a", axis=1, dtype='float32')
    res = Kinv(x, x, b, gamma, alpha=alpha)
    return res


def keops_np(x, b, gamma, alpha, callback=None):
    Kinv = KernelSolve_np(formula, aliases, "a", axis=1, dtype='float32')
    res = Kinv(x, x, b, gamma, alpha=alpha, callback=callback)
    return res

# Scipy :
# ----------


def scipy_cg(x, b, gamma, alpha, callback=None):
    K_ij = (-Pm(gamma) * Vi(x).sqdist(Vj(x))).exp()
    A = aslinearoperator(
        diags(alpha * np.ones(x.shape[0]))) + aslinearoperator(K_ij)
    A.dtype = np.dtype('float32')
    res = cg(A, b, callback=callback)
    return res

# Pythonized scipy :
# ---------------------


def dic_cg_np(x, b, gamma, alpha, callback=None, check_cond=False):
    Kinv = KernelSolve_np(formula, aliases, "a", axis=1, dtype='float32')
    ans = Kinv.dic_cg(x, x, b, gamma, alpha=alpha,
                      callback=callback, check_cond=check_cond)
    return ans


def dic_cg_tch(x, b, gamma, alpha, check_cond=False):
    Kinv = KernelSolve(formula, aliases, "a", axis=1, dtype='float32')
    ans = Kinv.dic_cg(x, x, b, gamma, alpha=alpha, check_cond=check_cond)
    return ans


#########################
# Benchmarking
#########################

functions = [(scipy_cg, "numpy"),
             (keops_np, "numpy"), (keops_tch, "torch"),
             (dic_cg_np, "numpy"), (dic_cg_tch, "torch")]

sizes = [50,  100, 500, 1000, 5000, 10000, 30000]
reps = [100,  100, 50,  10,   10,   5,     5]


def compute_error(func, pack, result, errors, x, b, alpha, gamma):

    code = "a = func(x, b, gamma, alpha)[0].reshape(b.shape);\
            err = ( (alpha * a + K(x, x, a, gamma) - b) ** 2).sum();\
            errors.append(err);"

    if pack == 'numpy':
        K = Genred_np(formula, aliases, axis=1, dtype='float32')
    else:
        K = Genred_tch(formula, aliases, axis=1, dtype='float32')

    exec(code, locals())
    return errors


def to_bench(funcpack, size, rep):
    global use_cuda
    importlib.reload(torch)
    if device == 'cuda':
        torch.cuda.manual_seed_all(112358)
    else:
        torch.manual_seed(112358)
    code = "func(x, b, gamma, alpha)"
    func, pack = funcpack

    times = []
    errors = []

    if use_cuda:
        torch.cuda.synchronize()
    for i in range(rep+1):  # 0 is a warmup

        x = torch.linspace(1/size, 1, size, dtype=torch.float32,
                           device=device).reshape(size, 1)
        b = torch.randn(size, 1, device=device, dtype=torch.float32)
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
        result = func(x, b, gamma, alpha)
        if use_cuda:
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)
        errors = compute_error(func, pack, result, errors, x, b, alpha, gamma)

    return sum(times)/rep, sum(errors)/rep


def global_bench(functions, sizes, reps):
    list_times = [[] for _ in range(len(functions))]
    list_errors = [[] for _ in range(len(functions))]

    for j, one_to_bench in enumerate(functions):
        for i in range(len(sizes)):
            try:
                time, err = to_bench(one_to_bench, sizes[i], reps[i])
                list_times[j].append(time)
                list_errors[j].append(err)
            except:
                while len(list_times[j]) != len(reps):
                    list_times[j].append(np.nan)
                    list_errors[j].append(np.nan)
                break

        print("Finished", one_to_bench[0], "in a cumulated time of {:3.9f}s.".format(
            sum(list_times[j])))

    return list_times, list_errors

# Stability
# ------------
# Stability of the errors and norm of the iterated approximations of the answer


def norm_stability(size, funcpack):
    errk_scipy, iter_scipy, normx_scipy = [], [], []
    errk_dic, iter_dic, normx_dic = [], [], []
    errk_keops, iter_keops, normx_keops = [], [], []

    def callback_sci(x):
        env = inspect.currentframe().f_back
        errk_scipy.append(env.f_locals['resid'])
        iter_scipy.append(env.f_locals['iter_'])
        normx_scipy.append(sum(env.f_locals['x'] ** 2))

    def callback_kinv_keops(x):
        env = inspect.currentframe().f_back
        errk_keops.append(env.f_locals['nr2'])
        iter_keops.append(env.f_locals['k'])
        normx_keops.append(sum(env.f_locals['a'] ** 2))

    def callback_dic(x):
        env = inspect.currentframe().f_back
        errk_dic.append(env.f_locals["scal1"])
        iter_dic.append(env.f_locals['iter_'])
        normx_dic.append(sum(env.f_locals['x'] ** 2))

    callback_list = [callback_sci, callback_kinv_keops, callback_dic]

    for i, funcpack in enumerate(funcpack):
        fun, pack = funcpack

        if device == 'cuda':
            torch.cuda.manual_seed_all(112358)
        else:
            torch.manual_seed(112358)

        x = torch.linspace(1/size, 1, size, dtype=torch.float32,
                           device=device).reshape(size, 1)
        b = torch.randn(size, 1, device=device, dtype=torch.float32)
        # kernel bandwidth
        gamma = torch.ones(
            1, device=device, dtype=torch.float32) * .5 / .01 ** 2
        # regularization
        alpha = torch.ones(1, device=device, dtype=torch.float32) * 2

        if pack == 'numpy':
            x, b = x.cpu().numpy().astype("float32"), b.cpu().numpy().astype("float32")
            gamma, alpha = gamma.cpu().numpy().astype(
                "float32"), alpha.cpu().numpy().astype("float32")

        fun(x, b, gamma, alpha, callback=callback_list[i])

    return errk_scipy, iter_scipy, normx_scipy, errk_dic, iter_dic, normx_keops, errk_keops, iter_keops, normx_dic


# Is the condition number too big ? 
# -------------------------------------
# The argument ``check_cond`` lets the user have an idea of the conditioning number of the matrix :math:`A=(K_{x,x} + \alpha Id)`. A warning appears
# if :math:`\mathrm{cond}(A)>500`. The user is also warned if the CG algorithm reached its maximum number of iterations *ie* did not converge.


def test_cond(device, size, pack, alpha):
    if device == 'cuda':
        torch.cuda.manual_seed_all(1234)
    else:
        torch.manual_seed(1234)

    x = torch.linspace(1/size, 1, size, dtype=torch.float32,
                       device=device).reshape(size, 1)
    b = torch.randn(size, 1, device=device, dtype=torch.float32)
    # kernel bandwidth
    gamma = torch.ones(1, device=device, dtype=torch.float32) * .5 / .01 ** 2
    alpha = torch.ones(1, device=device, dtype=torch.float32) * alpha  # regularization

    if pack == 'numpy':
        x, b = x.cpu().numpy().astype("float32"), b.cpu().numpy().astype("float32")
        gamma, alpha = gamma.cpu().numpy().astype(
            "float32"), alpha.cpu().numpy().astype("float32")
        ans = dic_cg_np(x, b, gamma, alpha, check_cond=True)
    else:
        ans = dic_cg_tch(x, b, gamma, alpha, check_cond=True)
    return ans


print("Condition number warnings tests")
print("Small matrix well conditioned (nothing should appear)")
ans = test_cond(device, 5, 'numpy', alpha=1e-6)
print("Large matrix unwell conditioned (a warning should appear)")
ans2 = test_cond(device, 1000, 'numpy', alpha=1e-6)
print("Large matrix unwell conditioned but with a large regularization (nothing should appear)")
ans3 = test_cond(device, 1000, 'numpy', alpha=100)


#########################################
# Plot the results of the benchmarking
#########################################

list_times, list_errors = global_bench(functions, sizes, reps)

onlynum = [(scipy_cg, "numpy"), (keops_np, "numpy"), (dic_cg_np, "numpy")]
errk_scipy, iter_scipy, normx_scipy, errk_dic, iter_dic,\
    normx_keops, errk_keops, iter_keops, normx_dic = norm_stability(
        1000, onlynum)

labels = ["scipy + keops", "keops_np", "keops_tch",
          "dico + keops_np", "dico + keops_tch"]
plt.style.use('ggplot')
plt.figure(figsize=(20, 7))
plt.subplot(221)
for i in range(len(functions)):
    plt.plot(sizes, list_times[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel of size $n\times n$")
plt.ylabel("Computational time (s)")
plt.legend()
plt.subplot(222)
for i in range(len(functions)):
    plt.plot(sizes, list_errors[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel of size $n\times n$")
plt.ylabel(r"Error $||Ax_{k_{end}} -b||^2$")
plt.legend()

plt.subplot(223)
plt.plot(iter_keops, errk_keops, 'o-', label=labels[1])
plt.plot(iter_scipy, errk_scipy, '^-', label=labels[0])
plt.plot(iter_dic, errk_dic, 'x-', label=labels[3])
plt.yscale('log')
plt.xlabel(r"Iteration k")
plt.ylabel(r"Iterates for the error $||r_k||^2$")
plt.legend()

plt.subplot(224)
plt.plot(iter_keops, normx_keops, 'o-', label=labels[1])
plt.plot(iter_scipy, normx_scipy, '^-', label=labels[0])
plt.plot(iter_dic, normx_dic, 'x-', label=labels[3])
plt.yscale('log')
plt.xlabel(r"Iteration k")
plt.ylabel(r"$||x_k||^2$")
plt.legend()

plt.tight_layout()
plt.show()
