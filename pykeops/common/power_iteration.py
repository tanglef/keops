import torch
import numpy as np
import warnings
from math import sqrt

from pykeops.common.utils import get_tools
from pykeops.common.cg import cg_dic as cg

#####################
# Power iteration
#####################


def random_draw_np(size, device, dtype='float32'):
    return np.random.rand(size, 1).astype(dtype)


def random_draw_torch(size,  device, dtype=torch.float32):
    return torch.rand(size, 1, device=device, dtype=dtype)


def power_it_ray(linop, size, binding, device, eps=1e-6):
    random = random_draw_np if binding == "numpy" else random_draw_torch
    x = random(size, device)
    maxiter = 10 * size
    k = 0
    while k <= maxiter:
        y = linop(x)
        norm_y = sqrt((y ** 2).sum())
        x = y / norm_y
        norm_x = ((x ** 2).sum())
        lambd_ = (y.T @ x) / norm_x
        if k > 0 and (old_lambd - lambd_) ** 2 <= eps ** 2:
            break
        old_lambd = lambd_
        k += 1
    if (k - 1) == maxiter:
        warnings.warn(
            "Warning ----------- Power iteration method did not converge !")
    return (y.T @ x) / (norm_x ** 2)


def bootleg_inv_power_cond_big(linop, size, binding, device, maxcond=500, maxiter=50):
    lambda_max = power_it_ray(linop, size, binding, device)
    thresh = lambda_max / maxcond
    k = 0
    vp = [maxcond]
    random = random_draw_np if binding == "numpy" else random_draw_torch
    x = random(size, device)
    while k <= maxiter:
        x = cg(linop, x, binding)[0]
        x = x / sqrt((x ** 2).sum())
        vp.append(x.T @ linop(x))
        if vp[k] <= thresh and vp[k-1] <= thresh:
            cond_too_big = True
            break
        k += 1
    if (k - 1) == maxiter:
        cond_too_big = False
    return cond_too_big

################################
# Tests that works
################################

# def linop(x):
#     np.random.seed(1234)
#     A = np.random.rand(10, 10)
#     mat = np.dot(A,A.transpose())
#     return mat @ x

# ans = power_it_ray(linop, 10, "numpy", torch.device('cpu'))
# small = bootleg_inv_power_cond_big(linop, 10, 'numpy', torch.device('cpu'))
# np.random.seed(1234)
# A = np.random.rand(10, 10)
# mat = np.dot(A,A.transpose())
# print(np.allclose(np.linalg.eig(mat)[0][0], ans))
# print("condition number", np.linalg.cond(mat))
# print("condition number too big ?", small)
# print(np.linalg.eig(mat)[0])

# print('##############" 2')

# def linop2(x):
#     return np.eye(10) @ x

# ans = power_it_ray(linop2, 10, "numpy", torch.device('cpu'))
# small = bootleg_inv_power_cond_big(linop2, 10, 'numpy', torch.device('cpu'))
# print(np.allclose(np.linalg.eig(np.eye(10))[0][0], ans))
# print("condition number ", np.linalg.cond(np.eye(10)))
# print("condition number too big ?", small)
# print(np.linalg.eig(np.eye(10))[0])
