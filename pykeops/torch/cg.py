from pykeops.common.utils import get_tools


###########################
###########################
# THIS CODE WONT WORK BC THE KERNEL CANT BE MULTIPPLIED LIKE THAT ! 
# take a look at linop and try to understand it??
# right now, every A @ sth can't be done....
#######################################
######################################



##################  Main routines

def cg(linop, b, binding, x=None, M=None,  eps=None, maxiter=None, regul=None, inv_precond=None):
    if binding not in ("torch", "numpy", "pytorch"):
        raise ValueError("Language not supported, please use numpy, torch or pytorch.")
    
    tools = get_tools(binding)
    ################################
    ##### Not as easy with linop
    ###############################
    b, x, M, replaced = check_dims(b, x, M, tools)
    x = x.reshape(-1)
    n = b.shape[0]

    if eps == None:
        eps = 1e-10
    
    if maxiter == None:
        maxiter = n
    
    if regul == None:
        regul = 1/n**.5

    # data is the only saved thing
    # first element is the residuals vectors computed : r_i size n
    # second is the direction vector : p_i size n
    # third are the q_i size n
    # fourth(added in the else of M is None) is the z_i size n

    data_vect = tools.zeros(4 * n, dtype=b.dtype)
    scal1, scal2 = None, None # init the scala values

    # cg subroutines will have a While not max iter reached
    iter_ = 0
    job, step = 1, 1

    while iter_ <= maxiter:
        print("cg", job, step, iter_)
        job, step, x, data_vect, iter_, scal1 ,scal2 = revcom(M, b, x, job, step, data_vect, iter_, n, eps, scal1, scal2)

        if job == 1: # matrix by vector product
            data_vect[2*n:3*n] = linop(data_vect[n:2*n].reshape(-1, 1))
            job, step = 2, 3

        elif job == 2: # precond solver, multiple steps ?
            data_vect[3*n:4*n] = inv_precond(M, data_vect[0:n])
            job = 2

        elif job == 3: # matrix by x product
            if iter_ == 0 and not replaced:
                data_vect[0:n] = linop(x.reshape(-1, 1))
            job, step = 1, 2

        elif job == 4: # check norm errors
            rando = tools.rand(1, 1)
            print("random", rando <= 0.1)
            data_vect[0:n] = b.reshape(-1) - linop(data_vect[0:n].reshape(-1)) if rando <= 0.1 else data_vect[0:n] # simple stocha condi regul residuals
            job, resid = should_stop(data_vect[0:n], eps, n)
            if job == -1:
                break;
            else:
                iter_ += 1
                step = 1  # in case precond needs mutliple steps

    return x, resid, iter_

def should_stop(data, eps, n):
    nrmresid2 = data[0:n].T @ data[0:n]
    njob = -1 if nrmresid2 <= eps**2 else 2 # the cycle restarts at the precond solver step
    print("stop", njob)
    return njob, nrmresid2

def revcom(M, b, x, job, step, data, iter_, n, eps, scal1, scal2):
    print("revcom", job, step, iter_)
    if job == 1: # init step
        if step == 1:
            njob = 3
        else:
            data[0:n] = b.reshape(-1) - data[0:n]
            njob = 4

    elif job == 2: # resume
        if step == 1:
            if M is not None: # we need the zi 
                njob = 2
            step += 1

        if step == 2:
            scal2 = scal1
            if M is None:
                scal1 = data[0:n].T @ data[0:n]
                if iter_ > 1:
                    print(scal1, scal2)
                    beta = scal1 / scal2 #rho i-1 / rho i-2
                    data[n:2*n] = data[0:n] + beta * data[n:2*n] # r i-1 + beta i-1 p i-1
                else:
                    data[n:2*n] = data[0:n] # p1=r0
            else:
                scal1 = data[0:n].T @ data[3*n:4*n]
                if iter_ > 1:
                    beta = scal1 / scal2 #rho i-1 / rho i-2
                    data[n:2*n] = data[3*n:4*n] + beta * data[n:2*n] # r i-1 + beta i-1 p i-1
                else:
                    data[n:2*n] = data[3*n:4*n] # p1=z0
            njob = 1
        elif step == 3:
            alpha = scal1 / (data[n:2*n].T @ data[2*n:3*n])
            x += alpha * data[n:2*n]
            data[0:n] -= alpha * data[2*n:3*n]
            njob = 4
    print("revcomfin", njob, step, iter_)
    return njob, step, x, data, iter_, scal1, scal2



############# Define the switchers
# each 'values' in the switcher will be
# cg -> revcom : a call to revcom, the only thing changing are in the key arguments
#                1-2 flag for init/resume + a step flag
# revcom -> cg : a flag for what job to do next

def switcher_cg(job):
    switcher = {
        1: "init", #revcom(M, 1, A, vecteurx....)
        2: "resume"
    }
    if job not in (1, 2):
        raise ValueError("Unavailable key called")
    return switcher.get(job)


def switcher_revcom(job, M):
    switcher = {
        -1: "Exit",
        1: "Do Ap_i",
        3: "Do Ax_0",
        4: "Stoping test",
        }
    if M is not None:
        switcher.update({2: "Do precond"})
        if job not in (-1, 1, 3, 4, 2):
            raise ValueError("Unavailable key called")
    elif job not in (-1, 1, 3, 4):
        if job == 2:
            raise ValueError("Preconditioner cannot be called if it's None")
        raise ValueError("Unavailable key called")
    return switcher.get(job)


############### SafeGuard routines

def check_dims(b, x, M, tools): # The actual kernel can't be used for the sizes (we only know the linop)
    nrow = b.shape[0]
    x_replaced = False

    if x is None: # check x shape and initiate it if needed
        x = tools.zeros((nrow, 1), dtype=b.dtype)
        x_replaced = True
    elif (nrow, 1) != x.shape:
            if x.shape == (nrow,):
                x = x.reshape((nrow, 1)) # just recast it
            else:
                raise ValueError("Mismatch between shapes of the kernel {} and shape of x {}.".format((nrow, nrow), x.shape))

    if x.shape != b.shape: # check RHS shape
        if b.shape == (nrow,):
            b = b.reshape((nrow, 1)) # not necessary to throw an error for that, just reshape
        else:
            raise ValueError("Mismatch between shapes of x {} and shape of b {}.".format(x.shape, b.shape))

    if M is not None: # check preconditioner dimsfrom pykeops.common.utils import get_tools

        if M.shape != (nrow, nrow):
            raise ValueError("Mismatch between shapes of M {} and shape of the kernel {}.".format(M.shape,(nrow, nrow)))
    return b, x, M, x_replaced 



############ test
import torch
import numpy as np
#from pykeops.torch import Genred
from pykeops.numpy import Genred
import pykeops


size = 5

formula = "Exp(-SqDist(x, y)) * a"  # Exponential kernel
aliases =  ["x = Vi(1)",  # 1st input: target points, one dim-3 vector per line
            "y = Vj(1)",  # 2nd input : dim-3 per columns
            "a = Vj(1)"]

#xi = torch.linspace(1/size, 1, size).view(-1,1)from pykeops.common.utils import get_tools

#b = torch.rand(size, 1).view(-1,1)
K = Genred(formula, aliases, axis = 1)
xnum = np.linspace(1/size, 1, size).reshape(-1,1)
bnum = np.random.rand(size, 1)

def linop(vect):
    global xnum
    vect = vect.reshape(-1,1)
    return K(xnum, xnum, vect).reshape(-1)
pykeops.verbose = True
print(linop(bnum))
print(cg(linop, bnum, "numpy"))
