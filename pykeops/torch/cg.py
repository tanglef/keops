import torch
from pykeops.common.utils import get_tools

##################  Main routines

def cg(linop, b, binding, x=None, M=None,  eps=None, maxiter=None, regul=None, inv_precond=None):
    if binding not in ("torch", "numpy", "pytorch"):
        raise ValueError("Language not supported, please use numpy, torch or pytorch.")
    
    tools = get_tools(binding)
    if binding == 'torch' or binding == 'pytorch':
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device = torch.device('cuda:0')
    else:
        is_cuda = False


    b, x, M, replaced = check_dims(b, x, M, tools, is_cuda)
    x = x.reshape(-1)
    n = b.shape[0]

    if eps == None:
        eps = 1e-6
    
    if maxiter == None:
        maxiter = 10*n
    
    if regul == None:
        regul = 1/n**.5

    # data is the only saved thing
    # first element is the residuals vectors computed : r_i size n
    # second is the direction vector : p_i size n
    # third are the q_i size n
    # fourth(added in the else of M is None) is the z_i size n

    scal1, scal2 = None, None # init the scala values

    if binding == 'numpy': # set the function to do the random draw in step 4
        random_draw = random_draw_np
        data_vect = tools.zeros(4*n, dtype=b.dtype) #device only for torch!!
    else:
        random_draw = random_draw_torch
        data_vect = tools.zeros(4*n, dtype=b.dtype, device=device) #device only for torch!!

    iter_ = 0
    job, step = 1, 1

    while iter_ <= maxiter:
        #print("cg", job, step, iter_)
        job, step, x, data_vect, iter_, scal1 ,scal2 = revcom(M, b, x, job, step, data_vect, iter_, n, eps, scal1, scal2)

        if job == 1: # matrix by vector product
            data_vect[2*n:3*n] = linop(data_vect[n:2*n].reshape(-1, 1)).reshape(-1)
            job, step = 2, 3

        elif job == 2: # precond solver, multiple steps ?
            data_vect[3*n:4*n] = inv_precond(M, data_vect[0:n])
            job = 2

        elif job == 3: # matrix by x product
            if iter_ == 0 and not replaced:
                data_vect[0:n] = linop(x.reshape(-1, 1)).reshape(-1)
            job, step = 1, 2

        elif job == 4: # check norm errors
            rando = random_draw(tools, b)
            data_vect[0:n] = b.reshape(-1) - linop(x.reshape(-1, 1)).reshape(-1) if rando <= 0.1 else data_vect[0:n] # stocha condi regul residuals
            job, resid = should_stop(data_vect[0:n], eps, n)
            if job == -1:
                break;
            else:
                iter_ += 1
                step = 1

    return x, resid, iter_

def should_stop(data, eps, n):
    nrmresid2 = data[0:n].T @ data[0:n]
    njob = -1 if nrmresid2 <= eps**2 else 2 # the cycle restarts at the precond solver step
    #print("stop", njob)
    return njob, nrmresid2

def random_draw_np(tools, b):
    return tools.rand(1, 1, dtype=b.dtype)

def random_draw_torch(tools, b):
    return torch.rand(1, 1, device=b.device, dtype=b.dtype)

def revcom(M, b, x, job, step, data, iter_, n, eps, scal1, scal2):
    #print("revcom", job, step, iter_)
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
    #print("revcomfin", njob, step, iter_)
    return njob, step, x, data, iter_, scal1, scal2



############### SafeGuard routines

def check_dims(b, x, M, tools, cuda_avlb): # The actual kernel can't be used for the sizes (we only know the linop)
    nrow = b.shape[0]
    x_replaced = False

    if x is None: # check x shape and initiate it if needed
        if cuda_avlb:
            x = tools.zeros((nrow, 1), dtype=b.dtype, device=torch.device('cuda:0'))
        else:
            x = tools.zeros((nrow, 1), dtype=b.dtype)
        x_replaced = True
    elif (nrow, 1) != x.shape: #add sth to check if x is on the same device as b if torch is used!
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