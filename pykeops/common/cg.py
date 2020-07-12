import torch
from pykeops.common.utils import get_tools

##################  Main routines

def cg(linop, b, binding, x=None, eps=None, maxiter=None, inv_precond=None):
    if binding not in ("torch", "numpy", "pytorch"):
        raise ValueError("Language not supported, please use numpy, torch or pytorch.")
    
    tools = get_tools(binding)

    # we don't need cuda with numpy (at least i think so)
    is_cuda = True if (binding == 'torch' or binding == 'pytorch') and torch.cuda.is_available() else False
    device = torch.device("cuda") if is_cuda else torch.device('cpu')

    b, x, replaced = check_dims(b, x, tools, is_cuda)
    n = b.shape[0]

    if eps == None:
        eps = 1e-6
    
    if maxiter == None:
        maxiter = 10*n

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
    data_vect[0:n] = x.reshape(-1)

    iter_ = 0
    job, step = 1, 1

    while iter_ <= maxiter:
        #print("cg", job, step, iter_)
        job, step, x, data_vect, iter_, scal1 ,scal2 = revcom(inv_precond, b, x, job, step, data_vect, iter_, n, eps, scal1, scal2)

        if job == 1: # matrix by vector product
            data_vect[2*n:3*n] = linop(data_vect[n:2*n].reshape(-1, 1)).reshape(-1)
            job, step = 2, 3

        elif job == 2: # precond solver, multiple steps ?
            data_vect[3*n:4*n] = inv_precond(data_vect[0:n])
            job = 2

        elif job == 3: # matrix by x product
            if iter_ == 0 and not replaced:
                data_vect[0:n] = linop(x.reshape(-1, 1)).reshape(-1)
            job, step = 1, 2

        elif job == 4: # check norm errors
            job, resid = should_stop(data_vect[0:n], eps, n)
            if job == -1:
                break;
            else:
                iter_ += 1
                step = 1

    return x#, iter_

def should_stop(data, eps, n):
    nrmresid2 = data.T @ data
    njob = -1 if nrmresid2 <= eps**2 else 2 # the cycle restarts at the precond solver step
    return njob, nrmresid2

def random_draw_np(tools, b):
    return tools.rand(1, 1, dtype=b.dtype)

def random_draw_torch(tools, b):
    return torch.rand(1, 1, device=b.device, dtype=b.dtype)

def revcom(invprecond, b, x, job, step, data, iter_, n, eps, scal1, scal2):
    #print("revcom", job, step, iter_)
    if job == 1: # init step
        if step == 1:
            njob = 3
        else:
            data[0:n] = b.reshape(-1) - data[0:n]
            njob = 4

    elif job == 2: # resume
        if step == 1:
            if invprecond is not None: # we need the zi 
                njob = 2
            step += 1

        if step == 2:
            scal2 = scal1
            if invprecond is None:
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
            x += alpha * data[n:2*n].reshape(-1,1)
            data[0:n] -= alpha * data[2*n:3*n]
            njob = 4
    #print("revcomfin", njob, step, iter_)
    return njob, step, x, data, iter_, scal1, scal2


############### SafeGuard routines

def check_dims(b, x, tools, cuda_avlb): # x is always of b's shape. If the error comes from b it isn't noticed...........
    try:
        nrow, ncol = b.shape
    except ValueError:
        b = b.reshape(-1, 1)

    x_replaced = False

    if x is None: # check x shape and initiate it if needed
        x = tools.zeros((nrow, ncol), dtype=b.dtype, device=torch.device('cuda')) if cuda_avlb \
            else  tools.zeros((nrow, ncol), dtype=b.dtype)
        x_replaced = True
    elif (nrow, ncol) != x.shape: #add sth to check if x is on the same device as b if torch is used!
            if x.shape == (nrow,):
                x = x.reshape((nrow, ncol)) # just recast it
            else:
                raise ValueError("Mismatch between shapes of b {} and shape of x {}.".format((nrow, nrow), x.shape))

    # if M is not None: # check preconditioner dimsfrom pykeops.common.utils import get_tools
    #     if M.shape != (nrow, nrow):
    #         raise ValueError("Mismatch between shapes of M {} and shape of the x {}.".format(M.shape,(nrow, nrow)))
    return b, x, x_replaced 



def another_cg(linop, b, binding, x=None,  eps=None, maxiter=None, inv_precond=None):
    if binding not in ("torch", "numpy", "pytorch"):
        raise ValueError("Language not supported, please use numpy, torch or pytorch.")
    
    tools = get_tools(binding)

    # we don't need cuda with numpy (at least i think so)
    is_cuda = True if (binding == 'torch' or binding == 'pytorch') and torch.cuda.is_available() else False
    if binding == 'torch' or binding == 'pytorch':
        device = torch.device("cuda") if is_cuda else torch.device('cpu')
    b, x, replaced = check_dims(b, x, tools, is_cuda)
    n, m = b.shape

    if eps == None:
        eps = 1e-6
    
    if maxiter == None:
        maxiter = 10*n
    
    rho2 = 1.
    resid = tools.copy(b) if replaced else (b - linop(x))
    iter_ = 1
    while iter_ <= maxiter:
        rho1 =  (resid ** 2).sum()
        if iter_ > 1:
            beta = rho1 / rho2
            p = resid + beta*p
            
        else:
            p = resid

        q = linop(p)
        alpha = rho1 / (p * q).sum()
        x += alpha * p
        resid -= alpha * q
        if (resid ** 2).sum() <= n*m*eps**2:
           break;
        
        rho2 = rho1
        iter_ += 1
    return x#, iter_



#############################################
####### CG_revcom with Python dictionnary
#############################################

def cg_dic(linop, b, binding, x=None, eps=None, maxiter=None, callback=None):
    if binding not in ("torch", "numpy", "pytorch"):
        raise ValueError("Language not supported, please use numpy, torch or pytorch.")
    
    tools = get_tools(binding)

    # we don't need cuda with numpy (at least i think so)
    is_cuda = True if (binding == 'torch' or binding == 'pytorch') and torch.cuda.is_available() else False
    device = torch.device("cuda") if is_cuda else torch.device('cpu')

    b, x, replaced = check_dims(b, x, tools, is_cuda)
    n, m = b.shape

    if eps == None:
        eps = 1e-6
    
    if maxiter == None:
        maxiter = 10*n

    # define the functions needed along the iterations
    if binding == "numpy":
        p, q, r= tools.zeros((n, m), dtype=b.dtype), tools.zeros((n, m), dtype=b.dtype), tools.zeros((n, m), dtype=b.dtype)
        scal1, scal2 = tools.zeros(1, dtype=b.dtype), tools.zeros(1, dtype=b.dtype) # init the scala values

    else:
        p, q, r= tools.zeros((n, m), dtype=b.dtype, device=device), tools.zeros((n, m), dtype=b.dtype, device=device), tools.zeros((n, m), dtype=b.dtype, device=device)
        scal1, scal2 = tools.zeros(1, dtype=b.dtype, device=device), tools.zeros(1, dtype=b.dtype, device=device) # init the scala values


    def init_iter(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_): # revc -> cg
        r = tools.copy(b) if replaced else (b - linop(x))
        job_cg = "check"
        return job_cg, x, r, p, q, scal1, scal2, iter_

    def check_resid(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_): #cg -> revc
        if (r ** 2).sum() <= n*m*eps**2:
           job_rev = "stop"
        else:
            scal2 = tools.copy(scal1)
            iter_ += 1
            job_rev = "direction_next" if iter_ > 1 else "direction_first"
        return job_rev, x, r, p, q, scal1, scal2, iter_ 

    def first_direct(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_): #revc -> cg
        p = tools.copy(r)
        scal1 =  (r ** 2).sum()
        job_cg = "matvec_p"
        return job_cg, x, r, p, q, scal1, scal2, iter_
    
    def matvec_p(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_): #cg -> revc
        q = linop(p)
        job_rev = "update"
        return job_rev, x, r, p, q, scal1, scal2, iter_ 

    def update(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_): #revc -> cg
        alpha = scal1 / (p * q).sum()
        x += alpha * p
        r -= alpha * q
        job_cg = "check"
        return job_cg, x, r, p, q, scal1, scal2, iter_

    def next_direct(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_): # revc -> cg
        scal1 = (r ** 2).sum()
        p = r + (scal1 / scal2) * p
        job_cg = "matvec_p"
        return job_cg, x, r, p, q, scal1, scal2, iter_


    jobs_cg = {"matvec_p" : matvec_p,
            "check" : check_resid
    }

    jobs_revcom = {
        "init" : init_iter,
        "update" : update,
        "direction_first" : first_direct,
        "direction_next" : next_direct
    }

    iter_ = 0
    job_rev = "init"
    job_cg = None

    while iter_ <= maxiter:
        if job_cg == "check" and callback is not None:
            callback(x)
        job_cg, x, r, p, q, scal1, scal2, iter_ = jobs_revcom[job_rev](linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_)
        job_rev, x, r, p, q, scal1, scal2, iter_ = jobs_cg[job_cg](linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_)

        if job_rev == "stop":
            break;

    return x#, iter_
