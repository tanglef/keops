import numpy as np #replace np by tools at the end bc of the binding 


################## Main routines

def cg(A, b, x=None, M=None,  eps=None, maxiter=None):
    check_dims(A, b, x, M)
    n = A.shape[0]
    if eps == None:
        eps = 1e-10
    
    if maxiter == None:
        maxiter = n

    # data is the only saved thing
    # first element is the residuals vectors computed : r_i size n
    # second is the direction vector : p_i size n
    # third are the q_i size n
    # fourth and fifth are rho (rho_i-1/2) size 2
    # sixth (added in the else of M is None) is the z_i size n
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # upmost an array of 4n + 2
    # ie an array of 4n for the vectors
    # and an array of 2 for the scalars
    data_vect = np.zeros(4 * n, dtype=A.dtype)
    data_scal = np.zeros(2, dtype=A.dtype)

    # cg subroutines will have a While converge do smth
    converge = True

    #replace the values with functions doing the corresponding job

    iter_ = 0

    while True:

        if iter_ > 0:
            switcher_cg(2)

        else:
            switcher_cg(1)


        # call to revcom 
        # returns multiple things including the job to do

    # do not return the whole data
    # only the norm of the last resid
    # the solution vector and whether the algo converged or not
    # be careful of the dim of the sol returned, must be (n, 1) !
    return x, converge


def revcom(M, job, stock_res):
    

    return 



############# Define the switchers
# each 'values' in the switcher will be
# cg -> revcom : a call to revcom, the only thing changing are in the key arguments
#                1-2 flag for init/resume + a step flag
# revcom -> cg : a flag for what job to do next

def switcher_cg(job):
    switcher = {
        1: "init",
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

def check_dims(A, b, x, M):
    nrow = A.shape[0]

    if nrow != A.shape[1]: # check symmetry
        raise ValueError("The matrix is expected to be squared, not of shape {}.".format(A.shape))

    if x is None: # check x shape and initiate it if needed
        x = np.zeros((nrow, 1), dtype=A.dtype)
    elif (nrow, 1) != x.shape:
            if x.shape == (nrow,):
                x = x.reshape((nrow, 1)) # just recast it
            else:
                raise ValueError("Mismatch between shapes of A {} and shape of x {}.".format(A.shape, x.shape))

    if x.shape != b.shape: # check RHS shape
        if b.shape == (nrow,):
            b = b.reshape((nrow, 1)) # not necessary to throw an error for that, just reshape
        else:
            raise ValueError("Mismatch between shapes of x {} and shape of b {}.".format(A.shape, b.shape))

    if M is not None: # check preconditioner dims
        if M.shape != A.shape:
            raise ValueError("Mismatch between shapes of M {} and shape of A {}.".format(M.shape, A.shape))


