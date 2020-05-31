import numpy as np

# https://sci-hub.tw/https://iopscience.iop.org/article/10.1088/1361-6420/ab7d2b
# https://www.sciencedirect.com/science/article/pii/S0377042708002252

#### Right now using numpy for tests, must use torch in the future!
############ ConjugateGradientSolver Function in common/operations #############

def ConjugateGradientSolver(linop, b, eps=1e-6):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    delta = np.size(b) * eps ** 2
    nb_iter = np.size(b)
    a = 0
    r = np.copy(b)
    nr2 = (r ** 2).sum()
    if nr2 < delta:
        return 0 * r
    p = np.copy(r)
    k = 0
    while k <= nb_iter:
        Mp = linop@p
        alp = nr2 / (p * Mp).sum()
        a += alp * p
        r -= alp * Mp
        nr2new = (r ** 2).sum()
        if nr2new < delta:
            break
        p = r + (nr2new / nr2) * p
        nr2 = nr2new
        k += 1
    return a


################# Arnoldi decomp (kinda Graam-Schmidt) ################

def arnoldi(A, b, nb_iter, thresh=1e-16):
    if b.ndim > 1:
        b = b.flatten()
    Q = np.zeros((A.shape[0], nb_iter + 1))
    Q[:, 0] = b / np.sqrt((b ** 2).sum())
    H = np.zeros((nb_iter + 1, nb_iter))
    for n in range(nb_iter):
        v = A @ Q[:, n]
        for j in range(n):
            H[j, n] = Q[:, j].T @ v
            v -= H[j, n] * Q[:, j]
        H[n+1, n] = np.sqrt((v **2 ).sum())
        if H[n+1, n] < thresh:
            print("Candidate vector too small at iteration" + str(n) + 
            ". Please note that the matrices are truncated.")
            return Q, H 
        else:
                Q[:, n+1] = v / H[n+1, n]
    return Q, H

def arnoldi_add_one_step(Q, H, A):
    rows, cols = Q.shape # cols = A.shape[0] + 1
    v = A @ Q[:, -1]
    Q = np.c_[Q, np.ones(rows)] # we add one more col
    H = np.c_[H, np.ones(cols)] # we add one more col
    H = np.r_['0, 2', H, np.zeros((1, cols))] #we add one more row (Hessenberg...)
    for j in range(cols):
        H[j, -1] = Q[:, j].T @ v
        v -= H[j, -1] * Q[:, j]
    H[-1, -1] = np.sqrt((v ** 2).sum())
    Q[:, -1] = v / H[-1, -1]
    return Q, H








if __name__ == '__main__':
    n = 102
    xi = np.linspace(1/n, 1, num=n, endpoint=True)
    K = np.exp(-(xi[:, np.newaxis] - xi)**2) # define Kernel
    b = np.random.rand(n, 1)
    alpha = .1 # goal is to avoid defining this little one
    if n > 100:
        nb_iter = int(n**.5)
    else:
        nb_iter = n

    # Arnoldi decomp
    Q, H = arnoldi(K, b, nb_iter)
    print(np.allclose(K@Q[:, :-1], Q@H)) # True, just had to make sure for my sanity's sake
    Q_step, H_step = arnoldi_add_one_step(Q, H, K)
    print(np.allclose(K@Q_step[:, :-1], Q_step@H_step))

    # Conjugate Gradient
    x = ConjugateGradientSolver(K, b)
    print(np.square(np.dot(K, x)- b).sum())   # Test compute MSE for non reg system
    x = ConjugateGradientSolver(K + alpha * np.eye(n), b)
    print(np.square(np.dot(K + alpha * np.eye(n), x) - b).sum()) # Test for reg system with tikho param = .1
    print(np.square(np.dot(K, x) - b).sum())   # Test MSE for solution found using the reg param