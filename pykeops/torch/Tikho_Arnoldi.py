import numpy as np
from numpy.linalg import qr, svd

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


def bicgstab(A, b, epsilon=1e-6): # 1 iter is *2 cost of cg
    nb_iter = max(A.shape[0], A.shape[1])
    x0 = np.zeros((A.shape[1], 1))
    r = b - A @ x0
    rchap = np.random.rand(r.shape[0], 1)
    rho = alpha = omega = 1
    nu = p = np.zeros((A.shape[1], 1))
    k = 0
    while k <= nb_iter:
        rho_step = rchap.T @ r
        beta = rho / rho_step * alpha / omega
        p = r + beta * (p - omega * nu)
        nu = A @ p
        alpha = rho / (rchap.T @ nu)
        s = r - alpha * nu
        t = A @ s
        omega = (t.T @ s) / (t.T @ t)
        x0 = x0 + alpha * p + omega * s
        accu = (A @ x0) - b
        if accu.T @ accu < epsilon:
            return x0
        r = s - omega * t
        k += 1
    return x0        



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


################# Stoch SVD ################

def svd_trunc_stoch(A, b, k=None):
    n = A.shape[0]
    if k == None:
        k = A.shape[1]
    S = np.random.normal(0, 1,(n, k))
    M = np.dot(A, np.dot(A.T, np.dot(A, S)))
    Q, _ = qr(M)
    A_tilde = np.dot(Q, np.dot(Q.T, A))
    U, Sigma, V = svd(Q.T @ A, full_matrices=False)
    U = Q @ U # U tilde = QU, Sigma_tilde = Sigma and V_tilde = V
    x_k = np.dot(V.T, np.dot(np.diag(1/Sigma), np.dot(U.T, b)))
    return (A_tilde, x_k)


################# Lanczos bidiag ################

def lanczos(A, b, l=None, sym=True):
    if l == None:
        l = A.shape[1]
    if sym:      # avoid having to transpose the matrix all the time: more efficient ?
        At = A
    else:
        At = A.T
    sigma = np.sqrt(b.T @ b)
    C = np.zeros((l+1, l))
    V = np.zeros((A.shape[1], l))
    U = np.zeros((A.shape[0], l+1))
    U[:, 0] = b.flatten() / sigma
    vtilde = At @ U[:, 0]
    C[0, 0] = np.sqrt(vtilde.T @ vtilde)
    V[:, 0] = vtilde / C[0, 0]
    for j in range(1, l):
        utilde = np.dot(A, V[:, j-1]) - C[j-1, j-1] * U[:, j-1]
        C[j, j-1] = np.sqrt(utilde.T @ utilde)
        U[:, j] = utilde / C[j, j-1]
        vtilde = np.dot(At, U[:, j]) - C[j, j-1] * V[:, j-1]
        C[j, j] = np.sqrt(vtilde.T @ vtilde)
        V[:, j] = vtilde / C[j, j]
    utilde = np.dot(A, V[:, l-1]) - C[l-1, l-1] * U[:, l-1]
    C[l, l-1] = np.sqrt(utilde.T @ utilde)
    U[:, l] = utilde / C[l, l-1]
    return (U, C, V)


if __name__ == '__main__':
    n = 150
    test_bicgstab, test_cg, test_arno, test_svdstoch = False, False, False, False
    test_lanczos = True
    xi = np.linspace(1/n, 1, num=n, endpoint=True)
    K = np.exp(-(xi[:, np.newaxis] - xi)**2) # define Kernel
    b = np.random.rand(n, 1)
    alpha = .1 # goal is to avoid defining this little one
    if n > 100:
        nb_iter = int(n**.5)
    else:
        nb_iter = n

    if test_arno:
        print("\n","######### Arnoldi decomp ########", "\n")
        Q, H = arnoldi(K, b, nb_iter)
        print(np.allclose(K@Q[:, :-1], Q@H)) # True, just had to make sure for my sanity's sake
        Q_step, H_step = arnoldi_add_one_step(Q, H, K)
        print(np.allclose(K@Q_step[:, :-1], Q_step@H_step))

    if test_svdstoch:
        print("\n","######### SVD Truncated ########", "\n")
        _, x = svd_trunc_stoch(K, b, k=nb_iter)
        print(np.square(np.dot(K, x)- b).sum() / n)
        _, x = svd_trunc_stoch(K + alpha * np.eye(n), b, k=nb_iter)
        print(np.square(np.dot(K + alpha * np.eye(n), x) - b).sum() / n) # Test for reg system with tikho param = .1
        print(np.square(np.dot(K, x) - b).sum() / n)   # Test MSE for solution found using the reg param compare with the init system


    if test_bicgstab:
        print("\n","######### BiCGstab ########", "\n")  # Slow,... you see it for n=1000....
        x = bicgstab(K, b)
        print(np.square(np.dot(K, x)- b).sum() / n)
        x = bicgstab(K + alpha * np.eye(n), b)
        print(np.square(np.dot(K + alpha * np.eye(n), x) - b).sum() / n) # Test for reg system with tikho param = .1
        print(np.square(np.dot(K, x) - b).sum() / n)   # Test MSE for solution found using the reg param compare with the init system


    if test_cg:
        print("\n","######### Conjugate Gradient ########", "\n")
        x = ConjugateGradientSolver(K, b)
        print(np.square(np.dot(K, x)- b).sum() / n)   # Test compute MSE for non reg system
        x = ConjugateGradientSolver(K + alpha * np.eye(n), b)
        print(np.square(np.dot(K + alpha * np.eye(n), x) - b).sum() / n) # Test for reg system with tikho param = .1
        print(np.square(np.dot(K, x) - b).sum() / n)   # Test MSE for solution found using the reg param compare with the init system

    if test_lanczos:
        print("\n","######### Lanczos decomp ########", "\n") # improve because of lost of orthogonality
        U, C, V = lanczos(K, b, nb_iter)
        print(np.allclose(K@V, U@C)) # True, just had to make sure for my sanity's sake
        print(np.allclose(U.T@ U, np.eye(nb_iter+1))) # I lose the orthogonality... Must reortho. How ? That's a good question: GS ?
                                                        # uj = uj - sum (uj-1, uj)uj-1 ??