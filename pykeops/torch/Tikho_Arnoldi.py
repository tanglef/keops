import numpy as np

class tikho_arnoldi:

    def __init__(self, A, b):
        self.A = A
        self.b = b.flatten()
        dim = A.shape[0]
        if dim <= 100:
            self.arno_iter = dim
        else:
            self.arno_iter = int(dim ** (.5))
    
    def __call__(self):
        return self.Arnoldi()

    def Arnoldi(self, thresh=1e-16):
        A, b, nb_iter = self.A, self.b, self.arno_iter
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






if __name__ == '__main__':
    n = 10000
    A = np.random.rand(n, n)
    b = np.random.rand(n, 1)
    solveur = tikho_arnoldi(A, b)
    Q, H = solveur()
    # print(np.allclose(A@Q[:, :-1], Q@H)) # True, just had to make sure for my sanity's sake