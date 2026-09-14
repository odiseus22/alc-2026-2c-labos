import numpy as np

def calculaLU(A):
    if A is None:
        return None, None, 0

    try:
        U = np.array(A, dtype = float)
    except:
        return None, None, 0

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return None, None, 0

    n = U.shape[0]
    L = np.eye(n)
    nops = 0

    for i in range(n - 1):
        pivot = U[i][i]
        
        if pivot == 0:
            return None, None, nops

        for j in range(i + 1, n):
            coef = U[j][i] / pivot
            nops += 1
            
            L[j][i] = coef
            U[j][i] = 0
            
            U[j][i + 1:] = U[j][i + 1:] - coef * U[i][i + 1:]
            
            nops += 2 * (n - i)

    return L, U, nops

def calculaLU(A):
    if A is None:
        return None, None, 0

    try:
        U = np.array(A, dtype=float)
    except:
        return None, None, 0

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return None, None, 0

    n = U.shape[0]
    L = np.eye(n)
    nops = 0

    for i in range(n - 1):
        pivot = U[i][i]
        
        if pivot == 0:
            return None, None, 0

        for j in range(i + 1, n):
            coef = U[j][i] / pivot
            nops += 1
            
            L[j][i] = coef
            U[j][i] = 0
            
            for k in range(i + 1, n):
                U[j][k] = U[j][k] - coef * U[i][k]
                nops += 2


    return L, U, nops

def res_tri(L, b, inferior = True):
    
    n = len(L[0])
    X = np.zeros(n)

    if inferior:
        for i in range(0, n):

            parcial = b[i]
            for j in range(0, i):
                parcial -= (L[i][j] * X[j])

            X[i] = parcial / L[i][i]

    else:
        for i in range(n - 1, -1, -1):
            parcial = b[i]
            for j in range(i + 1, n):
                parcial -= L[i][j] * X[j]
            X[i] = parcial / L[i][i]

    return X

def inversa(A):

    L, U, _ = calculaLU(A)

    if L is None: return None

    n = len(L)
    Y = []
    X = []

    for k in range(0, n):
        if U[k][k] == 0:
            return None    

    for i in range(0, n):
        Y.append(res_tri(L, np.eye(n)[i], inferior = True))

    for i in range(0, n):
        X.append(res_tri(U, Y[i], inferior = False))

    return np.array(X).T


def multMat(A, B):
    n = A.shape[0]
    AB = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                AB[i, j] += A[i, k] * B[k, j]
                
    return AB


def calculaLDV(A):
    LU = calculaLU(A)
    
    if LU == None:
        return None, None, None

    L = LU[0]
    U = LU[1]

    for i in range(0, len(L)):
        if L[i][i] == 0 or U[i][i] == 0:
            return None, None, None

    _, D, _ = calculaLU(U.T)

    if D is None:
        return None, None, None

    V = multMat(inversa(D), U)

    return (L, D, V)

def esSDP(A, atol = 1e-8):

    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > atol:
                return False

    LDV = calculaLDV(A)

    if LDV is None or LDV[0] is None: return False

    D = LDV[1]

    return all([D[i][i] > 0 for i in range(0, len(D))])