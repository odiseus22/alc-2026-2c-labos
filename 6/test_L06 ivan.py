# Test L06-metpot2k, Aval

import numpy as np

def norma(x: np.ndarray, p: int | str) -> float:
    if p == 1:
        return sum([abs(n) for n in x])

    elif p == "inf":
        max = 0
        for n in x:
            if abs(n) > max:
                max = abs(n)
        return max

    suma = sum([n**p for n in x])
    return np.power(suma, 1 / p)

def normaExacta(A: np.ndarray, p = [1, "inf"]):
    def normaM(A: np.ndarray, p: int | str) -> float:
        if p == "inf":
            sumas_inf = np.zeros(A.shape[0]) 
            for i in range(0, A.shape[0]):
                for j in range(0, A.shape[1]):
                    sumas_inf[i] += abs(A[i][j])
            return max(sumas_inf)

        if p == 1:
            sumas_1 = np.zeros(A.shape[1]) 
            for j in range(0, A.shape[1]):
                for i in range(0, A.shape[0]):
                    sumas_1[j] += abs(A[i][j])
            return max(sumas_1)
        
    if p == "inf": return normaM(A, "inf")
    
    if isinstance(p, int): return normaM(A, p)
    
    if p == [1, "inf"]: return [normaM(A, 1), normaM(A, "inf")]

    return None

def multMat(A, B):
    filasA = A.shape[0]
    columnasB = B.shape[1]
    columnasA = A.shape[1]
    AB = np.zeros((filasA, columnasB))
    
    for i in range(filasA):
        for j in range(columnasB):
            for k in range(columnasA):
                AB[i, j] += A[i, k] * B[k, j]
                
    return AB

def f(A, v):
    res = multMat(A, v)
    return res / norma(res, 2)

def metpot2k(A, tol = 1e-15, K = 1000):

    n = len(A)

    v = np.random.rand(n, 1)
    v_moño = f(A, f(A, v))

    e = multMat(v_moño.T, v)

    k = 0
    while (np.abs(e - 1) > tol) and (k < K):
        v = v_moño
        v_moño = f(A, f(A, v))
        e = multMat(v_moño.T, v)
        k += 1

    autovalor = multMat(v_moño.T, multMat(A, v_moño))
    
    return (v_moño, autovalor[0][0], k)

def diagRH(A, tol = 1e-15, K = 1000):

    n = len(A)
    if n == 1:
        return np.eye(1), A

    v_1, lambda_1, _ = metpot2k(A, tol, K)

    I = np.eye(n, n)
    e_1 = I[:, [0]]
    u = e_1 - v_1
    norma_u = norma(u, 2) ** 2

    if norma_u > tol:
        proyeccion = multMat(u, u.T)
        H_v_1 = I - 2 * (proyeccion / norma_u)
    else:
        H_v_1 = I

    B = multMat(H_v_1, multMat(A, H_v_1.T))
    A_moño = B[1:, 1:]
    
    S_moño, D_moño = diagRH(A_moño, tol, K)

    D = np.zeros((n, n))
    D[0, 0] = lambda_1
    D[1:, 1:] = D_moño

    S = np.eye(n)
    S[1:, 1:] = S_moño
    S = multMat(H_v_1, S)

    return (S, D)
    

#### TESTEOS
# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95



# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95



