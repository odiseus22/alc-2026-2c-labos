import numpy as np

#AUXILIAR LABO 3
def norma(v, p=2):
    if p == 'inf':
        res = -1
        for x in v:
            if abs(x) > res:
                res = abs(x)
        return res
    res = 0
    for x in v:
        res += abs(x)**p
    return res**(1/p)

#AUX LABO 0 PERO CON TOLERANCIA
def traspuesta(a):
    filas, columnas = a.shape
    # Inicializamos la matriz llena de ceros, PERO le aclaramos que sea de tipo float
    # para que al guardarle decimales no te los redondee a enteros.
    result = np.zeros((columnas, filas), dtype=float)
    
    for fila in range(filas):
        for columna in range(columnas): 
            result[columna][fila] = a[fila][columna]
    return result  


#AUX LOCAL
def productoEscalar(vectorA, vectorB):
    if vectorA.shape[0] != vectorB.shape[0]:
        return None
    else:
        res = 0.0
        for i in range(vectorA.shape[0]):
            res += vectorA[i] * vectorB[i]
        return res


def multiplicar(matrizA, matrizB):
    return matrizA @ matrizB
    filasA, columnasA = matrizA.shape
    filasB, columnasB = matrizB.shape

    if columnasA != filasB:
        raise Exception("No se puede multiplicar :(")

    res = matrizDeCeros(filasA, columnasB)

    for i in range(filasA):
        for j in range(columnasB):
            for k in range(columnasA):
                res[i, j] += matrizA[i, k] * matrizB[k, j]

    return res

def restar(a, b):
    # ponele que checkeamos que sean == las dim
    if a.shape != b.shape:
        raise Exception("No se puede :(")

    filas, columnas = a.shape
    res = [[0 for _ in range(filas)] for _ in range(columnas)]
    for i in range(filas):
        for j in range(columnas):
            res[i][j] = a[i][j] - b[i][j]

    return res

def esCuadrada(a):
    return a.ndim == 2 and a.shape[0] == a.shape[1]


def esSimetrica(a, tol=1e-10):
    if not esCuadrada(a):
        return False

    filas, columnas = a.shape
    tras = traspuesta(a)

    dif = restar(a, tras)

    for i in range(filas):
        for j in range(columnas):
            if np.abs(dif[i][j]) >= tol:
                return False

    return True




def aplica_normaliza(A,v, tol=1e-15, veces=2):

    for i in range(veces):
        w = A@v #todo ver si se acepta
        norma_w = norma(w,2)

        if norma_w <= tol:
            n = w.size
            w = np.zeros(n)
        else: 
            w = w / norma(w,2)
        v = w

    return w


def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2.(c).
    """
    if not p in [1, 'inf']:
        return None
    if p == 1:
        vectorSumas = []
        for j in range(A.shape[1]):
            sum = 0
            for i in range(A.shape[0]):
                sum += abs(A[i][j])
            vectorSumas.append(sum)
        return np.max(vectorSumas)

    if p == 'inf':
        vectorSumas = []
        for i in range(A.shape[0]):
            sum = 0
            for j in range(A.shape[1]):
                sum += abs(A[i][j])
            vectorSumas.append(sum)
        return np.max(vectorSumas)

#FIN FUNCIONES AUXILIARES

def metpot2k(A, tol=1e-15, K=1000):
    v = np.random.rand(A.shape[1])
    vPrima = aplica_normaliza(A,v,tol,2)
    e = np.dot(vPrima,v) #me sirve para ver cuanto cambio de angulo el vector
    k = 0


    while abs(e-1) > tol and k < K: # si no cambio casi nada, entonces ya esta convergiendo, o llego al maximo K
        v = vPrima
        vPrima = aplica_normaliza(A,v,tol,2)
        e = np.dot(vPrima,v)
        k += 1

    autovect = vPrima
    autoval = (vPrima@A)@vPrima # cociente de Rayleigh
    error = e-1

    return autovect, autoval, k #e # ver si lo piden en test online
    #e


#es recursiva podria hacer un early return guardando un input diciendo cuantos pasos hacer
def diagRH(A, tol=1e-15, K=1000):
    if not esSimetrica(A):
        return None
    n = A.shape[0]
    
    

    auto_v, auto_val, _ = metpot2k(A, tol, K)

    e1 = np.eye(n)[0]
    e1_menos_v = e1-auto_v
    e1_menos_v_norma = norma(e1_menos_v,2)
    HouseHolder = np.eye(n) - 2* np.outer(e1_menos_v,e1_menos_v)  / (e1_menos_v_norma**2)
    if n == 1:
        S = np.eye(1)
        D = A
    
    if n==2:
        S = HouseHolder
        D = HouseHolder @ A @ HouseHolder #np.transpose(HouseHolder) #en el pseudocodigo usa HH traspuesta pero como es simpetrica no es necesario
    else:
        B = HouseHolder @ A @  np.transpose(HouseHolder)
        A_prima = B[1:,1:]
        S_prima, D_prima = diagRH(A_prima,tol,K)
        
        D = np.zeros(B.shape)
        D[0,0] =  auto_val
        D[1:,1:] = D_prima
        S_AUX = np.eye(B.shape[0])
        S_AUX[1:,1:] = S_prima
        S = HouseHolder @ S_AUX

    return S, D

    #CONSIDERACIONES REEMPLAZAR @ Y NP.outer por versiones hechas a mano