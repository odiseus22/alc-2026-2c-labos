# Tests L08
import numpy as np

def productoEscalar(vectorA, vectorB):
    if vectorA.shape[0] != vectorB.shape[0]:
        return None
    else:
        res = 0.0
        for i in range(vectorA.shape[0]):
            res += vectorA[i] * vectorB[i]
        return res


def restar(a, b):
    # ponele que checkeamos que sean == las dim
    if a.shape != b.shape:
        raise Exception("No se puede :(")

    filas, columnas = a.shape
    res = matrizDeCeros(filas, columnas)
    for i in range(filas):
        for j in range(columnas):
            res[i][j] = a[i][j] - b[i][j]

    return res

def multiplacionMatricialDeVectores(vectorA, vectorB):
    res = np.zeros((vectorA.shape[0], vectorB.shape[0]))

    for i in range(vectorA.shape[0]):
        for j in range(vectorB.shape[0]):
            res[i, j] = vectorA[i] * vectorB[j]
    return res


def matrizPorEscalar(A, c):
    res = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            res[i][j] = A[i][j] * c

    return res


def aplicarMatrizYNormalizar(A, v):
    w = calcularAx(A, v)
    wNorma = norma(w, 2)
    if wNorma <= 1e-15:
        return np.zeros(w.shape[0])
    else:
        w = vectorPorEscalar(w, 1 / wNorma)
    return w

def norma(x, p):
    """
    Devuelve la norma p del vector x.
    """
    if p == 'inf':
        vectorAbs = [0 for _ in range(len(x))]
        for i in range(len(x)):
            vectorAbs[i] = abs(x[i])
        return np.max(vectorAbs)

    sum = 0
    for i in range(len(x)):
        sum += abs(x[i]) ** p

    return sum ** (1 / p)

def calcularAx(matriz, vector_x):
    tamVector = len(vector_x)
    if matriz.shape[1] != tamVector:
        raise Exception("No se puede :(")

    result = [0 for _ in range(matriz.shape[0])]

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            result[i] += matriz[i][j] * vector_x[j]

    return np.array(result) 

def aplicarMatrizKVecesYNormalizar(A, v, k):
    w = v
    for i in range(k):
        w = aplicarMatrizYNormalizar(A, w)
    return w

def metpot2k(A, tol=1e-15, K=1000):
    v = np.random.rand(A.shape[1])
    vPrima = aplicarMatrizKVecesYNormalizar(A, v, 2)

    e = productoEscalar(vPrima, v)
    k = 0
    #print("metpot2k     aplica matriz cantidad de veces: ", end="")
    while np.abs(e - 1) > tol and k < K:
        v = vPrima
        vPrima = aplicarMatrizKVecesYNormalizar(A, v, 2)
        #print(f"\r    {k * 2 + 2}", end="")

        e = productoEscalar(vPrima, v)
        #print(f"\terror: {e}", end="")
        k += 1
    #print()

    autovalor = productoEscalar(vPrima, calcularAx(A, vPrima))

    # if autovalor < tol:
    #     autovalor = 0.0

    error = e - 1
    #print(f"metpot2k     obtuvo autovalor: {autovalor}")

    return [vPrima, autovalor, error]

def restarVectores(a, b):
    if a.shape != b.shape:
        return None

    res = np.zeros(a.shape)

    for i in range(a.shape[0]):
        res[i] = a[i] - b[i]

    return res

def diagRH(A, tol=1e-15, K=1000, simetrica=False, iteracionesFaltantes="max"):
    if simetrica == False:
        # comentar esto si
        if not esSimetrica(A):
            return None
    else:
        simetrica = True

    n = A.shape[0]

    if iteracionesFaltantes != "max" and iteracionesFaltantes == 0:
        S = np.eye(n)
        D = A

    else:
        iteracionesFaltantesImprimir = iteracionesFaltantes
        if iteracionesFaltantesImprimir == "max":
            iteracionesFaltantesImprimir = n
            iteracionesFaltantes = n
        #print(f"metpot2k     es llamada desde diagRH, faltando {iteracionesFaltantesImprimir} recursiones de diagRH")
        autovector, lamda, _ = metpot2k(A, tol, K)

        u = restarVectores(np.eye(n)[0], autovector)
        uNormaAl2 = norma(u, 2) ** 2

        aRestar = matrizPorEscalar(multiplacionMatricialDeVectores(u, u), (2 / uNormaAl2))
        reflectorHouseholder = restar(np.eye(n), aRestar)

        if n == 2:
            S = reflectorHouseholder
            D = multiplicar(reflectorHouseholder, multiplicar(A, traspuesta(reflectorHouseholder)))

        else:
            B = multiplicar(reflectorHouseholder, multiplicar(A, traspuesta(reflectorHouseholder)))
            APrima = B[1:n, 1:n]
            SPrima, DPrima = diagRH(APrima, tol, K, simetrica, iteracionesFaltantes - 1)
            D = matrizPorEscalar(np.eye(A.shape[0]), lamda)
            D[1:n, 1:n] = DPrima
            S = np.eye(A.shape[0])
            S[1:n, 1:n] = SPrima
            S = multiplicar(reflectorHouseholder, S)

    return S, D

def traspuesta(a):
    if a.shape[0] == 0:
        return a

    # Transponer vector
    if len(a.shape) == 1:
        filas, = a.shape
        result = matrizDeCeros(1, filas)
        result = np.array(result)
        for fila in range(filas):
            result[0][fila] = a[fila]
        return result

    # Transponer matriz
    else:
        filas, columnas = a.shape
        result = matrizDeCeros(columnas, filas)
        result = np.array(result)
        for fila in range(filas):
            for columna in range(columnas):
                result[columna][fila] = a[fila][columna]
        return result

def multiplicar(matrizA, matrizB):
    #return matrizA @ matrizB
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


def vectorAMatriz(a):
    return traspuesta(traspuesta(a))


def vectorPorEscalar(x, s):
    res = []
    for i in range(len(x)):
        res.append(x[i] * s)
    return np.array(res)

def matrizDeCeros(filas, columnas):
    return np.array([[0.0 for _ in range(columnas)] for _ in range(filas)])



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


def svd_reducida(A, k="max",tol=1e-15 ):
    
    #print("traspuesta   es llamada desde svd_reducida")
    A_t = traspuesta(A)
    #print("multiplicar  es llamada desde svd_reducida")
    
    #optimizacion:
    fue_traspuesta  = False
    if A.shape[0] < A.shape[1]:
        fue_traspuesta  = True
        A_calculada = traspuesta(A)
        A_calculada_traspuesta = A
    else:
        A_calculada = A
        A_calculada_traspuesta = traspuesta(A)

    M = multiplicar(A_calculada_traspuesta,A_calculada)

    #print("diagRH       es llamada desde svd_reducida")
    S, D = diagRH(M, tol, K=1000, simetrica=True, iteracionesFaltantes=k)

    autovectores = []
    autovalores = []

    if k == "max":
        k = D.shape[0]

    i = 0
    while i < k and np.abs(D[i, i]) >= tol:

        if np.abs(D[i, i]) >= tol and not (D[i, i] < 0):
            autovectores.append(S[:, i])
            autovalores.append(np.sqrt(D[i, i]))

        i += 1

    res = matrizDeCeros(A_calculada.shape[1], len(autovectores))

    for i in range(len(autovectores)):
        for j in range(res.shape[0]):
            res[j][i] = autovectores[i][j]

    V_s = vectorAMatriz(res)
    E_s = np.array(autovalores)

    B = multiplicar(A_calculada, V_s)

    for j in range(B.shape[1]):
        B[:, j] = vectorPorEscalar(B[:, j], 1 / norma(B[:, j], 2))

    U_s = B

    if fue_traspuesta :
        U_final = V_s
        V_final = U_s
    else:
        U_final = U_s
        V_final = V_s

    return U_final, E_s, V_final
    
# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

for m in [2,5,10,20]:
    for n in [2,5,10,20]:
        for _ in range(10):
            A = genera_matriz_para_test(m,n)
            test_svd_reducida_mn(A)


# Matrices con nucleo

m = 12
for tam_nucleo in [2,4,6]:
    for _ in range(10):
        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)

# Tamaños de las reducidas
A = np.random.random((8,6))
for k in [1,3,5]:
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
    assert hU.shape[1] <= k, 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[1] <= k, 'Dimensiones de hV incorrectas(caso a)'
    assert len(hS) <= k, 'Tamaño de hS incorrecto'
