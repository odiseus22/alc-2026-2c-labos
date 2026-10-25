import numpy as np

#aux
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

def matrizDeCeros(filas, columnas):
    return np.array([[0.0 for _ in range(columnas)] for _ in range(filas)])


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
#fin aux


def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    res = np.zeros((n, n))
    for j in range(n):
        suma = 0
        while suma == 0:
            for i in range(n):
                res[i, j] = np.random.rand()
                suma += res[i, j]
        for i in range(n):
            res[i, j] = res[i, j] / suma
    return res


def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    res = np.zeros((n, n))
    for j in range(n):
        suma = 0

        while suma == 0 :
            for i in range(n):
                res[i, j] = 1 if np.random.rand() <= thres else 0
                suma += res[i, j]
        for i in range(n):
            res[i, j] = res[i, j] / suma   
    return res

def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    A_t = traspuesta(A)
    M = multiplicar(A_t, A) #demostrar que posee el mismo nucleo  T - I QUE (T-I )^t (T-I)
    S, D = diagRH(M, tol)

    autovectores_nucleo = []

    for i in range(D.shape[0]):
        if np.abs(D[i, i]) <= tol:
            autovectores_nucleo.append(S[:, i])

    if len(autovectores_nucleo) == 0:
        return np.array([])

    res = matrizDeCeros(A.shape[1], len(autovectores_nucleo))

    for i in range(len(autovectores_nucleo)):
        for j in range(res.shape[0]):
            res[j][i] = autovectores_nucleo[i][j]

    return vectorAMatriz(res)

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    elems = {}

    if not len(listado) == 0:
        coord_i, coord_j, valores = listado

        for k in range(len(coord_i)):
            if np.abs(valores[k]) > tol:
                elems[(coord_i[k], coord_j[k])] = valores[k]

    return elems, (m_filas, n_columnas)

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    res = np.zeros(v.shape[0])

    for key, value in A[0].items():
        i = key[0]
        j = key[1]
        res[i] += value * v[j]

    return res


def es_markov(T,tol=1e-6):
    """
    T una matriz cuadrada.
    tol la tolerancia para asumir que una suma es igual a 1.
    Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True

def es_markov_uniforme(T,thres=1e-6):
    """
    T una matriz cuadrada.
    thres la tolerancia para asumir que una entrada es igual a cero.
    Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    if not es_markov(T,thres):
        return False
    # cada columna debe tener entradas iguales entre si o iguales a cero
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        # all close
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True


def esNucleo(A,S,tol=1e-5):
    """
    A una matriz m x n
    S una matriz n x k
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
    """
    for col in S.T:
        res = A @ col
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True

## TESTS
# transiciones_al_azar_continuas
# transiciones_al_azar_uniformes
for i in range(1,100):
    T = transiciones_al_azar_continuas(i)
    assert es_markov(T), f"transiciones_al_azar_continuas fallo para n={i}"
    
    T = transiciones_al_azar_uniformes(i,0.3)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    # Si no atajan casos borde, pueden fallar estos tests. Recuerden que suma de columnas DEBE ser 1, no valen columnas nulas.
    T = transiciones_al_azar_uniformes(i,0.01)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    T = transiciones_al_azar_uniformes(i,0.01)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    
# nucleo
A = np.eye(3)
S = nucleo(A)
assert S.shape[0]==0, "nucleo fallo para matriz identidad"
A[1,1] = 0
S = nucleo(A)
msg = "nucleo fallo para matriz con un cero en diagonal"
assert esNucleo(A,S), msg
assert S.shape==(3,1), msg
assert abs(S[2,0])<1e-2, msg
assert abs(S[0,0])<1e-2, msg

v = np.random.random(5)
v = v / np.linalg.norm(v)
H = np.eye(5) - np.outer(v, v)  # proyección ortogonal
S = nucleo(H)
msg = "nucleo fallo para matriz de proyeccion ortogonal"
assert S.shape==(5,1), msg
v_gen = S[:,0]
v_gen = v_gen / np.linalg.norm(v_gen)
assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg

# crea rala
listado = [[0,17],[3,4],[0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,32,89)
assert dims == (32,89), "crea_rala fallo en dimensiones"
assert A_rala_dict[(0,3)] == 0.5, "crea_rala fallo"
assert A_rala_dict[(17,4)] == 0.25, "crea_rala fallo"
assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,50,50)
assert dims == (50,50), "crea_rala fallo en dimensiones con tol"
assert A_rala_dict.get((32,3)) == 7
assert A_rala_dict[(16,4)] == 0.5
assert A_rala_dict[(5,7)] == 0.25

listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con tol"
assert (1,4) not in A_rala_dict
assert A_rala_dict[(2,5)] == 0.5
assert A_rala_dict[(3,6)] == 0.25
assert len(A_rala_dict) == 2

# caso borde: lista vacia. Esto es una matriz de 0s
listado = []
A_rala_dict, dims = crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con lista vacia"
assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia"

# multiplica rala vector
listado = [[0,1,2],[0,1,2],[1,2,3]]
A_rala = crea_rala(listado,3,3)
v = np.random.random(3)
v = v / np.linalg.norm(v)
res = multiplica_rala_vector(A_rala,v)
A = np.array([[1,0,0],[0,2,0],[0,0,3]])
res_esperado = A @ v
assert np.allclose(res,res_esperado), "multiplica_rala_vector fallo"

A = np.random.random((5,5))
A = A * (A > 0.5) 
listado = [[],[],[]]
for i in range(5):
    for j in range(5):
        listado[0].append(i)
        listado[1].append(j)
        listado[2].append(A[i,j])
        
A_rala = crea_rala(listado,5,5)
v = np.random.random(5)
assert np.allclose(multiplica_rala_vector(A_rala,v), A @ v)