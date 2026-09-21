import numpy as np
#AUXILIAR LABO 0
def esCuadrada(a):
    return a.ndim == 2 and a.shape[0] == a.shape[1]
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
#Aux local
def signo(x):
    if x >=0:
        return 1
    else:
        return -1

def multiplacionMatricialDeVectores(vectorA, vectorB):
    res = np.zeros((vectorA.shape[0], vectorB.shape[0]))
    for i in range(vectorA.shape[0]):
        for j in range(vectorB.shape[0]):
            res[i, j] = vectorA[i] * vectorB[j]
    return res

def multiplicar(matrizA, matrizB, performance=True):
    if performance:
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


### Funciones L05-QR

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de m x n (m >= n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es m >= n, debe retornar None
    """
    nops = 0
    filas, columnas = A.shape
    if filas < columnas:
        return None
    Q_cols = []
    R = np.zeros((columnas, columnas))
    r = 0 # Contador de columnas linealmente independientes (el rango)
    for j in range(0, columnas):
        #en la primer columna Ortonormal utiliza la primer columna de A para inicar Q
        q_j = A[:,j]
        #vemos la sombra de A[:,j] en la direccion de las anteriores columnas calculadas
        #el tamaño de la sombra (proyeccion) la guardamos en R[k,j]
        # Proyectamos contra los q_k que YA aceptamos (que son r vectores, no j)
        for k in range(r):
            #R[k,j] es la projeccion_escalar de q_j sobre q_k
            R[k,j] = np.dot(Q_cols[k], A[:,j])
            nops += filas*2 - 1 # filas mult y filas-1 sumas
            #q_j es la resta de A[:,j] menos las projecciones anteriores escaladas
            q_j = q_j - R[k,j] * Q_cols[k]
            nops += 2*filas # filas mult y filas restas
        # en la diagonal de R se guarda la norma de cada columna q_j ya ortogonalizada
        norma_q = norma(q_j,2)
        nops += 2*filas # filas mult y filas-1 sumas y una raiz cuadrada    
        # Aca aplicamos la tolerancia para decidir si es LI o LD
        if norma_q > tol:
            R[r,j] = norma_q
            #se normaliza q_j para que sea ortonormal y se guarda en la matriz Q
            q_j = q_j / norma_q
            Q_cols.append(q_j)
            r += 1
            nops += filas # filas div
        else:
            # Es LD, la norma es casi cero. No lo agregamos a Q.
            pass
    # Transformamos la lista de columnas en una matriz de tamanio filas x r
    if r > 0:
        Q = np.zeros((filas, r))
        for idx in range(r):
            Q[:, idx] = Q_cols[idx]
    else:
        Q = np.zeros((filas, 0))
    # Recortamos R para quedarnos solo con las primeras r filas (matriz r x columnas)
    R = R[:r, :]
    for i in range(r):
        for j in range(columnas):
            if abs(R[i,j]) < tol:
                R[i,j] = 0.0
    if retorna_nops:
        return Q,R,nops
    else:
        return Q,R            
            

def QR_con_HH(A,tol=1e-12,extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve: 
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso


    """
    filas, columnas = A.shape
    if filas < columnas:
        return None
    
    R = np.copy(A)
    
    Q = np.eye(filas)
    
    extra_info = {
        'R_matrices': [],
        'Q_matrices': []
    }

    for k in range(columnas):
        if extras:
            extra_info['R_matrices'].append(np.copy(R))
            extra_info['Q_matrices'].append(np.copy(Q))
            
        # tajada vertical de la submatriz abajo a la derecha a partir de R[k,k]
        x = R[k:,k]

        # norma con - evita cancelacion numerica
        alpha = -signo(x[0]) * norma(x, 2)
        
        #definimos u perpendicular al espejo
        #columna x -  normaNegativa* primer canonico con dimension (m-k+1)
        u = x - alpha* np.eye(filas-k)[0]
        norma_u = norma(u,2)
        if norma_u > tol: #si no lo es, no lo procesamos, ya habia ceros debajo de la diagonal, pasamos a sig columna
            #si norma mayor a cero, reflejamos
            #normalizo
            u = u / norma_u 
            #submatriz de reflectores espejo, dimension (m-k)x(m-k)
            H_k = np.eye(filas-k) - 2 * multiplacionMatricialDeVectores(u,u)
            #expandimos a la dimension de la matriz original, agrgando identidad arriba a la izquierda
            H_kp = np.eye(filas)
            H_kp[k:filas, k:filas] = H_k
            R = multiplicar(H_kp,R)
            Q = multiplicar(Q, H_kp)

    if extras:
        extra_info['R_matrices'].append(np.copy(R))
        extra_info['Q_matrices'].append(np.copy(Q))
        return Q, R, extra_info

    return Q, R
            
            
            
            

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de m x n (m >= n)
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'GS':
        return QR_con_GS(A, tol)
    elif metodo == 'RH':
        return QR_con_HH(A, tol)
    else:
        return None