import numpy as np

#se entrega solo normaExacta y condMC, se permite usar la inversa segun el enunciado, esta en un pie de pagina
#CONDMC PASA, NORMA EXACTA NO

def condMC(A, p):
    A_inv = np.linalg.inv(A)
    norm_A = normaMatMC(A, p, p, 10000)[0]
    norm_A_inv = normaMatMC(A_inv, p, p, 10000)[0]
    return norm_A * norm_A_inv

def normaExacta(A, p=[1, 'inf']):
    if isinstance(p, int) and p == 1:
        return norma1Exacta(A)
    
    if isinstance(p, str) and p == 'inf':
        return normaInfExacta(A)
    
    if isinstance(p, list) and p == [1, 'inf']:
        return [norma1Exacta(A), normaInfExacta(A)]

    return None

def norma1Exacta(A):
    max_sum = 0
    for j in range(A.shape[1]):
        sum = 0 
        for i in range(A.shape[0]):
            sum += abs(A[i][j])
        if sum > max_sum:
            max_sum = sum
    return max_sum

def normaInfExacta(A):
    max_sum = 0
    for i in range(A.shape[0]):
        sum = 0 
        for j in range(A.shape[1]):
            sum += abs(A[i][j])
        if sum > max_sum:
            max_sum = sum
    return max_sum
            

def normaMatMC(A, q, p, Np):
    vectoresRandom = np.random.rand(Np, A.shape[1])
    vectoresNormalizados = normaliza(vectoresRandom, p)
    normas = []
    for x in vectoresNormalizados:
        normas.append(norma(A @ x, q))
        
    max_norma = -1
    mejor_vector = None
    for i in range(len(normas)):
        if normas[i] > max_norma:
            max_norma = normas[i]
            mejor_vector = vectoresNormalizados[i]
    return (max_norma, mejor_vector)

def normaliza(vectores,p=2):
    res = []
    for v in vectores:
        res.append(v/norma(v,p))
    return res

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


#aca debajo va lo que no se entrego o valido al 100% que cumpla con las funciones pedidas








def inversa(A):
    """
    Calcula la inversa de una matriz cuadrada A usando Eliminación de Gauss
    """
    n = A.shape[0]
    # Creamos la matriz aumentada [A | I]
    AI = np.zeros((n, 2*n))
    for i in range(n):
        for j in range(n):
            AI[i, j] = A[i, j]
        AI[i, i+n] = 1.0
        
    for i in range(n):
        # Pivoteo (busca el mayor valor en la columna para evitar inestabilidad)
        max_idx = i
        for k in range(i+1, n):
            if abs(AI[k, i]) > abs(AI[max_idx, i]):
                max_idx = k
        # Intercambio de filas
        if max_idx != i:
            for j in range(2*n):
                temp = AI[i, j]
                AI[i, j] = AI[max_idx, j]
                AI[max_idx, j] = temp
                
        # Hacer el pivote 1
        pivot = AI[i, i]
        for j in range(i, 2*n):
            AI[i, j] /= pivot
            
        # Hacer ceros en el resto de la columna i
        for k in range(n):
            if k != i:
                factor = AI[k, i]
                for j in range(i, 2*n):
                    AI[k, j] -= factor * AI[i, j]
                    
    # Extraer la matriz inversa de la mitad derecha
    A_inv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A_inv[i, j] = AI[i, n+j]
            
    return A_inv


def condExacto(A, p):
    A_inv = inversa(A) # Usamos nuestra propia funcion
    norm_A = normaExacta(A, p)
    norm_A_inv = normaExacta(A_inv, p)
    return norm_A * norm_A_inv




#versiones super optimizadas pero que usan funciones de numpy no autorizadas

# def norma(v,p=2):
#     if p == 'inf':
#         return np.max(np.abs(v))

#     return np.sum(np.abs(v)**p)**(1/p)


# def normaMatMC(A,q,p,Np) :
#     """Devuelve la norma ||A||-{q,p} y el vector x en el cual se alcanza el maximo."""

#     #vectores como columnas (tamaño: M x NP)
#     #usamos randn para generar negativos y positivos, cubriendo toda la esfera
#     vectoresRandom = np.random.randn(A.shape[1],Np)
#     #normalizo por columna
#     if p == 'inf':
#         normas_p = np.max(np.abs(vectoresRandom), axis=0, keepdims=True)
#     else:
#         normas_p = np.sum(np.abs(vectoresRandom)**p, axis=0, keepdims=True)**(1/p)
    

#     vectoresNormalizados = vectoresRandom / normas_p
#     #me da una matriz tamaño (N,Np)
#     Ax = A @ vectoresNormalizados

#     # calculo normas de Ax
#     if q == 'inf':
#         normas = np.max(np.abs(Ax), axis=0)
#     else:
#         normas = np.sum(np.abs(Ax)**q, axis=0)**(1/q)

#     indice_max = np.argmax(normas)
#     max_norma = normas[indice_max]

#     mejor_vector = vectoresNormalizados[:, indice_max]
#     return (max_norma, mejor_vector)