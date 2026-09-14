import numpy as np

#auxiliar labo 0
def traspuesta(a):
    filas, columnas = a.shape
    # Inicializamos la matriz llena de ceros, PERO le aclaramos que sea de tipo float
    # para que al guardarle decimales no te los redondee a enteros.
    result = np.zeros((columnas, filas), dtype=float)
    
    for fila in range(filas):
        for columna in range(columnas): 
            result[columna][fila] = a[fila][columna]
    return result  

def esCuadrada(a):
    return a.ndim == 2 and a.shape[0] == a.shape[1]

def esSimetrica(a,atol=1e-8):
    if not esCuadrada(a) :
        return False
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(a[i][j] - a[j][i]) > atol:
                return False
    return True
#fin auxiliar


def calculaLU(A):
    if A is None:
        return None, None, 0
        
    try:
        # Esto convierte listas a arrays y fuerza float
        Ac = np.array(A, dtype=float)
    except:
        return None, None, 0
        
    if Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
        return None, None, 0
        
    filas, columnas = Ac.shape
    
    L = np.eye(filas)
    U = Ac  # Usamos Ac directamente como U ya que es una copia segura
    cant_op = 0
    
    # Iterar hasta la anteúltima columna
    for k in range(filas - 1):
        if U[k, k] == 0:
            return None, None, cant_op
            
        for i in range(k + 1, filas):
            # Calcular multiplicador
            m = U[i, k] / U[k, k]
            
            # Guardar en L y vaciar en U
            L[i, k] = m
            U[i, k] = 0

            # Restar a las columnas siguientes usando NumPy (¡súper rápido!)
            U[i, k + 1:] = U[i, k + 1:] - m * U[k, k + 1:]

            cant_op += 1 + (filas - k) * 2

    return L, U, cant_op

def res_tri(L, b, inferior=True):
    n = len(L)
    x = np.zeros(n)

    if inferior:
        # Sustitución hacia adelante (Forward substitution)
        for i in range(n):
            if L[i,i] == 0:
                return None
            suma = 0
            for j in range(i):
                suma += L[i, j] * x[j]
            x[i] = (b[i] - suma) / L[i, i]
    else:
        # Sustitución hacia atrás (Backward substitution)
        for i in range(n - 1, -1, -1):
            if L[i,i] == 0:
                return None
            suma = 0
            for j in range(i + 1, n):
                suma += L[i, j] * x[j]
            x[i] = (b[i] - suma) / L[i, i]
            
    return x

def inversa(A):
    res = calculaLU(A)
    # Si falló la factorización, no podemos calcular la inversa por este método
    if res[0] is None:
        return None
        
    L, U, _ = res
    n = A.shape[0]
    
    # Check if U is singular
    for i in range(n):
        if U[i, i] == 0:
            return None
            
    A_inv = np.zeros((n, n))
    
    # Resolver A * x_i = e_i para cada columna de la identidad
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        
        # 1. Resolver L * y = e
        y = res_tri(L, e, inferior=True)
        # 2. Resolver U * x = y
        x = res_tri(U, y, inferior=False)
        
        # Guardar la solución en la columna i de la inversa
        for j in range(n):
            A_inv[j, i] = x[j]
            
    return A_inv

def calculaLDV(A):
    res = calculaLU(A)
    if res[0] is None:
        return None, None, None
        
    L, U, _ = res
    
    Ut = traspuesta(U)
    LU = calculaLU(Ut)
    if LU is None:
        return None, None, None
        
    V_t, D, _ = LU
    V = traspuesta(V_t)
    
    return L, D, V

def esSDP(A, atol=1e-8):
    if A is None:
        return None
        
    try:
        Ac = np.array(A, dtype=float)
    except:
        return None
        
    if Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
        return None

    n = Ac.shape[0]
    
    # 1. Chequear si es simétrica
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(Ac[i, j] - Ac[j, i]) > atol:
                return None
                
    # 2. Factorización LDV
    res = calculaLDV(Ac)
    if res is None or res[0] is None:
        return None
        
    L, D, V = res
    
    # 3. Chequear si los elementos de la diagonal son estrictamente mayores a cero
    for i in range(n):
        if D[i, i] <= -atol:
            return False
        if abs(D[i, i]) <= atol:
            return None 
            
    return True
