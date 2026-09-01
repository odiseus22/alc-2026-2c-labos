import numpy as np

def error(x, y):
    return abs(x - y)

def error_relativo(x, y):
    return abs(x - y) / abs(x)

def matricesIguales(A, B):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    if A.shape != B.shape:
        return False

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i][j] - B[i][j]) >= 1e-07:
                return False

    return True
