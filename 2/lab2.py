import numpy as np
import math

def rota(theta):
    """
    Recibe un angulo theta y retorna una matriz de 2x2
    que rota un vector dado en un angulo theta
    """
    res =np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])
    return res

def escala(s):
    """
    Recibe una tira de numeros s y retorna una matriz cuadrada de
    n x n, donde n es el tamaño de s.
    La matriz escala la componente i de un vector de Rn
    en un factor s[i]
    """
    res = np.diag(s)
    return res


def rota_y_escala(theta,s):
    """
    Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s
    """
    res = escala(s)@rota(theta)
    return res

def afin(theta,s,b):
    """
    Recibe un angulo theta , una tira de numeros s (en R2) , y un vector
    b en R2.
    Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    luego lo escala en un factor s y por ultimo lo mueve en un valor
    fijo b
    """
    matriz2x2 = rota_y_escala(theta,s)
    matriz3x3 = np.zeros((3, 3))
    matriz3x3[:2, :2] = matriz2x2
    matriz3x3[0][2] = b[0]
    matriz3x3[1][2] = b[1]
    matriz3x3[2][2] = 1
    return matriz3x3

def trans_afin(v,theta,s,b):
    """
    Recibe un vector v (en R2), un angulo theta,
    una tira de numeros s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformacion afin a
    v
    """
    transf = afin(theta,s,b)

    vectorCon1 =  np.append(v, 1)
    vectorColumna = vectorCon1.T
    vectorColumnaRes = transf@vectorColumna
    res = vectorColumnaRes.T[:2]
    return res