from lab04 import calculaLU, res_tri, inversa, calculaLDV, esSDP
import numpy as np

# TESTS L04-LU

# TESTS LU
print("TESTS calculaLU (Tu versión original optimizada)")

L0 = np.array([[1,0,0],
               [0,1,0],
               [1,1,1]])

U0 = np.array([[10,1,0],
               [0,2,1],
               [0,0,1]])

A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(np.allclose(L,L0))
assert(np.allclose(U,U0))


L0 = np.array([[1,0,0],
               [1,1.001,0],
               [1,1,1]])

U0 = np.array([[1,1,1],
               [0,1,1],
               [0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(not np.allclose(L,L0))
assert(not np.allclose(U,U0))
assert(np.allclose(L,L0,atol=1e-3))
assert(np.allclose(U,U0,atol=1e-3))
assert(nops == 19)

L0 = np.array([[1,0,0],
               [1,1,0],
               [1,1,1]])

U0 = np.array([[1,1,1],
               [0,0,1],
               [0,0,1]])

A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(L is None)
assert(U is None)
assert(nops == 14)

assert(calculaLU(None) == (None, None, 0))

assert(calculaLU(np.array([[1,2,3],[4,5,6]])) == (None, None, 0))

# Test matriz 1x1
A1 = np.array([[5]])
L1, U1, nops1 = calculaLU(A1)
assert(np.allclose(L1, [[1]]))
assert(np.allclose(U1, [[5]]))
assert(nops1 == 0)

# Test matriz 2x2
A2 = np.array([[2, 1],
               [4, 5]])
L2, U2, nops2 = calculaLU(A2)
assert(np.allclose(L2, [[1, 0],
                        [2, 1]]))
assert(np.allclose(U2, [[2, 1],
                        [0, 3]]))
assert(nops2 == 5)

# Test matriz con pivote 0
A_cero = np.array([[0, 1],
                   [1, 1]])
L_cero, U_cero, nops_cero = calculaLU(A_cero)
assert(L_cero is None)
assert(U_cero is None)

# Test riguroso de conteo de operaciones para matrices aleatorias NxN
def operaciones_teoricas_LU(N):
    total = 0
    for k in range(N - 1):
        for i in range(k + 1, N):
            total += 1 + 2 * (N - k)
    return total

for size in [3, 4, 10, 20]:
    # Crear matriz definida positiva para evitar pivotes nulos
    A_rand = np.random.rand(size, size)
    A_rand = A_rand.T @ A_rand + np.eye(size) 
    
    L_rand, U_rand, nops_rand = calculaLU(A_rand)
    
    assert(np.allclose(L_rand @ U_rand, A_rand))
    assert(np.allclose(np.tril(L_rand), L_rand))
    assert(np.allclose(np.triu(U_rand), U_rand))
    assert(np.allclose(np.diag(L_rand), np.ones(size)))
    assert(nops_rand == operaciones_teoricas_LU(size))

print("-----ÉXITO!!!!\n")



## TESTS res_tri
print("TESTS res_tri")

A = np.array([[1,0,0],
              [1,1,0],
              [1,1,1]])

b = np.array([1,1,1])
assert(np.allclose(res_tri(A,b),np.array([1,0,0])))

b = np.array([0,1,0])
assert(np.allclose(res_tri(A,b),np.array([0,1,-1])))

b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b),np.array([-1,2,-2])))

b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([-1,1,-1])))

A = np.array([[3,2,1],[0,2,1],[0,0,1]])
b = np.array([3,2,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1/3,1/2,1])))

A = np.array([[1,-1,1],[0,1,-1],[0,0,1]])
b = np.array([1,0,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1,1,1])))
print("-----ÉXITO!!!!\n")


# Test inversa
print("TESTS inversa")

def esSingular(A):
    try:
        np.linalg.inv(A)
        return False
    except:
        return True

# Por que no siempre es invertible, hacemos varios tests
ntest = 10
for i in range(ntest):
    A = np.random.random((4,4))
    A_ = inversa(A)
    if not esSingular(A):
        inversaConNumpy = np.linalg.inv(A)
        assert(A_ is not None)
        assert(np.allclose(inversaConNumpy,A_))
    else: 
        assert(A_ is None)

# Matriz singular devería devolver None
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
assert(inversa(A) is None)

print("-----ÉXITO!!!!\n")



# Test LDV:
print("TESTS calculaLDV")

L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
D0 = np.diag([1,2,3])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ D0 @ V0
L,D,V = calculaLDV(A)
assert(np.allclose(L,L0))
assert(np.allclose(D,D0))
assert(np.allclose(V,V0))


L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
D0 = np.diag([3,2,1])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
A =  L0 @ D0  @ V0
L,D,V = calculaLDV(A)
assert(np.allclose(L,L0,1e-3))
assert(np.allclose(D,D0,1e-3))
assert(np.allclose(V,V0,1e-3))

print("-----ÉXITO!!!!\n")

# TESTS SDP
print("TESTS esSDP")

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
A = L0 @ D0 @ L0.T
assert(esSDP(A))

D0 = np.diag([1,-1,1])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

D0 = np.diag([1,1,1e-16])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

L0 = np.array([[1,0,0],
               [1,1,0],
               [1,1,1]])
D0 = np.diag([1,1,1])
V0 = np.array([[1,0,0],
               [1,1,0],
               [1,1+1e-3,1]]).T
A = L0 @ D0 @ V0
assert(esSDP(A,1e-3))

print("-----ÉXITO!!!!\n")
print("---FINALIZADO LABO 4!---")

