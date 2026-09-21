# Tests L05-QR:
from lab05 import QR_con_GS,QR_con_HH,calculaQR
import numpy as np

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)

#arriba test original, abajo test nuevos

# --- TESTS PARA MATRICES LD (Linealmente Dependientes) ---
A_ld2 = np.array([[1., 2.], 
                  [2., 4.]])
Q_ld2, R_ld2 = QR_con_GS(A_ld2)
check_QR(Q_ld2, R_ld2, A_ld2)

A_ld3 = np.array([[1., 2., 3.], 
                  [4., 5., 6.], 
                  [7., 8., 9.]])
Q_ld3, R_ld3 = QR_con_GS(A_ld3)
check_QR(Q_ld3, R_ld3, A_ld3)

# --- TEST PARA MATRIZ CON NEGATIVOS ---
A_neg = np.array([[-1.,  2., -3.],
                  [ 4., -5.,  6.],
                  [-7.,  8., -9.]])
Q_neg, R_neg = QR_con_GS(A_neg)
check_QR(Q_neg, R_neg, A_neg)

# --- TEST DE CANTIDAD DE OPERACIONES ---
def test_nops():
    # Matriz full rank de n x n
    n = 5
    # Creamos una matriz que seguro sea full rank sumándole algo a la identidad
    A_full = np.eye(n) + np.ones((n, n)) * 0.1
    _, _, nops = QR_con_GS(A_full, retorna_nops=True)
    
    # Formula teorica de nops para matriz de nxn full rank:
    # Para cada columna j: 
    #   r = j (es full rank)
    #   El loop k corre j veces: (2n-1) + 2n = 4n - 1 operaciones por vuelta.
    #   Total loop interno: (4n-1) * (n*(n-1)/2)
    #   Fuera del loop k por cada columna: norma (2n ops) + div normalizacion (n ops) = 3n ops.
    #   Total fuera del loop: 3n^2
    expected_nops = n * (n - 1) * (4 * n - 1) // 2 + 3 * n**2
    assert nops == expected_nops, f"Esperado {expected_nops} ops, pero devolvió {nops}"

test_nops()


# --- TESTS PARA MATRICES RECTANGULARES (m > n y m < n) ---
# Caso 1: Rectangular 4x3 (m > n) estándar
A_rect_4x3 = np.array([[1., -1.,  4.],
                       [1.,  4., -2.],
                       [1.,  4.,  2.],
                       [1., -1.,  0.]])

# Caso 2: Rectangular 5x2 (m > n, más alta)
A_rect_5x2 = np.array([[2., 1.],
                       [0., 3.],
                       [1., -1.],
                       [4., 2.],
                       [-1., 0.]])

# Caso 3: Rectangular 4x3 con columnas Linealmente Dependientes (col 2 = 2 * col 0)
A_rect_ld = np.array([[1., 3., 2.],
                      [2., 0., 4.],
                      [0., 1., 0.],
                      [-1., 2., -2.]])

# Caso 4: Matriz ancha 2x3 (m < n) -> Debe retornar None según especificación
A_ancha = np.array([[1., 2., 3.],
                    [4., 5., 6.]])

def test_rectangulares():
    # 1. Test 4x3
    Q_gs, R_gs = QR_con_GS(A_rect_4x3)
    check_QR(Q_gs, R_gs, A_rect_4x3)
    Q_hh, R_hh = QR_con_HH(A_rect_4x3)
    check_QR(Q_hh, R_hh, A_rect_4x3)
    
    # 2. Test 5x2
    Q_gs2, R_gs2 = QR_con_GS(A_rect_5x2)
    check_QR(Q_gs2, R_gs2, A_rect_5x2)
    Q_hh2, R_hh2 = QR_con_HH(A_rect_5x2)
    check_QR(Q_hh2, R_hh2, A_rect_5x2)
    
    # 3. Test con columnas LD en matriz rectangular
    Q_gs_ld, R_gs_ld = QR_con_GS(A_rect_ld)
    check_QR(Q_gs_ld, R_gs_ld, A_rect_ld)
    # En GS debe filtrar la columna LD (rango = 2)
    assert Q_gs_ld.shape == (4, 2), f"Se esperaba shape (4, 2) en Q_gs_ld y dio {Q_gs_ld.shape}"
    assert R_gs_ld.shape == (2, 3), f"Se esperaba shape (2, 3) en R_gs_ld y dio {R_gs_ld.shape}"

    # 4. Test con matriz ancha (m < n): debe rechazar y devolver None
    assert QR_con_GS(A_ancha) is None, "QR_con_GS debería devolver None para m < n"
    assert QR_con_HH(A_ancha) is None, "QR_con_HH debería devolver None para m < n"

    # 5. Test de extras=True en HH con matriz rectangular
    Q_hh_ex, R_hh_ex, extras = QR_con_HH(A_rect_4x3, extras=True)
    check_QR(Q_hh_ex, R_hh_ex, A_rect_4x3)
    assert 'Q_matrices' in extras and 'R_matrices' in extras
    assert len(extras['Q_matrices']) == A_rect_4x3.shape[1] + 1

test_rectangulares()


print("¡Todos los nuevos tests (LD, negativos, nops) pasaron correctamente!")