"""
Soal 11.7 - Cholesky Decomposition dari Matriks Diagonal

[A] = [[9, 0, 0],
       [0,25, 0],
       [0, 0, 4]]

Apakah hasilnya masuk akal sesuai Eq. (11.3) dan (11.4)?
"""

import numpy as np

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

if __name__ == "__main__":
    A = np.array([
        [9,  0, 0],
        [0, 25, 0],
        [0,  0, 4]
    ], dtype=float)

    print("=" * 55)
    print("Soal 11.7 - Cholesky Decomposition Matriks Diagonal")
    print("=" * 55)
    print("\nMatriks A:")
    print(A)

    L = cholesky_decomposition(A)
    print("\nMatriks L (Cholesky):")
    print(np.round(L, 6))

    print("\nVerifikasi L @ L^T == A:", np.allclose(L @ L.T, A))
    print("\nL @ L^T =")
    print(np.round(L @ L.T, 6))

    print("""
Analisis (Eq. 11.3 dan 11.4):
  Untuk matriks diagonal, elemen L[i,i] = sqrt(A[i,i]).
  L[0,0] = sqrt(9)  = 3.0
  L[1,1] = sqrt(25) = 5.0
  L[2,2] = sqrt(4)  = 2.0
  Semua elemen di luar diagonal = 0, sesuai dengan matriks diagonal.
  Hasilnya masuk akal: Cholesky dari matriks diagonal menghasilkan
  matriks diagonal dengan elemen = akar kuadrat diagonal A.
""")
