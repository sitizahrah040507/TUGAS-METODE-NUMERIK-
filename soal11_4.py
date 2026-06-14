"""
Soal 11.4 - Verifikasi Cholesky Decomposition dari Example 11.2

Dari Example 11.2:
[A] = [L][L]^T  →  verifikasi bahwa [L][L]^T = [A]

Matriks dari Example 11.2:
A = [[2.5, 1, 3  ],
     [1,   4, 0.5],
     [3, 0.5, 6  ]]
"""

import numpy as np

def cholesky_decomposition(A):
    """Cholesky decomposition manual: A = L @ L.T"""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

if __name__ == "__main__":
    # Matriks dari Example 11.2
    A = np.array([
        [2.5, 1.0, 3.0],
        [1.0, 4.0, 0.5],
        [3.0, 0.5, 6.0]
    ])

    print("=" * 50)
    print("Soal 11.4 - Verifikasi Cholesky Decomposition")
    print("=" * 50)
    print("\nMatriks A:")
    print(A)

    L = cholesky_decomposition(A)
    print("\nMatriks L (lower triangular):")
    print(np.round(L, 6))

    print("\nMatriks L^T:")
    print(np.round(L.T, 6))

    product = L @ L.T
    print("\nVerifikasi [L][L]^T:")
    print(np.round(product, 6))

    print("\n[L][L]^T == [A]?", np.allclose(product, A))
    print("\nPerbandingan elemen-elemen:")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            print(f"  A[{i},{j}] = {A[i,j]:.4f}, "
                  f"(L@L^T)[{i},{j}] = {product[i,j]:.6f}, "
                  f"Selisih = {abs(A[i,j]-product[i,j]):.2e}")
