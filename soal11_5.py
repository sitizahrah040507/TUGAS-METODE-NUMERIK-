"""
Soal 11.5 - Cholesky Decomposition dan Solusi Sistem Simetris.

Sistem:
[  6   15   55 ] [a0]   [ 152.6 ]
[ 15   55  225 ] [a1] = [ 585.6 ]
[ 55  225  979 ] [a2]   [2488.8 ]
"""

import numpy as np

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val < 0:
                    raise ValueError(f"Matriks tidak positif definit pada indeks ({i},{i})")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x

if __name__ == "__main__":
    A = np.array([
        [  6,  15,   55],
        [ 15,  55,  225],
        [ 55, 225,  979]
    ], dtype=float)

    b = np.array([152.6, 585.6, 2488.8])

    print("=" * 55)
    print("Soal 11.5 - Cholesky Decomposition & Solusi Sistem Simetris")
    print("=" * 55)
    print("\nMatriks A:")
    print(A)
    print("\nVektor b:", b)

    L = cholesky_decomposition(A)
    print("\nMatriks L:")
    print(np.round(L, 6))

    print("\nVerifikasi L @ L^T == A:", np.allclose(L @ L.T, A))

    # Solve Ly = b (forward substitution)
    y = forward_substitution(L, b)
    print("\nSolusi y (dari Ly = b):", np.round(y, 6))

    # Solve L^T x = y (backward substitution)
    x = backward_substitution(L.T, y)
    print("\nSolusi x:")
    labels = ['a0', 'a1', 'a2']
    for i, (lbl, xi) in enumerate(zip(labels, x)):
        print(f"  {lbl} = {xi:.6f}")

    print("\nVerifikasi A @ x = b:", np.allclose(A @ x, b))
    print("Residual:", np.round(A @ x - b, 8))
