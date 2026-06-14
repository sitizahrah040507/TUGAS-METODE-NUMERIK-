"""
Soal 11.6 - Cholesky Decomposition (dikerjakan langkah demi langkah)

Sistem simetris:
[  8  20  15] [x1]   [ 50]
[ 20  80  50] [x2] = [250]
[ 15  50  60] [x3]   [100]
"""

import numpy as np

def cholesky_step_by_step(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    print("\nProses Cholesky Decomposition (langkah demi langkah):")
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                L[i, j] = np.sqrt(val)
                print(f"  L[{i},{j}] = sqrt(A[{i},{i}] - {s:.4f}) = sqrt({val:.4f}) = {L[i,j]:.6f}")
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
                print(f"  L[{i},{j}] = (A[{i},{j}] - {s:.4f}) / L[{j},{j}] = {L[i,j]:.6f}")
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
        [ 8, 20, 15],
        [20, 80, 50],
        [15, 50, 60]
    ], dtype=float)

    b = np.array([50.0, 250.0, 100.0])

    print("=" * 55)
    print("Soal 11.6 - Cholesky Decomposition (by hand)")
    print("=" * 55)
    print("\nMatriks A:")
    print(A)

    L = cholesky_step_by_step(A)
    print("\nMatriks L:")
    print(np.round(L, 6))

    print("\nVerifikasi L @ L^T == A:", np.allclose(L @ L.T, A))

    y = forward_substitution(L, b)
    x = backward_substitution(L.T, y)

    print("\nSolusi:")
    for i, xi in enumerate(x):
        print(f"  x{i+1} = {xi:.6f}")

    print("\nVerifikasi A @ x = b:", np.allclose(A @ x, b))
