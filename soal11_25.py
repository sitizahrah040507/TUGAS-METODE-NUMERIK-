"""
Soal 11.25 - Program Cholesky Decomposition (User-Friendly)

Diuji dengan sistem dari Example 11.2.
"""

import numpy as np

def cholesky_decomposition(A, verbose=True):
    """
    Cholesky Decomposition: A = L @ L^T
    Hanya valid untuk matriks simetris positif definit.

    Parameter:
      A       : matriks simetris positif definit (n x n)
      verbose : tampilkan langkah-langkah

    Return:
      L : matriks lower triangular
    """
    n = A.shape[0]
    if not np.allclose(A, A.T):
        raise ValueError("Matriks A harus simetris!")

    L = np.zeros((n, n), dtype=float)
    if verbose:
        print("\n  Proses Decomposisi:")

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    raise ValueError(f"Matriks tidak positif definit (negatif pada [{i},{i}])")
                L[i, j] = np.sqrt(val)
                if verbose:
                    print(f"    L[{i},{j}] = sqrt({A[i,i]:.4f} - {s:.4f}) = {L[i,j]:.6f}")
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
                if verbose:
                    print(f"    L[{i},{j}] = ({A[i,j]:.4f} - {s:.4f}) / {L[j,j]:.6f} = {L[i,j]:.6f}")
    return L

def cholesky_solve(L, b):
    """Selesaikan A x = b menggunakan L dari Cholesky (L L^T x = b)."""
    n = len(b)
    # Forward: L y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    # Backward: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j, i] * x[j] for j in range(i + 1, n))) / L[i, i]
    return x

if __name__ == "__main__":
    print("=" * 60)
    print("Soal 11.25 - Program Cholesky Decomposition")
    print("=" * 60)

    # Example 11.2 dari buku
    A = np.array([
        [2.5, 1.0, 3.0],
        [1.0, 4.0, 0.5],
        [3.0, 0.5, 6.0]
    ])
    b = np.array([10.0, 6.5, 14.5])  # contoh RHS

    print("\nMatriks A (Example 11.2):")
    print(A)
    print("Vektor b:", b)

    L = cholesky_decomposition(A)

    print("\nMatriks L:")
    print(np.round(L, 6))

    print(f"\nVerifikasi L @ L^T == A: {np.allclose(L @ L.T, A)}")

    x = cholesky_solve(L, b)
    print("\nSolusi:")
    for i, xi in enumerate(x):
        print(f"  x{i+1} = {xi:.6f}")

    print(f"\nVerifikasi A @ x = b: {np.allclose(A @ x, b)}")
    print(f"  Solusi numpy: {np.round(np.linalg.solve(A, b), 6)}")
