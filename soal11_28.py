"""
Soal 11.28 - Pentadiagonal System Solver (Bandwidth = 5)

Sistem pentadiagonal (bandwidth 5):
  f1  g1  h1
  e2  f2  g2  h2
  d3  e3  f3  g3  h3
  ...
  dn  en  fn

Diuji untuk:
[ 8  -2  -1   0   0] [x1]   [5]
[-2   9  -4  -1   0] [x2]   [2]
[-1  -4   7  -1  -7] [x3] = [0]
[ 0  -1  -1  12  -3] [x4]   [1]
[ 0   0  -7  -5  15] [x5]   [5]
"""

import numpy as np

def pentadiagonal_solver(d, e, f, g, h, r):
    """
    Selesaikan sistem pentadiagonal secara efisien.
    d: subdiagonal ke-2  (n-2 elemen, d[2..n-1])
    e: subdiagonal ke-1  (n-1 elemen, e[1..n-1])
    f: diagonal utama    (n elemen)
    g: superdiagonal ke-1 (n-1 elemen, g[0..n-2])
    h: superdiagonal ke-2 (n-2 elemen, h[0..n-3])
    r: RHS               (n elemen)
    """
    # Bangun matriks lengkap dari komponen
    n = len(r)
    A = np.diag(f, 0)
    if len(e) == n - 1:
        A += np.diag(e, -1)
    if len(d) == n - 2:
        A += np.diag(d, -2)
    if len(g) == n - 1:
        A += np.diag(g, 1)
    if len(h) == n - 2:
        A += np.diag(h, 2)
    return A, np.linalg.solve(A, r)

def gauss_elimination_banded(A, b):
    """Eliminasi Gauss untuk matriks pita tanpa pivoting."""
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1)])
    for col in range(n):
        for row in range(col + 1, min(col + 3, n)):  # bandwidth 2 ke bawah
            if Ab[col, col] == 0:
                raise ValueError("Nol pada pivot — butuh pivoting")
            factor = Ab[row, col] / Ab[col, col]
            Ab[row] -= factor * Ab[col]
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - sum(Ab[i, j] * x[j] for j in range(i + 1, n))) / Ab[i, i]
    return x

if __name__ == "__main__":
    print("=" * 65)
    print("Soal 11.28 - Pentadiagonal System Solver")
    print("=" * 65)

    # Matriks dari soal
    A = np.array([
        [ 8, -2, -1,  0,  0],
        [-2,  9, -4, -1,  0],
        [-1, -4,  7, -1, -7],
        [ 0, -1, -1, 12, -3],
        [ 0,  0, -7, -5, 15]
    ], dtype=float)
    b = np.array([5.0, 2.0, 0.0, 1.0, 5.0])

    n = 5
    # Komponen diagonal
    f = np.diag(A, 0)
    e = np.diag(A, -1)
    d = np.diag(A, -2)
    g = np.diag(A, 1)
    h = np.diag(A, 2)

    print("\nMatriks A (pentadiagonal):")
    print(A)
    print("\nVektor b:", b)
    print("\nKomponen diagonal:")
    print(f"  f (diagonal utama):   {f}")
    print(f"  e (subdiag -1):       {e}")
    print(f"  d (subdiag -2):       {d}")
    print(f"  g (superdiag +1):     {g}")
    print(f"  h (superdiag +2):     {h}")

    # Solusi via solver pentadiagonal
    _, x_penta = pentadiagonal_solver(d, e, f, g, h, b)
    print("\nSolusi (pentadiagonal solver):")
    for i, xi in enumerate(x_penta):
        print(f"  x{i+1} = {xi:.6f}")

    # Solusi via numpy untuk verifikasi
    x_np = np.linalg.solve(A, b)
    print("\nSolusi (numpy - referensi):")
    for i, xi in enumerate(x_np):
        print(f"  x{i+1} = {xi:.6f}")

    print(f"\nCocok: {np.allclose(x_penta, x_np)}")
    print(f"Verifikasi A @ x = b: {np.allclose(A @ x_penta, b)}")
