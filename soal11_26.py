"""
Soal 11.26 - Program Gauss-Seidel (User-Friendly)

Diuji dengan Example 11.3.
"""

import numpy as np

def gauss_seidel(A, b, x0=None, lam=1.0, es=1.0, max_iter=100, verbose=True):
    """
    Gauss-Seidel dengan relaxasi opsional.

    Parameter:
      A        : matriks koefisien (n x n)
      b        : vektor RHS (n)
      x0       : tebakan awal (default = nol)
      lam      : faktor relaxasi (default = 1.0 = tanpa relaxasi)
      es       : toleransi persen relatif (misal 1.0 = 1%)
      max_iter : maks iterasi
      verbose  : tampilkan tabel iterasi

    Return:
      x, iterations, converged
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    if verbose:
        header = " | ".join([f"x{i+1:>10}" for i in range(n)])
        print(f"\n  {'Iter':>5} | {header} | {'max_err%':>10}")
        print("  " + "-" * (5 + 3 + 13 * n + 14))

    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new_i = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_new_i + (1 - lam) * x[i]

        with np.errstate(divide='ignore', invalid='ignore'):
            ea = np.where(x != 0, np.abs((x - x_old) / x) * 100, np.abs(x - x_old) * 100)
        max_ea = np.max(ea)

        if verbose:
            vals = " | ".join([f"{xi:>10.5f}" for xi in x])
            print(f"  {it:>5} | {vals} | {max_ea:>10.4f}%")

        if max_ea < es:
            if verbose:
                print(f"\n  ✓ Konvergen pada iterasi {it} (error maks = {max_ea:.4f}%)")
            return x, it, True

    if verbose:
        print(f"\n  ✗ Tidak konvergen dalam {max_iter} iterasi")
    return x, max_iter, False

if __name__ == "__main__":
    print("=" * 65)
    print("Soal 11.26 - Program Gauss-Seidel (User-Friendly)")
    print("=" * 65)

    # Example 11.3 dari buku
    A = np.array([
        [8,  2, -2],
        [1,  3,  1],
        [-2, 1,  6]
    ], dtype=float)
    b = np.array([9.0, 6.0, 6.0])

    print("\nMatriks A (Example 11.3):")
    print(A)
    print("Vektor b:", b)

    # Cek diagonal dominan
    print("\nCek diagonal dominan:")
    for i in range(3):
        off = sum(abs(A[i, j]) for j in range(3) if j != i)
        print(f"  Baris {i}: |{A[i,i]}| = {abs(A[i,i])} > {off:.1f} → {'OK' if abs(A[i,i]) > off else 'TIDAK'}")

    x, iters, conv = gauss_seidel(A, b, es=1.0, verbose=True)

    print("\nSolusi Gauss-Seidel:")
    for i, xi in enumerate(x):
        print(f"  x{i+1} = {xi:.6f}")

    x_exact = np.linalg.solve(A, b)
    print("\nSolusi eksak (numpy):")
    for i, xi in enumerate(x_exact):
        print(f"  x{i+1} = {xi:.6f}")
