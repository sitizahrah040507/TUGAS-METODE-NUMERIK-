"""
Soal 11.11 - Gauss-Seidel untuk Sistem 3x3 hingga es < 5%

10x1 + 2x2  - x3  = 27
-3x1 - 6x2 + 2x3  = -61.5
  x1 +  x2 + 5x3  = -21.5
"""

import numpy as np

def gauss_seidel(A, b, x0=None, es=5.0, max_iter=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy().astype(float)
    header = " ".join([f"{'x'+str(i+1):>12}" for i in range(n)])
    print(f"\n{'Iter':>5} {header} {'max_err%':>12}")
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i, i]
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.where(x != 0, np.abs((x - x_old) / x) * 100, 0)
        max_err = np.max(errors)
        vals = " ".join([f"{xi:>12.6f}" for xi in x])
        print(f"{it:>5} {vals} {max_err:>12.4f}%")
        if max_err < es:
            print(f"\nKonvergen pada iterasi {it}")
            break
    return x

if __name__ == "__main__":
    A = np.array([
        [10,  2, -1],
        [-3, -6,  2],
        [ 1,  1,  5]
    ], dtype=float)
    b = np.array([27.0, -61.5, -21.5])

    print("=" * 65)
    print("Soal 11.11 - Gauss-Seidel 3x3 (es = 5%)")
    print("=" * 65)
    print("\nMatriks A:")
    print(A)
    print("Vektor b:", b)

    # Cek diagonal dominan
    print("\nCek diagonal dominan:")
    for i in range(3):
        off = sum(abs(A[i, j]) for j in range(3) if j != i)
        print(f"  Baris {i}: |{A[i,i]}| vs {off:.1f} → {'OK' if abs(A[i,i]) > off else 'TIDAK'}")

    x = gauss_seidel(A, b, es=5.0)
    print("\nSolusi Gauss-Seidel:")
    for i, xi in enumerate(x):
        print(f"  x{i+1} = {xi:.6f}")

    x_exact = np.linalg.solve(A, b)
    print("\nSolusi eksak:")
    for i, xi in enumerate(x_exact):
        print(f"  x{i+1} = {xi:.6f}")
