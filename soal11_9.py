"""
Soal 11.9 - Gauss-Seidel untuk Sistem Reaktor Berpasangan

15c1 -  3c2 -   c3 = 3800
-3c1 + 18c2 -  6c3 = 1200
-4c1 -   c2 + 12c3 = 2350

es = 5%
"""

import numpy as np

def gauss_seidel(A, b, x0=None, es=5.0, max_iter=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy().astype(float)
    print(f"\n{'Iter':>5} {'c1':>12} {'c2':>12} {'c3':>12} {'max_err%':>12}")
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i, i]
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.where(x != 0, np.abs((x - x_old) / x) * 100, 0)
        max_err = np.max(errors)
        print(f"{it:>5} {x[0]:>12.4f} {x[1]:>12.4f} {x[2]:>12.4f} {max_err:>12.4f}%")
        if max_err < es:
            print(f"\nKonvergen pada iterasi {it}")
            break
    return x

if __name__ == "__main__":
    A = np.array([
        [15, -3,  -1],
        [-3, 18,  -6],
        [-4, -1,  12]
    ], dtype=float)
    b = np.array([3800.0, 1200.0, 2350.0])

    print("=" * 60)
    print("Soal 11.9 - Gauss-Seidel: Sistem Reaktor Berpasangan")
    print("=" * 60)
    print("\nMatriks A:")
    print(A)
    print("Vektor b:", b)

    # Cek diagonal dominan
    for i in range(len(b)):
        off = sum(abs(A[i, j]) for j in range(len(b)) if j != i)
        print(f"  Baris {i}: |A[{i},{i}]|={abs(A[i,i])} vs sum off-diag={off:.2f} → {'OK' if abs(A[i,i]) > off else 'TIDAK dominan'}")

    x = gauss_seidel(A, b, es=5.0)
    print("\nSolusi Gauss-Seidel (konsentrasi g/m³):")
    for i, ci in enumerate(x):
        print(f"  c{i+1} = {ci:.4f}")

    x_exact = np.linalg.solve(A, b)
    print("\nSolusi eksak:")
    for i, ci in enumerate(x_exact):
        print(f"  c{i+1} = {ci:.4f}")
