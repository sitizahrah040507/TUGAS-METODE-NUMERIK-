"""
Soal 11.12 - Gauss-Seidel (a) tanpa relaxasi (b) dengan relaxasi λ=0.95

-3x1 +  x2 + 12x3 = 50
 6x1 -  x2 -   x3 = 3
 6x1 + 9x2 +   x3 = 40

es = 5% — atur ulang jika perlu agar konvergen.
"""

import numpy as np

def gauss_seidel_relaxation(A, b, lam=1.0, x0=None, es=5.0, max_iter=100, label=""):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy().astype(float)
    print(f"\n--- {label} ---")
    header = " ".join([f"{'x'+str(i+1):>10}" for i in range(n)])
    print(f"{'Iter':>5} {header} {'max_err%':>10}")
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new_i = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_new_i + (1 - lam) * x[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.where(x != 0, np.abs((x - x_old) / x) * 100, 0)
        max_err = np.max(errors)
        vals = " ".join([f"{xi:>10.5f}" for xi in x])
        print(f"{it:>5} {vals} {max_err:>10.4f}%")
        if max_err < es:
            print(f"Konvergen pada iterasi {it}")
            return x
    print("TIDAK konvergen dalam batas iterasi")
    return x

if __name__ == "__main__":
    # Sistem asli (tidak diagonal dominan — perlu diatur ulang)
    A_orig = np.array([
        [-3,  1, 12],
        [ 6, -1, -1],
        [ 6,  9,  1]
    ], dtype=float)
    b_orig = np.array([50.0, 3.0, 40.0])

    # Atur ulang agar diagonal dominan (tukar baris 0 dan 1 kemudian cek)
    # Baris 0: 6x1 - x2 - x3 = 3  → |6| > |-1|+|-1|=2 ✓
    # Baris 1: -3x1 + x2 + 12x3 = 50 → |12| > |-3|+|1|=4 ✓ (pakai baris ini untuk x3)
    # Baris 2: 6x1 + 9x2 + x3 = 40 → |9| > |6|+|1|=7 ✓
    A = np.array([
        [ 6, -1, -1],
        [-3,  1, 12],
        [ 6,  9,  1]
    ], dtype=float)
    b = np.array([3.0, 50.0, 40.0])

    print("=" * 65)
    print("Soal 11.12 - Gauss-Seidel tanpa & dengan Relaxasi (λ=0.95)")
    print("=" * 65)
    print("\nSistem SETELAH diatur ulang (diagonal dominan):")
    print(A)
    print("b:", b)

    # Cek diagonal dominan
    print("\nCek diagonal dominan setelah pengaturan ulang:")
    for i in range(3):
        off = sum(abs(A[i, j]) for j in range(3) if j != i)
        print(f"  Baris {i}: |{A[i,i]}| vs {off:.1f} → {'OK' if abs(A[i,i]) > off else 'TIDAK'}")

    x_exact = np.linalg.solve(A, b)
    print("\nSolusi eksak:", np.round(x_exact, 6))

    # (a) Tanpa relaxasi
    x_a = gauss_seidel_relaxation(A, b, lam=1.0, es=5.0, label="(a) Tanpa Relaxasi (λ=1.0)")
    print("Solusi (a):", np.round(x_a, 6))

    # (b) Dengan relaxasi λ=0.95
    x_b = gauss_seidel_relaxation(A, b, lam=0.95, es=5.0, label="(b) Dengan Relaxasi (λ=0.95)")
    print("Solusi (b):", np.round(x_b, 6))
