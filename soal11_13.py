"""
Soal 11.13 - Gauss-Seidel (a) tanpa relaxasi (b) dengan relaxasi λ=1.2.

 2x1 - 6x2 -   x3 = -38
-3x1 -  x2 +  7x3 = -34
-8x1 +  x2 -  2x3 = -20

es = 5% — atur ulang jika perlu agar konvergen.
"""

import numpy as np

def gauss_seidel_relaxation(A, b, lam=1.0, x0=None, es=5.0, max_iter=100, label=""):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy().astype(float)
    print(f"\n--- {label} ---")
    print(f"{'Iter':>5} {'x1':>12} {'x2':>12} {'x3':>12} {'max_err%':>12}")
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new_i = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_new_i + (1 - lam) * x[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.where(x != 0, np.abs((x - x_old) / x) * 100, 0)
        max_err = np.max(errors)
        print(f"{it:>5} {x[0]:>12.6f} {x[1]:>12.6f} {x[2]:>12.6f} {max_err:>12.4f}%")
        if max_err < es:
            print(f"Konvergen pada iterasi {it}")
            return x
    print("TIDAK konvergen dalam batas iterasi")
    return x

if __name__ == "__main__":
    # Sistem asli
    A_orig = np.array([
        [ 2, -6, -1],
        [-3, -1,  7],
        [-8,  1, -2]
    ], dtype=float)
    b_orig = np.array([-38.0, -34.0, -20.0])

    print("=" * 65)
    print("Soal 11.13 - Gauss-Seidel tanpa & dengan Relaxasi (λ=1.2)")
    print("=" * 65)

    # Cek diagonal dominan sistem asli
    print("\nCek diagonal dominan sistem ASLI:")
    for i in range(3):
        off = sum(abs(A_orig[i, j]) for j in range(3) if j != i)
        print(f"  Baris {i}: |{A_orig[i,i]}| vs {off:.1f} → {'OK' if abs(A_orig[i,i]) > off else 'TIDAK'}")

    # Atur ulang: baris terbesar diagonal dulu
    # x2: baris 1 dominan? |-1| < |-3|+|7|=10 TIDAK
    # Ubah: baris 2 untuk x1 (|-8|>|-6|+|-1|=7 TIDAK juga?)
    # Pindah: coba urutan x3 dari baris 1, x1 dari baris 2, x2 dari baris 0
    # Baris 1: -3x1 - x2 + 7x3 = -34 → |7| > |-3|+|-1|=4 ✓ → x3
    # Baris 2: -8x1 + x2 - 2x3 = -20 → |-8| > |1|+|-2|=3 ✓ → x1
    # Baris 0: 2x1 - 6x2 - x3 = -38  → |-6| > |2|+|-1|=3 ✓ → x2
    A = np.array([
        [-8,  1, -2],
        [ 2, -6, -1],
        [-3, -1,  7]
    ], dtype=float)
    b = np.array([-20.0, -38.0, -34.0])

    print("\nSistem SETELAH diatur ulang:")
    print(A)
    print("b:", b)

    print("\nCek diagonal dominan setelah pengaturan ulang:")
    for i in range(3):
        off = sum(abs(A[i, j]) for j in range(3) if j != i)
        print(f"  Baris {i}: |{A[i,i]}| vs {off:.1f} → {'OK' if abs(A[i,i]) > off else 'TIDAK'}")

    x_exact = np.linalg.solve(A, b)
    print("\nSolusi eksak:", np.round(x_exact, 6))

    # (a) Tanpa relaxasi
    x_a = gauss_seidel_relaxation(A, b, lam=1.0, es=5.0, label="(a) Tanpa Relaxasi (λ=1.0)")
    print("Solusi (a):", np.round(x_a, 6))

    # (b) Dengan relaxasi λ=1.2
    x_b = gauss_seidel_relaxation(A, b, lam=1.2, es=5.0, label="(b) Dengan Relaxasi (λ=1.2)")
    print("Solusi (b):", np.round(x_b, 6))
