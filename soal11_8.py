"""
Soal 11.8 - Gauss-Seidel dengan Overrelaxasi (λ=1.2) untuk Sistem Tridiagonal dari Prob. 11.1

Sistem:
[ 0.8  -0.4   0  ] [x1]   [ 41 ]
[-0.4   0.8  -0.4] [x2] = [ 25 ]
[ 0    -0.4   0.8] [x3]   [105 ]

Gunakan Gauss-Seidel dengan overrelaxasi λ=1.2, es=5%
"""

import numpy as np

def gauss_seidel_relaxation(A, b, x0=None, lam=1.0, es=0.05, max_iter=100):
    """
    Gauss-Seidel dengan overrelaxasi.
    lam: faktor relaxasi (lambda)
    es : toleransi error (persen relatif)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy().astype(float)
    print(f"\nIterasi Gauss-Seidel (λ={lam}, es={es*100}%):")
    print(f"{'Iter':>5} {'x1':>12} {'x2':>12} {'x3':>12} {'max_err%':>12}")

    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new_i = (b[i] - sigma) / A[i, i]
            # Relaxasi
            x[i] = lam * x_new_i + (1 - lam) * x[i]

        # Hitung error relatif maksimum
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.where(x != 0, np.abs((x - x_old) / x) * 100, 0)
        max_err = np.max(errors)

        print(f"{it:>5} {x[0]:>12.6f} {x[1]:>12.6f} {x[2]:>12.6f} {max_err:>12.4f}%")

        if max_err < es * 100:
            print(f"\nKonvergen pada iterasi {it}")
            break

    return x

if __name__ == "__main__":
    A = np.array([
        [0.8, -0.4,  0.0],
        [-0.4, 0.8, -0.4],
        [0.0, -0.4,  0.8]
    ])
    b = np.array([41.0, 25.0, 105.0])

    print("=" * 60)
    print("Soal 11.8 - Gauss-Seidel dengan Overrelaxasi (λ=1.2)")
    print("=" * 60)

    x = gauss_seidel_relaxation(A, b, lam=1.2, es=0.05)

    print("\nSolusi akhir:")
    for i, xi in enumerate(x):
        print(f"  x{i+1} = {xi:.6f}")

    print("\nVerifikasi A @ x = b:", np.allclose(A @ x, b, atol=1e-3))

    # Solusi eksak untuk referensi
    x_exact = np.linalg.solve(A, b)
    print("\nSolusi eksak (numpy):")
    for i, xi in enumerate(x_exact):
        print(f"  x{i+1} = {xi:.6f}")
