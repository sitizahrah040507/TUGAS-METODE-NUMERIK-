"""
Soal 11.24 - Program Thomas Algorithm (User-Friendly)

Implementasi Thomas algorithm yang user-friendly.
Diuji dengan sistem dari Example 11.1.
"""

import numpy as np

def thomas_algorithm(diag_lower, diag_main, diag_upper, rhs):
    """
    Menyelesaikan sistem tridiagonal Ax = b menggunakan Thomas Algorithm.

    Parameter:
      diag_lower : array subdiagonal (n-1 elemen)
      diag_main  : array diagonal utama (n elemen)
      diag_upper : array superdiagonal (n-1 elemen)
      rhs        : vektor RHS (n elemen)

    Return:
      x : vektor solusi (n elemen)
    """
    n = len(rhs)
    # Salin agar tidak mengubah input
    a = np.zeros(n); a[1:] = diag_lower  # subdiagonal, a[0] tidak dipakai
    b = diag_main.copy().astype(float)
    c = np.zeros(n); c[:-1] = diag_upper  # superdiagonal, c[-1] tidak dipakai
    d = rhs.copy().astype(float)

    print("\n  Forward Sweep:")
    for i in range(1, n):
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
        print(f"    i={i}: faktor={factor:.6f}, b[{i}]={b[i]:.6f}, d[{i}]={d[i]:.6f}")

    print("\n  Back Substitution:")
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    print(f"    x[{n-1}] = {d[-1]:.6f} / {b[-1]:.6f} = {x[-1]:.6f}")
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
        print(f"    x[{i}] = ({d[i]:.6f} - {c[i]:.6f}*{x[i+1]:.6f}) / {b[i]:.6f} = {x[i]:.6f}")

    return x

def build_tridiagonal_matrix(a, b, c, n):
    """Bangun matriks penuh dari komponen tridiagonal untuk verifikasi."""
    A = np.diag(b) + np.diag(a, -1) + np.diag(c, 1)
    return A

if __name__ == "__main__":
    print("=" * 60)
    print("Soal 11.24 - Program Thomas Algorithm (User-Friendly)")
    print("=" * 60)

    # Test dengan Example 11.1 dari buku
    print("\nTest: Example 11.1")
    print("Sistem:")
    print("[ 2.04  -1     0     0   ] [T1]   [40.8]")
    print("[-1    2.04  -1     0   ] [T2] = [0.8 ]")
    print("[ 0    -1    2.04  -1   ] [T3]   [0.8 ]")
    print("[ 0     0    -1   2.04  ] [T4]   [200.8]")

    diag_main  = np.array([2.04, 2.04, 2.04, 2.04])
    diag_lower = np.array([-1.0, -1.0, -1.0])       # n-1 elemen
    diag_upper = np.array([-1.0, -1.0, -1.0])       # n-1 elemen
    rhs = np.array([40.8, 0.8, 0.8, 200.8])

    x = thomas_algorithm(diag_lower, diag_main, diag_upper, rhs)

    A = build_tridiagonal_matrix(diag_lower, diag_main, diag_upper, 4)
    print("\nSolusi Thomas Algorithm:")
    for i, xi in enumerate(x):
        print(f"  T{i+1} = {xi:.6f}")

    print(f"\nVerifikasi A @ x = b: {np.allclose(A @ x, rhs)}")
    print(f"  Residual maks: {np.max(np.abs(A @ x - rhs)):.2e}")

    # Bandingkan dengan numpy
    x_np = np.linalg.solve(A, rhs)
    print("\nReferensi (numpy solve):")
    for i, xi in enumerate(x_np):
        print(f"  T{i+1} = {xi:.6f}")
