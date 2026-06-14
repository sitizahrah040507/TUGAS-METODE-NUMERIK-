"""
Soal 11.1 - Thomas Algorithm untuk Sistem Tridiagonal

Sistem:
[ 0.8  -0.4   0  ] [x1]   [ 41 ]
[-0.4   0.8  -0.4] [x2] = [ 25 ]
[ 0    -0.4   0.8] [x3]   [105 ]

(a) Selesaikan dengan Thomas algorithm
(b) Selesaikan dengan LU Decomposition (seperti Example 11.3)
"""

import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Thomas Algorithm (TDMA) untuk sistem tridiagonal.
    a = subdiagonal (bawah)
    b = diagonal utama
    c = superdiagonal (atas)
    d = vektor RHS
    Mengembalikan vektor solusi x.
    """
    n = len(d)
    # Salin agar tidak mengubah input asli
    a = a.copy().astype(float)
    b = b.copy().astype(float)
    c = c.copy().astype(float)
    d = d.copy().astype(float)

    # Forward sweep
    for i in range(1, n):
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def lu_decomposition_solve(A, b):
    """Solusi sistem Ax=b menggunakan scipy LU decomposition."""
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    return x


if __name__ == "__main__":
    # Definisi sistem tridiagonal
    # Diagonal utama
    b = np.array([0.8, 0.8, 0.8])
    # Subdiagonal (a[0] tidak dipakai)
    a = np.array([0.0, -0.4, -0.4])
    # Superdiagonal (c[-1] tidak dipakai)
    c = np.array([-0.4, -0.4, 0.0])
    # RHS
    d = np.array([41.0, 25.0, 105.0])

    # Matriks lengkap untuk verifikasi
    A = np.array([
        [0.8, -0.4,  0.0],
        [-0.4, 0.8, -0.4],
        [0.0, -0.4,  0.8]
    ])

    print("=" * 50)
    print("Soal 11.1 - Sistem Tridiagonal")
    print("=" * 50)
    print("\nMatriks A:")
    print(A)
    print("\nVektor b:", d)

    # (a) Thomas Algorithm
    x_thomas = thomas_algorithm(a, b, c, d)
    print("\n(a) Solusi dengan Thomas Algorithm:")
    for i, xi in enumerate(x_thomas):
        print(f"  x{i+1} = {xi:.6f}")

    # (b) LU Decomposition
    x_lu = lu_decomposition_solve(A, d)
    print("\n(b) Solusi dengan LU Decomposition:")
    for i, xi in enumerate(x_lu):
        print(f"  x{i+1} = {xi:.6f}")

    # Verifikasi
    print("\nVerifikasi A @ x_thomas =", np.allclose(A @ x_thomas, d))
    print("Verifikasi A @ x_lu    =", np.allclose(A @ x_lu, d))
