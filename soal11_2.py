"""
Soal 11.2 - Invers Matriks dari Soal 11.1 menggunakan LU Decomposition + Unit Vectors
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

def matrix_inverse_lu(A):
    """
    Menghitung invers matriks A menggunakan LU decomposition
    dengan unit vectors (vektor identitas) sebagai RHS.
    """
    n = A.shape[0]
    lu, piv = lu_factor(A)
    A_inv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        A_inv[:, i] = lu_solve((lu, piv), e)
    return A_inv

if __name__ == "__main__":
    A = np.array([
        [0.8, -0.4,  0.0],
        [-0.4, 0.8, -0.4],
        [0.0, -0.4,  0.8]
    ])

    print("=" * 50)
    print("Soal 11.2 - Invers Matriks via LU + Unit Vectors")
    print("=" * 50)
    print("\nMatriks A:")
    print(A)

    A_inv = matrix_inverse_lu(A)
    print("\nInvers A (A^-1):")
    print(np.round(A_inv, 6))

    # Verifikasi: A @ A_inv harus = I
    product = A @ A_inv
    print("\nVerifikasi A @ A^-1 = I:")
    print(np.round(product, 6))
    print("Benar?", np.allclose(product, np.eye(3)))

    # Bandingkan dengan numpy
    print("\nInvers A (numpy):")
    print(np.round(np.linalg.inv(A), 6))
