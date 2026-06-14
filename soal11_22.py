"""
Soal 11.22 - Penulisan Sistem Persamaan dalam Bentuk Matriks

50  = 5x3 - 7x2
4x2 + 7x3 + 30 = 0
x1  - 7x3 = 40 - 3x2 + 5x1

Tuliskan dalam bentuk Ax = b, lalu selesaikan.
"""

import numpy as np

if __name__ == "__main__":
    print("=" * 60)
    print("Soal 11.22 - Sistem ke Bentuk Matriks Ax = b")
    print("=" * 60)

    print("""
Persamaan asli:
  (1) 50  = 5x3 - 7x2
  (2) 4x2 + 7x3 + 30 = 0
  (3) x1 - 7x3 = 40 - 3x2 + 5x1

Atur ulang:
  (1)  0·x1 - 7·x2 + 5·x3 = 50
  (2)  0·x1 + 4·x2 + 7·x3 = -30
  (3) (1-5)·x1 + 3·x2 - 7·x3 = 40 → -4·x1 + 3·x2 - 7·x3 = 40
""")

    A = np.array([
        [ 0, -7,  5],
        [ 0,  4,  7],
        [-4,  3, -7]
    ], dtype=float)
    b = np.array([50.0, -30.0, 40.0])

    print("Matriks A:")
    print(A)
    print("\nVektor b:", b)

    # Cek singular (kolom pertama semua nol → singular!)
    det = np.linalg.det(A)
    print(f"\nDeterminan A: {det:.6f}")

    if abs(det) < 1e-10:
        print("PERHATIAN: Matriks A SINGULAR! Tidak ada solusi unik.")
        print("(Kolom pertama semua nol karena x1 hanya muncul di baris 3)")
    else:
        x = np.linalg.solve(A, b)
        print("\nSolusi:")
        for i, xi in enumerate(x):
            print(f"  x{i+1} = {xi:.6f}")

        # Invers dan transpose
        A_inv = np.linalg.inv(A)
        print("\nInvers A:")
        print(np.round(A_inv, 6))

    print("\nTranspose A^T:")
    print(A.T)
