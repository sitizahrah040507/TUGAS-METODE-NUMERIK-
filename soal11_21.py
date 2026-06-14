"""
Soal 11.21 - Membuat Matriks Augmented [A | I] dalam Python

Dalam MATLAB: Aug = [A, eye(size(A))]
Ekivalennya dalam Python dengan NumPy: np.hstack([A, np.eye(n)])
"""

import numpy as np

if __name__ == "__main__":
    print("=" * 60)
    print("Soal 11.21 - Matriks Augmented [A | I] dengan NumPy")
    print("=" * 60)

    # Contoh matriks A (3x3)
    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)

    n = A.shape[0]
    I = np.eye(n)

    # Perintah satu baris (ekivalen MATLAB: Aug = [A, eye(size(A))])
    Aug = np.hstack([A, I])

    print(f"\nMatriks A ({n}x{n}):")
    print(A)

    print(f"\nMatriks Identitas I ({n}x{n}):")
    print(I)

    print(f"\nPerintah satu baris Python:")
    print(f"  Aug = np.hstack([A, np.eye(A.shape[0])])")

    print(f"\nMatriks Augmented [A | I] ({n}x{2*n}):")
    print(Aug)

    print("""
Kegunaan matriks augmented [A | I]:
  - Dalam eliminasi Gauss-Jordan, mereduksi [A|I] menjadi [I|A^-1]
  - Menghasilkan invers matriks A setelah proses eliminasi selesai
""")

    # Demonstrasi: invers via Gauss-Jordan
    Aug2 = np.hstack([A.copy(), np.eye(n)])
    for col in range(n):
        # Pivot
        pivot = Aug2[col, col]
        Aug2[col] /= pivot
        for row in range(n):
            if row != col:
                Aug2[row] -= Aug2[row, col] * Aug2[col]

    A_inv_manual = Aug2[:, n:]
    print("Invers A via Gauss-Jordan manual:")
    print(np.round(A_inv_manual, 6))
    print("Invers A via numpy:")
    print(np.round(np.linalg.inv(A), 6))
    print("Cocok?", np.allclose(A_inv_manual, np.linalg.inv(A)))
