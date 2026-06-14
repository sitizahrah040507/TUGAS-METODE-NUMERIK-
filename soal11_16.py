"""
Soal 11.16 - Solusi, Invers, dan Condition Number (Row-Sum Norm)

(a) Matriks 3x3:
[1  4  9 ] [x1]   [14]
[4  9 16 ] [x2] = [29]
[9 16 25 ] [x3]   [50]

(b) Matriks Hilbert 4x4 (semua x = 1)
"""

import numpy as np

def row_sum_norm(A):
    """Infinity norm (max baris): kondisi tanpa scaling."""
    return np.max(np.sum(np.abs(A), axis=1))

def condition_number_row_sum(A):
    A_inv = np.linalg.inv(A)
    return row_sum_norm(A) * row_sum_norm(A_inv)

def analyze(A, b, label):
    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")
    print(f"Matriks A:\n{A}")
    print(f"Vektor b: {b}")

    x = np.linalg.solve(A, b)
    print(f"\nSolusi x: {np.round(x, 6)}")

    A_inv = np.linalg.inv(A)
    print(f"\nInvers A:\n{np.round(A_inv, 6)}")

    cond_row = condition_number_row_sum(A)
    cond_np = np.linalg.cond(A, np.inf)
    print(f"\nCondition Number (row-sum norm, manual): {cond_row:.4f}")
    print(f"Condition Number (numpy inf-norm):        {cond_np:.4f}")
    print(f"  → ~{int(np.log10(cond_np))} digit presisi yang hilang akibat ill-conditioning")

if __name__ == "__main__":
    print("Soal 11.16 - Solusi, Invers, Condition Number")

    # (a)
    A_a = np.array([
        [1,  4,  9],
        [4,  9, 16],
        [9, 16, 25]
    ], dtype=float)
    b_a = np.array([14.0, 29.0, 50.0])
    analyze(A_a, b_a, "(a) Matriks 3x3")

    # (b) Hilbert 4x4
    n = 4
    A_b = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)], dtype=float)
    b_b = np.sum(A_b, axis=1)   # sehingga semua x = 1
    analyze(A_b, b_b, "(b) Matriks Hilbert 4x4 (solusi semua x=1)")

    print("\nNota: Matriks Hilbert sangat ill-conditioned. Makin besar n, makin buruk.")
