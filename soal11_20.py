"""
Soal 11.20 - Spectral Condition Number Matriks Vandermonde 6x6

x1=4, x2=2, x3=7, x4=10, x5=3, x6=5
Vandermonde[i,j] = x_i^j  (j=0..n-1)
"""

import numpy as np

def vandermonde_matrix(x):
    n = len(x)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            V[i, j] = x[i] ** j
    return V

if __name__ == "__main__":
    x_vals = np.array([4.0, 2.0, 7.0, 10.0, 3.0, 5.0])
    n = len(x_vals)

    V = vandermonde_matrix(x_vals)
    b = np.sum(V, axis=1)  # solusi eksak semua 1

    print("=" * 65)
    print("Soal 11.20 - Spectral Condition Number Vandermonde 6x6")
    print("=" * 65)
    print(f"\nNilai x: {x_vals}")
    print(f"\nMatriks Vandermonde V:\n{V}")

    cond = np.linalg.cond(V)
    digits_lost = np.log10(cond)
    print(f"\nSpectral condition number: {cond:.6e}")
    print(f"Digit presisi yang hilang: ~{digits_lost:.1f} digit")
    print(f"Dari ~15 digit float64, tersisa: ~{15 - digits_lost:.1f} digit akurat")

    x = np.linalg.solve(V, b)
    x_exact = np.ones(n)

    print(f"\nSolusi (seharusnya semua 1):")
    print(f"  Nilai: {np.round(x, 6)}")
    max_err = np.max(np.abs(x - x_exact))
    rel_err = np.max(np.abs(x - x_exact) / np.abs(x_exact)) * 100
    print(f"  Error absolut maks: {max_err:.2e}")
    print(f"  Error relatif maks: {rel_err:.4f}%")

    print(f"""
Perbandingan dengan Hilbert (Soal 11.19):
  Vandermonde bisa juga sangat ill-conditioned, terutama jika
  nilai x menyebar lebar. Semakin besar rentang nilai x dan semakin
  besar n, makin buruk kondisinya.
""")
