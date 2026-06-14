"""
Soal 11.19 - Spectral Condition Number Matriks Hilbert 10x10

Tentukan spectral condition number Hilbert 10x10.
Berapa digit presisi yang hilang?
Selesaikan untuk b = jumlah baris (solusi seharusnya semua 1).
"""

import numpy as np

def hilbert_matrix(n):
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])

if __name__ == "__main__":
    n = 10
    H = hilbert_matrix(n)
    b = np.sum(H, axis=1)  # sehingga solusi eksak = semua 1

    print("=" * 65)
    print(f"Soal 11.19 - Spectral Condition Number Hilbert {n}x{n}")
    print("=" * 65)

    # Spectral condition number = ratio nilai singular terbesar/terkecil
    # = sama dengan cond(A, 2) = ||A||_2 * ||A^-1||_2
    cond = np.linalg.cond(H)
    print(f"\nSpectral condition number: {cond:.6e}")
    digits_lost = np.log10(cond)
    print(f"Digit presisi yang hilang: ~{digits_lost:.1f} digit")
    print(f"Dari ~15 digit float64, tersisa: ~{15 - digits_lost:.1f} digit akurat")

    # Solusi numerik
    x = np.linalg.solve(H, b)
    x_exact = np.ones(n)  # solusi yang diharapkan

    print(f"\nSolusi (seharusnya semua 1):")
    max_err = np.max(np.abs(x - x_exact))
    rel_err = np.max(np.abs(x - x_exact) / np.abs(x_exact)) * 100
    print(f"  Nilai: {np.round(x, 6)}")
    print(f"  Error absolut maks: {max_err:.2e}")
    print(f"  Error relatif maks: {rel_err:.4f}%")

    print(f"""
Kesimpulan:
  Matriks Hilbert 10x10 sangat ill-conditioned (cond ≈ {cond:.1e}).
  Prediksi digit hilang: {digits_lost:.0f}.
  Error numerik yang diamati sesuai ekspektasi dari condition number.
""")
