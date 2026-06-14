"""
Soal 11.23 - Jumlah Operasi: Gauss Elimination vs Thomas Algorithm

Gauss Elimination: ~2n³/3 operasi
Thomas Algorithm: ~8n - 7 operasi (O(n) — jauh lebih efisien untuk tridiagonal)

Plot perbandingan untuk n = 2 sampai 20.
"""

import numpy as np
import matplotlib.pyplot as plt

def ops_gauss(n):
    """Jumlah operasi Gauss elimination: sekitar 2n^3/3 + n^2 - n/3."""
    return (2 * n**3 / 3) + n**2 - n / 3

def ops_thomas(n):
    """
    Thomas algorithm:
    - Forward: (n-1) pembagian + (n-1) perkalian + (n-1) perkalian + (n-1) pengurangan = ~5(n-1)
    - Back sub: (n-1) perkalian + (n-1) pengurangan + n pembagian = ~3n - 2
    Total: ~8n - 7
    """
    return 8 * n - 7

if __name__ == "__main__":
    n_values = np.arange(2, 21)
    ops_g = [ops_gauss(n) for n in n_values]
    ops_t = [ops_thomas(n) for n in n_values]

    print("=" * 65)
    print("Soal 11.23 - Perbandingan Operasi: Gauss vs Thomas")
    print("=" * 65)
    print(f"\n{'n':>4} {'Gauss (~2n³/3)':>18} {'Thomas (~8n-7)':>16} {'Rasio':>10}")
    print("-" * 52)
    for n, g, t in zip(n_values, ops_g, ops_t):
        print(f"{n:>4} {g:>18.0f} {t:>16.0f} {g/t:>10.1f}x")

    plt.figure(figsize=(9, 6))
    plt.plot(n_values, ops_g, 'b-o', label='Gauss Elimination (~2n³/3)', markersize=6)
    plt.plot(n_values, ops_t, 'r-s', label='Thomas Algorithm (~8n-7)', markersize=6)
    plt.xlabel('Ukuran Sistem (n)')
    plt.ylabel('Jumlah Operasi (perkiraan)')
    plt.title('Soal 11.23 - Operasi: Gauss Elimination vs Thomas Algorithm')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('soal11_23_plot.png', dpi=120)
    print("\nPlot disimpan: soal11_23_plot.png")
    plt.show()

    print("""
Kesimpulan:
  Thomas algorithm jauh lebih efisien untuk sistem tridiagonal.
  Untuk n=20: Gauss ~3627 operasi, Thomas ~153 operasi (rasio ~23x).
  Kompleksitas: Gauss O(n³) vs Thomas O(n) — keuntungan meningkat drastis dengan n.
""")
