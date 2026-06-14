"""
Soal 11.18 - Sistem Linear: Produksi Elektronik

Komponen:   Tembaga  Seng  Kaca
Transistor:   4       1     2
Resistor:     3       3     1
Chip:         2       1     3

Total material: Tembaga=960, Seng=510, Kaca=610

4T + 3R + 2C = 960
T  + 3R + C  = 510
2T + R  + 3C = 610
"""

import numpy as np

if __name__ == "__main__":
    A = np.array([
        [4, 3, 2],
        [1, 3, 1],
        [2, 1, 3]
    ], dtype=float)
    b = np.array([960.0, 510.0, 610.0])

    print("=" * 55)
    print("Soal 11.18 - Produksi Elektronik (Sistem Linear)")
    print("=" * 55)
    print("\nMatriks koefisien A (bahan per komponen):")
    print("  Kolom: Transistor, Resistor, Chip")
    print("  Baris: Tembaga, Seng, Kaca")
    print(A)
    print("\nVektor ketersediaan bahan (b):", b)

    x = np.linalg.solve(A, b)
    print("\nSolusi jumlah komponen yang diproduksi:")
    print(f"  Transistor: {x[0]:.2f} unit")
    print(f"  Resistor:   {x[1]:.2f} unit")
    print(f"  Chip:       {x[2]:.2f} unit")

    # Verifikasi
    b_check = A @ x
    print("\nVerifikasi penggunaan bahan (harus = b):")
    print(f"  Tembaga: {b_check[0]:.2f} (target {b[0]})")
    print(f"  Seng:    {b_check[1]:.2f} (target {b[1]})")
    print(f"  Kaca:    {b_check[2]:.2f} (target {b[2]})")
    print(f"\n  Akurat: {np.allclose(b_check, b)}")

    # Invers matriks untuk fleksibilitas produksi
    A_inv = np.linalg.inv(A)
    print("\nInvers A (berguna jika material berubah minggu depan):")
    print(np.round(A_inv, 6))
