"""
Soal 11.27 - ODE Steady-State Kanal 1D → Sistem Linear

ODE: 0 = D·d²c/dx² - U·dc/dx - k·c
D=2, U=1, k=0.2
Syarat batas: c(0)=80, c(10)=20
Diskretisasi dengan Δx=2 (interior: x=2,4,6,8)

Aproksimasi beda hingga:
  d²c/dx² ≈ (c[i+1] - 2c[i] + c[i-1]) / Δx²
  dc/dx   ≈ (c[i+1] - c[i-1]) / (2Δx)
"""

import numpy as np
import matplotlib.pyplot as plt

def build_system(D, U, k, c0, cL, dx, n_interior):
    """
    Bangun sistem Ac=b untuk ODE steady-state 1D.
    n_interior: jumlah titik interior
    """
    n = n_interior
    A = np.zeros((n, n))
    b = np.zeros(n)

    r1 = D / dx**2
    r2 = U / (2 * dx)

    for i in range(n):
        # Koefisien dari diskretisasi beda hingga
        coef_im1 = r1 + r2   # c[i-1]
        coef_i   = -2*r1 - k # c[i]
        coef_ip1 = r1 - r2   # c[i+1]

        A[i, i] = coef_i
        if i > 0:
            A[i, i - 1] = coef_im1
        if i < n - 1:
            A[i, i + 1] = coef_ip1

        # Kondisi batas ke RHS
        if i == 0:
            b[i] -= coef_im1 * c0
        if i == n - 1:
            b[i] -= coef_ip1 * cL

    return A, b

if __name__ == "__main__":
    D   = 2.0
    U   = 1.0
    k   = 0.2
    c0  = 80.0  # c(x=0)
    cL  = 20.0  # c(x=10)
    dx  = 2.0
    L   = 10.0

    n_interior = int(L / dx) - 1  # 4 titik interior: x=2,4,6,8
    x_interior = np.arange(dx, L, dx)  # [2, 4, 6, 8]

    print("=" * 65)
    print("Soal 11.27 - ODE Kanal 1D ke Sistem Linear (Δx=2)")
    print("=" * 65)
    print(f"\nParameter: D={D}, U={U}, k={k}, c(0)={c0}, c(10)={cL}")
    print(f"Titik interior: {x_interior}")

    A, b = build_system(D, U, k, c0, cL, dx, n_interior)
    print(f"\nMatriks A ({n_interior}x{n_interior}):")
    print(np.round(A, 4))
    print(f"\nVektor b: {np.round(b, 4)}")

    c_interior = np.linalg.solve(A, b)
    print(f"\nSolusi konsentrasi di titik interior:")
    for xi, ci in zip(x_interior, c_interior):
        print(f"  c({xi:.0f}) = {ci:.6f}")

    # Titik penuh termasuk batas
    x_full = np.concatenate([[0], x_interior, [L]])
    c_full = np.concatenate([[c0], c_interior, [cL]])

    print(f"\nProfil konsentrasi lengkap:")
    for xi, ci in zip(x_full, c_full):
        print(f"  c({xi:.0f}) = {ci:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(x_full, c_full, 'bo-', label='Solusi numerik (Δx=2)', markersize=8)
    plt.xlabel('Jarak x (m)')
    plt.ylabel('Konsentrasi c (g/m³)')
    plt.title('Soal 11.27 - Profil Konsentrasi Kanal 1D Steady-State')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('soal11_27_plot.png', dpi=120)
    print("\nPlot disimpan: soal11_27_plot.png")
    plt.show()
