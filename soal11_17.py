"""
Soal 11.17 - Sistem Persamaan Nonlinier
Buku: Numerical Methods for Engineers, Chapra & Canale, 7th Ed.

f(x,y) = 4 - y - 2x^2 = 0
g(x,y) = 8 - y^2 - 4x = 0

(a) Temukan dua pasang solusi (x,y)
(b) Petakan wilayah initial guess yang menghasilkan masing-masing solusi
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def system(vars):
    x, y = vars
    f = 4 - y - 2*x**2
    g = 8 - y**2 - 4*x
    return [f, g]

if __name__ == "__main__":
    print("=" * 60)
    print("Soal 11.17 - Sistem Nonlinier (2 solusi)")
    print("=" * 60)

    # (a) Cari dua solusi dengan initial guess berbeda
    solutions = []
    guesses = [(1, 1), (-2, -2), (1, -3), (-1, 2)]
    for g in guesses:
        sol = fsolve(system, g, full_output=True)
        x_sol = sol[0]
        resid = np.max(np.abs(system(x_sol)))
        if resid < 1e-8:
            rounded = tuple(np.round(x_sol, 6))
            # Hindari duplikat
            new = True
            for s in solutions:
                if np.allclose(s, x_sol, atol=1e-4):
                    new = False
                    break
            if new:
                solutions.append(x_sol)

    print("\n(a) Solusi yang ditemukan:")
    for i, s in enumerate(solutions):
        print(f"  Solusi {i+1}: x = {s[0]:.6f}, y = {s[1]:.6f}")
        print(f"    Verifikasi f = {4 - s[1] - 2*s[0]**2:.2e}, g = {8 - s[1]**2 - 4*s[0]:.2e}")

    # (b) Peta basin of attraction
    print("\n(b) Membuat peta wilayah initial guess...")
    xs = np.linspace(-6, 6, 80)
    ys = np.linspace(-6, 6, 80)
    basin = np.zeros((len(ys), len(xs)), dtype=int)

    for j, xi in enumerate(xs):
        for i, yi in enumerate(ys):
            sol = fsolve(system, [xi, yi], full_output=True)
            x_sol = sol[0]
            resid = np.max(np.abs(system(x_sol)))
            if resid < 1e-6 and len(solutions) > 0:
                for k, s in enumerate(solutions):
                    if np.allclose(s, x_sol, atol=1e-3):
                        basin[i, j] = k + 1
                        break

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(xs, ys, basin, levels=[-0.5, 0.5, 1.5, 2.5],
                     colors=['white', 'lightblue', 'lightsalmon'], alpha=0.8)
    for k, s in enumerate(solutions):
        ax.plot(s[0], s[1], 'k*', markersize=15, label=f'Solusi {k+1}: ({s[0]:.2f},{s[1]:.2f})')

    # Gambar kurva f=0 dan g=0
    xp = np.linspace(-3, 3, 300)
    ax.plot(xp, 4 - 2*xp**2, 'b-', lw=2, label='f(x,y)=0')
    yp = np.linspace(-4, 4, 300)
    ax.plot((8 - yp**2)/4, yp, 'r-', lw=2, label='g(x,y)=0')

    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Soal 11.17 - Basin of Attraction')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('soal11_17_plot.png', dpi=120)
    print("Plot disimpan: soal11_17_plot.png")
    plt.show()
