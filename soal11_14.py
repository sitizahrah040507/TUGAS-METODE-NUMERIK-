"""
Soal 11.14 - Gambar ulang Fig. 11.5 untuk slope 1 dan -1.

Pertanyaan: Apa yang terjadi jika dua persamaan memiliki slope 1 dan -1?
Secara grafis, iterasi Gauss-Seidel membentuk pola spiral — bisa divergen
atau osilasi bergantung pada kondisi matriks.
"""

import numpy as np
import matplotlib.pyplot as plt

def gauss_seidel_trace(A, b, x0, max_iter=10):
    """Lacak jalur iterasi Gauss-Seidel dalam ruang 2D."""
    x = x0.copy().astype(float)
    trace = [x.copy()]
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(2):
            sigma = sum(A[i, j] * x[j] for j in range(2) if j != i)
            x[i] = (b[i] - sigma) / A[i, i]
        trace.append(x.copy())
    return np.array(trace)

if __name__ == "__main__":
    # Sistem dengan slope 1 dan -1:
    # x1 + x2 = 2  → x2 = 2 - x1 (slope -1)
    # x1 - x2 = 0  → x2 = x1      (slope +1)
    # Solusi: x1 = 1, x2 = 1
    A = np.array([
        [1.0,  1.0],
        [1.0, -1.0]
    ])
    b = np.array([2.0, 0.0])

    print("=" * 60)
    print("Soal 11.14 - Gauss-Seidel: Slope 1 dan -1")
    print("=" * 60)

    # Cek diagonal dominan
    print("\nMatriks A:", A)
    print("Cek diagonal dominan:")
    for i in range(2):
        off = sum(abs(A[i, j]) for j in range(2) if j != i)
        print(f"  Baris {i}: |{A[i,i]}| vs {off} → {'OK' if abs(A[i,i]) > off else 'TIDAK dominan'}")

    print("\nKarena diagonal TIDAK dominan (|1| tidak > |1|), Gauss-Seidel kemungkinan TIDAK konvergen.")
    print("Setiap iterasi akan berosilasi di sekitar solusi tanpa konvergen.")

    x0 = np.array([0.0, 0.0])
    trace = gauss_seidel_trace(A, b, x0, max_iter=10)
    print("\nJalur iterasi:")
    for i, pt in enumerate(trace):
        print(f"  Iter {i}: x1={pt[0]:.4f}, x2={pt[1]:.4f}")

    # Plot
    x1_range = np.linspace(-1, 3, 200)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x1_range, 2 - x1_range, 'b-', label='x1 + x2 = 2 (slope -1)')
    ax.plot(x1_range, x1_range, 'r-', label='x1 - x2 = 0 (slope +1)')
    ax.plot(trace[:, 0], trace[:, 1], 'go-', label='Jalur Gauss-Seidel', markersize=5)
    ax.plot(trace[0, 0], trace[0, 1], 'ks', markersize=10, label='Titik awal')
    ax.plot(1, 1, 'r*', markersize=15, label='Solusi eksak (1,1)')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Soal 11.14: Gauss-Seidel dengan Slope 1 dan -1')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('soal11_14_plot.png', dpi=120)
    print("\nPlot disimpan: soal11_14_plot.png")
    plt.show()
