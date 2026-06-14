"""
Soal 11.15 - Identifikasi Set Persamaan yang Tidak Konvergen dengan Gauss-Seidel

Set One:   8x + 3y + z = 12 | -6x + 7z = 1  | 2x + 4y - z = 5
Set Two:   x + y + 5z = 7   | x + 4y - z = 4  | 3x + y - z = 4
Set Three: 2x + 3y + 5z = 7 | -2x+4y-5z = -3  | 2y - z = 1
"""

import numpy as np

def gauss_seidel(A, b, x0=None, es=5.0, max_iter=50):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy().astype(float)
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i, i]
        with np.errstate(divide='ignore', invalid='ignore'):
            errors = np.where(x != 0, np.abs((x - x_old) / x) * 100, np.abs(x - x_old) * 100)
        max_err = np.max(errors)
        if max_err < es:
            return x, it, True
    return x, max_iter, False

def check_diagonal_dominance(A):
    n = A.shape[0]
    ok = True
    for i in range(n):
        off = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= off:
            ok = False
    return ok

if __name__ == "__main__":
    sets = {
        "Set One": (
            np.array([[8, 3, 1], [-6, 0, 7], [2, 4, -1]], dtype=float),
            np.array([12.0, 1.0, 5.0])
        ),
        "Set Two": (
            np.array([[1, 1, 5], [1, 4, -1], [3, 1, -1]], dtype=float),
            np.array([7.0, 4.0, 4.0])
        ),
        "Set Three": (
            np.array([[2, 3, 5], [-2, 4, -5], [0, 2, -1]], dtype=float),
            np.array([7.0, -3.0, 1.0])
        )
    }

    print("=" * 65)
    print("Soal 11.15 - Identifikasi Sistem yang Tidak Konvergen")
    print("=" * 65)

    for name, (A, b) in sets.items():
        print(f"\n{name}:")
        print(f"  A = {A.tolist()}")
        print(f"  b = {b.tolist()}")

        dom = check_diagonal_dominance(A)
        print(f"  Diagonal dominan: {'YA' if dom else 'TIDAK'}")

        x, iters, converged = gauss_seidel(A, b, max_iter=50)
        print(f"  Konvergen: {'YA (iter ' + str(iters) + ')' if converged else 'TIDAK (50 iterasi)'}")
        print(f"  x akhir: {np.round(x, 4)}")

        x_exact = np.linalg.solve(A, b)
        print(f"  Solusi eksak: {np.round(x_exact, 4)}")

    print("""
Kesimpulan:
  - Set yang tidak diagonal dominan kemungkinan tidak konvergen dengan Gauss-Seidel.
  - Diagonal dominan adalah SYARAT CUKUP (bukan syarat perlu) untuk konvergensi.
  - Set yang tidak konvergen ditandai dengan nilai yang terus membesar atau berosilasi.
""")
