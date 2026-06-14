"""
Soal 11.3 - Thomas Algorithm untuk Sistem Tridiagonal Crank-Nicolson

Sistem 4x4:
[2.01475  -0.020875   0         0       ] [T1]   [4.175 ]
[-0.020875  2.01475  -0.020875  0       ] [T2] = [0     ]
[0         -0.020875  2.01475  -0.020875] [T3]   [0     ]
[0          0        -0.020875  2.01475 ] [T4]   [2.0875]
"""

import numpy as np

def thomas_algorithm(a, b, c, d):
    n = len(d)
    a = a.copy().astype(float)
    b = b.copy().astype(float)
    c = c.copy().astype(float)
    d = d.copy().astype(float)
    for i in range(1, n):
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x

if __name__ == "__main__":
    # Diagonal utama
    b = np.array([2.01475, 2.01475, 2.01475, 2.01475])
    # Subdiagonal
    a = np.array([0.0, -0.020875, -0.020875, -0.020875])
    # Superdiagonal
    c = np.array([-0.020875, -0.020875, -0.020875, 0.0])
    # RHS
    d = np.array([4.175, 0.0, 0.0, 2.0875])

    A = np.array([
        [2.01475, -0.020875,  0.0,       0.0      ],
        [-0.020875, 2.01475, -0.020875,  0.0      ],
        [0.0,      -0.020875, 2.01475,  -0.020875 ],
        [0.0,       0.0,     -0.020875,  2.01475  ]
    ])

    print("=" * 55)
    print("Soal 11.3 - Tridiagonal Crank-Nicolson (Thomas Algorithm)")
    print("=" * 55)
    print("\nMatriks A:")
    print(A)
    print("\nVektor RHS:", d)

    x = thomas_algorithm(a, b, c, d)
    print("\nSolusi dengan Thomas Algorithm:")
    for i, xi in enumerate(x):
        print(f"  T{i+1} = {xi:.8f}")

    print("\nVerifikasi A @ x = b:", np.allclose(A @ x, d))
