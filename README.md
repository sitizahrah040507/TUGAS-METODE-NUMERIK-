### Thomas Algorithm (TDMA)
Algoritma efisien untuk sistem **tridiagonal** berukuran n. Terdiri dari dua tahap:
1. **Forward sweep**: eliminasi subdiagonal
2. **Back substitution**: solusi mundur

Kompleksitas: **O(n)** — jauh lebih efisien dari Gauss Elimination O(n³).  
Jumlah operasi ≈ 8n − 7.

### Cholesky Decomposition
Untuk matriks **simetris positif definit (SPD)**: `[A] = [L][L]ᵀ`  
- `L[i,i] = sqrt(A[i,i] - Σ L[i,k]²)`  
- `L[i,j] = (A[i,j] - Σ L[i,k]·L[j,k]) / L[j,j]`  
Lebih efisien ~2× dari LU decomposition umum karena memanfaatkan simetri.

### Gauss-Seidel
Metode **iteratif** untuk sistem linear besar. Pada tiap iterasi:
```
x[i]_baru = (b[i] - Σ A[i,j]·x[j]) / A[i,i]
```
Langsung menggunakan nilai terbaru yang sudah dihitung (beda dari Jacobi).

**Konvergensi dijamin** jika matriks diagonal dominan:
```
|A[i,i]| > Σ|A[i,j]|  untuk semua i (j≠i)
```

### Overrelaxasi (SOR)
```
x[i] = λ·x[i]_GS + (1−λ)·x[i]_lama
```
- `λ = 1`: Gauss-Seidel biasa  
- `λ > 1`: overrelaxasi (mempercepat konvergensi)  
- `λ < 1`: underrelaxasi (menstabilkan sistem yang hampir tidak konvergen)

### Condition Number
Mengukur sensitivitas solusi terhadap gangguan kecil pada data:  
`cond(A) = ||A|| · ||A⁻¹||`  
Digit presisi yang hilang ≈ log₁₀(cond(A)).

---

## Daftar Soal

### Soal 11.1 — Thomas Algorithm & LU Decomposition
**File:** `soal11_1.py`  
Menyelesaikan sistem tridiagonal 3×3 berikut:
```
[ 0.8  -0.4   0  ] [x1]   [ 41]
[-0.4   0.8  -0.4] [x2] = [ 25]
[ 0    -0.4   0.8] [x3]   [105]
```
(a) Thomas Algorithm — O(n), sangat efisien.  
(b) LU Decomposition via `scipy.linalg.lu_factor` + unit vectors.

---

### Soal 11.2 — Invers Matriks via LU + Unit Vectors
**File:** `soal11_2.py`  
Menghitung `A⁻¹` dari soal 11.1 dengan cara menyelesaikan `A·x = eᵢ` untuk setiap vektor satuan `eᵢ`. Kolom ke-i dari invers adalah solusi untuk kolom ke-i identitas.

---

### Soal 11.3 — Thomas Algorithm Crank-Nicolson (4×4)
**File:** `soal11_3.py`  
Sistem tridiagonal 4×4 dari metode Crank-Nicolson untuk PDE:
```
Diagonal: 2.01475 | Sub/Superdiag: -0.020875
b = [4.175, 0, 0, 2.0875]
```

---

### Soal 11.4 — Verifikasi Cholesky Decomposition
**File:** `soal11_4.py`  
Membuktikan bahwa `[L][L]ᵀ = [A]` dari Example 11.2. Menampilkan perbandingan elemen per elemen.

---

### Soal 11.5 — Cholesky: Sistem Simetris dengan Solusi
**File:** `soal11_5.py`  
Sistem simetris 3×3:
```
[  6  15   55] [a0]   [ 152.6]
[ 15  55  225] [a1] = [ 585.6]
[ 55 225  979] [a2]   [2488.8]
```
Cholesky decomposition + forward/backward substitution.

---

### Soal 11.6 — Cholesky "By Hand" (Langkah demi Langkah)
**File:** `soal11_6.py`  
Sistem:
```
[ 8  20  15] [x1]   [ 50]
[20  80  50] [x2] = [250]
[15  50  60] [x3]   [100]
```
Setiap langkah ditampilkan dengan detail rumus dan nilai numeriknya.

---

### Soal 11.7 — Cholesky Matriks Diagonal
**File:** `soal11_7.py`  
```
A = diag(9, 25, 4)
```
Untuk matriks diagonal: `L[i,i] = sqrt(A[i,i])` dan semua elemen luar diagonal = 0. Hasilnya masuk akal sesuai Eq. (11.3) dan (11.4).

---

### Soal 11.8 — Gauss-Seidel + Overrelaxasi (λ=1.2) untuk Prob. 11.1
**File:** `soal11_8.py`  
Menyelesaikan sistem tridiagonal dari Prob. 11.1 dengan Gauss-Seidel + overrelaxasi λ=1.2 hingga eₛ = 5%.

---

### Soal 11.9 — Gauss-Seidel: Sistem Reaktor Berpasangan
**File:** `soal11_9.py`  
```
15c1 -  3c2 -   c3 = 3800
-3c1 + 18c2 -  6c3 = 1200
-4c1 -   c2 + 12c3 = 2350
```
Konsentrasi (g/m³) dalam tiga reaktor berpasangan. Toleransi eₛ = 5%.

---

### Soal 11.10 — Jacobi Iteration: Sistem Reaktor
**File:** `soal11_10.py`  
Mengulangi Prob. 11.9 dengan **Jacobi iteration**.  
*Perbedaan utama*: Jacobi menggunakan nilai iterasi sebelumnya untuk semua variabel, sedangkan Gauss-Seidel langsung memakai nilai baru. Gauss-Seidel biasanya konvergen lebih cepat.

---

### Soal 11.11 — Gauss-Seidel Sistem 3×3 (eₛ = 5%)
**File:** `soal11_11.py`  
```
10x1 + 2x2  -  x3 =  27
-3x1 - 6x2 + 2x3  = -61.5
  x1 +  x2 + 5x3  = -21.5
```

---

### Soal 11.12 — Gauss-Seidel: Tanpa & Dengan Relaxasi (λ=0.95)
**File:** `soal11_12.py`  
Sistem yang perlu diatur ulang agar diagonal dominan, lalu dibandingkan:
- (a) λ = 1.0 (tanpa relaxasi)
- (b) λ = 0.95 (underrelaxasi)

---

### Soal 11.13 — Gauss-Seidel: Tanpa & Dengan Relaxasi (λ=1.2)
**File:** `soal11_13.py`  
```
 2x1 - 6x2 -  x3 = -38
-3x1 -  x2 + 7x3 = -34
-8x1 +  x2 - 2x3 = -20
```
- (a) λ = 1.0 (tanpa relaxasi)
- (b) λ = 1.2 (overrelaxasi)

---

### Soal 11.14 — Analisis Konvergensi: Slope 1 dan -1
**File:** `soal11_14.py`  
Menggambar ulang Fig. 11.5 untuk slope 1 dan −1. Ketika slope 1 dan −1 berpotongan, matriks tidak diagonal dominan (|1| ≤ |1|). Gauss-Seidel **tidak konvergen** — iterasi berosilasi. Disertai plot visual jalur iterasi.

---

### Soal 11.15 — Identifikasi Set Tidak Konvergen
**File:** `soal11_15.py`  
Tiga set persamaan dianalisis cek diagonal dominan dan diuji Gauss-Seidel selama 50 iterasi. Set yang tidak konvergen diidentifikasi berdasarkan divergensi atau osilasi nilainya.

---

### Soal 11.16 — Solusi, Invers, dan Condition Number
**File:** `soal11_16.py`  
(a) Matriks 3×3 Hilbert terpotong  
(b) Matriks Hilbert 4×4 (solusi semua x=1)  
Menghitung condition number (row-sum norm manual + numpy), invers, dan solusi.

---

### Soal 11.17 — Sistem Nonlinier: Dua Solusi
**File:** `soal11_17.py`  
```
f(x,y) = 4 - y - 2x² = 0
g(x,y) = 8 - y² - 4x = 0
```
(a) Dua solusi ditemukan menggunakan `scipy.optimize.fsolve`  
(b) Peta *basin of attraction* — wilayah initial guess yang menghasilkan masing-masing solusi

---

### Soal 11.18 — Produksi Elektronik: Sistem Linear
**File:** `soal11_18.py`  
Material (Tembaga=960, Seng=510, Kaca=610) dibagi untuk memproduksi Transistor, Resistor, Chip. Matriks koefisien dibangun dan diselesaikan dengan `numpy.linalg.solve`.

---

### Soal 11.19 — Condition Number Hilbert 10×10
**File:** `soal11_19.py`  
Matriks Hilbert 10×10: condition number ≈ 10¹³. Digit yang hilang ≈ 13 dari 15 digit float64. Solusi numerik dibandingkan dengan solusi eksak (semua 1).

---

### Soal 11.20 — Condition Number Vandermonde 6×6
**File:** `soal11_20.py`  
Matriks Vandermonde dengan x = [4, 2, 7, 10, 3, 5]. Analisis ill-conditioning dan perbandingan dengan Hilbert.

---

### Soal 11.21 — Matriks Augmented [A | I]
**File:** `soal11_21.py`  
```python
Aug = np.hstack([A, np.eye(A.shape[0])])
```
Demonstrasi Gauss-Jordan manual untuk menghitung invers dari matriks augmented.

---

### Soal 11.22 — Penulisan Sistem ke Bentuk Matriks
**File:** `soal11_22.py`  
Persamaan:
```
50  = 5x3 - 7x2
4x2 + 7x3 + 30 = 0
x1  - 7x3 = 40 - 3x2 + 5x1
```
Setelah diatur: x₁ muncul hanya di baris 3, membuat kolom pertama A hampir nol → singular!

---

### Soal 11.23 — Perbandingan Operasi: Gauss vs Thomas
**File:** `soal11_23.py`  
- Gauss Elimination: ≈ 2n³/3 operasi  
- Thomas Algorithm: ≈ 8n − 7 operasi  
Plot log-log untuk n = 2 sampai 20 menunjukkan perbedaan drastis kompleksitas.

---

### Soal 11.24 — Program Thomas Algorithm
**File:** `soal11_24.py`  
Implementasi Thomas algorithm yang bersih dengan output verbose (forward sweep + back substitution ditampilkan langkah demi langkah). Diuji dengan Example 11.1.

---

### Soal 11.25 — Program Cholesky Decomposition
**File:** `soal11_25.py`  
Implementasi Cholesky user-friendly dengan validasi (simetri & positif definit), verbose output, dan forward/backward substitution untuk menyelesaikan Ax = b.

---

### Soal 11.26 — Program Gauss-Seidel
**File:** `soal11_26.py`  
Program Gauss-Seidel dengan parameter: relaxasi λ, toleransi eₛ, dan verbose mode (tabel iterasi). Diuji dengan Example 11.3.

---

### Soal 11.27 — ODE Kanal 1D → Sistem Linear
**File:** `soal11_27.py`  
ODE steady-state:
```
0 = D·d²c/dx² - U·dc/dx - k·c
D=2, U=1, k=0.2, c(0)=80, c(10)=20
```
Diskretisasi beda hingga dengan Δx=2 menghasilkan 4 persamaan linear. Solusi dan plot profil konsentrasi.

---

### Soal 11.28 — Pentadiagonal System Solver
**File:** `soal11_28.py`  
Sistem pentadiagonal (bandwidth 5) 5×5:
```
[ 8  -2  -1   0   0] [x1]   [5]
[-2   9  -4  -1   0] [x2]   [2]
[-1  -4   7  -1  -7] [x3] = [0]
[ 0  -1  -1  12  -3] [x4]   [1]
[ 0   0  -7  -5  15] [x5]   [5]
```
Algoritma eliminasi Gauss untuk sistem pita diterapkan secara efisien dengan memanfaatkan struktur bandwidth.

---

## Library yang Digunakan

| Library | Kegunaan |
|---------|----------|
| `numpy` | Operasi matriks, solver linear, norm |
| `scipy.linalg` | LU decomposition, factorization |
| `scipy.optimize` | Solver sistem nonlinier (`fsolve`) |
| `matplotlib` | Visualisasi dan plot |

---

