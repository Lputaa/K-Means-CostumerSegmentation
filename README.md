# Customer Segmentation Using K-Means Clustering
Repositori ini berisi implementasi metode unsupervised learning menggunakan algoritma **K-Means Clustering** untuk melakukan segmentasi pelanggan berdasarkan dataset *Mall Customers*. Analisis disusun dalam format akademik dan mencakup pemrosesan data, penentuan jumlah cluster optimal, serta visualisasi hasil clustering.

## 1. Latar Belakang
Segmentasi pelanggan merupakan proses penting dalam analisis data pemasaran untuk memahami pola perilaku konsumen. Dengan mengelompokkan pelanggan berdasarkan karakteristik tertentu, perusahaan dapat menyusun strategi pemasaran yang lebih akurat. Metode **K-Means Clustering** digunakan karena efektif untuk data numerik dan mampu mengidentifikasi pola kelompok secara alami.

## 2. Tujuan
1. Menerapkan algoritma K-Means untuk mengelompokkan pelanggan berdasarkan fitur numerik.
2. Menentukan jumlah cluster optimal menggunakan **Elbow Method** dan **Silhouette Score**.
3. Menyajikan visualisasi hasil clustering.
4. Memberikan interpretasi umum terhadap segmentasi pelanggan.

## 3. Dataset
Dataset *Mall Customers* berisi fitur:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

Dataset disimpan pada `Mall_Customers.csv`.

## 4. File Utama dalam Repositori
| File | Deskripsi |
|------|-----------|
| `kmeans_customer_segmentation_mall_customers.ipynb` | Notebook analisis utama berisi preprocessing, pemilihan k, implementasi K-Means, dan visualisasi. |
| `*.py` | Implementasi Python mandiri (opsional). |
| `hasil_visualisasi/` | Menyimpan grafik Elbow, Silhouette, dan plot cluster (jika tersedia). |

## 5. Metodologi
### 5.1 Preprocessing
- Seleksi fitur utama: *Annual Income* dan *Spending Score*.
- Normalisasi menggunakan **StandardScaler**.
- Eksplorasi awal distribusi variabel dan scatterplot.

### 5.2 Penentuan Jumlah Cluster
Dua metode digunakan:
- **Elbow Method**: Mengamati penurunan WCSS (Within Cluster Sum of Squares).
- **Silhouette Score**: Mengukur kualitas pemisahan cluster.

Hasil evaluasi menunjukkan **k = 5** sebagai jumlah cluster optimal.

### 5.3 Implementasi K-Means
- Model dibangun dengan `n_clusters=5`.
- Data dipetakan berdasarkan jarak Euclidean ke centroid.
- Label cluster ditambahkan ke dataset.

## 6. Visualisasi
Visualisasi mencakup:
- Elbow Plot
- Silhouette Plot
- Scatter Plot 2D (Income vs Spending Score)
- Scatter Plot 3D (opsional)

## 7. Kesimpulan
Dataset *Mall Customers* secara alami membentuk **5 cluster** pelanggan. Segmentasi memberikan wawasan mengenai pola pembelanjaan seperti:
- Pelanggan bernilai tinggi (income tinggi, spending tinggi)
- Pelanggan impulsif (income rendah, spending tinggi)
- Pelanggan konservatif (income tinggi, spending rendah)
- Pelanggan reguler (nilai moderat)
- Pelanggan berdaya beli rendah

Metode K-Means efektif untuk analisis segmentasi berbasis fitur numerik sederhana.

## 8. Cara Menjalankan Notebook
1. Instal dependensi:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Pastikan file `Mall_Customers.csv` berada pada direktori yang sama.
3. Jalankan notebook:
   ```bash
   jupyter notebook
   ```

## 9. Lisensi
Repositori ini dibuat untuk tujuan akademik dan pembelajaran. Pengguna bebas memodifikasi atau memperluas analisis.
