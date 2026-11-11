# Indonesian Traffic Sign Recognition Using YOLOv8 for Autonomous Driving & ADAS Navigation

Proyek **Computer Vision & Deep Learning** ini mengembangkan sistem deteksi dan pengenalan rambu lalu lintas Indonesia (*Traffic Sign Recognition / TSR*) secara *real-time* menggunakan arsitektur **[Ultralytics YOLOv8](https://docs.ultralytics.com/usage/settings)**. Sistem ini dirancang sebagai komponen persepsi visual pada **Advanced Driver Assistance Systems (ADAS)** dan mendukung sistem navigasi kendaraan otonom di jalan raya maupun jalan tol Indonesia.

Model dilatih selama **100 epoch** pada **1.589 citra rambu jalan Indonesia** menggunakan lingkungan **Kaggle Notebook (GPU NVIDIA Tesla T4)**, serta menghasilkan performa sangat tinggi pada *Test Set* dengan **Precision mencapai 99,30%**, **Recall mencapai 98,80%**, **mAP@50 mencapai 99,46% (99,50%)**, **mAP@75 mencapai 76,95%**, **mAP@50-95 mencapai 73,27%**, dan waktu inferensi rata-rata **7,7 ms per citra** (total pemrosesan **9,6 ms/frame** atau **~104,2 FPS**).

## Tautan Proyek & Artefak Eksperimen

* **Kaggle Dataset**: `nanadiaa14/stc-project` (`/kaggle/input/datasets/nanadiaa14/stc-project`)
* **Sumber Kurasi Awal Dataset (Roboflow)**: [Indonesian Traffic Sign Dataset (`tes-f4pkg/stc-project-jmlkj` v2)](https://universe.roboflow.com/tes-f4pkg/stc-project-jmlkj)

## Latar Belakang Masalah & Motivasi

Tingginya angka kecelakaan lalu lintas di jalan raya dan ruas jalan tol sering kali dipicu oleh faktor kelalaian pengemudi dalam memperhatikan rambu peringatan, larangan, maupun petunjuk jalur pada kecepatan tinggi ([Mulyanto et al., 2020](https://doi.org/10.1109/ISRITI51436.2020.9315368); [Arief, Nurtanio \& Samman, 2021](https://www.google.com/search?q=https://doi.org/10.1109/AIMS51415.2021.9466006)). Dalam ekosistem transportasi cerdas (*smart mobility*) dan kendaraan otonom ([ITB Instrumentation and Control Research Group](https://itb.ac.id/news/itb-instrumentation-and-control-research-group-to-develop-autonomous-vehicle-in-indonesia/57932)), sistem **Traffic Sign Recognition (TSR)** berperan krusial untuk:

1. **Mendeteksi dan mengklasifikasikan rambu jalan secara otomatis** dari tangkapan kamera depan kendaraan (*dashcam*) dalam kondisi berkendara waktu nyata (*real-time*).
2. **Mencapai keseimbangan optimal antara akurasi deteksi dan efisiensi komputasi (FPS)** agar dapat dijalankan pada perangkat komputasi tepi (*edge device*) di dalam kendaraan ([Flores-Calero et al., 2024](https://doi.org/10.3390/math12020297); [Xing, Sun & Yang, 2025](https://doi.org/10.3390/wevj16080478)).
3. **Memberikan peringatan dini berbasis suara (*Text-to-Speech Audio Feedback*)** dengan mekanisme *debounce/cooldown* dan *object tracking* agar pengemudi memperoleh informasi tepat waktu tanpa gangguan berulang.

## Spesifikasi Dataset & Pembagian Kelas

Dataset disusun dari gabungan citra rambu lalu lintas Indonesia yang telah difilter, dianotasi dalam format *bounding box* standar YOLOv8 (`640 × 640` piksel), diunggah ke **Kaggle Datasets (`nanadiaa14/stc-project`)**, dan dibagi ke dalam tiga *split* evaluasi:

| Subset Data | Jumlah Citra | Proporsi | Keterangan |
| :--- | ---: | ---: | :--- |
| **Training Set** | 1.388 | 87,35% | Dilengkapi pipeline augmentasi (*Mosaic*, *Blur*, *MedianBlur*, *ToGray*, *CLAHE*, *Horizontal Flip*) |
| **Validation Set** | 133 | 8,37% | Digunakan untuk pemantauan *loss* dan pemilihan bobot terbaik (`best.pt`) di setiap *epoch* |
| **Test Set** | 68 | 4,28% | Data uji independen untuk evaluasi akhir performa generalisasi model |
| **Total** | **1.589** | **100,00%** | Mencakup **5 kelas rambu lalu lintas Indonesia** |

### Daftar 5 Kelas Rambu yang Dideteksi (`nc = 5`)

1. `dilarang-belok-kanan` — Rambu larangan berbelok ke arah kanan.
2. `dilarang-berhenti` — Rambu larangan menghentikan kendaraan di area terlarang.
3. `dilarang-masuk` — Rambu larangan masuk bagi kendaraan bermotor (*No Entry*).
4. `jalur-penyebrangan` — Rambu petunjuk fasilitas penyeberangan pejalan kaki (*Zebra Cross*).
5. `peringatan-lampu-merah` — Rambu peringatan adanya alat pemberi isyarat lalu lintas (APILL) di depan.

## Konfigurasi Arsitektur & Pelatihan Model (Kaggle Environment)

Eksperimen pelatihan dijalankan di **Kaggle Notebook** menggunakan model *pre-trained* **YOLOv8n (`yolov8n.pt`)** dari pustaka `ultralytics` versi `8.4.210` dengan Python `3.13.16` dan PyTorch `2.11.0+cu130` pada akselerator GPU **NVIDIA Tesla T4 (14.912 MiB VRAM)**:

| Parameter Konfigurasi | Nilai / Spesifikasi |
| :--- | :--- |
| **Base Architecture** | YOLOv8n (`CSPDarknet` Backbone + `C2f` + `SPPF` + Decoupled Head) |
| **Jumlah Parameter & Komputasi** | 3.011.823 parameter (*training*, 129 layer, 8,2 GFLOPs) • 3.133.679 parameter (*fused*, 73 layer, 8,6 GFLOPs) |
| **Ukuran Input Citra (`imgsz`)** | `640 × 640` piksel |
| **Jumlah Epoch & Batch Size** | 100 Epoch (`close_mosaic = 10` pada 10 epoch terakhir) • Batch Size = 16 |
| **Optimizer & Learning Rate** | `AdamW` (`lr = 0.001111`, `momentum = 0.9`, `weight_decay = 0.0005`) |
| **Fungsi Loss** | Box Regression Loss (`box = 7.5`), Classification Loss (`cls = 0.5`), Distribution Focal Loss (`dfl = 1.5`) |
| **Durasi Pelatihan & Ukuran Bobot** | 0,284 jam (~17,0 menit untuk 100 epoch) • Ukuran model `best.pt`: 6,5 MB |

## Hasil Evaluasi Model (Validation & Test Set)

### 1. Performa Keseluruhan: Validation Set vs. Test Set

| Subset Evaluasi | Jumlah Citra | Precision (P) | Recall (R) | mAP@50 | mAP@75 | mAP@50-95 | Kecepatan Inferensi |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Validation Set** | 133 | 99,20% | 98,70% | 99,10% | — | 71,90% | 4,3 ms / citra |
| **Test Set (Final)** | **68** | **99,30%** | **98,80%** | **99,46%** | **76,95%** | **73,27%** | **7,7 ms / citra** |

### 2. Rincian Performa Per Kelas pada Data Validasi (Validation Set — 133 Citra)

| Kelas Rambu Lalu Lintas | Jumlah Instans Validasi | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `dilarang-belok-kanan` | 50 | 99,90% | 100,00% | 99,50% | 66,70% |
| `dilarang-berhenti` | 42 | 100,00% | 93,50% | 97,50% | 72,50% |
| `dilarang-masuk` | 20 | 99,10% | 100,00% | 99,50% | 60,00% |
| `jalur-penyebrangan` | 11 | 98,80% | 100,00% | 99,50% | 76,60% |
| `peringatan-lampu-merah` | 10 | 98,40% | 100,00% | 99,50% | 83,60% |
| **Rata-Rata Semua Kelas (`all`)** | **133** | **99,20%** | **98,70%** | **99,10%** | **71,90%** |

### 3. Rincian Performa Per Kelas pada Data Uji Independen (Test Set — 68 Citra)

| Kelas Rambu Lalu Lintas | Jumlah Instans Uji | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `dilarang-belok-kanan` | 13 | 99,50% | 100,00% | 99,50% | 66,70% |
| `dilarang-berhenti` | 30 | 100,00% | 93,80% | 99,30% | 62,80% |
| `dilarang-masuk` | 11 | 99,50% | 100,00% | 99,50% | 69,40% |
| `jalur-penyebrangan` | 2 | 97,80% | 100,00% | 99,50% | 79,50% |
| `peringatan-lampu-merah` | 12 | 99,60% | 100,00% | 99,50% | 87,90% |
| **Rata-Rata Semua Kelas (`all`)** | **68** | **99,30%** | **98,80%** | **99,50% (99,46%)** | **73,30% (73,27%)** |

> **Analisis Hasil**: Empat dari lima kelas rambu mencapai **Recall 100%** dan **mAP@50 sebesar 99,50%** pada *Test Set*, sementara kelas `dilarang-berhenti` mencapai **Precision 100%** dan **mAP@50 sebesar 99,30%**. Waktu pemrosesan rata-rata pada *Test Set* hanya memerlukan **9,6 ms per frame** (`0,8 ms` *preprocess* + `7,7 ms` *inference* + `1,1 ms` *postprocess*), yang setara dengan **~104,2 FPS**—jauh melampaui standar deteksi *real-time* (30 FPS).

## Rancangan Pengembangan Sistem Navigasi Jalan Tol (ADAS Roadmap)

Sebagai kelanjutan dari model deteksi dasar ini, rancangan arsitektur sistem navigasi kendaraan otonom untuk skenario jalan tol mencakup integrasi modul-modul berikut ([Zhao, Li & Gong, 2025](https://doi.org/10.3390/info16090798); [Wang et al., 2025](https://doi.org/10.1038/s41598-025-03792-0)):

* **Multi-Scale Small Object Fusion (`BiFPN` + `P2 Detection Layer`)**: Menambahkan lapisan deteksi resolusi tinggi `P2` (skala 1/4 citra masukan) yang dipadukan dengan *Bidirectional Feature Pyramid Network (BiFPN)* guna mempertahankan detail spasial rambu berukuran kecil dari jarak jauh di jalan tol.
* **Fungsi Loss `Wise-IoU v3 (WIoU v3)`**: Menerapkan mekanisme *dynamic non-monotonic focusing* untuk menekan distorsi gradien dari sampel berkualitas ekstrem atau rambu yang tertutup sebagian (*occluded*).
* **Post-Processing, Object Tracking & Audio Feedback (TTS)**: Menggunakan pelacak objek ringan (*SORT / ByteTrack*), penyaringan jarak berbasis luas *bounding box*, serta aturan *debounce* (jeda minimal 3 detik antarperingatan untuk rambu yang sama) sebelum memicu peringatan suara *offline Text-to-Speech* (`pyttsx3`) dengan target latensi `< 0,5 detik`.

## Struktur Repositori

```text
indonesian-traffic-sign-recognition-yolov8/
├── assets/
│   ├── val_batch0_pred.jpg          # Contoh hasil deteksi bounding box pada rambu jalan
│   ├── confusion_matrix.png         # Matriks konfusi hasil evaluasi model
│   └── training_curves.png          # Grafik kurva Box/Class/DFL Loss dan mAP selama 100 epoch
├── notebooks/
│   └── train-stc.ipynb              # Notebook Kaggle lengkap (Training, Test Evaluation & Visualization)
├── weights/
│   └── best.pt                      # Bobot terbaik YOLOv8n (6,5 MB)
├── requirements.txt                 # Daftar dependensi pustaka Python
└── README.md                        # Dokumentasi utama proyek
```

## Cara Menjalankan Proyek (Kaggle & Lokal)

### 1. Menjalankan di Kaggle Notebook
1. Tambahkan dataset Kaggle `nanadiaa14/stc-project` ke *notebook* (`/kaggle/input/datasets/nanadiaa14/stc-project`).
2. Jalankan seluruh sel pada `notebooks/train-stc.ipynb` menggunakan akselerator **GPU Tesla T4**.
3. Hasil pelatihan (`weights/best.pt`, `results.csv`, dan kurva visualisasi) otomatis tersimpan di `/kaggle/working/Traffic_Sign_Recognition_YOLOv8_Result/traffic-sign-recognition_yolov8_training/augmentation_training_v2`.

### 2. Menjalankan Evaluasi Model secara Lokal
```bash
git clone https://github.com/nanadiaa14/indonesian-traffic-sign-recognition-yolov8.git
cd indonesian-traffic-sign-recognition-yolov8
pip install -r requirements.txt
```

```python
from ultralytics import YOLO

# Memuat bobot terbaik untuk evaluasi pada Test Set
model = YOLO("weights/best.pt")
metrics = model.val(
    data="/kaggle/input/datasets/nanadiaa14/stc-project/data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    device=0,
)

print(f"mAP@50-95 : {metrics.box.map:.4f}")
print(f"mAP@50    : {metrics.box.map50:.4f}")
print(f"mAP@75    : {metrics.box.map75:.4f}")
```

## Referensi Ilmiah

1. Arief, R. W., Nurtanio, I., & Samman, F. A. (2021). Traffic Signs Detection and Recognition System Using the YOLOv4 Algorithm. *2021 International Conference on Artificial Intelligence and Mechatronics Systems (AIMS)*, 1–6. [https://doi.org/10.1109/AIMS51415.2021.9466006](https://www.google.com/search?q=https://doi.org/10.1109/AIMS51415.2021.9466006)
2. Flores-Calero, M., et al. (2024). Traffic Sign Detection and Recognition Using YOLO Object Detection Algorithm: A Systematic Review. *Mathematics*, 12(2), 297. [https://doi.org/10.3390/math12020297](https://doi.org/10.3390/math12020297)
3. Mulyanto, A., et al. (2020). Indonesian Traffic Sign Recognition for Advanced Driver Assistant (ADAS) Using YOLOv4. *2020 3rd International Seminar on Research of Information Technology and Intelligent Systems (ISRITI)*, 520–524. [https://doi.org/10.1109/ISRITI51436.2020.9315368](https://doi.org/10.1109/ISRITI51436.2020.9315368)
4. Wang, G., et al. (2025). Traffic sign detection method based on improved YOLOv8. *Scientific Reports*, 15, 19385. [https://doi.org/10.1038/s41598-025-03792-0](https://doi.org/10.1038/s41598-025-03792-0)
5. Xing, C., Sun, H., & Yang, J. (2025). A Lightweight Traffic Sign Detection Model Based on Improved YOLOv8s for Edge Deployment in Autonomous Driving Systems Under Complex Environments. *World Electric Vehicle Journal*, 16(8), 478. [https://doi.org/10.3390/wevj16080478](https://doi.org/10.3390/wevj16080478)
6. Zhao, W., Li, L., & Gong, X. (2025). CPB-YOLOv8: An Enhanced Multi-Scale Traffic Sign Detector for Complex Road Environment. *Information*, 16(9), 798. [https://doi.org/10.3390/info16090798](https://doi.org/10.3390/info16090798)

## Penulis

**Nadia Indriani Sumardi**  
Departemen Teknik Informatika, Fakultas Teknik, Universitas Hasanuddin