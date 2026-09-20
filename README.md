# Segmentasi Polip Kolorektal Kvasir-SEG

Project ini berisi pipeline segmentasi citra medis polip kolorektal menggunakan
UNet dan UNet++.

## Isi Repository

- `Kvasir-SEG/`: dataset utama berisi citra kolonoskopi, mask segmentasi, dan
  metadata bounding box.
- `sessile-main-Kvasir-SEG/`: subset citra polip sessile beserta mask-nya.
- `unet.py`: training dan evaluasi model UNet.
- `unet++.py`: training dan evaluasi model UNet++.
- `requirements.txt`: dependency Python yang dibutuhkan.

## Konsep Dataset

Setiap data punya pasangan:

```text
images/nama_file.jpg  -> citra kolonoskopi asli
masks/nama_file.jpg   -> label/mask area polip
```

Mask dipakai sebagai ground truth untuk tugas semantic segmentation:

- piksel putih: area polip
- piksel hitam: background

## Loss Function

Script mendukung empat loss function:

- Jaccard Loss
- Tversky Loss
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

## Metrik Evaluasi

Output evaluasi yang dihitung:

- accuracy
- precision
- recall
- Dice coefficient
- Intersection over Union (IoU)

## Cara Menjalankan

### Opsi 1: Docker

Cara ini paling gampang untuk laptop client karena dependency Python dan
TensorFlow dipasang di dalam container.

Install Docker Desktop, lalu dari folder project jalankan smoke test:

```bash
docker compose run --rm polyp
```

Build pertama bisa lama karena Docker mengunduh Python, TensorFlow, dan library
pendukung. Setelah itu run berikutnya biasanya lebih cepat.

Jalankan UNet lengkap:

```bash
docker compose run --rm polyp python unet.py --loss all --epochs 30 --image-size 224 --batch-size 8 --base-filters 16
```

Jalankan UNet++ lengkap:

```bash
docker compose run --rm polyp python "unet++.py" --loss all --epochs 30 --image-size 224 --batch-size 8 --base-filters 16
```

Jalankan perbandingan dataset generalis vs spesialis:

```bash
docker compose run --rm polyp python compare_datasets.py --loss all --epochs 30 --image-size 224 --batch-size 8 --base-filters 16
```

Hasilnya tetap masuk ke folder `outputs/` di project lokal.

### Opsi 2: Install Manual

Install dependency:

```bash
python3 -m pip install -r requirements.txt
```

Smoke test cepat:

```bash
python3 unet.py --loss jaccard --epochs 1 --limit 4 --image-size 32 --batch-size 1 --base-filters 4
python3 'unet++.py' --loss jaccard --epochs 1 --limit 4 --image-size 32 --batch-size 1 --base-filters 4
```

Eksperimen penuh:

```bash
python3 unet.py --loss all --epochs 30 --image-size 224 --batch-size 8 --base-filters 16
python3 'unet++.py' --loss all --epochs 30 --image-size 224 --batch-size 8 --base-filters 16
```

Perbandingan dataset generalis vs spesialis:

```bash
python3 compare_datasets.py --loss all --epochs 30 --image-size 224 --batch-size 8 --base-filters 16
```

Smoke test cepat perbandingan dataset:

```bash
python3 compare_datasets.py --smoke-test
```

## Pencegahan Overfitting

Secara default script sekarang memakai beberapa teknik untuk mengurangi
overfitting:

- augmentasi data train: flip horizontal, flip vertical, rotasi 90 derajat, dan
  variasi brightness ringan.
- `SpatialDropout2D` dengan default `--dropout 0.15`.
- L2 regularization pada convolution layer dengan default `--l2 1e-5`.
- early stopping berdasarkan validation Dice.
- evaluasi test memakai `best.weights.h5`, yaitu bobot terbaik dari validation
  Dice, bukan bobot epoch terakhir.

Kalau mau eksperimen tanpa regularisasi untuk pembanding, jalankan:

```bash
python3 unet.py --loss jaccard --epochs 30 --no-augment --dropout 0 --l2 0
python3 'unet++.py' --loss jaccard --epochs 30 --no-augment --dropout 0 --l2 0
```

Hasil training tersimpan di folder `outputs/`, misalnya:

```text
outputs/unet/hasil-unet-2026-09-15_18-09-45/
outputs/unetplusplus/hasil-unetplusplus-2026-09-11_23-11-33/
```

Folder output berisi:

- `01_visualisasi_resize_normalisasi.png`: contoh image asli, resize,
  normalisasi, mask asli, dan binary mask.
- `02_visualisasi_train_val_test.png`: contoh pasangan image-mask dari train,
  validation, dan test.
- `03_distribusi_train_val_test.png`: grafik jumlah data train/validation/test.
- `04_hyperparameters.csv` dan `04_hyperparameters.json`: hyperparameter numerik
  model, split data, optimizer, loss function, epoch, batch size, learning rate,
  image size, dan base filters.
- `<loss>/05_history_training.png`: grafik train loss, validation loss, Dice, dan
  IoU per epoch untuk tiap loss function.
- `06_performa_loss_function.png`: grafik perbandingan performa loss function
  berdasarkan test set.
- `best.weights.h5`
- `last.weights.h5`
- `history.csv`
- `metrics.json`
- `summary_metrics.csv`
- `predictions/*.png`

Output khusus perbandingan dataset tersimpan di:

```text
outputs/perbandingan-dataset/
```

Isinya:

- `ringkasan_dataset.csv`: jumlah data generalis dan spesialis.
- `ringkasan_perbandingan_generalis_vs_spesialis.csv`: tabel perbandingan
  performa dataset, model, dan loss function.
- `grafik_perbandingan_generalis_vs_spesialis.png`: grafik Dice dan IoU.

## Catatan Penting

`sessile-main-Kvasir-SEG` adalah subset dari `Kvasir-SEG`, bukan dataset tambahan
yang sepenuhnya berbeda. Jika keduanya digabung mentah-mentah, 196 sampel bisa
terduplikasi dan menyebabkan data leakage.
