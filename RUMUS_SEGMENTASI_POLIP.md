# Rumus Segmentasi Polip

Dokumen ini merangkum rumus yang dipakai dalam project segmentasi polip
kolorektal menggunakan UNet dan UNet++.

## 1. Input Dataset

Setiap data terdiri dari pasangan:

```text
image = citra kolonoskopi
mask  = label area polip
```

Secara matematis:

```text
X = image
Y = mask ground truth
```

Untuk segmentasi biner:

```text
Y(i, j) = 1 jika piksel adalah polip
Y(i, j) = 0 jika piksel adalah background
```

Keterangan:

- `i, j` adalah posisi piksel.
- `1` berarti area polip.
- `0` berarti bukan polip.

## 2. Resize Image dan Mask

Semua image dan mask disamakan ukurannya menjadi:

```text
H x W
```

Pada script default:

```text
H = 224
W = 224
```

Rumus resize secara konsep:

```text
X_resized = resize(X, H, W)
Y_resized = resize(Y, H, W)
```

Pada image digunakan interpolasi area:

```text
X_resized = INTER_AREA(X)
```

Pada mask digunakan nearest neighbor agar label tidak berubah:

```text
Y_resized = INTER_NEAREST(Y)
```

## 3. Normalisasi Image

Nilai piksel image awal berada pada rentang:

```text
0 sampai 255
```

Kemudian dinormalisasi menjadi:

```text
0 sampai 1
```

Rumus:

```text
X_norm = X_resized / 255
```

Contoh:

```text
pixel = 128
pixel_norm = 128 / 255 = 0.5019
```

## 4. Binarisasi Mask

Mask asli dibaca sebagai grayscale, lalu dijadikan biner.

Rumus:

```text
Y_bin(i, j) = 1 jika Y(i, j) > 127
Y_bin(i, j) = 0 jika Y(i, j) <= 127
```

Dalam bentuk fungsi:

```text
Y_bin = threshold(Y, 127)
```

Artinya:

- putih menjadi `1`
- hitam menjadi `0`

## 5. Split Dataset

Dataset dibagi menjadi:

```text
train      = 70%
validation = 15%
test       = 15%
```

Jika total data adalah `N`, maka:

```text
N_train = 0.70 x N
N_val   = 0.15 x N
N_test  = 0.15 x N
```

Contoh pada `Kvasir-SEG`:

```text
N = 1000
N_train = 700
N_val   = 150
N_test  = 150
```

Contoh pada `sessile-main-Kvasir-SEG`:

```text
N = 196
N_train kira-kira = 137
N_val kira-kira   = 29
N_test kira-kira  = 30
```

Angka aktual bisa sedikit berbeda karena pembagian dilakukan oleh fungsi
`train_test_split`.

## 6. Output Model

Model menerima input image:

```text
X_norm
```

Lalu menghasilkan prediksi mask:

```text
Y_pred = model(X_norm)
```

Karena output layer memakai sigmoid, nilai prediksi berada pada rentang:

```text
0 sampai 1
```

Rumus sigmoid:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

Maka:

```text
Y_pred(i, j) = sigmoid(z(i, j))
```

Interpretasi:

- mendekati `1` berarti model yakin piksel adalah polip
- mendekati `0` berarti model yakin piksel adalah background

## 7. Threshold Prediksi

Untuk mengubah probabilitas menjadi mask biner, digunakan threshold:

```text
threshold = 0.5
```

Rumus:

```text
Y_pred_bin(i, j) = 1 jika Y_pred(i, j) >= 0.5
Y_pred_bin(i, j) = 0 jika Y_pred(i, j) < 0.5
```

## 8. Confusion Matrix Piksel

Karena ini segmentasi, confusion matrix dihitung berdasarkan piksel.

### True Positive

Piksel benar-benar polip dan diprediksi polip.

```text
TP = jumlah piksel dengan Y_true = 1 dan Y_pred = 1
```

### True Negative

Piksel benar-benar background dan diprediksi background.

```text
TN = jumlah piksel dengan Y_true = 0 dan Y_pred = 0
```

### False Positive

Piksel background tetapi diprediksi sebagai polip.

```text
FP = jumlah piksel dengan Y_true = 0 dan Y_pred = 1
```

### False Negative

Piksel polip tetapi diprediksi sebagai background.

```text
FN = jumlah piksel dengan Y_true = 1 dan Y_pred = 0
```

## 9. Accuracy

Accuracy mengukur jumlah piksel yang benar dibanding seluruh piksel.

Rumus:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Keterangan:

- semakin tinggi semakin baik
- tetapi untuk segmentasi medis, accuracy saja tidak cukup karena background
  biasanya jauh lebih banyak daripada area polip

## 10. Precision

Precision mengukur seberapa banyak prediksi polip yang benar-benar polip.

Rumus:

```text
Precision = TP / (TP + FP)
```

Makna:

```text
Precision tinggi = model jarang salah menandai background sebagai polip
```

## 11. Recall

Recall mengukur seberapa banyak area polip asli yang berhasil ditemukan model.

Rumus:

```text
Recall = TP / (TP + FN)
```

Makna:

```text
Recall tinggi = model sedikit melewatkan area polip
```

## 12. Dice Coefficient

Dice mengukur tingkat kemiripan antara mask asli dan mask prediksi.

Rumus:

```text
Dice = (2 x TP) / (2 x TP + FP + FN)
```

Atau dalam bentuk himpunan:

```text
Dice = (2 x |Y_true ∩ Y_pred|) / (|Y_true| + |Y_pred|)
```

Makna:

- `Dice = 1` berarti prediksi sempurna
- `Dice = 0` berarti tidak ada overlap

Dice sangat sering dipakai untuk segmentasi medis.

## 13. IoU / Jaccard Index

IoU mengukur overlap antara mask asli dan mask prediksi dibanding gabungannya.

Rumus:

```text
IoU = TP / (TP + FP + FN)
```

Atau dalam bentuk himpunan:

```text
IoU = |Y_true ∩ Y_pred| / |Y_true ∪ Y_pred|
```

Makna:

- `IoU = 1` berarti prediksi sempurna
- `IoU = 0` berarti tidak overlap

## 14. Jaccard Loss

Jaccard Loss berasal dari IoU.

Rumus:

```text
Jaccard Loss = 1 - Jaccard Index
```

Dengan smoothing:

```text
Jaccard Loss = 1 - ((intersection + smooth) / (union + smooth))
```

Di mana:

```text
intersection = sum(Y_true x Y_pred)
union = sum(Y_true) + sum(Y_pred) - intersection
smooth = 1e-6
```

Makin kecil nilai loss, makin baik.

## 15. Tversky Loss

Tversky Index adalah variasi dari Dice/IoU yang bisa memberi bobot berbeda pada
false positive dan false negative.

Rumus:

```text
Tversky = (TP + smooth) / (TP + alpha x FP + beta x FN + smooth)
```

Tversky Loss:

```text
Tversky Loss = 1 - Tversky
```

Pada script:

```text
alpha = 0.3
beta  = 0.7
```

Artinya `FN` diberi bobot lebih besar daripada `FP`.

Maknanya:

```text
Model lebih dihukum ketika melewatkan area polip asli.
```

## 16. Mean Squared Error

MSE menghitung rata-rata kuadrat selisih antara mask asli dan prediksi.

Rumus:

```text
MSE = (1 / n) x sum((Y_true - Y_pred)^2)
```

Keterangan:

- `n` adalah jumlah piksel
- makin kecil makin baik

## 17. Mean Absolute Error

MAE menghitung rata-rata nilai absolut selisih antara mask asli dan prediksi.

Rumus:

```text
MAE = (1 / n) x sum(|Y_true - Y_pred|)
```

Keterangan:

- `n` adalah jumlah piksel
- makin kecil makin baik

## 18. Loss Training

Pada setiap epoch, model menghitung loss dari prediksi terhadap mask asli.

Rumus umum:

```text
Loss = L(Y_true, Y_pred)
```

Bobot model diperbarui agar loss semakin kecil:

```text
weights_new = weights_old - learning_rate x gradient
```

Dalam project ini optimizer yang digunakan adalah Adam:

```text
optimizer = Adam
learning_rate = 0.0001
```

## 19. Validation Loss

Validation loss dihitung pada data validation.

Rumus:

```text
Val Loss = L(Y_val_true, Y_val_pred)
```

Fungsinya:

```text
mengecek apakah model hanya menghafal data train atau benar-benar belajar pola
```

## 20. Test Metrics

Setelah training selesai, model diuji pada data test.

Rumus:

```text
Y_test_pred = model(X_test)
```

Lalu dihitung:

```text
Accuracy
Precision
Recall
Dice
IoU
TP
TN
FP
FN
```

Hasil ini disimpan di:

```text
metrics.json
summary_metrics.csv
```

## 21. Rumus Perbandingan Loss Function

Untuk setiap model, loss function dibandingkan berdasarkan metrik test.

Contoh tabel:

```text
Model   Loss      Dice    IoU
UNet    Jaccard   ...
UNet    Tversky   ...
UNet    MSE       ...
UNet    MAE       ...
```

Secara konsep:

```text
Best Loss = loss function dengan Dice tertinggi dan IoU tertinggi
```

Jika ingin memilih berdasarkan Dice:

```text
Best Loss = argmax(Dice)
```

Jika ingin memilih berdasarkan IoU:

```text
Best Loss = argmax(IoU)
```

## 22. Rumus Perbandingan Model

Model dibandingkan berdasarkan performa test.

Contoh:

```text
UNet_score = Dice_UNet
UNet++_score = Dice_UNet++
```

Selisih performa:

```text
Delta Dice = Dice_UNet++ - Dice_UNet
```

Jika:

```text
Delta Dice > 0
```

maka UNet++ lebih baik berdasarkan Dice.

Jika:

```text
Delta Dice < 0
```

maka UNet lebih baik berdasarkan Dice.

## 23. Rumus Perbandingan Dataset Generalis dan Spesialis

Dalam project ini:

```text
Dataset generalis = Kvasir-SEG
Dataset spesialis = sessile-main-Kvasir-SEG
```

Perbandingan dilakukan dengan menjalankan model pada kedua dataset, lalu
membandingkan metrik test.

Contoh:

```text
Dice_generalis = Dice model pada Kvasir-SEG
Dice_spesialis = Dice model pada sessile-main-Kvasir-SEG
```

Selisih performa:

```text
Delta Dice Dataset = Dice_spesialis - Dice_generalis
```

Jika:

```text
Delta Dice Dataset > 0
```

maka performa pada dataset spesialis lebih tinggi.

Jika:

```text
Delta Dice Dataset < 0
```

maka performa pada dataset generalis lebih tinggi.

Rumus yang sama bisa dipakai untuk IoU:

```text
Delta IoU Dataset = IoU_spesialis - IoU_generalis
```

## 24. Output File yang Berkaitan dengan Rumus

### `history.csv`

Berisi metrik per epoch:

```text
epoch
loss
accuracy
precision
recall
dice
iou
val_loss
val_accuracy
val_precision
val_recall
val_dice
val_iou
```

### `metrics.json`

Berisi hasil akhir pada test set:

```text
accuracy
precision
recall
dice
iou
tp
tn
fp
fn
```

### `summary_metrics.csv`

Berisi ringkasan performa loss function.

### `ringkasan_perbandingan_generalis_vs_spesialis.csv`

Berisi ringkasan perbandingan:

```text
dataset_role
dataset_name
model
loss_name
accuracy
precision
recall
dice
iou
total_data
train_data
validation_data
test_data
run_dir
```

## 25. Ringkasan Alur Rumus

Alur dari awal sampai akhir:

```text
Image asli
-> resize
-> normalisasi
-> masuk ke UNet / UNet++
-> output sigmoid
-> prediksi mask probabilitas
-> threshold 0.5
-> prediksi mask biner
-> bandingkan dengan mask asli
-> hitung loss
-> update bobot model
-> evaluasi test set
-> hitung Accuracy, Precision, Recall, Dice, IoU
-> bandingkan model, loss function, dan dataset
```

## 26. Rumus Paling Penting untuk Laporan

Untuk laporan skripsi, rumus yang paling penting biasanya:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

```text
Precision = TP / (TP + FP)
```

```text
Recall = TP / (TP + FN)
```

```text
Dice = (2 x TP) / (2 x TP + FP + FN)
```

```text
IoU = TP / (TP + FP + FN)
```

```text
Jaccard Loss = 1 - ((intersection + smooth) / (union + smooth))
```

```text
Tversky Loss = 1 - ((TP + smooth) / (TP + alpha x FP + beta x FN + smooth))
```

```text
MSE = (1 / n) x sum((Y_true - Y_pred)^2)
```

```text
MAE = (1 / n) x sum(|Y_true - Y_pred|)
```
