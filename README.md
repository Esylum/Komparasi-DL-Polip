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

Hasil training tersimpan di folder `outputs/`, misalnya:

```text
outputs/unet/<timestamp>/
outputs/unetplusplus/<timestamp>/
```

Folder output berisi:

- `best.weights.h5`
- `last.weights.h5`
- `history.csv`
- `metrics.json`
- `summary_metrics.csv`
- `predictions/*.png`

## Catatan Penting

`sessile-main-Kvasir-SEG` adalah subset dari `Kvasir-SEG`, bukan dataset tambahan
yang sepenuhnya berbeda. Jika keduanya digabung mentah-mentah, 196 sampel bisa
terduplikasi dan menyebabkan data leakage.

# Komparasi-DL-Polip
