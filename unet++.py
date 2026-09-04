import os as _boot_os
import sys as _boot_sys

if __name__ == "__main__" and _boot_os.environ.get("UNETPP_BOOTSTRAPPED") != "1":
    _boot_os.environ["UNETPP_BOOTSTRAPPED"] = "1"
    _boot_os.environ["UNETPP_ENTRY_FILE"] = __file__
    _boot_code = """
import importlib.util, os, sys, traceback
path = os.environ["UNETPP_ENTRY_FILE"]
spec = importlib.util.spec_from_file_location("unetplusplus", path)
runtime = importlib.util.module_from_spec(spec)
sys.modules["unetplusplus"] = runtime
spec.loader.exec_module(runtime)
try:
    runtime.main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    runtime.tf.keras.backend.clear_session()
    runtime.plt.close("all")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
"""
    _boot_os.execvpe(
        _boot_sys.executable,
        [_boot_sys.executable, "-u", "-c", _boot_code] + _boot_sys.argv[1:],
        _boot_os.environ,
    )

import argparse
import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


LOSS_NAMES = ("jaccard", "tversky", "mse", "mae")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def find_pairs(dataset_dir):
    dataset_dir = Path(dataset_dir)
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Dataset harus punya folder images/ dan masks/: {dataset_dir}"
        )

    pairs = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"Tidak ada pasangan image-mask di {dataset_dir}")

    return pairs


def read_image(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Gagal membaca image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def read_mask(path, image_size):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Gagal membaca mask: {path}")
    mask = (mask > 127).astype(np.uint8)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(mask.astype(np.float32), axis=-1)


def load_arrays(pairs, image_size):
    images = np.stack([read_image(image_path, image_size) for image_path, _ in pairs])
    masks = np.stack([read_mask(mask_path, image_size) for _, mask_path in pairs])
    return images, masks


def conv_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def upsample_like(inputs, filters):
    x = tf.keras.layers.UpSampling2D(interpolation="bilinear")(inputs)
    x = tf.keras.layers.Conv2D(filters, 2, padding="same")(x)
    return x


def build_unet_plus_plus(input_shape, base_filters):
    inputs = tf.keras.Input(input_shape)
    filters = [
        base_filters,
        base_filters * 2,
        base_filters * 4,
        base_filters * 8,
        base_filters * 16,
    ]

    x00 = conv_block(inputs, filters[0])
    x10 = conv_block(tf.keras.layers.MaxPooling2D()(x00), filters[1])
    x20 = conv_block(tf.keras.layers.MaxPooling2D()(x10), filters[2])
    x30 = conv_block(tf.keras.layers.MaxPooling2D()(x20), filters[3])
    x40 = conv_block(tf.keras.layers.MaxPooling2D()(x30), filters[4])

    x01 = conv_block(
        tf.keras.layers.Concatenate()([x00, upsample_like(x10, filters[0])]),
        filters[0],
    )
    x11 = conv_block(
        tf.keras.layers.Concatenate()([x10, upsample_like(x20, filters[1])]),
        filters[1],
    )
    x21 = conv_block(
        tf.keras.layers.Concatenate()([x20, upsample_like(x30, filters[2])]),
        filters[2],
    )
    x31 = conv_block(
        tf.keras.layers.Concatenate()([x30, upsample_like(x40, filters[3])]),
        filters[3],
    )

    x02 = conv_block(
        tf.keras.layers.Concatenate()([x00, x01, upsample_like(x11, filters[0])]),
        filters[0],
    )
    x12 = conv_block(
        tf.keras.layers.Concatenate()([x10, x11, upsample_like(x21, filters[1])]),
        filters[1],
    )
    x22 = conv_block(
        tf.keras.layers.Concatenate()([x20, x21, upsample_like(x31, filters[2])]),
        filters[2],
    )

    x03 = conv_block(
        tf.keras.layers.Concatenate()([x00, x01, x02, upsample_like(x12, filters[0])]),
        filters[0],
    )
    x13 = conv_block(
        tf.keras.layers.Concatenate()([x10, x11, x12, upsample_like(x22, filters[1])]),
        filters[1],
    )

    x04 = conv_block(
        tf.keras.layers.Concatenate()(
            [x00, x01, x02, x03, upsample_like(x13, filters[0])]
        ),
        filters[0],
    )

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x04)
    return tf.keras.Model(inputs, outputs, name="UNetPlusPlus")


def flatten_binary(y_true, y_pred, threshold=None):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if threshold is not None:
        y_pred = tf.cast(y_pred >= threshold, tf.float32)
    return tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])


def dice_metric(y_true, y_pred, smooth=1e-6):
    y_true, y_pred = flatten_binary(y_true, y_pred, threshold=0.5)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true, y_pred = flatten_binary(y_true, y_pred, threshold=0.5)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_metric(y_true, y_pred, smooth=1e-6):
    y_true, y_pred = flatten_binary(y_true, y_pred, threshold=0.5)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    return (tp + smooth) / (tp + fp + smooth)


def recall_metric(y_true, y_pred, smooth=1e-6):
    y_true, y_pred = flatten_binary(y_true, y_pred, threshold=0.5)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return (tp + smooth) / (tp + fn + smooth)


def binary_accuracy_metric(y_true, y_pred):
    y_true, y_pred = flatten_binary(y_true, y_pred, threshold=0.5)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def jaccard_loss(y_true, y_pred, smooth=1e-6):
    y_true, y_pred = flatten_binary(y_true, y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return 1.0 - ((intersection + smooth) / (union + smooth))


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    y_true, y_pred = flatten_binary(y_true, y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky


def get_loss(name):
    if name == "jaccard":
        return jaccard_loss
    if name == "tversky":
        return tversky_loss
    if name == "mse":
        return tf.keras.losses.MeanSquaredError()
    if name == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    raise ValueError(f"Loss tidak dikenal: {name}")


def batch_metrics(y_true, y_prob):
    y_true, y_pred = flatten_binary(y_true, y_prob, threshold=0.5)
    tp = tf.reduce_sum(y_true * y_pred)
    tn = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    eps = tf.constant(1e-6, dtype=tf.float32)
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "dice": (2.0 * tp) / (2.0 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
    }


def average_logs(logs):
    keys = logs[0].keys()
    return {key: float(np.mean([float(log[key]) for log in logs])) for key in keys}


def evaluate_predictions(model, images, masks, batch_size):
    preds = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        preds.append(model(batch, training=False).numpy())
    preds = np.concatenate(preds, axis=0)
    pred_bin = (preds >= 0.5).astype(np.float32)
    true = masks.astype(np.float32)

    tp = np.sum((pred_bin == 1) & (true == 1))
    tn = np.sum((pred_bin == 0) & (true == 0))
    fp = np.sum((pred_bin == 1) & (true == 0))
    fn = np.sum((pred_bin == 0) & (true == 1))
    eps = 1e-6

    return {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "dice": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def save_prediction_samples(model, sample_pairs, image_size, out_dir, max_samples):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (image_path, mask_path) in enumerate(sample_pairs[:max_samples], start=1):
        image = read_image(image_path, image_size)
        mask = read_mask(mask_path, image_size)[:, :, 0]
        pred = model(np.expand_dims(image, axis=0), training=False).numpy()[0, :, :, 0]
        pred_bin = (pred >= 0.5).astype(np.float32)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[3].imshow(image)
        axes[3].imshow(pred_bin, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        axes[3].set_title("Overlay")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:02d}_{image_path.stem}.png", dpi=150)
        plt.close(fig)


def iterate_batches(images, masks, batch_size, shuffle):
    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        yield images[batch_ids], masks[batch_ids]


def run_training_loop(model, loss_fn, optimizer, x_train, y_train, x_val, y_val, args, loss_dir):
    loss_dir.mkdir(parents=True, exist_ok=True)
    history_rows = []
    best_val_dice = -1.0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_batch_count = int(np.ceil(len(x_train) / args.batch_size))
        val_batch_count = int(np.ceil(len(x_val) / args.batch_size))
        print(f"    epoch {epoch}: training {train_batch_count} batch...")
        train_logs = []
        for batch_index, (xb, yb) in enumerate(
            iterate_batches(x_train, y_train, args.batch_size, shuffle=True), start=1
        ):
            if args.verbose_batches:
                print(f"      train batch {batch_index}/{train_batch_count}: forward/backward...")
            xb = tf.convert_to_tensor(xb, dtype=tf.float32)
            yb = tf.convert_to_tensor(yb, dtype=tf.float32)
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                loss_value = loss_fn(yb, pred)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            logs = {"loss": float(loss_value)}
            logs.update({key: float(value) for key, value in batch_metrics(yb, pred).items()})
            train_logs.append(logs)
            if args.verbose_batches:
                print(f"      train batch {batch_index}/{train_batch_count}: selesai")

        print(f"    epoch {epoch}: validasi {val_batch_count} batch...")
        val_logs = []
        for batch_index, (xb, yb) in enumerate(
            iterate_batches(x_val, y_val, args.batch_size, shuffle=False), start=1
        ):
            if args.verbose_batches:
                print(f"      val batch {batch_index}/{val_batch_count}: forward...")
            xb = tf.convert_to_tensor(xb, dtype=tf.float32)
            yb = tf.convert_to_tensor(yb, dtype=tf.float32)
            pred = model(xb, training=False)
            loss_value = loss_fn(yb, pred)
            logs = {"loss": float(loss_value)}
            logs.update({key: float(value) for key, value in batch_metrics(yb, pred).items()})
            val_logs.append(logs)
            if args.verbose_batches:
                print(f"      val batch {batch_index}/{val_batch_count}: selesai")

        row = {"epoch": epoch}
        row.update(average_logs(train_logs))
        row.update({f"val_{key}": value for key, value in average_logs(val_logs).items()})
        history_rows.append(row)

        val_dice = row.get("val_dice", 0.0)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            wait = 0
            print(f"    epoch {epoch}: menyimpan best weights...")
            model.save_weights(loss_dir / "best.weights.h5")
            checkpoint_text = "best saved"
        else:
            wait += 1
            checkpoint_text = f"no improve ({wait}/{args.patience})"

        print(
            f"Epoch {epoch:03d}/{args.epochs} - "
            f"loss={row.get('loss', 0):.4f} "
            f"dice={row.get('dice', 0):.4f} "
            f"iou={row.get('iou', 0):.4f} - "
            f"val_loss={row.get('val_loss', 0):.4f} "
            f"val_dice={val_dice:.4f} "
            f"val_iou={row.get('val_iou', 0):.4f} - "
            f"{checkpoint_text}"
        )

        if wait >= args.patience:
            print(f"Early stopping aktif pada epoch {epoch}.")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(loss_dir / "history.csv", index=False)
    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training UNet++ untuk segmentasi polip Kvasir-SEG."
    )
    parser.add_argument("--dataset", default="Kvasir-SEG", help="Folder dataset.")
    parser.add_argument(
        "--loss",
        default="all",
        help="Loss: all, jaccard, tversky, mse, atau mae.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--base-filters",
        type=int,
        default=16,
        help="Jumlah filter awal UNet++. Naikkan ke 32 untuk eksperimen lebih besar.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Batasi jumlah data untuk smoke test. 0 berarti semua data.",
    )
    parser.add_argument("--output-dir", default="outputs/unetplusplus")
    parser.add_argument("--save-samples", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--verbose-batches", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.loss == "all":
        losses = list(LOSS_NAMES)
    else:
        losses = [args.loss.lower()]
    invalid = [name for name in losses if name not in LOSS_NAMES]
    if invalid:
        raise ValueError(f"Loss tidak valid: {invalid}. Pilih dari {LOSS_NAMES}.")

    pairs = find_pairs(args.dataset)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=0.30, random_state=args.seed
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.50, random_state=args.seed
    )

    print("\n=== PROJECT SEGMENTASI POLIP - UNet++ ===")
    print(f"Dataset              : {Path(args.dataset).resolve()}")
    print(f"Total pasangan data  : {len(pairs)}")
    print(f"Train / Val / Test   : {len(train_pairs)} / {len(val_pairs)} / {len(test_pairs)}")
    print(f"Image size           : {args.image_size} x {args.image_size}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Epoch                : {args.epochs}")
    print(f"Learning rate        : {args.learning_rate}")
    print(f"Base filters         : {args.base_filters}")
    print(f"Loss yang diuji      : {', '.join(losses)}")
    print("Catatan mask         : grayscale -> threshold > 127 -> biner 0/1")

    print("\n[1/4] Loading dataset ke memory...")
    x_train, y_train = load_arrays(train_pairs, args.image_size)
    x_val, y_val = load_arrays(val_pairs, args.image_size)
    x_test, y_test = load_arrays(test_pairs, args.image_size)
    print(f"Train arrays         : X{x_train.shape} y{y_train.shape}")
    print(f"Val arrays           : X{x_val.shape} y{y_val.shape}")
    print(f"Test arrays          : X{x_test.shape} y{y_test.shape}")

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for index, loss_name in enumerate(losses, start=1):
        print(f"\n[2/4] Training UNet++ dengan {loss_name.upper()} ({index}/{len(losses)})")
        loss_dir = run_dir / loss_name
        loss_dir.mkdir(parents=True, exist_ok=True)

        print("  - Membangun arsitektur model...")
        model = build_unet_plus_plus(
            (args.image_size, args.image_size, 3), args.base_filters
        )
        print(f"  - Total parameter model: {model.count_params():,}")
        print("  - Menyiapkan optimizer, loss, dan metrik manual...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss_fn = get_loss(loss_name)

        print("  - Mulai training loop manual")
        history = run_training_loop(
            model, loss_fn, optimizer, x_train, y_train, x_val, y_val, args, loss_dir
        )
        print("  - Training selesai")
        history.to_csv(loss_dir / "history_final.csv", index=False)

        print(f"\n[3/4] Evaluasi test set untuk loss {loss_name.upper()}...")
        metrics = evaluate_predictions(model, x_test, y_test, args.batch_size)
        metrics["loss_name"] = loss_name
        summary_rows.append(metrics)

        with open(loss_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        model.save_weights(loss_dir / "last.weights.h5")
        save_prediction_samples(
            model,
            test_pairs,
            args.image_size,
            loss_dir / "predictions",
            args.save_samples,
        )

        print("Hasil test:")
        print(
            f"  accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"dice={metrics['dice']:.4f} "
            f"iou={metrics['iou']:.4f}"
        )
        print(
            f"  confusion pixels: TP={metrics['tp']} TN={metrics['tn']} "
            f"FP={metrics['fp']} FN={metrics['fn']}"
        )
        print(f"  output folder: {loss_dir}")

    print("\n[4/4] Ringkasan komparasi loss")
    summary_df = pd.DataFrame(summary_rows)
    metric_cols = ["loss_name", "accuracy", "precision", "recall", "dice", "iou"]
    summary_df[metric_cols].to_csv(run_dir / "summary_metrics.csv", index=False)
    print(summary_df[metric_cols].sort_values("dice", ascending=False).to_string(index=False))
    print(f"\nSemua output tersimpan di: {run_dir.resolve()}")


if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location("unetplusplus", Path(__file__))
    runtime = importlib.util.module_from_spec(spec)
    sys.modules["unetplusplus"] = runtime
    spec.loader.exec_module(runtime)

    try:
        runtime.main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        runtime.tf.keras.backend.clear_session()
        runtime.plt.close("all")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
