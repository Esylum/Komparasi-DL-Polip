import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


DATASETS = [
    ("generalis", "Kvasir-SEG"),
    ("spesialis", "sessile-main-Kvasir-SEG"),
]

MODELS = {
    "unet": ("UNet", "unet.py"),
    "unetplusplus": ("UNet++", "unet++.py"),
}


def slugify_name(value):
    slug = []
    last_dash = False
    for char in str(value).lower():
        if char.isalnum():
            slug.append(char)
            last_dash = False
        elif not last_dash:
            slug.append("-")
            last_dash = True
    return "".join(slug).strip("-") or "dataset"


def count_pairs(dataset_dir):
    image_dir = Path(dataset_dir) / "images"
    mask_dir = Path(dataset_dir) / "masks"
    return sum(1 for image_path in image_dir.glob("*.jpg") if (mask_dir / image_path.name).exists())


def latest_child(parent):
    children = [path for path in parent.iterdir() if path.is_dir()]
    if not children:
        raise RuntimeError(f"Tidak ada folder output di {parent}")
    return max(children, key=lambda path: path.stat().st_mtime)


def load_model_module(model_key):
    _, script_name = MODELS[model_key]
    module_name = f"{model_key}_runtime"
    spec = importlib.util.spec_from_file_location(module_name, script_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_model(module, model_key, input_shape, base_filters):
    if model_key == "unet":
        return module.build_unet(input_shape, base_filters)
    if model_key == "unetplusplus":
        return module.build_unet_plus_plus(input_shape, base_filters)
    raise ValueError(f"Model tidak dikenal: {model_key}")


def run_experiment(args, dataset_role, dataset_dir, model_key):
    model_name, _ = MODELS[model_key]
    module = load_model_module(model_key)
    output_dir = args.output_root / dataset_role / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    run_args = SimpleNamespace(**vars(args))
    run_args.dataset = dataset_dir
    run_args.output_dir = str(output_dir)

    print("\n=== RUN EKSPERIMEN ===")
    print(f"Dataset role : {dataset_role}")
    print(f"Dataset      : {dataset_dir}")
    print(f"Model        : {model_name}")
    print(f"Loss         : {args.loss}")

    module.set_seed(args.seed)
    losses = module.losses_from_args(args.loss)
    invalid = [name for name in losses if name not in module.LOSS_NAMES]
    if invalid:
        raise ValueError(f"Loss tidak valid: {invalid}. Pilih dari {module.LOSS_NAMES}.")

    pairs = module.find_pairs(dataset_dir)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=0.30, random_state=args.seed
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.50, random_state=args.seed
    )

    print(f"Total pasangan data  : {len(pairs)}")
    print(f"Train / Val / Test   : {len(train_pairs)} / {len(val_pairs)} / {len(test_pairs)}")
    print("Loading dataset ke memory...")
    x_train, y_train = module.load_arrays(train_pairs, args.image_size)
    x_val, y_val = module.load_arrays(val_pairs, args.image_size)
    x_test, y_test = module.load_arrays(test_pairs, args.image_size)

    run_dir = module.make_run_dir(output_dir, model_key, dataset_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    split_map = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    split_counts = {
        "train": len(train_pairs),
        "validation": len(val_pairs),
        "test": len(test_pairs),
    }

    for index, loss_name in enumerate(losses, start=1):
        print(f"\nTraining {model_name} - {dataset_role} - {loss_name.upper()} ({index}/{len(losses)})")
        loss_dir = run_dir / loss_name
        loss_dir.mkdir(parents=True, exist_ok=True)
        model = build_model(module, model_key, (args.image_size, args.image_size, 3), args.base_filters)
        print(f"Total parameter model: {model.count_params():,}")
        optimizer = module.tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss_fn = module.get_loss(loss_name)
        history = module.run_training_loop(
            model,
            loss_fn,
            optimizer,
            x_train,
            y_train,
            x_val,
            y_val,
            run_args,
            loss_dir,
        )
        history.to_csv(loss_dir / "history_final.csv", index=False)
        module.save_history_plot(history, loss_dir)

        metrics = module.evaluate_predictions(model, x_test, y_test, args.batch_size)
        metrics["loss_name"] = loss_name
        summary_rows.append(metrics)
        with open(loss_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        model.save_weights(loss_dir / "last.weights.h5")
        module.save_prediction_samples(
            model,
            test_pairs,
            args.image_size,
            loss_dir / "predictions",
            args.save_samples,
        )
        print(
            f"Hasil test: accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"dice={metrics['dice']:.4f} iou={metrics['iou']:.4f}"
        )
        module.tf.keras.backend.clear_session()

    summary_df = pd.DataFrame(summary_rows)
    metric_cols = ["loss_name", "accuracy", "precision", "recall", "dice", "iou"]
    summary_df[metric_cols].to_csv(run_dir / "summary_metrics.csv", index=False)
    module.save_hyperparameters(run_args, model_name, pairs, train_pairs, val_pairs, test_pairs, run_dir)
    module.save_preprocessing_visualization(pairs, args.image_size, run_dir)
    module.save_split_visualization(split_map, args.image_size, run_dir)
    module.save_split_distribution(split_counts, run_dir)
    module.save_performance_plot(summary_df, run_dir)
    module.plt.close("all")

    return {
        "dataset_role": dataset_role,
        "dataset_name": Path(dataset_dir).name,
        "dataset_path": str(Path(dataset_dir).resolve()),
        "model": model_name,
        "model_key": model_key,
        "run_dir": str(run_dir),
    }


def run_isolated_experiment(args, dataset_role, dataset_dir, model_key):
    output_dir = args.output_root / dataset_role / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--single-run",
        "--single-dataset-role",
        dataset_role,
        "--single-dataset-dir",
        dataset_dir,
        "--single-model-key",
        model_key,
        "--loss",
        args.loss,
        "--epochs",
        str(args.epochs),
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--base-filters",
        str(args.base_filters),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--patience",
        str(args.patience),
        "--save-samples",
        str(args.save_samples),
        "--limit",
        str(args.limit),
        "--output-root",
        str(args.output_root),
    ]
    if args.verbose_batches:
        command.append("--verbose-batches")

    subprocess.run(command, check=True)
    model_name, _ = MODELS[model_key]
    return {
        "dataset_role": dataset_role,
        "dataset_name": Path(dataset_dir).name,
        "dataset_path": str(Path(dataset_dir).resolve()),
        "model": model_name,
        "model_key": model_key,
        "run_dir": str(latest_child(output_dir)),
    }


def load_run_summary(run_info):
    run_dir = Path(run_info["run_dir"])
    summary_path = run_dir / "summary_metrics.csv"
    hyperparameter_path = run_dir / "04_hyperparameters.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary_metrics.csv tidak ditemukan: {summary_path}")

    summary = pd.read_csv(summary_path)
    summary.insert(0, "model", run_info["model"])
    summary.insert(0, "dataset_name", run_info["dataset_name"])
    summary.insert(0, "dataset_role", run_info["dataset_role"])
    summary["run_dir"] = run_info["run_dir"]

    if hyperparameter_path.exists():
        with open(hyperparameter_path, "r", encoding="utf-8") as file:
            hyperparameters = json.load(file)
        summary["total_data"] = hyperparameters.get("total_data")
        summary["train_data"] = hyperparameters.get("train_data")
        summary["validation_data"] = hyperparameters.get("validation_data")
        summary["test_data"] = hyperparameters.get("test_data")

    return summary


def save_comparison_plot(summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary.copy()
    plot_df["label"] = (
        plot_df["dataset_role"]
        + " | "
        + plot_df["model"]
        + " | "
        + plot_df["loss_name"].astype(str)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(plot_df["label"], plot_df["dice"], color="#2f6f9f")
    axes[0].set_title("Perbandingan Dice")
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(plot_df["label"], plot_df["iou"], color="#4f8f5f")
    axes[1].set_title("Perbandingan IoU")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_dir / "grafik_perbandingan_generalis_vs_spesialis.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Menjalankan perbandingan dataset generalis Kvasir-SEG dan dataset spesialis sessile."
    )
    parser.add_argument("--loss", default="all", help="Loss: all, jaccard, tversky, mse, atau mae.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--save-samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 berarti pakai seluruh data.")
    parser.add_argument("--verbose-batches", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unet", "unetplusplus"],
        choices=sorted(MODELS.keys()),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/perbandingan-dataset"),
    )
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-dataset-role", default="", help=argparse.SUPPRESS)
    parser.add_argument("--single-dataset-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--single-model-key",
        choices=sorted(MODELS.keys()),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Mode cepat: jaccard, 1 epoch, limit 4, image size 32, batch size 1, base filters 4.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.loss = "jaccard"
        args.epochs = 1
        args.limit = 4
        args.image_size = 32
        args.batch_size = 1
        args.base_filters = 4
        args.save_samples = 1
        args.output_root = Path("outputs/perbandingan-dataset-smoke-test")

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.single_run:
        if not args.single_dataset_role or not args.single_dataset_dir or not args.single_model_key:
            raise ValueError("single-run membutuhkan dataset role, dataset dir, dan model key.")
        run_experiment(
            args,
            args.single_dataset_role,
            args.single_dataset_dir,
            args.single_model_key,
        )
        return

    dataset_report = []
    for dataset_role, dataset_dir in DATASETS:
        pair_count = count_pairs(dataset_dir)
        dataset_report.append(
            {
                "dataset_role": dataset_role,
                "dataset_name": Path(dataset_dir).name,
                "total_pairs": pair_count,
                "catatan": "dataset utama" if dataset_role == "generalis" else "subset polip sessile",
            }
        )
    dataset_report_df = pd.DataFrame(dataset_report)
    dataset_report_df.to_csv(args.output_root / "ringkasan_dataset.csv", index=False)

    run_infos = []
    for dataset_role, dataset_dir in DATASETS:
        for model_key in args.models:
            run_infos.append(run_isolated_experiment(args, dataset_role, dataset_dir, model_key))

    summary = pd.concat([load_run_summary(run_info) for run_info in run_infos], ignore_index=True)
    selected_columns = [
        "dataset_role",
        "dataset_name",
        "model",
        "loss_name",
        "accuracy",
        "precision",
        "recall",
        "dice",
        "iou",
        "total_data",
        "train_data",
        "validation_data",
        "test_data",
        "run_dir",
    ]
    summary = summary[selected_columns]
    summary_path = args.output_root / "ringkasan_perbandingan_generalis_vs_spesialis.csv"
    summary.to_csv(summary_path, index=False)
    save_comparison_plot(summary, args.output_root)

    print("\n=== RINGKASAN DATASET ===")
    print(dataset_report_df.to_string(index=False))
    print("\n=== RINGKASAN PERBANDINGAN ===")
    print(summary.sort_values(["dataset_role", "model", "loss_name"]).to_string(index=False))
    print(f"\nCSV ringkasan : {summary_path.resolve()}")
    print(
        "Grafik        : "
        f"{(args.output_root / 'grafik_perbandingan_generalis_vs_spesialis.png').resolve()}"
    )


if __name__ == "__main__":
    main()
