"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title("ELEC 475: Master Ablation Study (Validation Accuracy)", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")


def load_run_histories(results_dir: str = "./results") -> Dict[str, dict]:
    """Loads every per-epoch history JSON written by train.py, keyed by file name (e.g. 'loss_alexnet_bn')."""
    import glob
    import json
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            runs[os.path.splitext(os.path.basename(path))[0]] = json.load(f)
    return runs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ELEC 475 Lab 2 plotting utilities")
    parser.add_argument("--plot-ablation", action="store_true",
                        help="Plot validation accuracy of every CIFAR-10 run in --results-dir on one figure")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Only include runs on this dataset")
    parser.add_argument("--save-path", type=str, default="./plots/ablation_summary.png")
    args = parser.parse_args()

    if args.plot_ablation:
        runs = load_run_histories(args.results_dir)
        # The seven rows of the master ablation table (Table 7.3a), in ladder order.
        # loss_no_aug is left out: its first 10 epochs are identical to loss_alexnet_bn (same seed).
        ladder = [
            ("loss_lenet_mps", "1. Base LeNet-5"),
            ("loss_lenet_modern_mps", "2. Modernized LeNet"),
            ("loss_alexnet_none", "3. AlexNet (No Norm)"),
            ("loss_alexnet_lrn", "4. AlexNet + LRN"),
            ("loss_alexnet_bn", "5. AlexNet + BatchNorm"),
            ("loss_with_aug", "6. AlexNet + BN + Aug"),
            ("loss_zoo_alexnet", "7. PyTorch Zoo AlexNet"),
        ]
        selected = {label: {"val_accs": runs[name]["val_accs"]}
                    for name, label in ladder
                    if name in runs and runs[name]["args"]["dataset"] == args.dataset}
        if not selected:
            raise SystemExit(f"No {args.dataset} histories found in {args.results_dir}; train with --plot-file first.")
        plot_ablation_summary(selected, save_path=args.save_path)
    else:
        parser.print_help()
