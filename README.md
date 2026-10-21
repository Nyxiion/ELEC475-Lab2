# ELEC 475 Lab 2: From LeNet-5 to AlexNet

ELEC 475 (Computer Vision with Deep Learning), Fall 2026, Queen's University.

This repository contains my completed Lab 2: a base and modernized LeNet-5, a 5-stage AlexNet-64 with a normalization study (none vs LRN vs BatchNorm2d), a PyTorch data augmentation pipeline, and a benchmark of every model on CIFAR-10 and Tiny ImageNet-200 at 64x64 resolution. The full analysis is in the submitted report (`ELEC475_Lab2_Report.pdf`, submitted on OnQ).

## Results summary

Best-epoch validation accuracy (seed 42; values in brackets are Mean ± Std over seeds 42, 123 and 999).

| Model | Params (M) | FLOPs (M) | CIFAR-10 acc (%) | Tiny ImageNet acc (%) |
|---|---|---|---|---|
| Base LeNet-5 (Tanh + AvgPool) | 0.338 | 7.17 | 53.06 | 11.59 |
| Modernized LeNet (ReLU + overlapping MaxPool) | 0.446 | 9.19 | 64.74 | 20.67 |
| AlexNet-64, no normalization | 6.609 | 436.70 | 73.23 (72.75 ± 2.08) | 15.67 |
| AlexNet-64 + LRN | 6.609 | 436.70 | 71.86 (73.75 ± 2.78) | 16.96 |
| AlexNet-64 + BatchNorm | 6.611 | 437.97 | 79.19 (79.47 ± 0.25) | 30.03 |
| AlexNet-64 + BatchNorm + augmentation | 6.611 | 437.97 | 81.81 | 30.57 |
| PyTorch Model Zoo AlexNet | 57.045 | 188.20 | 67.12 | 20.49 (lr 1e-4) |

CIFAR-10 runs use 5 epochs for the LeNets, 10 for the AlexNet normalization study and 15 for the augmentation study; all Tiny ImageNet runs use 10 epochs. Hardware: Apple M3 Pro, trained on the GPU through PyTorch MPS.

## Repository layout

```text
configs/config.py        Experiment configuration
models/lenet.py          Base LeNet-5 for 3x64x64 input (in-lab Step 2)
models/lenet_modern.py   Modernized LeNet: ReLU + overlapping MaxPool (in-lab Step 4)
models/alexnet.py        AlexNet-64 with --norm none | lrn | bn (post-lab Steps 1-2)
models/zoo.py            torchvision AlexNet adapter (Model Zoo baseline)
utils/dataset.py         CIFAR-10 and Tiny ImageNet loaders (download automatically)
utils/transforms.py      Augmentation: RandomCrop, RandomHorizontalFlip, ColorJitter (post-lab Step 3)
utils/eda.py             Exploratory data analysis (class balance, channel statistics)
utils/metrics.py         Accuracy, Macro/Micro F1, confusion matrix, early stopping
utils/profiler.py        Parameters, MACs/FLOPs and a device-synchronized Timer
utils/seed.py            Global seeding and Mean ± Std aggregation
utils/visualization.py   Training curves and the ablation summary plot
train.py                 Training script (CLI)
evaluate.py              Checkpoint evaluation
compare_models.py        Complexity benchmark across all models
demo_transforms.py       Pre-lab transform benchmark
run_postlab.sh           Resumable script that runs every post-lab experiment
commands.txt             Exact commands used for every result
plots/  logs/  results/  Figures, training logs and per-epoch histories (JSON)
```

The `data/` folder (datasets) and `checkpoints/` (model weights) are not tracked in git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CIFAR-10 and Tiny ImageNet are downloaded into `data/` automatically on first use.

## Reproducing the results

Every command used is listed in [`commands.txt`](commands.txt). The main steps:

```bash
# Check the architectures
python -m models.lenet
python -m models.lenet_modern
python -m models.alexnet

# In-lab: baseline vs modernized LeNet (5 epochs)
python train.py --model lenet        --dataset cifar10 --epochs 5 --seed 42 --no-augment --save-model checkpoints/lenet.pth        --plot-file plots/loss_lenet.png
python train.py --model lenet_modern --dataset cifar10 --epochs 5 --seed 42 --no-augment --save-model checkpoints/lenet_modern.pth --plot-file plots/loss_lenet_modern.png

# Post-lab: all experiments (normalization, augmentation, Tiny ImageNet, Zoo AlexNet, extra seeds)
./run_postlab.sh                 # or pick stages: ./run_postlab.sh norm aug tiny zoo lenet seeds zoofix

# Complexity benchmark and ablation plot
python compare_models.py --dataset cifar10 --device mps --reps 200
python -m utils.visualization --plot-ablation
```

`train.py` picks the device automatically (CUDA, then Apple MPS, then CPU); override it with `--device`.

## Changes to the starter template

A few problems in the provided template were fixed; they are described in Appendix A of the report:

- Tiny ImageNet validation now uses the official `val/` split (the template validated on training images).
- The normalization study is run with `--no-augment` so the three variants differ only in normalization.
- FLOPs are profiled on a copy of the model, so thop no longer leaves extra buffers in saved checkpoints.
- Added `--device` (Apple MPS support) to `train.py` and `compare_models.py`, and a working `--plot-ablation` command.
