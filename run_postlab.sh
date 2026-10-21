#!/usr/bin/env bash
# ELEC 475 Lab 2 - runs every post-lab experiment in order.
# Resumable: a run is skipped if its results/*.json history already exists.
# Usage (from the project root, venv active):   caffeinate -i ./run_postlab.sh [stage ...]
# Stages: norm aug tiny zoo lenet seeds zoofix   (default: all, in that order)
set -u
mkdir -p logs results

run() {  # run <history json> <train.py args...>
    local hist="$1"; shift
    if [ -f "$hist" ]; then echo "[skip] $hist exists"; return; fi
    local log="logs/$(basename "${hist%.json}").log"
    echo "[run ] python train.py $* --history-file $hist   (log: $log)"
    python -u train.py "$@" --history-file "$hist" 2>&1 | grep --line-buffered -v -i warn > "$log"
}

COMMON="--seed 42 --batch-size 128"

stage_norm() {  # Step 2: normalization study (no augmentation so only the norm layer differs)
    for n in none lrn bn; do
        run results/loss_alexnet_$n.json --model alexnet --norm $n --dataset cifar10 --epochs 10 $COMMON --no-augment \
            --save-model checkpoints/alexnet_$n.pth --plot-file plots/loss_alexnet_$n.png
    done
}

stage_aug() {  # Step 3: BN AlexNet with vs without augmentation, 15 epochs
    run results/loss_no_aug.json --model alexnet --norm bn --dataset cifar10 --epochs 15 $COMMON --no-augment \
        --save-model checkpoints/alexnet_no_aug.pth --plot-file plots/loss_no_aug.png
    run results/loss_with_aug.json --model alexnet --norm bn --dataset cifar10 --epochs 15 $COMMON \
        --save-model checkpoints/alexnet_with_aug.pth --plot-file plots/loss_with_aug.png
}

stage_tiny() {  # Step 4 + Table 7.3b Tiny ImageNet column (10 epochs each)
    run results/loss_tiny_imagenet.json --model alexnet --norm bn --dataset tiny_imagenet --epochs 10 $COMMON \
        --save-model checkpoints/alexnet_tiny_imagenet.pth --plot-file plots/loss_tiny_imagenet.png
    run results/loss_tiny_lenet.json --model lenet --dataset tiny_imagenet --epochs 10 $COMMON --no-augment \
        --plot-file plots/loss_tiny_lenet.png
    run results/loss_tiny_lenet_modern.json --model lenet_modern --dataset tiny_imagenet --epochs 10 $COMMON --no-augment \
        --plot-file plots/loss_tiny_lenet_modern.png
    for n in none lrn bn; do
        run results/loss_tiny_alexnet_$n.json --model alexnet --norm $n --dataset tiny_imagenet --epochs 10 $COMMON --no-augment \
            --plot-file plots/loss_tiny_alexnet_$n.png
    done
    run results/loss_tiny_zoo_alexnet.json --model zoo_alexnet --dataset tiny_imagenet --epochs 10 $COMMON --no-augment \
        --plot-file plots/loss_tiny_zoo_alexnet.png
}

stage_zoo() {  # Table 7.3 row 7 on CIFAR-10 (same 10-epoch, no-augment setup as the norm study)
    # no --save-model: a 57M-param checkpoint is ~230 MB and not a required deliverable
    run results/loss_zoo_alexnet.json --model zoo_alexnet --dataset cifar10 --epochs 10 $COMMON --no-augment \
        --plot-file plots/loss_zoo_alexnet.png
}

stage_lenet() {  # Re-time both LeNets on the same device as AlexNet so Table 7.3a epoch times are comparable
    run results/loss_lenet_mps.json --model lenet --dataset cifar10 --epochs 5 $COMMON --no-augment \
        --plot-file plots/loss_lenet_mps.png
    run results/loss_lenet_modern_mps.json --model lenet_modern --dataset cifar10 --epochs 5 $COMMON --no-augment \
        --plot-file plots/loss_lenet_modern_mps.png
}

stage_seeds() {  # Multi-seed Mean +/- Std for the normalization study (seed 42 comes from stage_norm)
    mkdir -p results/seeds plots/seeds
    for s in 123 999; do
        for n in none lrn bn; do
            run results/seeds/alexnet_${n}_s$s.json --model alexnet --norm $n --dataset cifar10 --epochs 10 \
                --seed $s --batch-size 128 --no-augment --plot-file plots/seeds/alexnet_${n}_s$s.png
        done
    done
}

stage_zoofix() {  # Zoo AlexNet collapsed to chance on Tiny ImageNet at lr 1e-3 (loss stuck at ln 200); retry at lr 1e-4
    run results/loss_tiny_zoo_alexnet_lr1e-4.json --model zoo_alexnet --dataset tiny_imagenet --epochs 10 $COMMON --no-augment \
        --lr 1e-4 --plot-file plots/loss_tiny_zoo_alexnet_lr1e-4.png
}

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(norm aug tiny zoo lenet seeds zoofix)
for st in "${STAGES[@]}"; do
    echo "===== stage: $st ====="
    "stage_$st"
done
echo "===== all requested stages done ====="
