#!/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

python main.py \
    --phase train \
    --gpu $gpu \
    --mode branch \
    --branch artifact \
    --train_dataset $dataset \
    --epochs 20

python main.py \
    --phase train \
    --gpu $gpu \
    --mode branch \
    --branch semantic \
    --train_dataset $dataset \
    --epochs 10

python main.py \
    --phase train \
    --gpu $gpu \
    --mode fusion \
    --train_dataset $dataset \
    --epochs 2
