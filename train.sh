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

# Artifact branch training
python main.py \
    --phase train \
    --gpu $gpu \
    --mode branch \
    --branch artifact \
    --train_dataset $dataset \
    --epochs 20

# Semantic branch training (feature interpolation is only applied for sd-v1_4)
semantic_extra=""
if [[ "$dataset" == "sd-v1_4" ]]; then
    semantic_extra="--feat_interp"
fi

python main.py \
    --phase train \
    --gpu $gpu \
    --mode branch \
    --branch semantic \
    --train_dataset $dataset \
    --epochs 10 \
    $semantic_extra

# Fusion
python main.py \
    --phase train \
    --gpu $gpu \
    --mode fusion \
    --train_dataset $dataset \
    --epochs 1
