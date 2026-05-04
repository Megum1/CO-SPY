#!/bin/bash
########### Training & Validation Dataset Download Script (CNNDet) ###########
# Download and unzip the synthetic training dataset from CNNDet
# Reference: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf
# Data source: https://github.com/peterwang512/CNNDetection

# Preflight: check required tools. 7z is needed to extract .7z.NNN split
# archives, unzip for the inner .zip file.
_missing=0
for tool in wget 7z unzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not found in PATH."
        case "$tool" in
            7z)
                echo "Install inside the active conda env:"
                echo "    conda install -c conda-forge p7zip"
                echo "(Alternatively: apt install p7zip-full  /  dnf install p7zip)"
                ;;
            *)
                echo "Install inside the active conda env:"
                echo "    conda install -c conda-forge $tool"
                echo "(Alternatively: apt install $tool  /  dnf install $tool)"
                ;;
        esac
        _missing=1
    fi
done
[ "$_missing" = 1 ] && exit 1

# Create directory and download training dataset
# mkdir train
cd train

wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.002 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.003 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.004 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.005 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.006 &
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.007 &
wait $(jobs -p)

7z x progan_train.7z.001
unzip progan_train.zip
rm progan_train.zip
rm progan_train.7z.*
cd ..

# Create directory and download validation dataset
mkdir val
cd val
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip

unzip progan_val.zip
rm progan_val.zip
cd ..
