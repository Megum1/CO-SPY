#!/bin/bash
########### Training & Validation Dataset Download Script (DRCT-2M) ###########
# Download and unzip the synthetic training dataset from DRCT
# Reference: https://icml.cc/virtual/2024/poster/33086
# Data source: https://github.com/beibuwandeluori/DRCT

# Preflight: check required tools.
for tool in wget unzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not found in PATH."
        echo "Install inside the active conda env:"
        echo "    conda install -c conda-forge $tool"
        echo "(Alternatively: apt install $tool  /  dnf install $tool)"
        exit 1
    fi
done

wget --no-check-certificate https://modelscope.cn/datasets/BokingChen/DRCT-2M/resolve/master/images/stable-diffusion-v1-4.zip
unzip stable-diffusion-v1-4.zip
rm stable-diffusion-v1-4.zip

# Download the real training dataset from MSCOCO2017
# Reference: https://arxiv.org/pdf/1405.0312
# Data source: https://cocodataset.org/#download
wget https://huggingface.co/datasets/ruojiruoli/Co-Spy-Misc/resolve/main/mscoco2017.zip
unzip mscoco2017.zip
rm mscoco2017.zip
