#!/bin/bash
# Source: https://modelscope.cn/datasets/aemilia/AIGCDetectionBenchmark/tree/master/AIGCDetectionBenchMark
# Reference: https://github.com/Ekko-zn/AIGCDetectBenchmark

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

wget --no-check-certificate https://modelscope.cn/datasets/aemilia/AIGCDetectionBenchmark/resolve/master/AIGCDetectionBenchMark/test_set.zip
unzip test_set.zip
rm test_set.zip
# Should contain an additional directory "test" after unzipping
