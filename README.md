<div align="center">
    <img src="imgs/logo.png" width="220px" />
</div>

# CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI

![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 2.4.1](https://img.shields.io/badge/pytorch-2.4.1-DodgerBlue.svg?style=plastic)
![Torchvision 0.19.1](https://img.shields.io/badge/torchvision-0.18.1-DodgerBlue.svg?style=plastic)
![CUDA 12.1](https://img.shields.io/badge/cuda-12.1-DodgerBlue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/License-MIT-DodgerBlue.svg?style=plastic)

Table of Contents
=================
- [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [CO-SPY-Bench](#co-spy-bench)
  - [Main Code Architecutre](#main-code-architecture)
  - [Environments](#environments)
  - [Prerequisites](#prerequisites)
    - [Download Datasets](#download-datasets)
    - [Download Pre-trained Weights](#download-pre-trained-weights)
  - [Experiments](#experiments)
    - [Evaluation of Pre-trained Detectors](#evaluation-of-pre-trained-detectors)
    - [Inference on a Single Image](#inference-on-a-single-image)
    - [Training](#training)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Overview
- This is the official implementation for CVPR 2025 paper "[CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_CO-SPY_Combining_Semantic_and_Pixel_Features_to_Detect_Synthetic_Images_CVPR_2025_paper.html)".
- [[arXiv](https://arxiv.org/abs/2503.18286)\] | \[[poster](https://www.cs.purdue.edu/homes/cheng535/static/slides/COSPY_poster.pdf)\]

<img src="imgs/overview.png" width="900px"/>

## CO-SPY-Bench
We have released our benchmark on [Huggingface](https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench), designed to offer diverse and comprehensive coverage of the latest generative models:
- Captions are sourced from five real-world datasets: MSCOCO2017, CC3M, Flickr, TextCaps, and SBU.
- Synthetic images are generated using 22 different models, covering a wide range of architectures.
- Diverse generation parameters, such as diffusion steps and guidance scales, are used to enrich variability.

## Main Code Architecture
    .
    ├── data              # Dataset folder
    │   ├── in_the_wild   # CO-SPY-Bench in-the-wild synthetic samples
    │   ├── test          # Test dataset (CO-SPY-Bench & AIGCDetectionBenchMark)
    │   └── train         # Training dataset (DRCT-2M & CNNDet)
    ├── dataSets          # Various dataset classes
    ├── detectors         # Various detector classes
    │   ├── progan        # Detectors for CNNDet training set
    │   └── sd-v1_4       # Detectors for DRCT-2M training set
    ├── pretrained        # Pre-trained weights
    ├── main.py           # Main function
    ├── evaluate.py       # Evaluation function
    ├── main.py           # Main function (Entry point)
    ├── train.py          # Training function
    ├── train.sh          # Recommended training script
    └── utils.py          # Utility functions

## Environments
```bash
# Create python environment (optional)
conda env create -f environment.yml
source activate cospy
```

## Prerequisites
Please download the required datasets and pre-trained weights for full evaluation.

### Download Datasets
Make sure you have `7z` an `unzip` installed. You can install them via conda:
```bash
conda install p7zip
conda install unzip
```

To download the training and test datasets, run the following commands respectively:
```bash
###############################
# Download the training dataset
###############################
# Download CNNDet (ProGAN) training set
cd data/train/progan
sh download.sh

# Download DRCT-2M (Stable Diffusion v1.4) training set
cd data/train/sd-v1_4
sh download.sh

###############################
# Download the test dataset
###############################
# Download AIGCDetectionBenchMark test set
cd data/test/AIGCDetectionBenchMark
python download.py

# Download CO-SPY-Bench test set
cd data/test/Co-Spy-Bench
sh download.sh
```

Finally, the directory structure should look like this:
```
.
├── data
│   ├── test
│   │   ├── AIGCDetectionBenchMark
│   │   │   └── test
│   │   └── Co-Spy-Bench
│   │       ├── real_image_examples
│   │       └── synthetic
│   └── train
│       ├── progan
│       │   ├── train
│       │   └── val
│       └── sd-v1_4
│           ├── mscoco2017
│           └── stable_diffusion_v1-4
```
*Note: Please ensure the use of these datasets complies with the original licenses.*

### Download Pre-trained Weights
```bash
# Download the pre-trained weights
cd pretrained
sh download.sh
# It contains the pre-trained weights on CNNDet (`progan`) and DRCT-2M (`sd-v1_4`) training sets.
```



## Experiments  
We provide the source code for training and evaluating the CO-SPY detector.

### Evaluation of Pre-trained Detectors
To evaluate the pre-trained detector, trained on DRCT-2M (`sd-v1_4`), on [CO-SPY-Bench](https://github.com/your-repo/CO-SPY-Bench), run
```bash
python main.py --gpu 0 --phase eval --train_dataset sd-v1_4 --pretrain
```
When finished, evaluation results will be saved to `ckpt/sd-v1_4/fusion/pretrain_Co-Spy-Bench`.

To evaluate the pre-trained detector, trained on CNNDet (`progan`), on [AIGCDetectionBenchMark](https://github.com/Ekko-zn/AIGCDetectBenchmark), run
```bash
python main.py --gpu 1 --phase eval --train_dataset progan --pretrain
```
When finished, evaluation results will be saved to `ckpt/progan/fusion/pretrain_AIGCDetectionBenchMark`.

Evaluation results contain three files, including:
- `evaluation.log`: detailed evaluation log.
- `output.json`: predicted synthetic probabilities for each sample.
- `result.json`: evaluation metrics (dataset size, AP and accuracy) for each test source.

### Inference on a Single Image
To run inference on a single image (e.g., using the pre-trained detector trained on DRCT-2M):
```bash
python main.py --gpu 0 --phase test --train_dataset sd-v1_4 --pretrain
# The script will prompt for the image file path:
# "Please enter the image filepath for scanning: "
imgs/test.png
# Output (probability - decision):
# "CO-SPY Prediction: 0.854 - AI-Generated"
```

### Training
We provide two training pipelines: (1) end-to-end training and (2) best practice of training semantic and artifact branches separately, followed by calibrating the combined detector.

```bash
# Train an end-to-end CO-SPY detector for 10 epochs on DRCT-2M
python main.py --gpu 0 --phase train --mode end2end --train_dataset sd-v1_4 --epochs 10
# Train an end-to-end CO-SPY detector for 10 epochs on CNNDet
python main.py --gpu 1 --phase train --mode end2end --train_dataset progan --epochs 10
```
The trained model will be saved to `ckpt/<train_dataset>/end2end`.

*The end-to-end training may not yield optimal performance due to the conflicting nature of semantic and artifact features. We recommend the following best practice for training.*
```bash
# Train an optimal CO-SPY detector using the best practice script on DRCT-2M
bash train.sh --gpu 2 --dataset sd-v1_4
# Train an optimal CO-SPY detector using the best practice script on CNNDet
bash train.sh --gpu 2 --dataset progan
```
The trained models will be saved to `ckpt/<train_dataset>/fusion`.

You can also customize the training parameters for the training of each branch and the calibration step:
```bash
# Train the semantic component (on ProGAN training set as an example)
python main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch semantic \
    --train_dataset progan \
    --epochs 10
# Train the artifact component
python main.py \
    --phase train \
    --gpu 1 \
    --mode branch \
    --branch artifact \
    --train_dataset progan \
    --epochs 20
# Calibrate the combined CO-SPY detector
python main.py \
    --phase train \
    --gpu 2 \
    --mode fusion \
    --train_dataset progan \
    --epochs 2
```
The trained branch models and the calibrated fusion model will be saved to `ckpt/progan/semantic`, `ckpt/progan/artifact`, and `ckpt/progan/fusion`, respectively.

After training, you can evaluate the trained model on the test datasets by just changing the `--phase` to `eval` and specifying the `--train_dataset` and `--mode` accordingly. For example, to evaluate the fusion model (best practice), trained on CNNDet, on AIGCDetectionBenchMark, run:
```bash
python main.py --gpu 0 --phase eval --train_dataset progan --mode fusion
```

## Citation
Please cite our paper if you find it useful for your research.😀

```bibtex
@inproceedings{cheng2025co,
  title={CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI},
  author={Cheng, Siyuan and Lyu, Lingjuan and Wang, Zhenting and Zhang, Xiangyu and Sehwag, Vikash},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={13455--13465},
  year={2025}
}
```

## Acknowledgement
We gratefully acknowledge these outstanding works, which have deeply inspired our project!
- [CNNDetection](https://github.com/PeterWang512/CNNDetection)
- [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
- [NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
- [DRCT](https://github.com/beibuwandeluori/DRCT)
