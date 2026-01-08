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


> Please refer to `data/in_the_wild/README.md` for detailed instructions on accessing the **CO-SPY-Bench in-the-wild synthetic samples**.
>
> *Note: Some samples are temporarily unavailable (e.g., [instavibe.ai](https://www.instavibe.ai/)). As we cannot confirm whether redistributing these images would violate the original sources’ intellectual property rights, we choose not to release them at this time. This decision and its rationale are discussed in [Issue #6](https://github.com/Megum1/CO-SPY/issues/6). We apologize for any inconvenience this may cause.*


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
To evaluate the pre-trained detector (trained on DRCT-2M (`sd-v1_4`)) on [CO-SPY-Bench](https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench), run
```bash
python main.py --gpu 0 --phase eval --train_dataset sd-v1_4 --pretrain
```
When finished, evaluation results will be saved to `ckpt/sd-v1_4/fusion/pretrain_Co-Spy-Bench`.

<details>
<summary><strong>📊 Results (click to expand)</strong></summary>

<p><em>
Results are highly consistent with the original results reported in Table 2, with slight differences arising from improvements in hyper-parameter settings.
Original results can be referred to in <a href="https://github.com/Megum1/CO-SPY/issues/7">Issue #7</a>.
</em></p>

## Average Precision (AP)
| Detector | CC3M | FLICKR | MSCOCO | TEXTCAPS | SBU |
|----------|------|--------|---------|----------|-----|
| ldm-text2im-large-256 | 95.61 | 99.93 | 99.87 | 98.82 | 99.73 |
| stable-diffusion-v1-4 | 91.95 | 99.74 | 99.81 | 97.82 | 98.66 |
| stable-diffusion-v1-5 | 91.44 | 99.71 | 99.67 | 97.46 | 98.66 |
| SSD-1B | 89.07 | 99.33 | 99.17 | 95.17 | 98.22 |
| tiny-sd | 85.52 | 99.11 | 98.92 | 95.41 | 98.01 |
| SegMoE-SD-4x2-v0 | 90.09 | 99.49 | 99.49 | 96.93 | 98.72 |
| small-sd | 86.91 | 99.08 | 99.14 | 95.62 | 98.28 |
| stable-diffusion-2-1 | 90.83 | 99.61 | 99.73 | 97.74 | 98.53 |
| stable-diffusion-3-medium-diffusers | 86.56 | 99.13 | 99.06 | 94.67 | 97.86 |
| sdxl-turbo | 97.09 | 99.86 | 99.81 | 97.69 | 99.81 |
| stable-diffusion-2 | 86.63 | 99.44 | 99.33 | 95.87 | 97.59 |
| stable-diffusion-xl-base-1.0 | 80.06 | 98.53 | 98.40 | 90.05 | 94.92 |
| playground-v2.5-1024px-aesthetic | 90.87 | 99.75 | 99.73 | 96.92 | 98.65 |
| playground-v2-1024px-aesthetic | 91.18 | 99.81 | 99.73 | 97.46 | 98.82 |
| playground-v2-512px-base | 84.11 | 98.47 | 98.66 | 94.62 | 96.98 |
| playground-v2-256px-base | 86.73 | 99.24 | 99.03 | 96.64 | 97.80 |
| PixArt-XL-2-1024-MS | 93.31 | 99.94 | 99.90 | 98.55 | 99.53 |
| PixArt-XL-2-512x512 | 94.39 | 99.93 | 99.93 | 98.54 | 99.58 |
| lcm-lora-sdxl | 97.00 | 99.98 | 99.96 | 99.14 | 99.84 |
| lcm-lora-sdv1-5 | 98.28 | 99.98 | 99.97 | 99.51 | 99.92 |
| FLUX.1-schnell | 88.58 | 99.52 | 99.44 | 94.69 | 98.28 |
| FLUX.1-dev | 88.44 | 99.61 | 99.50 | 94.57 | 98.09 |
| **Average** | **89.39** | **99.46** | **99.40** | **96.52** | **98.50** |

## Accuracy
| Detector | CC3M | FLICKR | MSCOCO | TEXTCAPS | SBU |
|----------|------|--------|---------|----------|-----|
| ldm-text2im-large-256 | 88.45 | 97.10 | 96.73 | 94.68 | 94.65 |
| stable-diffusion-v1-4 | 83.73 | 93.65 | 96.03 | 92.10 | 84.45 |
| stable-diffusion-v1-5 | 82.73 | 92.85 | 95.25 | 91.33 | 83.23 |
| SSD-1B | 80.33 | 87.58 | 88.88 | 85.23 | 80.93 |
| tiny-sd | 76.38 | 83.25 | 85.43 | 86.63 | 78.85 |
| SegMoE-SD-4x2-v0 | 82.23 | 88.88 | 92.20 | 89.38 | 84.03 |
| small-sd | 77.78 | 84.23 | 87.45 | 87.05 | 80.00 |
| stable-diffusion-2-1 | 82.60 | 92.58 | 94.03 | 91.65 | 83.48 |
| stable-diffusion-3-medium-diffusers | 77.70 | 86.95 | 88.23 | 83.85 | 79.10 |
| sdxl-turbo | 90.45 | 95.53 | 96.35 | 91.95 | 96.40 |
| stable-diffusion-2 | 78.80 | 88.68 | 90.55 | 87.80 | 76.58 |
| stable-diffusion-xl-base-1.0 | 70.13 | 80.60 | 83.53 | 76.68 | 67.33 |
| playground-v2.5-1024px-aesthetic | 83.90 | 93.93 | 94.65 | 89.70 | 84.00 |
| playground-v2-1024px-aesthetic | 83.45 | 95.15 | 94.58 | 90.93 | 84.95 |
| playground-v2-512px-base | 74.28 | 83.23 | 86.35 | 84.58 | 74.50 |
| playground-v2-256px-base | 77.50 | 87.50 | 88.10 | 89.35 | 78.95 |
| PixArt-XL-2-1024-MS | 86.80 | 97.73 | 97.83 | 94.00 | 93.38 |
| PixArt-XL-2-512x512 | 87.18 | 97.13 | 98.13 | 93.65 | 93.35 |
| lcm-lora-sdxl | 91.40 | 99.03 | 98.85 | 95.25 | 96.98 |
| lcm-lora-sdv1-5 | 91.98 | 99.15 | 99.28 | 96.33 | 98.28 |
| FLUX.1-schnell | 80.05 | 90.15 | 90.88 | 84.58 | 80.68 |
| FLUX.1-dev | 79.93 | 92.43 | 92.05 | 83.68 | 81.40 |
| **Average** | **82.22** | **91.24** | **92.24** | **89.20** | **84.34** 

</details>



To evaluate the pre-trained detector (trained on CNNDet (`progan`)) on [AIGCDetectionBenchMark](https://github.com/Ekko-zn/AIGCDetectBenchmark), run
```bash
python main.py --gpu 1 --phase eval --train_dataset progan --pretrain
```
When finished, evaluation results will be saved to `ckpt/progan/fusion/pretrain_AIGCDetectionBenchMark`.

<details>
<summary><strong>📊 Results (click to expand)</strong></summary>

<p><em>
Results are highly consistent with the original results reported in Table 8 in Appendix K, with slight differences arising from improvements in hyper-parameter settings.
</em></p>

| Model | AP | Accuracy |
|-------|------|----------|
| ADM | 89.98 | 79.17 |
| biggan | 98.00 | 94.30 |
| cyclegan | 99.47 | 98.98 |
| DALLE2 | 97.14 | 87.80 |
| gaugan | 98.24 | 95.05 |
| Glide | 96.71 | 90.40 |
| Midjourney | 94.55 | 87.19 |
| progan | 100.00 | 100.00 |
| stable_diffusion_v_1_4 | 92.99 | 85.34 |
| stable_diffusion_v_1_5 | 92.96 | 85.44 |
| stargan | 99.98 | 99.45 |
| stylegan | 99.69 | 94.81 |
| stylegan2 | 99.85 | 94.89 |
| VQDM | 91.96 | 85.83 |
| whichfaceisreal | 82.27 | 80.55 |
| wukong | 89.59 | 80.06 |
| **Average** | **95.21** | **90.58** |

</details>

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
# Train the semantic component (on CNNDet training set as an example)
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
@inproceedings{Cheng_2025_CVPR,
    author    = {Cheng, Siyuan and Lyu, Lingjuan and Wang, Zhenting and Zhang, Xiangyu and Sehwag, Vikash},
    title     = {CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {13455-13465}
}
```

## Acknowledgement
We gratefully acknowledge these outstanding works, which have deeply inspired our project!
- [CNNDetection](https://github.com/PeterWang512/CNNDetection)
- [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
- [NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
- [DRCT](https://github.com/beibuwandeluori/DRCT)
