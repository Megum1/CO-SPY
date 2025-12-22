########### Training & Validation Dataset Download Script (DRCT-2M) ###########
# Download and unzip the synthetic training dataset from DRCT
# Reference: https://icml.cc/virtual/2024/poster/33086
# Data source: https://github.com/beibuwandeluori/DRCT
wget --no-check-certificate https://modelscope.cn/datasets/BokingChen/DRCT-2M/resolve/master/images/stable-diffusion-v1-4.zip
unzip stable-diffusion-v1-4.zip
rm stable-diffusion-v1-4.zip

# Download the real training dataset from MSCOCO2017
# Reference: https://arxiv.org/pdf/1405.0312
# Data source: https://cocodataset.org/#download
mkdir mscoco2017
cd mscoco2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip
rm train2017.zip
rm val2017.zip
cd ..
