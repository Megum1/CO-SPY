########### Training & Validation Dataset Download Script (CNNDet) ###########
# Download and unzip the synthetic training dataset from CNNDet
# Reference: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf
# Data source: https://github.com/peterwang512/CNNDetection

# Create directory and download training dataset
mkdir train
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
rm progan_train.7z.*
unzip progan_train.zip
rm progan_train.zip
cd ..

# Create directory and download validation dataset
mkdir val
cd val
wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip

unzip progan_val.zip
rm progan_val.zip
cd ..
