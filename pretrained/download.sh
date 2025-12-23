# Download pre-trained weights for CO-SPY
# Pre-trained weights on CNNDet (ProGAN)
wget https://huggingface.co/ruojiruoli/Co-Spy-Pretrained-Weights/resolve/main/progan.zip
unzip progan.zip
rm progan.zip

# Pre-trained weights on DRCT-2M (Stable Diffusion v1.4)
wget https://huggingface.co/ruojiruoli/Co-Spy-Pretrained-Weights/resolve/main/sd-v1_4.zip
unzip sd-v1_4.zip
rm sd-v1_4.zip

# Clean up unnecessary files
rm -rf __MACOSX
