# Source: https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench
# Source: https://huggingface.co/datasets/ruojiruoli/Co-Spy-Misc
import os
import urllib
import tarfile
import zipfile
import requests
from tqdm import tqdm


# List of evaluated datasets
DATASET_LIST = ['mscoco', 'flickr', 'cc3m', 'textcaps', 'sbu']

# List of evaluated generative models
MODEL_LIST = [
    # CompVis
    "CompVis@ldm-text2im-large-256",
    "CompVis@stable-diffusion-v1-4",
    # runwayml
    "runwayml@stable-diffusion-v1-5",
    # segmind
    "segmind@SSD-1B",
    "segmind@tiny-sd",
    "segmind@SegMoE-SD-4x2-v0",
    "segmind@small-sd",
    # stabilityai
    "stabilityai@stable-diffusion-2-1",
    "stabilityai@stable-diffusion-3-medium-diffusers",
    "stabilityai@sdxl-turbo",
    "stabilityai@stable-diffusion-2",
    "stabilityai@stable-diffusion-xl-base-1.0",
    # playgroundai
    "playgroundai@playground-v2.5-1024px-aesthetic",
    "playgroundai@playground-v2-1024px-aesthetic",
    "playgroundai@playground-v2-512px-base",
    "playgroundai@playground-v2-256px-base",
    # PixArt-alpha
    "PixArt-alpha@PixArt-XL-2-1024-MS",
    "PixArt-alpha@PixArt-XL-2-512x512",
    # latent-consistency
    "latent-consistency@lcm-lora-sdxl",
    "latent-consistency@lcm-lora-sdv1-5",
    # black-forest-labs
    "black-forest-labs@FLUX.1-schnell",
    "black-forest-labs@FLUX.1-dev",
]

# URL format for downloading
URL_FORMAT = "https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench/resolve/main/{dataset}/{model}/images_with_metadata.tar.gz"


# Helper functions
def get_download_url(dataset, model):
    quoted_model = urllib.parse.quote(model)
    return URL_FORMAT.format(dataset=dataset, model=quoted_model)


def streaming_download(url, dest_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def extract_tar_gz_files(tar_filepath):
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=os.path.dirname(tar_filepath))


def extract_zip_file(zip_filepath, dest_dir):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)


if __name__ == "__main__":
    # Download synthetic images and metadata
    root_dir = "synthetic"
    for dataset in DATASET_LIST:
        print(f"Processing dataset: {dataset}")
        os.makedirs(os.path.join(root_dir, dataset), exist_ok=True)
        for model in tqdm(MODEL_LIST):
            url = get_download_url(dataset, model)
            dest_dir = os.path.join(root_dir, dataset)
            tar_filepath = os.path.join(dest_dir, "images_with_metadata.tar.gz")
            streaming_download(url, tar_filepath)
            extract_tar_gz_files(tar_filepath)
            os.remove(tar_filepath)
    print("All synthetic downloads and extractions completed.")

    # Download partial test real images
    # Note: Please follow the original license to use these images.
    real_images_url = "https://huggingface.co/datasets/ruojiruoli/Co-Spy-Misc/resolve/main/real_image_examples.zip"
    dest_zip_path = os.path.basename(real_images_url)
    streaming_download(real_images_url, dest_zip_path)
    extract_zip_file(dest_zip_path, "./")
    os.remove(dest_zip_path)
    print("Partial real image examples download and extraction completed.")
