import os
import json
import torch
import numpy as np
from PIL import Image
import datasets as ds


class MSCOCO2017(torch.utils.data.Dataset):
    def __init__(self, split='val', transform=None):
        # Split [train: 118287, val: 5000]
        self.dataset = ds.load_dataset(
            "shunk031/MSCOCO",
            year=2017,
            coco_task="captions"
            )[split]

        # Preprocess the images
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # PIL RGB image
        image = example['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
        # A list of valid captions
        caption_list = example['annotations']['caption']
        # Randomly select a caption
        caption = np.random.choice(caption_list)
        return image, caption


class Flickr30k(torch.utils.data.Dataset):
    def __init__(self, split='test', transform=None):
        # Split [test: 31014]
        self.dataset = ds.load_dataset("nlphuji/flickr30k")[split]

        # Preprocess the images
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # PIL RGB image
        image = example['image']
        if self.transform:
            image = self.transform(image)
        # A list of valid captions
        caption_list = example['caption']
        # Randomly select a caption
        caption = np.random.choice(caption_list)
        return image, caption


class OtherReal(torch.utils.data.Dataset):
    def __init__(self, dataset_name, transform=None):
        # Load the test samples from other real datasets (CC3M, SBU, TextCaps)
        # Note: there are only 2,000 examples for each dataset
        root_dir = "data/test/Co-Spy-Bench/real_image_examples"
        if dataset_name == "cc3m":
            self.image_dir = os.path.join(root_dir, "CC3M")
        elif dataset_name == "sbu":
            self.image_dir = os.path.join(root_dir, "SBU")
        elif dataset_name == "textcaps":
            self.image_dir = os.path.join(root_dir, "TextCaps")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported in OtherReal class.")

        self.image_list = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg'))]
        self.image_list.sort()

        # Preprocess the images
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption_filepath = img_path.replace('.jpg', '.json')
        with open(caption_filepath, 'r') as f:
            caption_data = json.load(f)
        caption = caption_data['caption']
        return image, caption
