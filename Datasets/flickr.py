import os
import json
import torch
import numpy as np
from PIL import Image
import datasets as ds


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
