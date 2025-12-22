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
