import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import get_list, png_to_jpeg
from .mscoco import MSCOCO2017
from .flickr import Flickr30k


class TrainDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None, add_jpeg=True):
        assert split in ["train", "val"]

        # Load the dataset for training
        real_list = get_list(os.path.join(data_path, "mscoco2017", f"{split}2017"))
        fake_list = get_list(os.path.join(data_path, "stable-diffusion-v1-4", f"{split}2017"))

        # Setting the labels for the dataset
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        # Construct the entire dataset
        self.total_list = real_list + fake_list
        np.random.shuffle(self.total_list)

        # JPEG compression
        self.add_jpeg = add_jpeg

        # Transformations
        self.transform = transform

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        image = Image.open(img_path).convert("RGB")

        # Add JPEG compression
        if self.add_jpeg:
            image = png_to_jpeg(image, quality=95)

        # Apply the transformation
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, dataset, model, root_path, transform=None, add_jpeg=True):
        # Load fake images
        fake_dir = os.path.join(root_path, dataset, model)
        fake_list = [i for i in os.listdir(fake_dir) if i.endswith(".png")]
        fake_list.sort()
        self.fake = [os.path.join(fake_dir, i) for i in fake_list]

        # Take the real images from the dataset
        if dataset == "mscoco":
            self.real = MSCOCO2017()
        elif dataset == "flickr":
            self.real = Flickr30k()
        else:
            raise ValueError(f"Invalidr real dataset name: {dataset}")
        
        # Ensure the number of real and fake images are the same
        self.image_idx = list(range(len(self.fake) * 2))
        # First half is real, second half is fake
        self.labels = [0] * len(self.fake) + [1] * len(self.fake)

        # JPEG compression
        self.add_jpeg = add_jpeg

        # Transformations
        self.transform = transform

    def __len__(self):
        return len(self.image_idx)
    
    def __getitem__(self, idx):
        if idx < len(self.fake):
            image, _ = self.real[idx]
        else:
            image = Image.open(self.fake[idx - len(self.fake)]).convert("RGB")

        # JPEG compression
        if self.add_jpeg:
            image = png_to_jpeg(image, quality=95)

        # Transformations
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label
