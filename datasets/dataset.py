import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import get_list, png_to_jpeg
from .cospy_real import MSCOCO2017, Flickr30k, OtherReal


class TrainDataset(Dataset):
    def __init__(self, train_dataset, split="train", add_jpeg=False, transform=None):
        assert split in ["train", "val"]
        # Root directory of the training datasets
        root_dir = f"data/train/{train_dataset}"

        # Load the dataset for training
        if train_dataset == "progan":
            real_list = get_list(os.path.join(root_dir, split), must_contain='0_real')
            fake_list = get_list(os.path.join(root_dir, split), must_contain='1_fake')
        elif train_dataset == "sd-v1.4":
            root_dir = "data/train/sd-v1.4"
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


class CoSpyBenchTestDataset(Dataset):
    def __init__(self, dataset, model, num_real=2000, add_jpeg=True, transform=None):
        # Root path of the Co-Spy-Bench dataset
        # root_path = "data/test/Co-Spy-Bench/synthetic"
        # TODO: Remove
        root_path = "/data4/user/cheng535/sony_intern/sy_custom_deepfake"

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
            self.real = OtherReal(dataset)
        
        # Ensure the number of real and fake images are the same
        self.num_real = min(num_real, len(self.real), len(self.fake))
        self.image_idx = list(range(self.num_real * 2))
        # First half is real, second half is fake
        self.labels = [0] * self.num_real + [1] * self.num_real

        # JPEG compression
        self.add_jpeg = add_jpeg

        # Transformations
        self.transform = transform

    def __len__(self):
        return len(self.image_idx)
    
    def __getitem__(self, idx):
        if idx < self.num_real:
            image, _ = self.real[idx]
        else:
            image = Image.open(self.fake[idx - self.num_real]).convert("RGB")

        # JPEG compression
        if self.add_jpeg:
            image = png_to_jpeg(image, quality=95)

        # Transformations
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label


class AIGCDetectTestDataset(Dataset):
    def __init__(self, dataset, model, transform=None):
        # Root path of the AIGCDetectionBenchMark dataset
        root_path = "data/test/AIGCDetectionBenchMark"

        # Load images
        image_dir = os.path.join(root_path, dataset, model)
        real_list = get_list(image_dir, must_contain='0_real')
        fake_list = get_list(image_dir, must_contain='1_fake')

        # Setting the labels for the dataset
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1
        
        # Construct the entire dataset
        self.total_list = real_list + fake_list
        np.random.shuffle(self.total_list)

        # Transformations
        self.transform = transform

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        image = Image.open(img_path).convert("RGB")

        # Transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, label
