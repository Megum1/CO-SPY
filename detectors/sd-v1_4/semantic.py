import torch
import open_clip
from torchvision import transforms
from utils import data_augment
from .base import BaseDetector


class SemanticDetector(BaseDetector):
    def __init__(self, dim_clip=1152, num_classes=1):
        super(SemanticDetector, self).__init__()

        # Get the pre-trained CLIP (SigLIP)
        model_name = "ViT-SO400M-14-SigLIP-384"
        version = "webli"
        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=version)

        # Freeze the CLIP visual encoder
        self.clip.requires_grad_(False)

        # Classifier
        self.fc = torch.nn.Linear(dim_clip, num_classes)

        # Build transforms
        self._build_transforms()
    
    def _build_transforms(self):
        # Normalization
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # Resolution
        self.loadSize = 384
        self.cropSize = 384

        # Data augmentation
        self.blur_prob = 0.5
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(30, 101))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            rz_func,
            aug_func,
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def forward(self, x, return_feat=False):
        feat = self.clip.encode_image(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        # Only save the fc layer (CLIP is frozen)
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        # Only load the fc layer
        weights = torch.load(weights_path, map_location='cpu')
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]
