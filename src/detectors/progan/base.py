import torch
from abc import ABC, abstractmethod
from torchvision import transforms
from utils import data_augment


class BaseDetector(torch.nn.Module, ABC):
    """Abstract base class for all detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()

        # Default normalization (ImageNet)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Default resolution
        self.loadSize = 256
        self.cropSize = 224

        # Default data augmentation config
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.0
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(30, 101))

    @property
    def aug_config(self):
        return {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

    def _build_transforms(self):
        """Build train and test transforms based on current config."""
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

    @abstractmethod
    def forward(self, x, return_feat=False):
        """Forward pass of the detector."""
        pass

    def predict(self, inputs):
        """Prediction function."""
        inputs = inputs.to(next(self.parameters()).device)
        outputs = self.forward(inputs)
        prediction = outputs.sigmoid().flatten().tolist()
        return prediction

    def save_weights(self, weights_path):
        """Save model weights to a file."""
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        """Load model weights from a file."""
        weights = torch.load(weights_path, map_location='cpu')
        self.load_state_dict(weights, strict=False)
