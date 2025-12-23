import torch
from torchvision import transforms
from utils import data_augment
from .base import BaseDetector
from .semantic import SemanticDetector
from .artifact import ArtifactDetector


class CoSpyFusionDetector(BaseDetector):
    """CO-SPY Fusion Detector (calibrate and fuse semantic and artifact detectors)."""

    def __init__(self, semantic_weights_path, artifact_weights_path, num_classes=1):
        super(CoSpyFusionDetector, self).__init__()

        # Load the semantic detector
        self.sem = SemanticDetector()
        self.sem.load_weights(semantic_weights_path)

        # Load the artifact detector
        self.art = ArtifactDetector()
        self.art.load_weights(artifact_weights_path)

        # Freeze the two pre-trained models
        for param in self.sem.parameters():
            param.requires_grad = False
        for param in self.art.parameters():
            param.requires_grad = False

        # Classifier
        self.fc = torch.nn.Linear(2, num_classes)

        # Transformations inside the forward function
        # Including the normalization and resizing (only for the artifact detector)
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Resize(self.art.cropSize, antialias=False),
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Build transforms (no normalization, done in forward)
        self._build_transforms()

    def _build_transforms(self):
        """Build transforms without normalization (normalization done per-branch in forward)."""

        # Resolution
        self.loadSize = 384
        self.cropSize = 384

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        crop_func = transforms.RandomCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            flip_func,
            aug_func,
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

    def forward(self, x, return_feat=False):
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)
        pred_sem = self.sem(x_sem)
        pred_art = self.art(x_art)
        feat = torch.cat([pred_sem, pred_art], dim=1)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        # Only save the fc layer (sub-detectors are frozen)
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        # Load only the fc layer
        weights = torch.load(weights_path, map_location='cpu')
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]
