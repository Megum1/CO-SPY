import torch
import random
from torchvision import transforms
from utils import data_augment, weights2cpu
from .base import BaseDetector
from .semantic import SemanticDetector
from .artifact import ArtifactDetector


class End2EndDetector(BaseDetector):
    """CO-SPY End-to-End Detector."""

    def __init__(self, num_classes=1):
        super(End2EndDetector, self).__init__()

        # Load the semantic detector
        self.sem = SemanticDetector()
        self.sem_dim = self.sem.fc.in_features

        # Load the artifact detector
        self.art = ArtifactDetector()
        self.art_dim = self.art.fc.in_features

        # Classifier
        self.fc = torch.nn.Linear(self.sem_dim + self.art_dim, num_classes)

        # Transformations inside the forward function
        # Including the normalization and resizing (only for the artifact detector)
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Build transforms (no normalization, done in forward)
        self._build_transforms()

    def _build_transforms(self):
        """Build transforms without normalization (normalization done per-branch in forward)."""

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

    def forward(self, x, return_feat=False, dropout_rate=0.3):
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)

        # Forward pass
        sem_feat, sem_coeff = self.sem(x_sem, return_feat=True)
        art_feat, art_coeff = self.art(x_art, return_feat=True)

        # Dropout during training
        if self.training:
            if random.random() < dropout_rate:
                # Randomly select a feature to drop
                idx_drop = random.randint(0, 1)
                if idx_drop == 0:
                    sem_coeff = torch.zeros_like(sem_coeff)
                else:
                    art_coeff = torch.zeros_like(art_coeff)

        # Concatenate the features
        feat = torch.cat([sem_coeff * sem_feat, art_coeff * art_feat], dim=1)
        out = self.fc(feat)

        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        save_params = {
            "sem_fc": weights2cpu(self.sem.fc.state_dict()),
            "art_fc": weights2cpu(self.art.fc.state_dict()),
            "art_encoder": weights2cpu(self.art.artifact_encoder.state_dict()),
            "classifier": weights2cpu(self.fc.state_dict()),
        }
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        self.sem.fc.load_state_dict(weights["sem_fc"])
        self.art.fc.load_state_dict(weights["art_fc"])
        self.art.artifact_encoder.load_state_dict(weights["art_encoder"])
        self.fc.load_state_dict(weights["classifier"])
