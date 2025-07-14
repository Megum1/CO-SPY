import torch
import random
from torchvision import transforms
from utils import data_augment, weights2cpu
from .semantic_detector import SemanticDetector
from .artifact_detector import ArtifactDetector


# CO-SPY Detector
class CospyDetector(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(CospyDetector, self).__init__()

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
            transforms.Resize(self.art.cropSize, antialias=False),
            transforms.Normalize(self.art.mean, self.art.std)
        ])

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

        # Pre-processing
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

    def forward(self, x, dropout_rate=0.3):
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)

        # Forward pass
        sem_feat, sem_coeff = self.sem(x_sem, return_feat=True)
        art_feat, art_coeff = self.art(x_art, return_feat=True)

        # Dropout
        if self.train():
            # Random dropout
            if random.random() < dropout_rate:
                # Randomly select a feature to drop
                idx_drop = random.randint(0, 1)
                if idx_drop == 0:
                    sem_coeff = torch.zeros_like(sem_coeff)
                else:
                    art_coeff = torch.zeros_like(art_coeff)

        # Concatenate the features
        x = torch.cat([sem_coeff * sem_feat, art_coeff * art_feat], dim=1)
        x = self.fc(x)

        return x

    def save_weights(self, weights_path):
        save_params = {
            "sem_fc": weights2cpu(self.sem.fc.state_dict()),
            "art_fc": weights2cpu(self.art.fc.state_dict()),
            "art_encoder": weights2cpu(self.art.artifact_encoder.state_dict()),
            "classifier": weights2cpu(self.fc.state_dict()),
        }
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.sem.fc.load_state_dict(weights["sem_fc"])
        self.art.fc.load_state_dict(weights["art_fc"])
        self.art.artifact_encoder.load_state_dict(weights["art_encoder"])
        self.fc.load_state_dict(weights["classifier"])


# Define the label smoothing loss
class LabelSmoothingBCEWithLogits(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCEWithLogits, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        target = target.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        return loss
