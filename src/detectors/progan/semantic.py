import torch
from transformers import CLIPModel

from .base import BaseDetector


class SemanticDetector(BaseDetector):
    def __init__(self, dim_clip=768, num_classes=1):
        super(SemanticDetector, self).__init__()

        # Get the pre-trained CLIP
        model_name = "openai/clip-vit-large-patch14"
        self.clip = CLIPModel.from_pretrained(model_name)

        # Freeze the CLIP visual encoder
        self.clip.requires_grad_(False)

        # Classifier
        self.fc = torch.nn.Linear(dim_clip, num_classes)

        # CLIP normalization
        self.mean = [0.48145466, 0.45782750, 0.40821073]
        self.std  = [0.26862954, 0.26130258, 0.27577711]

        # Data augmentation (override defaults)
        self.blur_prob = 0.5
        self.jpg_prob  = 0.5

        # Build transforms
        self._build_transforms()

    def forward(self, x, return_feat=False):
        feat = self.clip.get_image_features(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        # Only save the fc layer (CLIP is frozen)
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        # Load only the fc layer
        weights = torch.load(weights_path, map_location='cpu')
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]
