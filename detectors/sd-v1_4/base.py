import torch
from abc import ABC, abstractmethod


class BaseDetector(torch.nn.Module, ABC):
    """Abstract base class for all detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()

    @abstractmethod
    def _build_transforms(self):
        """Build train and test transforms."""
        pass

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
