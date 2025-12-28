"""ResNet-50 model wrapper for inference benchmarking.

ResNet-50 is a convolutional neural network that is 50 layers deep.
It uses residual connections to enable training of very deep networks.
This implementation uses the pretrained ImageNet model from torchvision.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Any, Dict, Union, List
from PIL import Image
import numpy as np
from .base import BaseModel


class ResNet50ModelWrapper(BaseModel):
    """ResNet-50 wrapper for inference benchmarking.
    
    ResNet-50 architecture:
    - Input: 224×224×3 RGB images
    - 50 layers with residual connections
    - ~25.6M parameters
    - Output: 1000 class predictions (ImageNet)
    
    Attributes:
        device (str): Device model runs on
        model (nn.Module): The ResNet-50 model
        transform (transforms.Compose): Image preprocessing pipeline
    """
    
    def __init__(self, device: str = "cpu", pretrained: bool = True):
        """Initialize ResNet-50 model.
        
        Args:
            device: Device to run model on ('cpu', 'cuda', or 'mps')
            pretrained: Whether to load pretrained ImageNet weights
        """
        self.pretrained = pretrained
        # Preprocessing needs to be set up before super().__init__ calls _setup_model
        self._setup_preprocessing()
        super().__init__(device)
    
    def _setup_preprocessing(self):
        """Setup ImageNet preprocessing pipeline."""
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])
    
    def _setup_model(self):
        """Load pretrained ResNet-50 model."""
        print(f"Loading ResNet-50 on {self.device}...")
        
        if self.pretrained:
            # Use weights parameter (modern torchvision API)
            try:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except AttributeError:
                # Fallback for older torchvision versions
                self.model = models.resnet50(pretrained=True)
        else:
            self.model = models.resnet50(pretrained=False)
        
        self.model.eval()
        self.model.to(self.device)
        
        print(f"ResNet-50 loaded successfully on {self.device}")

    def prepare_input(self, batch_size: int = 1) -> torch.Tensor:
        """Generate random image input (normalized).
        
        Returns:
            Tensor of shape [batch_size, 3, 224, 224] on correct device
        """
        # Generate random images in [0, 1] range
        images = torch.rand(batch_size, 3, 224, 224, device=self.device)
        
        # Apply ImageNet normalization manually to avoid CPU/GPU transfer
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        return (images - mean) / std
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference on ResNet-50."""
        with torch.no_grad():
            return self.model(inputs)

    def predict(self, inputs: torch.Tensor, top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Get top-K predictions with class names."""
        logits = self.forward(inputs)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Format results
        results = []
        for i in range(inputs.size(0)):
            image_results = []
            for j in range(top_k):
                class_idx = top_indices[i, j].item()
                image_results.append({
                    "class": class_idx,
                    "probability": top_probs[i, j].item()
                })
            results.append(image_results)
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get ResNet-50 specific model information."""
        # Check if model is initialized to avoid recursion in super call
        if self.model is None:
            return {"status": "loading"}
            
        info = super().get_model_info()
        info.update({
            "model_type": "ResNet-50",
            "architecture": "CNN",
            "input_size": "224x224x3",
            "layers": 50
        })
        return info

    def get_flops(self, batch_size: int = 1) -> int:
        """Estimate FLOPs for ResNet-50 (~4.1 GFLOPs per image)."""
        return int(4.1e9 * batch_size)

    def get_layer_info(self) -> Dict[str, int]:
        """Get parameter counts per layer type."""
        layer_params = {"conv": 0, "bn": 0, "fc": 0, "total": 0}
        for module in self.model.modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if isinstance(module, torch.nn.Conv2d):
                layer_params["conv"] += params
            elif isinstance(module, torch.nn.BatchNorm2d):
                layer_params["bn"] += params
            elif isinstance(module, torch.nn.Linear):
                layer_params["fc"] += params
            layer_params["total"] += params
        return layer_params


if __name__ == "__main__":
    # Quick sanity test
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = ResNet50ModelWrapper(device=device)
    wrapper.print_summary()
    
    test_input = wrapper.prepare_input(batch_size=2)
    output = wrapper.forward(test_input)
    print(f"Output shape: {output.shape}")
    print("ResNet-50 module test passed!")