"""ResNet-50 model wrapper for inference benchmarking.

ResNet-50 is a convolutional neural network that is 50 layers deep.
It uses residual connections to enable training of very deep networks.
This implementation uses the pretrained ImageNet model from torchvision.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Any, Dict, Union, List
from PIL import Image
import numpy as np
from .base import BaseModel


class ResNet50Model(BaseModel):
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
    
    def __init__(self, device: str = "cuda", pretrained: bool = True):
        """Initialize ResNet-50 model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            pretrained: Whether to load pretrained ImageNet weights
        """
        self.pretrained = pretrained
        self._setup_preprocessing()
        super().__init__(device)
    
    def _setup_preprocessing(self):
        """Setup ImageNet preprocessing pipeline."""
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Full preprocessing pipeline
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
            print("  Using pretrained ImageNet weights")
            # Use weights parameter (new torchvision API)
            try:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except AttributeError:
                # Fallback for older torchvision versions
                self.model = models.resnet50(pretrained=True)
        else:
            print("  Using random initialization")
            self.model = models.resnet50(pretrained=False)
        
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✓ ResNet-50 loaded successfully")
        print(f"  Model size: {self.get_model_info()['num_parameters']:,} parameters")
    
    def prepare_input(self, batch_size: int = 1) -> torch.Tensor:
        """Generate random image input (normalized).
        
        Args:
            batch_size: Number of images in batch
            
        Returns:
            Tensor of shape [batch_size, 3, 224, 224] on correct device
        """
        # Generate random images in [0, 1] range
        images = torch.rand(batch_size, 3, 224, 224, device=self.device)
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        images = (images - mean) / std
        
        return images
    
    def prepare_real_input(
        self, 
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        batch_size: int = None
    ) -> torch.Tensor:
        """Prepare real image input with proper preprocessing.
        
        Args:
            images: PIL Image(s) or tensor
            batch_size: If provided, create batch by repeating image
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Handle single PIL image
        if isinstance(images, Image.Image):
            tensor = self.transform(images).unsqueeze(0)
            
            # Repeat to create batch if needed
            if batch_size and batch_size > 1:
                tensor = tensor.repeat(batch_size, 1, 1, 1)
        
        # Handle list of PIL images
        elif isinstance(images, list) and isinstance(images[0], Image.Image):
            tensors = [self.transform(img).unsqueeze(0) for img in images]
            tensor = torch.cat(tensors, dim=0)
        
        # Handle tensor input
        elif isinstance(images, torch.Tensor):
            # Assume already preprocessed if 4D
            if images.dim() == 4:
                tensor = images
            # If 3D, add batch dimension
            elif images.dim() == 3:
                tensor = images.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected tensor shape: {images.shape}")
        
        else:
            raise TypeError(f"Unsupported image type: {type(images)}")
        
        # Move to device
        return tensor.to(self.device)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference on ResNet-50.
        
        Args:
            inputs: Image tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            Logits tensor of shape [batch_size, 1000]
        """
        with torch.no_grad():
            return self.model(inputs)
    
    def predict(
        self, 
        inputs: torch.Tensor, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top-K predictions with class names.
        
        Args:
            inputs: Image tensor
            top_k: Number of top predictions to return
            
        Returns:
            List of dicts with 'class', 'class_name', 'probability'
        """
        # Forward pass
        logits = self.forward(inputs)
        probabilities = torch.softmax(logits, dim=1)
        
        # Get top-K predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Load ImageNet class names
        class_names = self._load_imagenet_classes()
        
        # Format results
        results = []
        for i in range(inputs.size(0)):
            image_results = []
            for j in range(top_k):
                class_idx = top_indices[i, j].item()
                prob = top_probs[i, j].item()
                
                image_results.append({
                    "class": class_idx,
                    "class_name": class_names.get(class_idx, f"class_{class_idx}"),
                    "probability": prob
                })
            
            results.append(image_results)
        
        return results
    
    def _load_imagenet_classes(self) -> Dict[int, str]:
        """Load ImageNet class names.
        
        Returns:
            Dictionary mapping class index to class name
        """
        # Simplified class names (first 10 for demo)
        # In production, you'd load the full 1000 classes from a file
        class_names = {
            0: "tench",
            1: "goldfish",
            2: "great white shark",
            3: "tiger shark",
            4: "hammerhead",
            5: "electric ray",
            6: "stingray",
            7: "cock",
            8: "hen",
            9: "ostrich",
            # Add remaining 990 classes...
        }
        
        # Return all indices with default names
        return {i: class_names.get(i, f"class_{i}") for i in range(1000)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ResNet-50 specific model information."""
        base_info = super().get_model_info()
        base_info.update({
            "model_type": "ResNet-50",
            "architecture": "Convolutional Neural Network",
            "input_size": "224×224×3",
            "output_size": "1000 (ImageNet classes)",
            "pretrained": self.pretrained,
            "layers": 50,
            "residual_blocks": "3+4+6+3 blocks"
        })
        return base_info
    
    def get_layer_info(self) -> Dict[str, int]:
        """Get information about ResNet layers.
        
        Returns:
            Dictionary with parameter counts per layer type
        """
        layer_params = {
            "conv_layers": 0,
            "batch_norm_layers": 0,
            "linear_layers": 0,
            "total": 0
        }
        
        for name, module in self.model.named_modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            
            if isinstance(module, torch.nn.Conv2d):
                layer_params["conv_layers"] += params
            elif isinstance(module, torch.nn.BatchNorm2d):
                layer_params["batch_norm_layers"] += params
            elif isinstance(module, torch.nn.Linear):
                layer_params["linear_layers"] += params
            
            layer_params["total"] += params
        
        return layer_params
    
    def get_flops(self, batch_size: int = 1) -> int:
        """Estimate FLOPs for ResNet-50.
        
        Args:
            batch_size: Batch size for FLOP calculation
            
        Returns:
            Estimated FLOPs (approximate)
        """
        # Approximate FLOPs for ResNet-50: ~4.1 GFLOPs per image
        flops_per_image = 4.1e9
        return int(flops_per_image * batch_size)


# Test code
if __name__ == "__main__":
    print("Testing ResNet50Model implementation...")
    
    # Test on CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Initialize model
    print("\n1. Creating ResNet-50...")
    model = ResNet50Model(device=device, pretrained=True)
    
    # Print detailed summary
    print("\n2. Model Summary:")
    model.print_summary()
    
    # Print layer info
    print("3. Layer Information:")
    layer_info = model.get_layer_info()
    for layer_type, params in layer_info.items():
        print(f"   {layer_type}: {params:,} parameters")
    
    # Test with random input
    print("\n4. Testing with random input...")
    batch_size = 4
    inputs = model.prepare_input(batch_size=batch_size)
    print(f"   Input shape: {inputs.shape}")
    print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    
    outputs = model.forward(inputs)
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    
    # Test predictions
    print("\n5. Testing predictions...")
    predictions = model.predict(inputs, top_k=3)
    for i, pred in enumerate(predictions[:2]):  # Show first 2 images
        print(f"\n   Image {i+1} - Top 3 predictions:")
        for p in pred:
            print(f"     {p['class_name']:20s}: {p['probability']:.4f}")
    
    # Test with real image (if PIL available)
    print("\n6. Testing with synthetic PIL image...")
    try:
        from PIL import Image
        
        # Create synthetic RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_array)
        
        real_inputs = model.prepare_real_input(pil_img, batch_size=2)
        print(f"   Real input shape: {real_inputs.shape}")
        
        real_outputs = model.forward(real_inputs)
        print(f"   Real output shape: {real_outputs.shape}")
    except ImportError:
        print("   PIL not available, skipping real image test")
    
    # Test FLOPs calculation
    print("\n7. FLOPs Calculation:")
    flops = model.get_flops(batch_size=1)
    print(f"   FLOPs per image: {flops / 1e9:.2f} GFLOPs")
    
    # Test warmup
    print("\n8. Testing warmup...")
    model.warmup(num_iterations=5, batch_size=2)
    
    # Performance info
    print("\n9. Performance Characteristics:")
    print(f"   Compute-bound: Yes (Conv2d dominates)")
    print(f"   Memory-bound: Low (efficient residual design)")
    print(f"   Best for: Image classification, feature extraction")
    
    print("\n✓ All ResNet-50 tests passed!")
    print("\nReady for Week 1 benchmarking! 🚀")