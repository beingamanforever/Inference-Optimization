"""Abstract base class for all models.

This module provides a base interface that all model wrappers must implement.
It ensures consistency across different model types (CNN, Transformer, etc.)
and makes it easy to add new models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple
import torch
import torch.nn as nn


class BaseModel(ABC):
    """Abstract base class for model wrappers.
    
    All model implementations should inherit from this class and implement
    the abstract methods. This ensures a consistent interface for benchmarking
    and profiling across different model architectures.
    
    Attributes:
        device (str): Device to run model on ('cuda' or 'cpu')
        model (nn.Module): The actual PyTorch model
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize the model wrapper.
        
        Args:
            device: Target device ('cpu', 'cuda', or 'mps')
        """
        # Validate and normalize device
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = "cpu"
        
        self.device = device
        self.backend = "PyTorch"  # Required by benchmarking suite
        self.model: nn.Module = None
        
        # Setup model (implemented by subclass)
        self._setup_model()
        
        if self.model is None:
            raise RuntimeError(f"{self.__class__.__name__}._setup_model() did not create a model")
    
    @abstractmethod
    def _setup_model(self):
        """Load and prepare the model.
        
        This method should:
        1. Load/create the model
        2. Set model to evaluation mode
        3. Move model to the specified device
        
        Must set self.model to a valid nn.Module instance.
        
        Example:
            def _setup_model(self):
                self.model = torchvision.models.resnet50(pretrained=True)
                self.model.eval()
                self.model.to(self.device)
        """
        pass
    
    @abstractmethod
    def prepare_input(self, batch_size: int = 1) -> Any:
        """Generate sample input for the model.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Model input (tensor or dict of tensors) on the correct device
            
        Example:
            For vision models:
                return torch.randn(batch_size, 3, 224, 224, device=self.device)
            
            For NLP models:
                return {
                    "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                    "attention_mask": torch.ones(batch_size, seq_len)
                }
        """
        pass
    
    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Run inference on the model.
        
        Args:
            inputs: Model inputs (from prepare_input)
            
        Returns:
            Model outputs
            
        Note:
            Should use torch.no_grad() context for inference.
            
        Example:
            def forward(self, inputs):
                with torch.no_grad():
                    return self.model(inputs)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and statistics.
        
        Returns:
            Dictionary containing model information:
                - name: Model class name
                - num_parameters: Total number of parameters
                - device: Device model is on
                - dtype: Data type of parameters
                - trainable_parameters: Number of trainable parameters
                - memory_footprint_mb: Approximate memory usage
        """
        if self.model is None:
            return {"error": "Model not initialized"}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        # Estimate memory footprint (parameters only)
        param_memory_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        memory_mb = param_memory_bytes / (1024 ** 2)
        
        # Get dtype
        try:
            dtype = next(self.model.parameters()).dtype
        except StopIteration:
            dtype = "unknown (no parameters)"
        
        return {
            "name": self.__class__.__name__,
            "num_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device,
            "dtype": str(dtype),
            "memory_footprint_mb": f"{memory_mb:.2f}"
        }
    
    def get_flops(self, batch_size: int = 1) -> Union[int, str]:
        """Estimate FLOPs for a forward pass (optional).
        
        Args:
            batch_size: Batch size for FLOP calculation
            
        Returns:
            Estimated FLOPs or "Not implemented"
            
        Note:
            Subclasses can override this for accurate FLOP counting.
        """
        return "Not implemented"
    
    def warmup(self, num_iterations: int = 10, batch_size: int = 1) -> None:
        """Warmup the model with dummy forward passes.
        
        This is important for accurate benchmarking as the first few
        iterations are typically slower due to CUDA initialization,
        memory allocation, etc.
        
        Args:
            num_iterations: Number of warmup iterations
            batch_size: Batch size for warmup
        """
        print(f"Warming up {self.__class__.__name__} for {num_iterations} iterations...")
        
        inputs = self.prepare_input(batch_size=batch_size)
        
        for i in range(num_iterations):
            _ = self.forward(inputs)
        
        # Synchronize based on device
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
        
        print("Warmup complete")
    
    def to(self, device: str) -> 'BaseModel':
        """Move model to a different device.
        
        Args:
            device: Target device ('cpu', 'cuda', or 'mps')
            
        Returns:
            Self for chaining
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available. Staying on current device.")
            return self
        elif device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Staying on current device.")
            return self
        
        self.device = device
        self.model.to(device)
        
        return self
    
    def eval(self) -> 'BaseModel':
        """Set model to evaluation mode.
        
        Returns:
            Self for chaining
        """
        self.model.eval()
        return self
    
    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return (
            f"{info['name']}(\n"
            f"  parameters={info['num_parameters']:,}\n"
            f"  device={info['device']}\n"
            f"  dtype={info['dtype']}\n"
            f"  memory={info['memory_footprint_mb']} MB\n"
            f")"
        )
    
    def print_summary(self) -> None:
        """Print a detailed summary of the model."""
        info = self.get_model_info()
        
        print("\n" + "="*60)
        print(f"Model Summary: {info['name']}")
        print("="*60)
        print(f"Total Parameters:      {info['num_parameters']:>15,}")
        print(f"Trainable Parameters:  {info['trainable_parameters']:>15,}")
        print(f"Device:                {info['device']:>15}")
        print(f"Data Type:             {info['dtype']:>15}")
        print(f"Memory Footprint:      {info['memory_footprint_mb']:>15} MB")
        print(f"Backend:               {self.backend:>15}")
        print("="*60 + "\n")
