"""Data loading utilities for inference benchmarking.

This module provides utilities for generating dummy data for benchmarking
different types of models (vision, NLP, etc.).
"""

import torch
from typing import Tuple, Optional, Iterator


class DummyDataLoader:
    """Simple data loader that generates random tensors.
    
    This is useful for benchmarking when you don't need real data
    and just want to measure inference performance.
    
    Example:
        loader = DummyDataLoader(
            data_shape=(3, 224, 224),
            num_samples=100,
            batch_size=8,
            device="cuda"
        )
        
        for batch in loader:
            outputs = model(batch)
    """
    
    def __init__(
        self,
        data_shape: Tuple[int, ...],
        num_samples: int = 100,
        batch_size: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32
    ):
        """Initialize dummy data loader.
        
        Args:
            data_shape: Shape of each sample (e.g., (3, 224, 224) for images)
            num_samples: Total number of samples
            batch_size: Batch size
            device: Device to create tensors on
            dtype: Data type of tensors
        """
        self.data_shape = data_shape
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        # Calculate number of batches
        self.num_batches = (num_samples + batch_size - 1) // batch_size
    
    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over batches."""
        for i in range(self.num_batches):
            # Calculate actual batch size (last batch might be smaller)
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_samples)
            actual_batch_size = end_idx - start_idx
            
            # Generate random batch
            batch_shape = (actual_batch_size,) + self.data_shape
            batch = torch.randn(batch_shape, device=self.device, dtype=self.dtype)
            
            yield batch
    
    def get_batch(self, batch_idx: int = 0) -> torch.Tensor:
        """Get a specific batch by index.
        
        Args:
            batch_idx: Index of batch to retrieve
            
        Returns:
            Random tensor batch
        """
        if batch_idx >= self.num_batches:
            raise IndexError(f"Batch index {batch_idx} out of range (0-{self.num_batches-1})")
        
        # Calculate batch size
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, self.num_samples)
        actual_batch_size = end_idx - start_idx
        
        # Generate batch
        batch_shape = (actual_batch_size,) + self.data_shape
        return torch.randn(batch_shape, device=self.device, dtype=self.dtype)


class ImageDataGenerator:
    """Generate dummy image data for vision models.
    
    Example:
        # Generate a batch of ImageNet-sized images
        images = ImageDataGenerator.generate_batch(
            batch_size=8,
            image_size=224,
            channels=3,
            device="cuda"
        )
    """
    
    @staticmethod
    def generate_batch(
        batch_size: int = 1,
        image_size: int = 224,
        channels: int = 3,
        device: str = "cuda",
        normalize: bool = True
    ) -> torch.Tensor:
        """Generate a batch of random images.
        
        Args:
            batch_size: Number of images
            image_size: Height and width of images
            channels: Number of channels (3 for RGB)
            device: Device to create tensors on
            normalize: Whether to normalize to [0, 1] range
            
        Returns:
            Tensor of shape (batch_size, channels, image_size, image_size)
        """
        images = torch.randn(
            batch_size, channels, image_size, image_size,
            device=device
        )
        
        if normalize:
            # Normalize to [0, 1] range
            images = (images - images.min()) / (images.max() - images.min())
        
        return images
    
    @staticmethod
    def generate_imagenet_batch(
        batch_size: int = 1,
        device: str = "cuda"
    ) -> torch.Tensor:
        """Generate ImageNet-sized batch (224x224 RGB).
        
        Args:
            batch_size: Number of images
            device: Device to create tensors on
            
        Returns:
            Tensor of shape (batch_size, 3, 224, 224)
        """
        return ImageDataGenerator.generate_batch(
            batch_size=batch_size,
            image_size=224,
            channels=3,
            device=device,
            normalize=True
        )


class TextDataGenerator:
    """Generate dummy text data for NLP models.
    
    Example:
        # Generate a batch of token sequences
        input_ids, attention_mask = TextDataGenerator.generate_batch(
            batch_size=8,
            seq_length=128,
            vocab_size=30522,
            device="cuda"
        )
    """
    
    @staticmethod
    def generate_batch(
        batch_size: int = 1,
        seq_length: int = 128,
        vocab_size: int = 30522,
        device: str = "cuda",
        pad_token_id: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of random token sequences.
        
        Args:
            batch_size: Number of sequences
            seq_length: Length of each sequence
            vocab_size: Size of vocabulary
            device: Device to create tensors on
            pad_token_id: ID for padding token
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Generate random token IDs
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_length),
            device=device
        )
        
        # Generate attention mask (all ones for no padding)
        attention_mask = torch.ones(
            batch_size, seq_length,
            dtype=torch.long,
            device=device
        )
        
        return input_ids, attention_mask
    
    @staticmethod
    def generate_batch_with_padding(
        batch_size: int = 1,
        seq_length: int = 128,
        vocab_size: int = 30522,
        device: str = "cuda",
        pad_token_id: int = 0,
        padding_ratio: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch with variable-length sequences (with padding).
        
        Args:
            batch_size: Number of sequences
            seq_length: Maximum length of sequences
            vocab_size: Size of vocabulary
            device: Device to create tensors on
            pad_token_id: ID for padding token
            padding_ratio: Approximate ratio of padding (0.0 to 1.0)
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        input_ids = torch.zeros(
            batch_size, seq_length,
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.zeros(
            batch_size, seq_length,
            dtype=torch.long,
            device=device
        )
        
        for i in range(batch_size):
            # Random sequence length
            actual_length = int(seq_length * (1.0 - padding_ratio * torch.rand(1).item()))
            actual_length = max(1, actual_length)  # At least 1 token
            
            # Generate tokens
            input_ids[i, :actual_length] = torch.randint(
                low=1,  # Avoid pad token
                high=vocab_size,
                size=(actual_length,),
                device=device
            )
            
            # Set attention mask
            attention_mask[i, :actual_length] = 1
            
            # Pad the rest (already zeros)
            input_ids[i, actual_length:] = pad_token_id
        
        return input_ids, attention_mask
    
    @staticmethod
    def generate_gpt2_batch(
        batch_size: int = 1,
        seq_length: int = 128,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate GPT-2 compatible batch.
        
        Args:
            batch_size: Number of sequences
            seq_length: Length of sequences
            device: Device to create tensors on
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        return TextDataGenerator.generate_batch(
            batch_size=batch_size,
            seq_length=seq_length,
            vocab_size=50257,  # GPT-2 vocab size
            device=device
        )
    
    @staticmethod
    def generate_bert_batch(
        batch_size: int = 1,
        seq_length: int = 128,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate BERT compatible batch.
        
        Args:
            batch_size: Number of sequences
            seq_length: Length of sequences
            device: Device to create tensors on
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        return TextDataGenerator.generate_batch(
            batch_size=batch_size,
            seq_length=seq_length,
            vocab_size=30522,  # BERT vocab size
            device=device
        )


# Test code
if __name__ == "__main__":
    print("Testing data loading utilities...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Test DummyDataLoader
    print("=== Testing DummyDataLoader ===")
    loader = DummyDataLoader(
        data_shape=(3, 224, 224),
        num_samples=25,
        batch_size=8,
        device=device
    )
    
    print(f"Number of batches: {len(loader)}")
    
    for i, batch in enumerate(loader):
        print(f"Batch {i}: shape={batch.shape}, device={batch.device}")
    
    # Test ImageDataGenerator
    print("\n=== Testing ImageDataGenerator ===")
    images = ImageDataGenerator.generate_batch(
        batch_size=4,
        image_size=224,
        channels=3,
        device=device
    )
    print(f"Generated images: {images.shape}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
    
    imagenet_batch = ImageDataGenerator.generate_imagenet_batch(
        batch_size=8,
        device=device
    )
    print(f"ImageNet batch: {imagenet_batch.shape}")
    
    # Test TextDataGenerator
    print("\n=== Testing TextDataGenerator ===")
    input_ids, attention_mask = TextDataGenerator.generate_batch(
        batch_size=4,
        seq_length=128,
        vocab_size=30522,
        device=device
    )
    print(f"Input IDs: {input_ids.shape}")
    print(f"Attention mask: {attention_mask.shape}")
    print(f"Sample tokens: {input_ids[0, :10]}")
    
    # Test with padding
    input_ids_pad, attention_mask_pad = TextDataGenerator.generate_batch_with_padding(
        batch_size=4,
        seq_length=128,
        vocab_size=30522,
        device=device,
        padding_ratio=0.3
    )
    print(f"\nWith padding:")
    print(f"Input IDs: {input_ids_pad.shape}")
    for i in range(4):
        actual_length = attention_mask_pad[i].sum().item()
        print(f"  Sequence {i}: {actual_length} tokens (padding: {128 - actual_length})")
    
    # Test model-specific generators
    print("\n=== Testing Model-Specific Generators ===")
    gpt2_ids, gpt2_mask = TextDataGenerator.generate_gpt2_batch(
        batch_size=2,
        seq_length=128,
        device=device
    )
    print(f"GPT-2 batch: {gpt2_ids.shape}")
    
    bert_ids, bert_mask = TextDataGenerator.generate_bert_batch(
        batch_size=2,
        seq_length=128,
        device=device
    )
    print(f"BERT batch: {bert_ids.shape}")
    
    print("\n✓ All data loading tests passed!")
