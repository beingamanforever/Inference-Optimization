"""DistilBERT model wrapper for inference benchmarking."""
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from typing import Any, Dict
from .base import BaseModel


class DistilBERTModel(BaseModel):
    """DistilBERT wrapper for inference benchmarking.
    
    DistilBERT is a distilled version of BERT - 40% smaller,
    60% faster, while retaining 97% of BERT's performance.
    Uses encoder-only architecture (66M parameters).
    """
    
    def __init__(self, device: str = "cuda", seq_length: int = 128):
        """Initialize DistilBERT model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            seq_length: Input sequence length for benchmarking
        """
        self.seq_length = seq_length
        super().__init__(device)
    
    def _setup_model(self):
        """Load pretrained DistilBERT model and tokenizer."""
        print(f"Loading DistilBERT on {self.device}...")
        
        # Load model
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.eval()
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        print(f"✓ DistilBERT loaded successfully")
        print(f"  Model size: {self.get_model_info()['num_parameters']:,} parameters")
    
    def prepare_input(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Generate random token IDs as input.
        
        Args:
            batch_size: Number of sequences in batch
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Generate random token IDs (vocab size is 30522 for DistilBERT)
        input_ids = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=(batch_size, self.seq_length),
            device=self.device
        )
        
        # Create attention mask (all ones for random input)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def prepare_real_input(self, texts: list, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Prepare real text input (optional, for actual inference).
        
        Args:
            texts: List of text strings
            batch_size: Number of sequences in batch
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # If texts not provided, use dummy text
        if not texts:
            texts = [
                "This is a sample sentence for testing DistilBERT."
            ] * batch_size
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.seq_length,
            return_tensors="pt"
        )
        
        # Move to device
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device)
        }
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Any:
        """Run inference on DistilBERT.
        
        Args:
            inputs: Dictionary with input_ids and attention_mask
            
        Returns:
            Model outputs (last hidden states)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        return outputs.last_hidden_state
    
    def get_embeddings(self, texts: list) -> torch.Tensor:
        """Get sentence embeddings using mean pooling.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of sentence embeddings [batch_size, hidden_size]
        """
        # Prepare inputs
        inputs = self.prepare_real_input(texts, batch_size=len(texts))
        
        # Get outputs
        outputs = self.forward(inputs)
        
        # Mean pooling - take attention mask into account
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
        
        sum_embeddings = torch.sum(outputs * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        embeddings = sum_embeddings / sum_mask
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata including DistilBERT specific info."""
        base_info = super().get_model_info()
        base_info.update({
            "model_type": "DistilBERT Base",
            "architecture": "Encoder-only Transformer (distilled from BERT)",
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": self.model.config.max_position_embeddings,
            "num_layers": self.model.config.n_layers,
            "num_heads": self.model.config.n_heads,
            "hidden_size": self.model.config.dim,
            "seq_length": self.seq_length
        })
        return base_info


# Test code
if __name__ == "__main__":
    # Test on CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing DistilBERTModel on {device}...")
    
    # Initialize model
    model = DistilBERTModel(device=device, seq_length=128)
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass with random input
    print("\nTesting forward pass with random input...")
    inputs = model.prepare_input(batch_size=2)
    print(f"  Input shape: {inputs['input_ids'].shape}")
    
    outputs = model.forward(inputs)
    print(f"  Output shape: {outputs.shape}")
    
    # Test with real text
    print("\nTesting with real text...")
    texts = [
        "DistilBERT is a fast and efficient model.",
        "It is great for inference optimization tasks."
    ]
    
    real_inputs = model.prepare_real_input(texts)
    print(f"  Real input shape: {real_inputs['input_ids'].shape}")
    
    real_outputs = model.forward(real_inputs)
    print(f"  Real output shape: {real_outputs.shape}")
    
    # Test embeddings
    print("\nTesting sentence embeddings...")
    embeddings = model.get_embeddings(texts)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    print("\n✓ All tests passed!")