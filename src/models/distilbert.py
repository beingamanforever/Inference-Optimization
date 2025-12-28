"""DistilBERT model wrapper for inference benchmarking."""
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from typing import Any, Dict, List
from .base import BaseModel


class DistilBERTModelWrapper(BaseModel):
    """DistilBERT wrapper for inference benchmarking.
    
    DistilBERT Architecture:
    - Type: Encoder-only Transformer (Distilled from BERT)
    - Parameters: ~66M
    - Efficiency: 40% fewer parameters, 60% faster than BERT-base.
    """
    
    def __init__(self, device: str = "cpu", seq_length: int = 128):
        """Initialize DistilBERT model."""
        self.seq_length = seq_length
        super().__init__(device)
    
    def _setup_model(self):
        """Load DistilBERT."""
        print(f"Loading DistilBERT on {self.device}...")
        
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        self.model.eval()
        self.model.to(self.device)
        
        print(f"DistilBERT loaded successfully on {self.device}")
    
    def prepare_input(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Generate token IDs for BERT-style input."""
        input_ids = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=(batch_size, self.seq_length),
            device=self.device
        )
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run bidirectional forward pass."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        return outputs.last_hidden_state
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Mean pooling to get sentence embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Tensor of shape [batch_size, hidden_size] containing sentence embeddings
        """
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.forward(encoded)
        
        # Mean pooling logic
        mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.size()).float()
        return torch.sum(outputs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_type": "DistilBERT Base",
            "architecture": "Encoder-only Transformer",
            "num_layers": self.model.config.n_layers,
            "hidden_size": self.model.config.dim,
            "seq_length": self.seq_length
        })
        return info


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = DistilBERTModelWrapper(device=device)
    model.print_summary()
    
    # Test embedding
    emb = model.get_embeddings(["This is a test sentence."])
    print(f"Embedding shape: {emb.shape}")