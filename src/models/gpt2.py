"""GPT-2 model wrapper for inference benchmarking."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Any, Dict, Union
from .base import BaseModel


class GPT2ModelWrapper(BaseModel):
    """GPT-2 wrapper for inference benchmarking.
    
    GPT-2 Architecture:
    - Type: Decoder-only Transformer
    - Parameters: ~124M (Small)
    - Context: Causal attention (tokens only look at the past)
    """
    
    def __init__(self, device: str = "cpu", seq_length: int = 128):
        """Initialize GPT-2 model.
        
        Args:
            device: Device to run model on ('cpu', 'cuda', or 'mps')
            seq_length: Input sequence length for benchmarking
        """
        self.seq_length = seq_length
        super().__init__(device)
    
    def _setup_model(self):
        """Load pretrained GPT-2 model and tokenizer."""
        print(f"Loading GPT-2 on {self.device}...")
        
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Standard configuration for inference
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.model.to(self.device)
        
        print(f"GPT-2 loaded successfully on {self.device}")
    
    def prepare_input(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Generate random token IDs for benchmarking."""
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
        """Run autoregressive forward pass."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        return outputs.logits
    
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Helper for actual text generation testing."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_type": "GPT-2 Small",
            "architecture": "Decoder-only Transformer",
            "num_layers": self.model.config.n_layer,
            "hidden_size": self.model.config.n_embd,
            "seq_length": self.seq_length
        })
        return info


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GPT2ModelWrapper(device=device)
    model.print_summary()
    
    # Test generation
    print(f"Sample generation: {model.generate('AI benchmarking is', max_length=20)}")