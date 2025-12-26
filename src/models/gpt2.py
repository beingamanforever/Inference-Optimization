"""GPT-2 model wrapper for inference benchmarking."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Any, Dict
from .base import BaseModel


class GPT2Model(BaseModel):
    """GPT-2 wrapper for inference benchmarking.
    
    This wrapper handles GPT-2 Small (124M parameters) for 
    text generation tasks with configurable sequence lengths.
    """
    
    def __init__(self, device: str = "cuda", seq_length: int = 128):
        """Initialize GPT-2 model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            seq_length: Input sequence length for benchmarking
        """
        self.seq_length = seq_length
        super().__init__(device)
    
    def _setup_model(self):
        """Load pretrained GPT-2 model and tokenizer."""
        print(f"Loading GPT-2 on {self.device}...")
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        self.model.to(self.device)
        
        # Load tokenizer (useful for real input later)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ GPT-2 loaded successfully")
        print(f"  Model size: {self.get_model_info()['num_parameters']:,} parameters")
    
    def prepare_input(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Generate random token IDs as input.
        
        Args:
            batch_size: Number of sequences in batch
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Generate random token IDs (vocab size is 50257 for GPT-2)
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
            texts = ["Hello, how are you doing today?"] * batch_size
        
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
        """Run inference on GPT-2.
        
        Args:
            inputs: Dictionary with input_ids and attention_mask
            
        Returns:
            Model outputs (logits)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        return outputs.logits
    
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Generate text from a prompt (for testing).
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text string
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata including GPT-2 specific info."""
        base_info = super().get_model_info()
        base_info.update({
            "model_type": "GPT-2 Small",
            "architecture": "Decoder-only Transformer",
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": self.model.config.n_positions,
            "num_layers": self.model.config.n_layer,
            "num_heads": self.model.config.n_head,
            "hidden_size": self.model.config.n_embd,
            "seq_length": self.seq_length
        })
        return base_info


# Test code
if __name__ == "__main__":
    # Test on CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing GPT-2Model on {device}...")
    
    # Initialize model
    model = GPT2Model(device=device, seq_length=128)
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    inputs = model.prepare_input(batch_size=2)
    print(f"  Input shape: {inputs['input_ids'].shape}")
    
    outputs = model.forward(inputs)
    print(f"  Output shape: {outputs.shape}")
    
    # Test text generation
    print("\nTesting text generation...")
    prompt = "Once upon a time"
    generated = model.generate(prompt, max_length=30)
    print(f"  Prompt: {prompt}")
    print(f"  Generated: {generated}")
    
    print("\n✓ All tests passed!")