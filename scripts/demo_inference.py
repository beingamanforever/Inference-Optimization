"""Demo script for testing model inference with real examples.

Demonstrates:
- ResNet50: Image classification on sample data
- GPT-2: Text generation
- DistilBERT: Text embedding extraction
"""
import torch
import numpy as np
from src.models import ResNet50ModelWrapper, GPT2ModelWrapper, DistilBERTModelWrapper


def set_deterministic(seed: int = 42):
    """Set seeds for reproducible inference."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def demo_resnet():
    """Demonstrate ResNet50 image classification."""
    print("\n" + "=" * 70)
    print("DEMO 1: ResNet50 Image Classification")
    print("=" * 70)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = ResNet50ModelWrapper(device=device, pretrained=True)
    
    # Prepare sample input (batch of 2 images)
    print("\nGenerating sample images (224x224x3)...")
    inputs = model.prepare_input(batch_size=2)
    
    # Run inference
    print("Running inference...")
    predictions = model.predict(inputs, top_k=3)
    
    # Display results
    for i, preds in enumerate(predictions):
        print(f"\nImage {i+1} - Top 3 Predictions:")
        for j, pred in enumerate(preds, 1):
            print(f"  {j}. Class {pred['class']:>4}: {pred['probability']:.4f}")
    
    print("\nNote: Using random images, so predictions are meaningless.")
    print("Replace with real images using PIL.Image.open() for actual use.")


def demo_gpt2():
    """Demonstrate GPT-2 text generation."""
    print("\n" + "=" * 70)
    print("DEMO 2: GPT-2 Text Generation")
    print("=" * 70)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = GPT2ModelWrapper(device=device, seq_length=128)
    
    # Generate text from prompt
    prompt = "Machine learning inference optimization is important because"
    print(f"\nPrompt: '{prompt}'")
    print("Generating text...\n")
    
    generated_text = model.generate(prompt, max_length=60)
    print(f"Generated: {generated_text}")
    
    # Show benchmark-ready inference (random tokens)
    print("\nBenchmark Mode (random tokens):")
    inputs = model.prepare_input(batch_size=4)
    logits = model.forward(inputs)
    print(f"  Input shape:  {inputs['input_ids'].shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Batch size: 4, Sequence length: 128, Vocab size: {logits.shape[-1]}")


def demo_distilbert():
    """Demonstrate DistilBERT embedding extraction."""
    print("\n" + "=" * 70)
    print("DEMO 3: DistilBERT Sentence Embeddings")
    print("=" * 70)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = DistilBERTModelWrapper(device=device, seq_length=128)
    
    # Extract embeddings from sample texts
    texts = [
        "PyTorch is a deep learning framework.",
        "Machine learning models require optimization.",
        "Apple M-series chips support MPS acceleration."
    ]
    
    print("\nExtracting embeddings for sample texts...")
    embeddings = model.get_embeddings(texts)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"  Number of texts: {embeddings.shape[0]}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Show similarity between first two texts
    cos_sim = torch.nn.functional.cosine_similarity(
        embeddings[0:1], embeddings[1:2]
    ).item()
    print(f"\nCosine similarity between text 1 and text 2: {cos_sim:.4f}")
    
    # Benchmark mode
    print("\nBenchmark Mode (random tokens):")
    inputs = model.prepare_input(batch_size=8)
    outputs = model.forward(inputs)
    print(f"  Input shape:  {inputs['input_ids'].shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Batch size: 8, Sequence length: 128, Hidden size: {outputs.shape[-1]}")


def main():
    """Run all demos."""
    set_deterministic(seed=42)
    
    print("\n" + "#" * 70)
    print("#" + " " * 15 + "INFERENCE DEMONSTRATION SUITE" + " " * 15 + "#")
    print("#" * 70)
    
    # Run demos
    demo_resnet()
    demo_gpt2()
    demo_distilbert()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nAll models loaded and tested successfully!")
    print("\nNext steps:")
    print("  1. Run benchmarks: python scripts/benchmark_model.py --model resnet50 --device cpu")
    print("  2. Run profiling:  python scripts/profile_model.py --model gpt2 --device mps --all")
    print("  3. Generate plots:  python scripts/plot_results.py")
    print("  4. View report:    python scripts/generate_report.py")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()