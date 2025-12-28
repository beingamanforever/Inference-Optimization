"""Python-level profiling for overhead analysis.

Uses cProfile to identify Python-level bottlenecks that may not be
visible in PyTorch profiler (e.g., data preprocessing, Python loops).
"""
import cProfile
import pstats
import io
import torch
from typing import Optional


class PythonOverheadAnalyzer:
    """Analyzer for Python-level overhead using cProfile.
    
    Attributes:
        model: Model wrapper instance
    """
    
    def __init__(self, model_wrapper):
        """Initialize analyzer with model wrapper.
        
        Args:
            model_wrapper: Model instance implementing BaseModel interface
        """
        self.model = model_wrapper

    def profile_inference(self, iterations: int = 50, batch_size: int = 1, top_n: int = 20) -> None:
        """Profile Python-level execution overhead.
        
        Uses cProfile to identify Python function calls that consume time.
        Useful for finding data loading, preprocessing, or framework overhead.
        
        Args:
            iterations: Number of inference iterations to profile
            batch_size: Batch size for inference
            top_n: Number of top functions to display
        """
        inputs = self.model.prepare_input(batch_size)
        
        print(f"Analyzing Python overhead for {iterations} iterations...")
        
        # Start profiling
        pr = cProfile.Profile()
        pr.enable()

        # Profile inference loop
        for _ in range(iterations):
            self.model.forward(inputs)
            
            # Ensure GPU operations complete
            if "cuda" in self.model.device:
                torch.cuda.synchronize()
            elif "mps" in self.model.device:
                torch.mps.synchronize()

        pr.disable()
        
        # Format and display results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(top_n)
        
        print(f"\nTop {top_n} Python Functions (Cumulative Time):")
        print("=" * 80)
        print(s.getvalue())
        print("=" * 80)