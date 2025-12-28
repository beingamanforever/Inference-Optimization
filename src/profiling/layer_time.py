"""Layer-by-layer execution time profiling.

Uses PyTorch hooks to measure the execution time of individual layers,
helping identify which layers are performance bottlenecks.
"""
import torch
import time
from typing import Dict, Any, List, Tuple


class LayerTimer:
    """Timer for measuring layer-by-layer execution time.
    
    Uses forward hooks to capture timing for each module in the model.
    
    Attributes:
        model_wrapper: Model wrapper instance
        model: The actual PyTorch model
    """
    
    def __init__(self, model_wrapper):
        """Initialize layer timer.
        
        Args:
            model_wrapper: Model instance implementing BaseModel interface
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model

    def profile_layers(self, batch_size: int = 1, top_n: int = 10) -> Dict[str, float]:
        """Measure execution time for each major layer.
        
        Registers forward hooks on all top-level modules to capture
        their execution time. Returns timing breakdown.
        
        Args:
            batch_size: Batch size for profiling
            top_n: Number of top layers to display
            
        Returns:
            Dictionary mapping layer names to execution times (seconds)
        """
        layer_times: Dict[str, float] = {}
        start_times: Dict[Any, float] = {}
        module_to_name: Dict[Any, str] = {}
        hooks: List[Any] = []

        # Step 1: Define hooks to capture start and end of each layer
        def pre_hook(module, input):
            """Record layer start time."""
            start_times[module] = time.perf_counter()

        def post_hook(module, input, output):
            """Record layer end time and calculate duration."""
            if module in start_times:
                duration = time.perf_counter() - start_times[module]
                name = module_to_name.get(module, "unknown")
                layer_times[name] = layer_times.get(name, 0.0) + duration

        # Step 2: Register hooks for top-level modules
        for name, module in self.model.named_children():
            module_to_name[module] = name
            hooks.append(module.register_forward_pre_hook(pre_hook))
            hooks.append(module.register_forward_hook(post_hook))

        # Step 3: Prepare inputs and run inference
        inputs = self.model_wrapper.prepare_input(batch_size)
        print(f"Profiling layer-by-layer execution (batch_size={batch_size})...")
        
        try:
            with torch.no_grad():
                if isinstance(inputs, dict):
                    self.model(**inputs)
                else:
                    self.model(inputs)
        finally:
            # Step 4: Clean up hooks (always execute, even if inference fails)
            for h in hooks:
                h.remove()

        # Step 5: Display results
        print(f"\nTop {top_n} Layers by Execution Time:")
        print("=" * 60)
        sorted_layers = sorted(layer_times.items(), key=lambda x: x[1], reverse=True)
        
        total_time = sum(layer_times.values())
        for name, duration in sorted_layers[:top_n]:
            pct = (duration / total_time * 100) if total_time > 0 else 0
            print(f"  {name:30s}: {duration*1000:>8.2f} ms ({pct:>5.1f}%)")
        
        print("=" * 60)
        print(f"Total layer time: {total_time*1000:.2f} ms\n")
        
        return layer_times