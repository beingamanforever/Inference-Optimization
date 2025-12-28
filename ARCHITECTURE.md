# Project Structure & Flow

## Architecture

```
Inference-Optimization/
│
├── setup.sh              # One-step setup (venv + deps + models)
├── test_models.py        # Quick validation test
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker image for CUDA
├── docker-compose.yml    # Docker orchestration
│
├── src/                  # Core library
│   ├── benchmarking/     # Benchmark class, metrics, reporters
│   ├── models/          # Model wrappers (ResNet50, GPT-2, DistilBERT)
│   ├── profiling/       # Profilers (PyTorch, roofline, memory, layers)
│   └── utils/           # Logging, timing, data loading
│
├── scripts/              # Experiment runners
│   ├── Python (single operations)
│   │   ├── benchmark_model.py
│   │   ├── profile_model.py
│   │   ├── plot_results.py
│   │   ├── generate_report.py
│   │   └── demo_inference.py
│   │
│   └── Shell (batch operations)
│       ├── benchmark_all.sh
│       ├── profile_all.sh
│       ├── report.sh
│       ├── run_full_suite.sh
│       └── help.sh
│
└── week1/               # Experiment outputs
    ├── benchmarks/      # CSV/JSON metrics
    ├── experiments/     # Profiling traces
    └── plots/          # Visualizations
```

## Workflow

### 1. Setup (One Time)
```bash
bash setup.sh            # Creates venv, installs deps, downloads models
source venv/bin/activate
python test_models.py    # Validate installation
```

### 2. Run Experiments

**Option A: Complete Suite**
```bash
bash scripts/run_full_suite.sh cpu    # Benchmark + profile + report
```

**Option B: Individual Steps**
```bash
# Benchmark
bash scripts/benchmark_all.sh cpu
# or
python scripts/benchmark_model.py --model resnet50 --device cpu --batch-sizes 1 4 8

# Profile
bash scripts/profile_all.sh cpu
# or
python scripts/profile_model.py --model gpt2 --device mps --all

# Report
bash scripts/report.sh
# or
python scripts/plot_results.py
python scripts/generate_report.py
```

### 3. Analyze Results

```bash
# View terminal report
python scripts/generate_report.py

# View plots
open week1/plots/performance_comparison.png

# View profiling traces
# Open chrome://tracing in Chrome
# Load week1/experiments/{model}/trace.json
```

## Code Flow

### Benchmarking Flow
```
benchmark_model.py
  └─> Benchmark(model_wrapper)
       └─> run_batch_sweep(batch_sizes)
            └─> _measure_batch()
                 ├─> Warmup iterations
                 ├─> Timed inference
                 ├─> Collect metrics (latency, throughput, memory)
                 └─> Return InferenceMetrics

  └─> BenchmarkReporter
       ├─> print_terminal_summary()
       └─> save_reports() -> CSV + JSON
```

### Profiling Flow
```
profile_model.py
  └─> InferenceProfiler(model_wrapper)
       └─> run_profile()
            ├─> PyTorch profiler
            ├─> Export trace.json
            └─> Calculate overhead

  └─> LayerTimer (if --all)
  └─> MemoryTracker (if --all)
  └─> PythonOverheadAnalyzer (if --all)
```

### Model Wrapper Pattern
```
BaseModel (abstract)
  ├─> prepare_input(batch_size) -> tensor/dict
  ├─> forward(input) -> output
  ├─> get_model_info() -> dict
  └─> to(device) -> self

Implementations:
  ├─> ResNet50ModelWrapper
  ├─> GPT2ModelWrapper
  └─> DistilBERTModelWrapper
```

## Key Components

### 1. Benchmarking
- **Benchmark**: Orchestrates timing and metric collection
- **InferenceMetrics**: Data class with latency percentiles, throughput, memory
- **BenchmarkReporter**: Terminal display and file export

### 2. Profiling
- **InferenceProfiler**: PyTorch profiler with Chrome trace export
- **LayerTimer**: Per-layer timing using hooks
- **MemoryTracker**: Device memory monitoring
- **PythonOverheadAnalyzer**: cProfile integration
- **RooflineCalculator**: Operational intensity analysis

### 3. Models
- **BaseModel**: Abstract interface
- All wrappers implement: prepare_input(), forward(), get_model_info()
- Device handling: CPU, MPS, CUDA

### 4. Utilities
- **timing.py**: Timing helpers
- **logger.py**: Logging setup
- **data_loader.py**: Dummy data generation

## Device Support

| Device | Availability | Performance | Use Case |
|--------|-------------|-------------|----------|
| CPU | Always | Baseline | Development, testing |
| MPS | Apple Silicon | Medium | M1/M2/M3 Macs |
| CUDA | NVIDIA GPU | Best | Production, training |

## Docker Usage

For CUDA support on systems with NVIDIA GPUs:

```bash
# Build and run
docker-compose up -d
docker-compose exec optimizer bash

# Inside container
python scripts/benchmark_model.py --model resnet50 --device cuda --batch-sizes 1 8 16
```

## Determinism

All experiments use fixed seeds (42) for reproducibility:
- torch.manual_seed(42)
- np.random.seed(42)
- torch.backends.cudnn.deterministic = True

Note: Some variance is normal due to OS scheduling and hardware variability.

## File Naming Convention

**Python scripts**: `{action}_{target}.py`
- benchmark_model.py, profile_model.py, generate_report.py

**Shell scripts**: `{action}_all.sh` or `{action}.sh`
- benchmark_all.sh, profile_all.sh, report.sh

**Output files**: `{model}_{device}_results.{ext}`
- resnet50_cpu_results.csv, gpt2_mps_results.json

## Common Issues

**Import errors**: Run from project root, ensure PYTHONPATH includes `.`
**Device not available**: Check hardware and drivers
**Out of memory**: Reduce batch sizes
**No results found**: Run benchmarks before plotting/reporting

## Two READMEs Explained

- **README.md** (root): Overview, quick start, high-level architecture
- **scripts/README.md**: Detailed documentation for each script, flags, usage examples

This separation keeps the main README concise while providing detailed docs where needed.
