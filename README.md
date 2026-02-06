<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="logo-light.svg">
    <img alt="DeepScan Logo" src="logo-dark.svg" width="200">
  </picture>

  <br>
  
  <h1>DeepScan: Diagnostic Framework for LLMs</h1>

  <!-- <p><b>DeepScan: Diagnostic Framework for LLMs</b></p> -->

  <p>
    <!-- <a href="https://github.com/your-org/DeepScan"><img alt="GitHub" -->
      <!-- src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white"></a> -->
    <a href="https://huggingface.co/your-org/DeepScan"><img alt="Hugging Face"
      src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img alt="arXiv"
      src="https://img.shields.io/badge/arXiv-b31b1b?style=flat-square&logo=arxiv&logoColor=white"></a>
    <a href="https://deepscan.readthedocs.io"><img alt="Documentation"
      src="https://img.shields.io/badge/Documentation-8CA1AF?style=flat-square&logo=readthedocs&logoColor=white"></a>
  </p>
</div>


A flexible and extensible framework for diagnosing Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Designed around the "Register → Configure → Execute → Summarize" workflow, this framework provides unified Runner, Evaluator, and Summarizer abstractions for quickly building or customizing diagnostic pipelines.

## ✨ Features

- **📦 Model Registry**: Register and manage model instances, supporting Qwen, Llama, Mistral, Gemma, GLM, InternLM, InternVL, and more
- **🚀 Unified Model Interface**: Consistent `generate`/`chat` abstraction across model families
- **📊 Dataset Registry**: Register and manage dataset instances, supporting multiple data formats
- **⚙️ Configuration Management**: Load and manage configurations from YAML/JSON files
- **🔍 Extensible Evaluators**: Built-in diagnostic evaluators:
  - **TELLME**: Quantify the degree of disentanglement between different concepts in representations using metrics
  - **X-Boundary**: Diagnose hidden representation spaces: geometric relationships between safe/harmful/boundary regions
  - **MI-Peaks**: Track information evolution in reasoning representations during generation based on mutual information
  - **SPIN**: Analyze potential conflicts between safety objectives such as fairness and privacy
- **📝 Customizable Summarizers**: Aggregate and format evaluation results for different benchmarks
- **🔌 Plugin Architecture**: Easy to extend with custom evaluators and summarizers
- **💻 CLI Support**: Run evaluations directly from command line without writing code

## 📥 Installation

**Minimal install** (core dependencies only):
```bash
pip install -e .
```

**Recommended install** (includes common dependencies for most use cases):
```bash
pip install -e ".[default]"
```

**For development:**
```bash
pip install -e ".[dev]"
```

**Additional optional dependencies:**

```bash
# 🤖 Model runner dependencies
pip install -e ".[qwen]"          # Qwen models
pip install -e ".[glm]"           # GLM models
pip install -e ".[ministral3]"    # Ministral 3 (multimodal) models

# 🔬 Evaluator dependencies
pip install -e ".[tellme]"        # TELLME evaluator + metrics stack
pip install -e ".[xboundary]"     # X-Boundary evaluator + visualization stack
pip install -e ".[mi_peaks]"      # MI-Peaks evaluator

# 🎁 Convenience extras
pip install -e ".[all]"           # All evaluator dependencies (tellme + xboundary + mi_peaks)

# 🌐 API server (for internal use - not included in open-source core)
pip install -e ".[api]"           # FastAPI + Uvicorn
```

## 🚀 Quick Start

### 🎯 End-to-end from a config (any evaluator)

**Python API:**
```python
from deepscan import run_from_config

# YAML/JSON or dict with model/dataset/evaluator sections
results = run_from_config("examples/config.tellme.yaml")

# With output directory and run ID
results = run_from_config(
    "examples/config.tellme.yaml",
    output_dir="results",
    run_id="my_experiment",
)
```

**CLI** (no Python code needed):
```bash
# ✅ Basic usage
python -m deepscan.run --config examples/config.tellme.yaml --output-dir runs

# 🏷️ With custom run ID (optional; default is run_<timestamp>)
python -m deepscan.run --config examples/config.tellme.yaml --output-dir runs --run-id experiment_001

# 🔍 Dry run (validate config without loading model/dataset)
python -m deepscan.run --config examples/config.tellme.yaml --dry-run

# 💾 Optional: also write a single consolidated JSON to a specific location
python -m deepscan.run --config examples/config.tellme.yaml --output results.json
```

### 1. Register Models and Datasets (global registries) 📝

#### 🔧 Registering Individual Models

```python
from deepscan.registry.model_registry import get_model_registry
from deepscan.registry.dataset_registry import get_dataset_registry

# IMPORTANT: use the global registries so `run_from_config()` can find your entries
model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

@model_registry.register_model("gpt2")
def create_gpt2():
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained("gpt2")

@dataset_registry.register_dataset("glue_sst2")
def create_sst2():
    from datasets import load_dataset
    return load_dataset("glue", "sst2", split="test")
```

#### Registering Model Families (e.g., Qwen) 🏗️



**Option 1: Register by generation (Recommended)** ⭐

```python
from deepscan.registry.model_registry import get_model_registry

registry = get_model_registry()

@registry.register_model(
    "qwen3",
    model_family="qwen",
    model_generation="qwen3",
)
def create_qwen3(model_name: str = "Qwen3-8B", device: str = "cuda", **kwargs):
    """Create Qwen3 model of specified name."""
    from transformers import AutoModelForCausalLM
    
    model_paths = {
        "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
        "Qwen3-1.5B": "Qwen/Qwen3-1.5B",
        "Qwen3-2B": "Qwen/Qwen3-2B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "Qwen3-14B": "Qwen/Qwen3-14B",
        "Qwen3-32B": "Qwen/Qwen3-32B",
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return AutoModelForCausalLM.from_pretrained(
        model_paths[model_name],
        device_map=device,
        **kwargs
    )

# Usage:
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

**Option 2: Register individual models with generation prefix** 🔑

```python
@registry.register_model(
    "qwen3/Qwen3-8B",
    model_family="qwen",
    model_generation="qwen3",
    model_name="Qwen3-8B",
)
def create_qwen3_8b(device: str = "cuda", **kwargs):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        device_map=device,
        **kwargs
    )

# Usage:
runner = registry.get_model("qwen3/Qwen3-8B", device="cuda")
```

**💡 Why organize by generation?** Different Qwen generations (qwen, qwen2, qwen3) may have different architectures, tokenizers, and configurations even at the same parameter count.

**Option 3: Use pre-registered models (Recommended)** ⭐

Qwen models are automatically registered when you import the framework:

```python
from deepscan.registry.model_registry import get_model_registry

# Models are already registered - just use them!
registry = get_model_registry()
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

Or simply import the framework and models are ready:

```python
import deepscan  # Qwen models are auto-registered
from deepscan.registry.model_registry import get_model_registry

registry = get_model_registry()
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

See `deepscan/models/` for model implementations and `examples/` for end-to-end evaluation pipelines.

#### Pre-registered resources (models + datasets) 🎁

The framework ships with ready-to-use registrations that are loaded automatically
when you `import deepscan`:

**🤖 Models** (see `deepscan/models/` for implementations):
- **Qwen**: qwen / qwen2 / qwen2.5 / qwen3 variants
- **Llama**: Llama 2/3 variants
- **Mistral**: Mistral and Ministral3 (multimodal)
- **Gemma**: Gemma and Gemma3 (multimodal)
- **GLM**: GLM-4 series
- **InternLM**: InternLM2/3 variants
- **InternVL**: InternVL3.5 (multimodal)

**📊 Datasets**:
- BeaverTails (HF dataset)
- `tellme/beaver_tails_filtered` (CSV loader)
- `xboundary/diagnostic` (X-Boundary diagnostic dataset)

```python
import deepscan
from deepscan.registry.model_registry import get_model_registry
from deepscan.registry.dataset_registry import get_dataset_registry

model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

# Use any registered model
model = model_registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
# Or Llama, Mistral, etc.

# Use registered datasets
dataset = dataset_registry.get_dataset("tellme/beaver_tails_filtered", test_path="/path/to/test.csv")
```

> 💡 The built-in dataset loaders rely on Hugging Face `datasets` (included in core dependencies).

To use a copy saved via `datasets.save_to_disk`, pass a local path:

```python
dataset = dataset_registry.get_dataset(
    "beaver_tails",
    split="330k_train",
    path="/path/to/BeaverTails",
)
```

### 🏃 Model Runners

Model registry lookups now return a **model runner**—an object that exposes a uniform
`generate()` interface and keeps the underlying Hugging Face model/tokenizer handy.

```python
from deepscan.models.base_runner import GenerationRequest, PromptMessage, PromptContent
from deepscan.registry.model_registry import get_model_registry

runner = get_model_registry().get_model(
    "qwen3",
    model_name="Qwen3-8B",
    device="cuda",
)

# Quick text generation
response = runner.generate("Explain what a registry pattern is in two sentences.")
print(response.text)

# Chat-style prompt with structured messages
chat_request = GenerationRequest.from_messages(
    [
        PromptMessage(role="system", content=[PromptContent(text="You are a math tutor.")]),
        PromptMessage(role="user", content=[PromptContent(text="Help me factor x^2 + 5x + 6.")]),
    ],
    temperature=0.1,
    max_new_tokens=128,
)
chat_response = runner.generate(chat_request)
print(chat_response.text)
```

Runners keep the raw model/tokenizer accessible via `runner.model` / `runner.tokenizer`
so existing diagnostic code can still reach low-level APIs when necessary.

### 2. Load Configuration ⚙️

```python
from deepscan import ConfigLoader

# Load from file
config = ConfigLoader.from_file("config.yaml")

# Or create from dictionary
config = ConfigLoader.from_dict({
    "model": {"generation": "qwen3", "model_name": "Qwen3-8B", "device": "cuda"},
    "dataset": {"name": "beaver_tails", "split": "330k_train"},
    "evaluator": {"type": "tellme", "batch_size": 4},
})
```

### 3. Create and Use Evaluators 🔍

Evaluators are typically used through `run_from_config`, but can also be used programmatically:

```python
from deepscan.evaluators.registry import get_evaluator_registry
from deepscan.registry.model_registry import get_model_registry
from deepscan.registry.dataset_registry import get_dataset_registry

# Get registries
evaluator_registry = get_evaluator_registry()
model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

# Create evaluator from registry (pass options via config=)
evaluator = evaluator_registry.create_evaluator(
    "tellme",
    config=dict(batch_size=4, layer_ratio=0.6666, token_position=-1),
)

# Get model and dataset
model = model_registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
dataset = dataset_registry.get_dataset("tellme/beaver_tails_filtered", test_path="/path/to/test.csv")

# Run evaluation (typically done via run_from_config)
# results = evaluator.evaluate(model, dataset, ...)
```

### 4. Create Custom Evaluators 🛠️

```python
from deepscan.evaluators.base import BaseEvaluator
from deepscan.evaluators.registry import get_evaluator_registry

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, model, dataset, **kwargs):
        # Your custom evaluation logic here
        results = {}
        # ... implementation ...
        return results

# Register the evaluator
registry = get_evaluator_registry()
registry.register_evaluator("custom_eval")(CustomEvaluator)

# Use it in config or programmatically (pass options via config=)
evaluator = registry.create_evaluator("custom_eval", config=dict(param1=value1))
```

### 5. Summarize Results 📊

```python
from deepscan.summarizers.base import BaseSummarizer

class SimpleSummarizer(BaseSummarizer):
    def summarize(self, results, benchmark=None, **kwargs):
        # Minimal example: keep a small subset of keys
        return {
            "benchmark": benchmark,
            "keys": sorted(results.keys()),
        }

summarizer = SimpleSummarizer(name="simple")

summary = summarizer.summarize(results, benchmark="beaver_tails")

# Format as markdown
report = summarizer.format_report(summary, format="markdown")
print(report)
```

## 🏗️ Architecture

### Core Components

1. **📦 Registry System** (`deepscan/registry/`)
   - `BaseRegistry`: Generic registry pattern
   - `ModelRegistry`: Model registration and retrieval
   - `DatasetRegistry`: Dataset registration and retrieval

2. **⚙️ Configuration** (`deepscan/config/`)
   - `ConfigLoader`: Load and manage YAML/JSON configurations
   - Supports dot notation for nested access
   - Merge multiple configurations

3. **🔍 Evaluators** (`deepscan/evaluators/`)
   - `BaseEvaluator`: Abstract base class for all evaluators
   - `TellMeEvaluator`: Disentanglement metrics on BeaverTails
   - `XBoundaryEvaluator`: Safety boundary analysis
   - `MiPeaksEvaluator`: Model introspection and peak analysis
   - `SpinEvaluator`: Self-play fine-tuning evaluation
   - `EvaluatorRegistry`: Registry for evaluator classes

4. **📝 Summarizers** (`deepscan/summarizers/`)
   - `BaseSummarizer`: Abstract base class for all summarizers
   - `SummarizerRegistry`: Registry for summarizer classes
   - Multiple output formats (dict, JSON, Markdown, text)

## 🔧 Extending the Framework

### ➕ Adding a New Evaluator

1. Inherit from `BaseEvaluator`
2. Implement the `evaluate` method
3. Register it using the evaluator registry

```python
from deepscan.evaluators.base import BaseEvaluator
from deepscan.evaluators.registry import get_evaluator_registry

class MyEvaluator(BaseEvaluator):
    def evaluate(self, model, dataset, **kwargs):
        # Your evaluation logic
        results = {}
        # ... implementation ...
        return results

# Register
registry = get_evaluator_registry()
registry.register_evaluator("my_evaluator")(MyEvaluator)
```

### ➕ Adding a New Summarizer

1. Inherit from `BaseSummarizer`
2. Implement the `summarize` method
3. Register it using the summarizer registry

```python
from deepscan.summarizers.base import BaseSummarizer
from deepscan.summarizers.registry import get_summarizer_registry

class MySummarizer(BaseSummarizer):
    def summarize(self, results, benchmark=None, **kwargs):
        # Your summarization logic
        return {"summary": "..."}

# Register
registry = get_summarizer_registry()
registry.register_summarizer("my_summarizer")(MySummarizer)
```

## 📋 Example Configuration File

```yaml
# 📝 Minimal TELLME-style config (see `examples/config.tellme.yaml`; see `examples/` for multi-evaluator configs)
model:
  generation: qwen3
  model_name: Qwen3-8B
  device: cuda
  dtype: float16
  # Optional: point to a local checkpoint dir to avoid downloads
  # path: /path/to/models--Qwen--Qwen3-8B

dataset:
  name: tellme/beaver_tails_filtered
  test_path: /path/to/test.csv
  # train_path: /path/to/train.csv
  # max_rows: 400

evaluator:
  type: tellme
  batch_size: 4
  layer_ratio: 0.6666
  token_position: -1
```

## 📚 Examples

See the `examples/` directory for complete usage examples:

**🔍 Single-evaluator:**
- `config.tellme.yaml`: Minimal TELLME disentanglement metrics

**🔗 Multi-evaluator (same model, multiple benchmarks):**
- `config.x-boundary.tellme.spin.mi-peaks.qwen2.5-7b-instruct.yaml`: TELLME, X-Boundary, SPIN, and MI-Peaks with Qwen2.5-7B-Instruct
- `config.xboundary-llama3.3-70b-instruct.yaml`: Same suite with Llama 3.3 70B Instruct

## 💻 Development

```bash
# 📦 Install in development mode
pip install -e ".[dev]"

# ✅ Run tests
pytest

```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## 📧 Contact

For questions or suggestions, please contact us:

  📧 Email: [shaojing@pjlab.org.cn](mailto:shaojing@pjlab.org.cn)
